from __future__ import annotations

import importlib
import json
import logging
import os
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any

import torch

from sglang.multimodal_gen.runtime.layers.lora.linear import (
    BaseLayerWithLoRA,
    MergedColumnParallelLinearWithLoRA,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import MiniMaxH3Pipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MiniMaxH3DenoiseBranch,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
    minimax_h3_patchify_video_latent,
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    MiniMaxH3DenoisingStage,
    _precompute_rope_cache,
    _resolve_denoise_model,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import VerificationResult
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3Attention,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def register_checkpoint_model() -> None:
    checkpoint_format = os.environ.get("COMFYUI_SGLANG_H3_CHECKPOINT_FORMAT")
    if checkpoint_format not in {"comfy_int8_convrot", "comfy_bf16"}:
        return
    from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

    from .comfyui_minimax_h3_pruned_model import (
        ComfyBF16MiniMaxH3DiTModel,
        ComfyPrunedMiniMaxH3DiTModel,
    )

    model_class = (
        ComfyPrunedMiniMaxH3DiTModel
        if checkpoint_format == "comfy_int8_convrot"
        else ComfyBF16MiniMaxH3DiTModel
    )
    ModelRegistry.register_model("MiniMaxH3DiTModel", model_class)

    if importlib.util.find_spec("cache_dit") is not None:
        from cache_dit import ForwardPattern
        from sglang.multimodal_gen.runtime.cache import cache_dit_integration

        cache_dit_integration._CUSTOM_BLOCK_ADAPTER_SPECS.setdefault(
            model_class.__name__,
            ("blocks", ForwardPattern.Pattern_3),
        )


register_checkpoint_model()

CONTRACT_VERSION = 1


def _attention_options() -> dict:
    value = json.loads(
        os.environ.get("COMFYUI_SGLANG_H3_ATTENTION_OPTIONS", "{}")
    )
    if not isinstance(value, dict):
        raise TypeError("MiniMax H3 attention options must be an object")
    return value


def _sage_function(mode: str):
    if mode == "auto":
        from sageattention import sageattn

        def function(q, k, v):
            return sageattn(
                q,
                k,
                v,
                is_causal=False,
                tensor_layout="NHD",
            )
    elif mode == "sageattn_qk_int8_pv_fp16_cuda":
        from sageattention import sageattn_qk_int8_pv_fp16_cuda

        def function(q, k, v):
            return sageattn_qk_int8_pv_fp16_cuda(
                q,
                k,
                v,
                is_causal=False,
                pv_accum_dtype="fp32",
                tensor_layout="NHD",
            )
    elif mode == "sageattn_qk_int8_pv_fp16_triton":
        from sageattention import sageattn_qk_int8_pv_fp16_triton

        def function(q, k, v):
            return sageattn_qk_int8_pv_fp16_triton(
                q,
                k,
                v,
                is_causal=False,
                tensor_layout="NHD",
            )
    elif mode in {
        "sageattn_qk_int8_pv_fp8_cuda",
        "sageattn_qk_int8_pv_fp8_cuda++",
    }:
        from sageattention import sageattn_qk_int8_pv_fp8_cuda

        accumulation = (
            "fp32+fp16"
            if mode.endswith("++")
            else "fp32+fp32"
        )

        def function(q, k, v):
            return sageattn_qk_int8_pv_fp8_cuda(
                q,
                k,
                v,
                is_causal=False,
                pv_accum_dtype=accumulation,
                tensor_layout="NHD",
            )
    elif mode in {"sageattn3", "sageattn3_per_block_mean"}:
        from sageattn3 import sageattn3_blackwell

        per_block_mean = mode == "sageattn3_per_block_mean"

        def function(q, k, v):
            output = sageattn3_blackwell(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=False,
                per_block_mean=per_block_mean,
            )
            return output.transpose(1, 2)
    else:
        raise ValueError(f"unsupported Sage Attention mode: {mode!r}")
    return function


class _SageAttentionImpl:
    def __init__(self, function) -> None:
        self.function = function

    def forward(
        self,
        query,
        key,
        value,
        attn_metadata,
        *,
        return_softmax_lse=False,
    ):
        del attn_metadata
        if return_softmax_lse:
            raise NotImplementedError(
                "MiniMax H3 Sage Attention does not return softmax LSE"
            )
        return self.function(query, key, value)

    def forward_varlen(
        self,
        query,
        key,
        value,
        *,
        cu_seqlens,
        max_seqlen,
        cu_seqlens_host=None,
    ):
        del max_seqlen
        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(item) for item in cu_seqlens.tolist())
        )
        output = torch.empty_like(query)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if start != stop:
                output[start:stop].copy_(
                    self.function(
                        query[start:stop].unsqueeze(0),
                        key[start:stop].unsqueeze(0),
                        value[start:stop].unsqueeze(0),
                    )[0]
                )
        return output


def _configure_sage_attention(model) -> None:
    if os.environ.get("COMFYUI_SGLANG_H3_ATTENTION_BACKEND") != "sage_attn":
        return
    options = _attention_options()
    model._resolve_attention_backend_once()
    function = _sage_function(options.get("sage_attention", "auto"))
    if not options.get("allow_compile", False):
        function = torch.compiler.disable()(function)
    count = 0
    for module in model.modules():
        if isinstance(module, MiniMaxH3Attention):
            module._attention_impl = _SageAttentionImpl(function)
            count += 1
    logging.info("MiniMax H3 Sage Attention configured on %d blocks", count)


def _parse_blocks(spec: str, count: int) -> frozenset[int]:
    blocks = set()
    for part in "".join(str(spec).split()).split(","):
        if not part:
            continue
        match = re.fullmatch(r"(-?\d+)(?:-(-?\d+))?", part)
        if match is None:
            raise ValueError(
                f"cannot parse block spec {part!r}; use indices and ranges "
                "like '0-3,47,-1'"
            )
        first = int(match.group(1))
        last = first if match.group(2) is None else int(match.group(2))
        first = first if first >= 0 else count + first
        last = last if last >= 0 else count + last
        if first > last:
            first, last = last, first
        blocks.update(range(max(first, 0), min(last, count - 1) + 1))
    return frozenset(blocks)


def _parse_tau_profile(spec: str, count: int) -> dict[int, float]:
    profile = {}
    for entry in re.split(r"[;\n]", str(spec)):
        entry = entry.split("#", 1)[0].strip()
        if not entry:
            continue
        blocks, separator, value = entry.partition("=")
        if not separator:
            raise ValueError(
                f"tau_profile entry {entry!r} needs '=', e.g. '39-42=0.9'"
            )
        try:
            level = float(value)
        except ValueError:
            raise ValueError(
                f"tau_profile entry {entry!r} has a non-numeric tau"
            )
        for block in _parse_blocks(blocks, count):
            profile[block] = level
    return profile


def _load_sol_kernels():
    provider = Path(os.environ["COMFYUI_SGLANG_H3_SOL_PROVIDER"])
    if not (provider / "_tri_fwd.py").is_file():
        raise RuntimeError(f"Sol Attention provider is incomplete: {provider}")
    package_name = "_comfyui_h3_sol_provider"
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [str(provider)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    bf16 = importlib.import_module(f"{package_name}._tri_fwd").sol_attn
    int8 = importlib.import_module(f"{package_name}._int8_fwd").sol_attn_int8
    return bf16, int8


_SOL_FAILURES = set()


class _SolAttentionImpl:
    def __init__(
        self,
        dense_impl,
        kernels,
        softmax_scale: float,
        block_index: int,
        options: dict,
        dense_blocks: frozenset[int],
        tau_profile: dict[int, float],
    ) -> None:
        self.dense_impl = dense_impl
        self.kernel = kernels[1] if options.get("int8_qk", True) else kernels[0]
        self.softmax_scale = softmax_scale
        self.block_index = block_index
        self.tau = tau_profile.get(block_index, float(options.get("tau", 1.3)))
        self.min_tokens = int(options.get("min_tokens", 4096))
        self.int8_qk = bool(options.get("int8_qk", True))
        self.int8_pv = bool(options.get("int8_pv", True))
        self.use_tma = bool(options.get("use_tma", False))
        self.verbose = bool(options.get("verbose", False))
        self.sink_conditioning = options.get(
            "sink_conditioning",
            "exact_kv_and_rows",
        )
        self.dense_block = block_index in dense_blocks
        self.sigma_start = float(options["sigma_start"])
        self.sigma_end = float(options["sigma_end"])
        self.active = False
        self.sink_blocks = (0, 0)
        self.sink_q = (0, 0)

    def set_sigma(self, sigma: float) -> None:
        self.active = self.sigma_end <= sigma <= self.sigma_start

    def set_conditioning_rows(self, stop: int) -> None:
        if self.sink_conditioning == "off":
            self.sink_blocks = self.sink_q = (0, 0)
            return
        self.sink_blocks = (0, (stop + 63) // 64)
        self.sink_q = (
            self.sink_blocks
            if self.sink_conditioning == "exact_kv_and_rows"
            else (0, 0)
        )

    def _dense(self, query, key, value):
        return self.dense_impl.forward(
            query,
            key,
            value,
            None,
        )

    def _segment(self, query, key, value):
        if (
            not self.active
            or self.dense_block
            or query.shape[1] < self.min_tokens
        ):
            return self._dense(query, key, value)
        arguments = {
            "scale": self.softmax_scale,
            "tau": self.tau,
            "sink_blocks": self.sink_blocks,
            "sink_q": self.sink_q,
            "use_tma": self.use_tma,
        }
        if self.int8_qk:
            arguments["int8_pv"] = self.int8_pv
        try:
            output = self.kernel(query, key, value, **arguments)
            message = ("sparse", self.block_index, query.shape[1])
            if (
                self.verbose
                and self.block_index == 0
                and message not in _SOL_FAILURES
            ):
                _SOL_FAILURES.add(message)
                logging.info(
                    "MiniMax H3 Sol-Attn active on block %d (%d tokens)",
                    self.block_index,
                    query.shape[1],
                )
            return output
        except Exception as error:
            message = (type(error).__name__, str(error))
            if message not in _SOL_FAILURES:
                _SOL_FAILURES.add(message)
                logging.exception(
                    "MiniMax H3 Sol-Attn failed; using the dense backend"
                )
            return self._dense(query, key, value)

    def forward(
        self,
        query,
        key,
        value,
        attn_metadata,
        *,
        return_softmax_lse=False,
    ):
        del attn_metadata
        if return_softmax_lse:
            raise NotImplementedError(
                "MiniMax H3 Sol-Attn does not return softmax LSE"
            )
        return self._segment(query, key, value)

    def forward_varlen(
        self,
        query,
        key,
        value,
        *,
        cu_seqlens,
        max_seqlen,
        cu_seqlens_host=None,
    ):
        del max_seqlen
        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(item) for item in cu_seqlens.tolist())
        )
        output = torch.empty_like(query)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if start != stop:
                output[start:stop].copy_(
                    self._segment(
                        query[start:stop].unsqueeze(0),
                        key[start:stop].unsqueeze(0),
                        value[start:stop].unsqueeze(0),
                    )[0]
                )
        return output


def _configure_sol_attention(model) -> None:
    if os.environ.get("COMFYUI_SGLANG_H3_ATTENTION_BACKEND") != "sol_attn":
        return
    options = _attention_options()
    model._resolve_attention_backend_once()
    kernels = _load_sol_kernels()
    blocks = model.blocks
    dense_blocks = _parse_blocks(options.get("dense_blocks", ""), len(blocks))
    tau_profile = _parse_tau_profile(
        options.get("tau_profile", ""),
        len(blocks),
    )
    for index, block in enumerate(blocks):
        module = block.attn
        module._attention_impl = _SolAttentionImpl(
            module._attention_impl,
            kernels,
            module.softmax_scale,
            index,
            options,
            dense_blocks,
            tau_profile,
        )
    logging.info("MiniMax H3 Sol-Attn configured on %d blocks", len(blocks))


def _configure_attention(model) -> None:
    _configure_sage_attention(model)
    _configure_sol_attention(model)


def _set_sol_conditioning_sink(model, video_target_start: int) -> None:
    for block in model.blocks:
        impl = block.attn._attention_impl
        if isinstance(impl, _SolAttentionImpl):
            impl.set_conditioning_rows(video_target_start)


def _set_sol_sigma(model, sigma: float) -> None:
    for block in model.blocks:
        impl = block.attn._attention_impl
        if isinstance(impl, _SolAttentionImpl):
            impl.set_sigma(sigma)


def _pack_outputs(video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (video.reshape(video.shape[0], -1), audio.reshape(audio.shape[0], -1)),
        dim=1,
    )


def _time_shift_sigma(
    sigma: torch.Tensor, from_shift: float, to_shift: float
) -> torch.Tensor:
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def _video_condition_rows(
    latents: list[torch.Tensor],
    *,
    seed: int,
    noise_aug: float,
) -> torch.Tensor:
    rows = []
    for latent in latents:
        packed = minimax_h3_patchify_video_latent(
            latent.to(dtype=torch.float32), patch_size=(1, 2, 2)
        )
        if noise_aug < 1.0:
            generator = torch.Generator("cpu").manual_seed(seed)
            noise = torch.randn(
                packed.shape, generator=generator, dtype=torch.float32
            )
            packed = noise_aug * packed + (1.0 - noise_aug) * noise
        rows.append(packed)
    if not rows:
        return torch.empty((0, 96), dtype=torch.float32)
    return torch.cat(rows, dim=0).contiguous()


def _audio_condition_rows(
    latents: list[torch.Tensor],
    *,
    seed: int,
    noise_aug: float,
) -> torch.Tensor:
    rows = []
    for latent in latents:
        packed = (
            latent.to(dtype=torch.float32)[0]
            .permute(1, 2, 0)
            .reshape(-1, latent.shape[1])
        )
        if noise_aug < 1.0:
            generator = torch.Generator("cpu").manual_seed(seed + 1)
            noise = torch.randn(
                packed.shape, generator=generator, dtype=torch.float32
            )
            packed = noise_aug * packed + (1.0 - noise_aug) * noise
        rows.append(packed)
    if not rows:
        return torch.empty((0, 32), dtype=torch.float32)
    return torch.cat(rows, dim=0).contiguous()


class ExternalH3ForwardStage(MiniMaxH3DenoisingStage):
    def __init__(self, transformer: Any, pipeline: Any) -> None:
        super().__init__(transformer=transformer, pipeline=pipeline)
        self._executions: dict[str, dict[str, Any]] = {}
        self._attention_configured = False

    def verify_input(self, batch, server_args):
        return VerificationResult()

    def verify_output(self, batch, server_args):
        return VerificationResult()

    def forward(self, batch, server_args):
        payload = batch.extra.get("external_h3")
        if not isinstance(payload, dict):
            raise ValueError("external_h3 payload is required")
        if payload.get("contract_version") != CONTRACT_VERSION:
            raise ValueError(
                f"contract version {payload.get('contract_version')} != "
                f"{CONTRACT_VERSION}"
            )
        action = payload.get("action")
        if action == "begin_execution":
            return self._begin_execution(batch, payload)
        if action == "evaluate":
            return self._evaluate(batch, payload)
        if action == "end_execution":
            return self._end_execution(batch, payload)
        raise ValueError(f"unsupported external_h3 action: {action!r}")

    def _model(self, batch):
        device = torch.device("cuda")
        placement_managed = self._component_residency_manager is not None
        if placement_managed:
            self._manage_dit_use_site(self.transformer, "transformer", batch)
        model = _resolve_denoise_model(
            self.transformer,
            device,
            placement_managed=placement_managed,
        )
        if not self._attention_configured:
            _configure_attention(model)
            self._attention_configured = True
        return model

    @staticmethod
    def _cache_parallel_groups():
        from sglang.multimodal_gen.runtime.distributed import (
            get_sp_group,
            get_tp_group,
            get_world_size,
        )

        if get_world_size() <= 1:
            return None, None
        sp_group = get_sp_group()
        tp_group = get_tp_group()
        sp = sp_group.device_group if sp_group.world_size > 1 else None
        tp = tp_group.device_group if tp_group.world_size > 1 else None
        return sp, tp

    def _enable_cache_dit(self, model, options: dict, num_steps: int) -> None:
        try:
            from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
                CacheDitConfig,
                enable_cache_on_transformer,
            )
        except ImportError as error:
            raise RuntimeError(
                "MiniMax H3 Cache-DiT requires the optional cache-dit package"
            ) from error
        sp_group, tp_group = self._cache_parallel_groups()
        enable_cache_on_transformer(
            model,
            CacheDitConfig(
                enabled=True,
                Fn_compute_blocks=1,
                Bn_compute_blocks=0,
                max_warmup_steps=int(options["max_warmup_steps"]),
                residual_diff_threshold=float(
                    options["residual_diff_threshold"]
                ),
                max_continuous_cached_steps=int(
                    options["max_continuous_cached_steps"]
                ),
                num_inference_steps=num_steps,
            ),
            model_name="minimax_h3",
            sp_group=sp_group,
            tp_group=tp_group,
        )

    @staticmethod
    def _disable_cache_dit(model) -> None:
        from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
            disable_cache_on_transformer,
        )

        disable_cache_on_transformer(model)

    def _refine_context(self, model, context: torch.Tensor, device) -> torch.Tensor:
        if context.ndim != 3 or context.shape[0] != 1:
            raise ValueError(
                f"context must be [1,L,D], got {list(context.shape)}"
            )
        if context.shape[-1] == 5376:
            return context[0].to(device=device, dtype=torch.bfloat16)
        if context.shape[-1] != 5120:
            raise ValueError(
                f"context width must be 5120 or 5376, got {context.shape[-1]}"
            )
        length = int(context.shape[1])
        cu_seqlens = torch.tensor([0, length], dtype=torch.int32, device=device)
        return model.refine_prompt_embeds(
            context[0],
            cu_seqlens,
            device=device,
        )

    def _begin_execution(self, batch, payload):
        execution_id = str(payload["execution_id"])
        if execution_id in self._executions:
            raise ValueError(f"execution already exists: {execution_id}")
        model = self._model(batch)
        cache_enabled = False
        try:
            cache_options = payload.get("cache_dit")
            if cache_options is not None:
                self._enable_cache_dit(
                    model,
                    cache_options,
                    int(payload["num_inference_steps"]),
                )
                cache_enabled = True
            raw_context = payload["context"]
            video_shape = tuple(int(v) for v in payload["video_shape"])
            audio_shape = tuple(int(v) for v in payload["audio_shape"])
            refs = payload["refs"]
            keyframes = payload["keyframes"]
            model_variant = payload["model_variant"]
            if refs and keyframes:
                raise ValueError("H3 conditioning cannot mix references and keyframes")
            if model_variant == "ref2va":
                if keyframes:
                    raise ValueError("Ref2VA checkpoints do not accept keyframes")
                packed = minimax_h3_packed_sequence_ref2va_blocks(
                    text_len=int(raw_context.shape[1]),
                    latent_t=video_shape[2],
                    latent_h=video_shape[3],
                    latent_w=video_shape[4],
                    audio_t=audio_shape[-1],
                    ref_blocks=refs,
                )
            elif model_variant == "fl2va":
                if refs:
                    raise ValueError("FL2VA checkpoints do not accept references")
                packed = minimax_h3_packed_sequence(
                    text_len=int(raw_context.shape[1]),
                    latent_t=video_shape[2],
                    latent_h=video_shape[3],
                    latent_w=video_shape[4],
                    audio_t=audio_shape[-1],
                    include_keyframe_cond=bool(keyframes),
                    keyframe_frame_indices=(
                        [
                            keyframe["frame_index"]
                            for keyframe in keyframes
                        ]
                        if keyframes
                        else None
                    ),
                    frame_count=payload.get("frame_count"),
                )
            else:
                raise ValueError(f"unsupported MiniMax H3 variant: {model_variant!r}")
            token_tags = packed["token_tags"]
            text_pos = packed["text_pos"].view(-1)
            text_tags = payload["text_token_tags"].view(-1).to(torch.long)
            if text_tags.numel() != text_pos.numel():
                raise ValueError("text token tag count does not match text rows")
            token_tags[text_pos] = text_tags
            device = torch.device("cuda")
            context = self._refine_context(model, raw_context, device)
            positive = MiniMaxH3DenoiseBranch(
                packed=packed,
                text_embeddings=context,
                token_tags=token_tags,
                device=device,
            )
            positive.static_kwargs["refined_prompt_embeds_length"] = int(
                context.shape[0]
            )
            _set_sol_conditioning_sink(model, positive.video_target_start)
            _precompute_rope_cache(model, positive, device=device)
            video_condition = _video_condition_rows(
                payload.get("cond_video_latents", []),
                seed=int(payload["seed"]),
                noise_aug=float(payload["visual_cond_noise_aug"]),
            ).to(device)
            if video_condition.shape[0] != positive.video_target_start:
                raise ValueError(
                    f"condition video rows {video_condition.shape[0]} != "
                    f"layout rows {positive.video_target_start}"
                )
            audio_condition = _audio_condition_rows(
                payload.get("cond_audio_latents", []),
                seed=int(payload["seed"]),
                noise_aug=float(payload["audio_cond_noise_aug"]),
            ).to(device)
            if audio_condition.shape[0] != positive.audio_target_start:
                raise ValueError(
                    f"condition audio rows {audio_condition.shape[0]} != "
                    f"layout rows {positive.audio_target_start}"
                )
            self._executions[execution_id] = {
                "positive": positive,
                "video_condition": video_condition,
                "audio_condition": audio_condition,
                "video_shape": video_shape,
                "audio_shape": audio_shape,
                "video_shift": float(payload["video_shift"]),
                "audio_shift": float(payload["audio_shift"]),
                "audio_scale": float(payload["audio_scale"]),
                "visual_cond_noise_aug": float(payload["visual_cond_noise_aug"]),
                "audio_cond_noise_aug": float(payload["audio_cond_noise_aug"]),
                "cache_model": model if cache_enabled else None,
            }
            batch.noise_pred = torch.tensor([CONTRACT_VERSION], dtype=torch.int32)
            return batch
        except Exception:
            if cache_enabled:
                self._disable_cache_dit(model)
            raise
        finally:
            self._finish_active_component_use()

    def _evaluate(self, batch, payload):
        execution_id = str(payload["execution_id"])
        state = self._executions.get(execution_id)
        if state is None:
            raise KeyError(f"unknown execution: {execution_id}")
        sigma_v = torch.tensor(
            float(payload["sigma_video"]), dtype=torch.float32, device="cuda"
        ).clamp(min=1e-6)
        sigma_a = _time_shift_sigma(
            sigma_v, state["video_shift"], state["audio_shift"]
        )
        carry = sigma_a / sigma_v

        video_x = payload["video_x"].to(device="cuda", dtype=torch.bfloat16)
        audio_carried = payload["audio_x_carried"].to(
            device="cuda", dtype=torch.bfloat16
        )
        scale = state["audio_scale"]
        audio_native = audio_carried
        if scale != 1.0:
            audio_native = audio_carried * carry.to(audio_carried.dtype)
        video_rows = minimax_h3_patchify_video_latent(
            video_x.to(torch.float32), patch_size=(1, 2, 2)
        )
        audio_rows = (
            audio_native[0]
            .permute(1, 2, 0)
            .reshape(-1, audio_native.shape[1])
            .to(torch.float32)
            .contiguous()
        )
        full_video_rows = torch.cat(
            (state["video_condition"], video_rows), dim=0
        )
        full_audio_rows = torch.cat(
            (state["audio_condition"], audio_rows), dim=0
        )
        t_v = float(1.0 - sigma_v)
        t_a = float(1.0 - sigma_a)
        plan = state["positive"].prepare_timestep_plan(
            video_timesteps=[t_v],
            audio_timesteps=[t_a],
            imgvid_cond_noise_aug=state["visual_cond_noise_aug"],
            audio_ref_cond_noise_aug=state["audio_cond_noise_aug"],
        )
        call_kwargs = state["positive"].forward_kwargs(
            video_rows=full_video_rows,
            audio_rows=full_audio_rows,
            step_timesteps=plan[0],
        )
        model = self._model(batch)
        _set_sol_sigma(model, float(sigma_v))
        try:
            with set_forward_context(
                current_timestep=int(payload["call_index"]),
                attn_metadata=None,
                forward_batch=batch,
            ):
                video_logits, audio_logits = model(**call_kwargs)
        finally:
            self._finish_active_component_use()

        video_out = -minimax_h3_unpatchify_video_tokens(
            video_logits,
            latent_shape=(
                state["video_shape"][2],
                state["video_shape"][3] // 2,
                state["video_shape"][4] // 2,
                state["video_shape"][1],
            ),
            patch_size=(1, 2, 2),
        ).to(torch.bfloat16)
        audio_native_output = -minimax_h3_unpack_audio_tokens(
            audio_logits[state["positive"].audio_target_slice],
            audio_t=state["audio_shape"][-1] * state["audio_shape"][2],
            audio_channel=state["audio_shape"][2],
        ).permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16)
        audio_out = audio_native_output
        if scale != 1.0:
            audio_out = (
                (1.0 - scale) * audio_native
                + (1.0 + (scale - 1.0) * sigma_a).to(
                    audio_native_output.dtype
                )
                * audio_native_output
            )
        batch.noise_pred = _pack_outputs(video_out, audio_out)
        return batch

    def _end_execution(self, batch, payload):
        execution_id = str(payload["execution_id"])
        state = self._executions.pop(execution_id, None)
        if state is None:
            raise KeyError(f"unknown execution: {execution_id}")
        if state["cache_model"] is not None:
            self._disable_cache_dit(state["cache_model"])
        batch.noise_pred = torch.tensor([CONTRACT_VERSION], dtype=torch.int32)
        return batch


class ComfyUIMiniMaxH3ExternalPipeline(MiniMaxH3Pipeline):
    pipeline_name = "ComfyUIMiniMaxH3ExternalPipeline"
    _required_config_modules = ["transformer"]

    def _apply_lora_to_layers(
        self,
        lora_layers: dict[str, BaseLayerWithLoRA],
        lora_nicknames: list[str],
        lora_paths: list[str | None],
        rank: int,
        strengths: list[float],
        clear_existing: bool = False,
        merge_weights: bool = True,
    ) -> int:
        for name, layer in lora_layers.items():
            if not isinstance(layer, MergedColumnParallelLinearWithLoRA):
                continue
            key = name + ".lora_B"
            output_sizes = layer.base_layer.output_sizes
            for nickname in lora_nicknames:
                weights = self.lora_adapters[nickname]
                weight = weights.get(key)
                if weight is None or weight.ndim != 2:
                    continue
                if weight.shape[0] != sum(output_sizes):
                    raise ValueError(
                        f"LoRA {key} rows {weight.shape[0]} do not match "
                        f"H3 fused output size {sum(output_sizes)}"
                    )
                weights[key] = torch.stack(
                    weight.split(output_sizes, dim=0), dim=0
                )
        return super()._apply_lora_to_layers(
            lora_layers,
            lora_nicknames,
            lora_paths,
            rank,
            strengths,
            clear_existing=clear_existing,
            merge_weights=merge_weights,
        )

    def unmerge_lora_weights(self, target: str = "all") -> None:
        self.deactivate_lora_weights(target)
        if target in {"all", "transformer"}:
            self.cur_adapter_name.pop("transformer", None)
            self.cur_adapter_path.pop("transformer", None)
        if target in {"all", "transformer_2"}:
            self.cur_adapter_name.pop("transformer_2", None)
            self.cur_adapter_path.pop("transformer_2", None)
        if target in {"all", "critic"}:
            self.cur_adapter_name.pop("critic", None)
            self.cur_adapter_path.pop("critic", None)

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(
            ExternalH3ForwardStage(
                transformer=self.get_module("transformer"),
                pipeline=self,
            )
        )


EntryClass = ComfyUIMiniMaxH3ExternalPipeline
