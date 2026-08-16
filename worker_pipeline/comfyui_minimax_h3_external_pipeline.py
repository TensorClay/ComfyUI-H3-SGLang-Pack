from __future__ import annotations

import os
from typing import Any

import torch

from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import MiniMaxH3Pipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MiniMaxH3DenoiseBranch,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
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

    ModelRegistry.register_model(
        "MiniMaxH3DiTModel",
        ComfyPrunedMiniMaxH3DiTModel
        if checkpoint_format == "comfy_int8_convrot"
        else ComfyBF16MiniMaxH3DiTModel,
    )


register_checkpoint_model()

CONTRACT_VERSION = 1


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
        return _resolve_denoise_model(
            self.transformer,
            device,
            placement_managed=placement_managed,
        )

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
        try:
            raw_context = payload["context"]
            video_shape = tuple(int(v) for v in payload["video_shape"])
            audio_shape = tuple(int(v) for v in payload["audio_shape"])
            refs = payload["refs"]
            packed = minimax_h3_packed_sequence_ref2va_blocks(
                text_len=int(raw_context.shape[1]),
                latent_t=video_shape[2],
                latent_h=video_shape[3],
                latent_w=video_shape[4],
                audio_t=audio_shape[-1],
                ref_blocks=refs,
            )
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
            }
            batch.noise_pred = torch.tensor([CONTRACT_VERSION], dtype=torch.int32)
            return batch
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
        scale = state["audio_scale"]
        audio_out = (
            (1.0 - scale) * audio_native
            + (1.0 + (scale - 1.0) * sigma_a).to(audio_native_output.dtype)
            * audio_native_output
        )
        batch.noise_pred = _pack_outputs(video_out, audio_out)
        return batch

    def _end_execution(self, batch, payload):
        execution_id = str(payload["execution_id"])
        if self._executions.pop(execution_id, None) is None:
            raise KeyError(f"unknown execution: {execution_id}")
        batch.noise_pred = torch.tensor([CONTRACT_VERSION], dtype=torch.int32)
        return batch


class ComfyUIMiniMaxH3ExternalPipeline(MiniMaxH3Pipeline):
    pipeline_name = "ComfyUIMiniMaxH3ExternalPipeline"
    _required_config_modules = ["transformer"]

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(
            ExternalH3ForwardStage(
                transformer=self.get_module("transformer"),
                pipeline=self,
            )
        )


EntryClass = ComfyUIMiniMaxH3ExternalPipeline
