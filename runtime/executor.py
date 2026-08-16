from __future__ import annotations

from collections import OrderedDict
import logging
import threading
import uuid

import torch

from comfy import patcher_extension
import comfy.ldm.common_dit
from comfy.ldm.minimax.model import MiniMaxH3Model

from .protocol import (
    CONTRACT_VERSION,
    ExecutionSignature,
    sanitize_keyframes,
    sanitize_refs,
    unpack_outputs,
)


LOGGER = logging.getLogger(__name__)
CACHE_DIT_OPTION = "sglang_h3_cache_dit"
LORAS_OPTION = "sglang_h3_loras"


def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").contiguous()


class _RemoteH3Attention(torch.nn.Module):
    pass


class _RemoteH3Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _RemoteH3Attention()


def _block_count(parameter_keys: tuple[str, ...]) -> int:
    indices = []
    for key in parameter_keys:
        parts = key.split(".", 2)
        if len(parts) > 1 and parts[0] == "blocks" and parts[1].isdigit():
            indices.append(int(parts[1]))
    return max(indices, default=49) + 1


class H3SGLangExecutor(MiniMaxH3Model):
    def __init__(self, runtime, parameter_keys: tuple[str, ...] = ()) -> None:
        torch.nn.Module.__init__(self)
        self.blocks = torch.nn.ModuleList(
            _RemoteH3Block() for _ in range(_block_count(parameter_keys))
        )
        self.runtime = runtime
        self._parameter_keys = parameter_keys
        self.dtype = torch.bfloat16
        self._execution_id: str | None = None
        self._signature: ExecutionSignature | None = None
        self._video_shape: tuple[int, ...] | None = None
        self._last_sigma: float | None = None
        self._call_index = 0
        self._lock = threading.RLock()

    def state_dict(
        self, *args, destination=None, prefix="", keep_vars=False
    ):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        placeholder = torch.empty(0)
        for key in self._parameter_keys:
            destination[prefix + key] = placeholder
        return destination

    def preprocess_text_embeds(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 3 or context.shape[-1] not in (5120, 5376):
            raise ValueError(
                "MiniMax H3 context must be [B,L,5120] raw Qwen states or "
                f"[B,L,5376] refined states, got {tuple(context.shape)}"
            )
        # Refinement is part of the SGLang transformer and runs once on workers
        # during begin_execution. The standard ComfyUI BaseModel still owns
        # conditioning collection and passes this tensor through c_crossattn.
        return context

    def _reject_unsupported(
        self,
        *,
        control,
        transformer_options: dict,
    ) -> None:
        if control is not None:
            raise NotImplementedError(
                "ControlNet-style control tensors are not yet supported by "
                "the MiniMax H3 SGLang executor"
            )
        if transformer_options.get("optimized_attention_override") is not None:
            raise NotImplementedError(
                "ComfyUI attention override callbacks cannot run in SGLang worker "
                "processes; use this pack's MiniMax H3 attention patch nodes"
            )
        if transformer_options.get("patches"):
            raise NotImplementedError(
                "ComfyUI transformer patches cannot run in SGLang worker processes"
            )
        if any(transformer_options.get("patches_replace", {}).values()):
            raise NotImplementedError(
                "ComfyUI transformer replacement patches cannot run in SGLang "
                "worker processes"
            )

    def _finish_execution(self) -> None:
        if self._execution_id is None or self._video_shape is None:
            return
        execution_id = self._execution_id
        video_shape = self._video_shape
        self._execution_id = None
        self._signature = None
        self._video_shape = None
        self._last_sigma = None
        self._call_index = 0
        try:
            self.runtime.end_execution(execution_id, video_shape)
        except Exception:
            LOGGER.exception("Failed to release SGLang H3 execution %s", execution_id)

    def _begin_execution(
        self,
        *,
        context: torch.Tensor,
        video: torch.Tensor,
        audio: torch.Tensor,
        payload: dict,
        transformer_options: dict,
        signature: ExecutionSignature,
    ) -> None:
        self._finish_execution()
        execution_id = str(uuid.uuid4())
        self.runtime.configure_loras(
            transformer_options.get(LORAS_OPTION, [])
        )
        video_shape = tuple(int(value) for value in video.shape)
        audio_shape = tuple(int(value) for value in audio.shape)
        begin = {
            "contract_version": CONTRACT_VERSION,
            "action": "begin_execution",
            "execution_id": execution_id,
            "model_variant": self.runtime.model_variant,
            "context": _cpu(context),
            "text_token_tags": _cpu(payload["text_token_tags"]),
            "cond_video_latents": [
                _cpu(value) for value in payload.get("cond_video_latents", [])
            ],
            "cond_audio_latents": [
                _cpu(value) for value in payload.get("cond_audio_latents", [])
            ],
            "refs": sanitize_refs(payload.get("refs", [])),
            "keyframes": sanitize_keyframes(
                payload.get("keyframes", []),
                payload.get("frame_count"),
            ),
            "frame_count": payload.get("frame_count"),
            "video_shape": video_shape,
            "audio_shape": audio_shape,
            "seed": int(payload.get("seed", 0)),
            "visual_cond_noise_aug": float(
                payload.get("visual_cond_noise_aug", 0.999)
            ),
            "audio_cond_noise_aug": float(
                payload.get("audio_cond_noise_aug", 1.0)
            ),
            "video_shift": float(
                transformer_options.get("minimax_h3_sigma_shift_video", 12.0)
            ),
            "audio_shift": float(
                transformer_options.get("minimax_h3_sigma_shift_audio", 3.0)
            ),
            "audio_scale": float(payload.get("audio_scale", 1.0)),
            "cache_dit": transformer_options.get(CACHE_DIT_OPTION),
            "num_inference_steps": max(
                1,
                int(transformer_options["sample_sigmas"].numel()) - 1,
            ),
        }
        self.runtime.send(begin, video_shape)
        self._execution_id = execution_id
        self._signature = signature
        self._video_shape = video_shape
        self._last_sigma = None
        self._call_index = 0

    def forward(
        self,
        x,
        timestep,
        context,
        control=None,
        transformer_options=None,
        minimax_payload=None,
        **kwargs,
    ):
        transformer_options = transformer_options or {}
        return patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            patcher_extension.get_all_wrappers(
                patcher_extension.WrappersMP.DIFFUSION_MODEL,
                transformer_options,
            ),
        ).execute(
            x,
            timestep,
            context,
            control=control,
            transformer_options=transformer_options,
            minimax_payload=minimax_payload,
            **kwargs,
        )

    def _forward(
        self,
        x,
        timestep,
        context,
        control=None,
        transformer_options=None,
        minimax_payload=None,
        **kwargs,
    ):
        transformer_options = transformer_options or {}
        self._reject_unsupported(
            control=control,
            transformer_options=transformer_options,
        )
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("MiniMax H3 requires [video, audio] latent streams")
        if not isinstance(minimax_payload, dict):
            raise ValueError("MiniMax H3 minimax_payload is required")
        if "text_token_tags" not in minimax_payload:
            raise ValueError("MiniMax H3 text token tags are required")
        video, audio = x
        if video.shape[0] != 1 or audio.shape[0] != 1:
            raise ValueError("MiniMax H3 supports batch size 1")
        original_video_shape = tuple(int(value) for value in video.shape)
        signature = ExecutionSignature.from_inputs(
            context,
            video,
            audio,
            minimax_payload,
            transformer_options,
        )
        video = comfy.ldm.common_dit.pad_to_patch_size(video, (1, 2, 2))
        sigma = float((timestep.detach().flatten()[0].float() / 1000.0).cpu())

        with self._lock:
            is_new = (
                self._execution_id is None
                or self._signature != signature
                or (
                    self._last_sigma is not None
                    and sigma > self._last_sigma + 1e-6
                )
            )
            if is_new:
                self._begin_execution(
                    context=context,
                    video=video,
                    audio=audio,
                    payload=minimax_payload,
                    transformer_options=transformer_options,
                    signature=signature,
                )
            evaluate = {
                "contract_version": CONTRACT_VERSION,
                "action": "evaluate",
                "execution_id": self._execution_id,
                "call_index": self._call_index,
                "sigma_video": sigma,
                "video_x": _cpu(video),
                "audio_x_carried": _cpu(audio),
            }
            try:
                output = self.runtime.send(evaluate, tuple(video.shape))
            except Exception:
                self._finish_execution()
                raise
            if not torch.is_tensor(output.noise_pred):
                raise RuntimeError("SGLang H3 executor returned no tensor")
            video_out, audio_out = unpack_outputs(
                output.noise_pred,
                tuple(int(value) for value in video.shape),
                tuple(int(value) for value in audio.shape),
            )
            self._last_sigma = sigma
            self._call_index += 1
            video_out = video_out[
                :,
                :,
                : original_video_shape[2],
                : original_video_shape[3],
                : original_video_shape[4],
            ]
            return [
                video_out.to(device=video.device, dtype=video.dtype),
                audio_out.to(device=audio.device, dtype=audio.dtype),
            ]

    def close(self) -> None:
        with self._lock:
            self._finish_execution()
