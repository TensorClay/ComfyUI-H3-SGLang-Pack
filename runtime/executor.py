from __future__ import annotations

import logging
import threading
import uuid

import torch

from .protocol import (
    CONTRACT_VERSION,
    ExecutionSignature,
    sanitize_refs,
    unpack_outputs,
)


LOGGER = logging.getLogger(__name__)


def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").contiguous()


class H3SGLangExecutor(torch.nn.Module):
    def __init__(self, runtime) -> None:
        super().__init__()
        self.runtime = runtime
        self.dtype = torch.bfloat16
        self._execution_id: str | None = None
        self._signature: ExecutionSignature | None = None
        self._video_shape: tuple[int, ...] | None = None
        self._last_sigma: float | None = None
        self._call_index = 0
        self._lock = threading.RLock()

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
        dit_patches = transformer_options.get("patches_replace", {}).get("dit", {})
        if dit_patches:
            raise NotImplementedError(
                "DiT patches_replace is not yet supported by the distributed "
                "MiniMax H3 executor"
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
        video_shape = tuple(int(value) for value in video.shape)
        audio_shape = tuple(int(value) for value in audio.shape)
        begin = {
            "contract_version": CONTRACT_VERSION,
            "action": "begin_execution",
            "execution_id": execution_id,
            "context": _cpu(context),
            "text_token_tags": _cpu(payload["text_token_tags"]),
            "cond_video_latents": [
                _cpu(value) for value in payload.get("cond_video_latents", [])
            ],
            "cond_audio_latents": [
                _cpu(value) for value in payload.get("cond_audio_latents", [])
            ],
            "refs": sanitize_refs(payload.get("refs", [])),
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
        sigma = float((timestep.detach().flatten()[0].float() / 1000.0).cpu())
        signature = ExecutionSignature.from_inputs(
            context, video, audio, minimax_payload
        )

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
            output = self.runtime.send(evaluate, tuple(video.shape))
            if not torch.is_tensor(output.noise_pred):
                raise RuntimeError("SGLang H3 executor returned no tensor")
            video_out, audio_out = unpack_outputs(
                output.noise_pred,
                tuple(int(value) for value in video.shape),
                tuple(int(value) for value in audio.shape),
            )
            self._last_sigma = sigma
            self._call_index += 1
            return [
                video_out.to(device=video.device, dtype=video.dtype),
                audio_out.to(device=audio.device, dtype=audio.dtype),
            ]

    def close(self) -> None:
        with self._lock:
            self._finish_execution()
