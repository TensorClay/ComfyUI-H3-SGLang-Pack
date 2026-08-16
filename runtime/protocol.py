from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


CONTRACT_VERSION = 1
PIPELINE_NAME = "ComfyUIMiniMaxH3ExternalPipeline"


def unpack_outputs(
    packed: torch.Tensor,
    video_shape: tuple[int, ...],
    audio_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    video_values = math.prod(video_shape[1:])
    audio_values = math.prod(audio_shape[1:])
    if packed.ndim != 2 or packed.shape[0] != video_shape[0]:
        raise ValueError(
            f"packed output must be [B,N], got {tuple(packed.shape)}"
        )
    if packed.shape[1] != video_values + audio_values:
        raise ValueError(
            f"packed output has {packed.shape[1]} values per batch; expected "
            f"{video_values + audio_values}"
        )
    video = packed[:, :video_values].reshape(video_shape)
    audio = packed[:, video_values:].reshape(audio_shape)
    return video, audio


def sanitize_refs(refs: list[dict[str, Any]]) -> list[dict[str, int | str]]:
    allowed = {
        "kind",
        "latent_t",
        "latent_h",
        "latent_w",
        "ref_audio_t",
        "resolved_frame_index",
    }
    return [
        {
            key: int(value) if key != "kind" else str(value)
            for key, value in ref.items()
            if key in allowed and value is not None
        }
        for ref in refs
    ]


def sanitize_keyframes(
    keyframes: list[dict[str, Any]],
    frame_count: int | None,
) -> list[dict[str, int]]:
    if not keyframes:
        return []
    if not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError("frame_count is required with MiniMax H3 keyframes")
    out = []
    for keyframe in keyframes:
        resolved = int(keyframe["resolved_frame_index"])
        if resolved == 0:
            frame_index = 0
        elif resolved == frame_count - 1:
            frame_index = -1
        else:
            raise ValueError("MiniMax H3 only supports first/last keyframes")
        out.append({"frame_index": frame_index})
    return out


@dataclass(frozen=True)
class ExecutionSignature:
    seed: int
    context_shape: tuple[int, ...]
    context_identity: int
    video_shape: tuple[int, ...]
    audio_shape: tuple[int, ...]
    reference_shapes: tuple[tuple[int, ...], ...]
    reference_identities: tuple[int, ...]
    text_tags_shape: tuple[int, ...]
    text_tags_identity: int
    keyframe_indices: tuple[int, ...]
    execution_options: tuple[Any, ...]

    @classmethod
    def from_inputs(
        cls,
        context: torch.Tensor,
        video: torch.Tensor,
        audio: torch.Tensor,
        payload: dict[str, Any],
        transformer_options: dict[str, Any] | None = None,
    ) -> "ExecutionSignature":
        transformer_options = transformer_options or {}
        references = tuple(
            tensor
            for key in ("cond_video_latents", "cond_audio_latents")
            for tensor in payload.get(key, [])
        )
        refs = tuple(
            tuple(int(v) for v in tensor.shape) for tensor in references
        )
        tags = payload.get("text_token_tags")
        loras = tuple(
            (str(lora["path"]), float(lora["strength"]))
            for lora in transformer_options.get("sglang_h3_loras", [])
        )
        cache_dit = transformer_options.get("sglang_h3_cache_dit")
        cache_options = (
            ()
            if cache_dit is None
            else (
                int(cache_dit["max_warmup_steps"]),
                float(cache_dit["residual_diff_threshold"]),
                int(cache_dit["max_continuous_cached_steps"]),
            )
        )
        return cls(
            seed=int(payload.get("seed", 0)),
            context_shape=tuple(int(v) for v in context.shape),
            context_identity=context.data_ptr(),
            video_shape=tuple(int(v) for v in video.shape),
            audio_shape=tuple(int(v) for v in audio.shape),
            reference_shapes=refs,
            reference_identities=tuple(
                tensor.data_ptr() for tensor in references
            ),
            text_tags_shape=(
                tuple(int(v) for v in tags.shape) if torch.is_tensor(tags) else ()
            ),
            text_tags_identity=tags.data_ptr() if torch.is_tensor(tags) else 0,
            keyframe_indices=tuple(
                int(keyframe["resolved_frame_index"])
                for keyframe in payload.get("keyframes", [])
            ),
            execution_options=(
                str(transformer_options.get("sglang_h3_attention_backend", "auto")),
                tuple(sorted(transformer_options.get("sglang_h3_attention_options", {}).items())),
                loras,
                cache_options,
                float(payload.get("visual_cond_noise_aug", 0.999)),
                float(payload.get("audio_cond_noise_aug", 1.0)),
                float(payload.get("audio_scale", 1.0)),
                float(transformer_options.get("minimax_h3_sigma_shift_video", 12.0)),
                float(transformer_options.get("minimax_h3_sigma_shift_audio", 3.0)),
            ),
        )
