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


@dataclass(frozen=True)
class ExecutionSignature:
    seed: int
    context_shape: tuple[int, ...]
    video_shape: tuple[int, ...]
    audio_shape: tuple[int, ...]
    reference_shapes: tuple[tuple[int, ...], ...]
    text_tags_shape: tuple[int, ...]

    @classmethod
    def from_inputs(
        cls,
        context: torch.Tensor,
        video: torch.Tensor,
        audio: torch.Tensor,
        payload: dict[str, Any],
    ) -> "ExecutionSignature":
        refs = tuple(
            tuple(int(v) for v in tensor.shape)
            for key in ("cond_video_latents", "cond_audio_latents")
            for tensor in payload.get(key, [])
        )
        tags = payload.get("text_token_tags")
        return cls(
            seed=int(payload.get("seed", 0)),
            context_shape=tuple(int(v) for v in context.shape),
            video_shape=tuple(int(v) for v in video.shape),
            audio_shape=tuple(int(v) for v in audio.shape),
            reference_shapes=refs,
            text_tags_shape=(
                tuple(int(v) for v in tags.shape) if torch.is_tensor(tags) else ()
            ),
        )
