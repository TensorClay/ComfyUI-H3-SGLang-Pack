from __future__ import annotations

from copy import deepcopy
import os
from typing import Iterable

import torch
import torch.nn as nn

from safetensors import safe_open
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3DiTModel,
    MiniMaxH3Attention,
)

from .comfyui_quantized_weights import QuantizedWeightRestorer


class _CurveCoordinates(torch.Tensor):
    """Carry native FP32 curve coordinates through SGLang's fixed prelude.

    Upstream H3 always applies ``silu(t_emb).to(bfloat16)``. A pruned ComfyUI
    checkpoint has already replaced that value with interpolated FP32 curve
    coordinates, so both operations must be identities for this one tensor.
    The result returned from ``to`` is an ordinary Tensor; the subclass never
    enters the distributed linear kernels.
    """

    @staticmethod
    def __new__(cls, data: torch.Tensor):
        return torch.Tensor._make_subclass(cls, data, require_grad=False)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func is nn.functional.silu:
            value = args[0].as_subclass(torch.Tensor)
            return cls(value)
        return super().__torch_function__(func, types, args, kwargs or {})

    def to(self, *args, **kwargs):
        value = self.as_subclass(torch.Tensor)
        requested_dtype = kwargs.get("dtype")
        if args and isinstance(args[0], torch.dtype):
            requested_dtype = args[0]
        if requested_dtype is torch.bfloat16:
            return value
        return value.to(*args, **kwargs)


class _CurveTimeEmbedder(nn.Module):
    def __init__(self, grid_size: int, width: int) -> None:
        super().__init__()
        self.register_buffer(
            "table",
            torch.empty((grid_size, width), dtype=torch.float32),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        values = timesteps.to(dtype=torch.float32).clamp(0.0, 1.0)
        positions = values * (self.table.shape[0] - 1)
        lower = positions.floor().long().clamp(max=self.table.shape[0] - 2)
        coordinates = torch.lerp(
            self.table[lower],
            self.table[lower + 1],
            (positions - lower).unsqueeze(1),
        )
        return _CurveCoordinates(coordinates)


def _fp32_adaln_linear(
    *, input_size: int, output_size: int, prefix: str
) -> ColumnParallelLinear:
    return ColumnParallelLinear(
        input_size,
        output_size,
        bias=True,
        gather_output=False,
        params_dtype=torch.float32,
        quant_config=None,
        prefix=prefix,
    )


def _restore_contiguous_qkv_loader(attention: MiniMaxH3Attention) -> None:
    loader = MergedColumnParallelLinear.weight_loader.__get__(
        attention.qkv_proj, MergedColumnParallelLinear
    )
    weight = attention.qkv_proj.weight
    if hasattr(weight, "_weight_loader"):
        weight._weight_loader = loader
    else:
        weight.weight_loader = loader


def _checkpoint_path() -> str:
    path = os.environ.get("COMFYUI_SGLANG_H3_CHECKPOINT_PATH")
    if not path:
        raise RuntimeError("MiniMax H3 checkpoint path was not provided to worker")
    return path


def _curve_shape() -> tuple[int, int]:
    with safe_open(_checkpoint_path(), framework="pt", device="cpu") as checkpoint:
        shape = tuple(checkpoint.get_slice("adaln_t_table").get_shape())
    if len(shape) != 2 or min(shape) <= 1:
        raise ValueError(f"invalid MiniMax H3 AdaLN curve shape: {shape}")
    return shape


def _configure_comfy_checkpoint(model: MiniMaxH3DiTModel) -> None:
    # MiniMax's original checkpoint interleaves Q/K/V rows per head, while
    # ComfyUI's H3 conversion stores fused Q/K/V contiguously. Avoid applying
    # SGLang's upstream reorder to an already-converted checkpoint.
    for module in model.modules():
        if isinstance(module, MiniMaxH3Attention):
            _restore_contiguous_qkv_loader(module)


def _target_parameter(model: MiniMaxH3DiTModel, name: str):
    parameter = model.get_parameter(name)
    return tuple(parameter.shape), parameter.dtype


def _restore_quantized_weights(
    model: MiniMaxH3DiTModel,
    weights: Iterable[tuple[str, torch.Tensor]],
):
    restorer = QuantizedWeightRestorer(
        _checkpoint_path(),
        lambda name: _target_parameter(model, name),
    )
    yield from restorer.iter_restored(weights)


class ComfyPrunedMiniMaxH3DiTModel(MiniMaxH3DiTModel):
    """Load reduced-AdaLN H3 checkpoints in formats supported by ComfyUI."""

    handles_checkpoint_quantization = True

    def __init__(self, config, hf_config, quant_config=None) -> None:
        if quant_config is not None:
            raise ValueError("Comfy H3 checkpoint adapter does not accept quant_config")
        curve_grid, curve_width = _curve_shape()
        config = deepcopy(config)
        config.arch_config.time_embed_dim = curve_width
        super().__init__(config=config, hf_config=hf_config, quant_config=None)
        _configure_comfy_checkpoint(self)

        self.time_embedder = _CurveTimeEmbedder(curve_grid, curve_width)
        for index, block in enumerate(self.blocks):
            block.adaln_proj.linear = _fp32_adaln_linear(
                input_size=curve_width,
                output_size=self.arch.adaln_out_features,
                prefix=f"blocks.{index}.adaln_proj.linear",
            )
        self.final_layer.adaln_proj.linear = _fp32_adaln_linear(
            input_size=curve_width,
            output_size=self.arch.final_adaln_out_features,
            prefix="final_layer.adaln_proj.linear",
        )
        self._mark_missing_params_required()

    def preprocess_loaded_state_dict(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ):
        for name, value in _restore_quantized_weights(self, weights):
            if name == "adaln_t_table":
                yield "time_embedder.table", value
            else:
                yield name, value

    def post_load_weights(self) -> None:
        required_fp32 = (
            "video_patch_proj.weight",
            "video_patch_proj.bias",
            "audio_patch_proj.weight",
            "audio_patch_proj.bias",
            "final_layer.video_out.weight",
            "final_layer.video_out.bias",
            "final_layer.audio_out.weight",
            "final_layer.audio_out.bias",
        )
        for name in required_fp32:
            if self.get_parameter(name).dtype is not torch.float32:
                raise ValueError(f"{name} must stay fp32")
        for index, block in enumerate(self.blocks):
            if block.adaln_proj.linear.weight.dtype is not torch.float32:
                raise ValueError(f"blocks.{index}.adaln_proj must stay fp32")
        if self.final_layer.adaln_proj.linear.weight.dtype is not torch.float32:
            raise ValueError("final_layer.adaln_proj must stay fp32")
        if self.time_embedder.table.dtype is not torch.float32:
            raise ValueError("adaln_t_table must stay fp32")
        if self.rope.inv_freq.dtype is not torch.float32:
            raise ValueError("rope.inv_freq must stay fp32")


class ComfyFullMiniMaxH3DiTModel(MiniMaxH3DiTModel):
    """Load full-AdaLN H3 checkpoints in formats supported by ComfyUI."""

    handles_checkpoint_quantization = True

    def __init__(self, config, hf_config, quant_config=None) -> None:
        if quant_config is not None:
            raise ValueError("Comfy H3 checkpoint adapter does not accept quant_config")
        super().__init__(config=config, hf_config=hf_config, quant_config=None)
        _configure_comfy_checkpoint(self)

    def preprocess_loaded_state_dict(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ):
        yield from _restore_quantized_weights(self, weights)


__all__ = ["ComfyFullMiniMaxH3DiTModel", "ComfyPrunedMiniMaxH3DiTModel"]
