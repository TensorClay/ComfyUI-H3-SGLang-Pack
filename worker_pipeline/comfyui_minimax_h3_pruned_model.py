from __future__ import annotations

from copy import deepcopy
import json
from typing import Iterable

import torch
import torch.nn as nn

from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3DiTModel,
    MiniMaxH3Attention,
)


CURVE_GRID = 1025
CURVE_WIDTH = 8
CONVROT_GROUP_SIZE = 256


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
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "table",
            torch.empty((CURVE_GRID, CURVE_WIDTH), dtype=torch.float32),
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


def _decode_quant_metadata(value: torch.Tensor, name: str) -> dict:
    try:
        metadata = json.loads(bytes(value.tolist()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid ComfyUI quantization metadata: {name}") from error
    expected = {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": CONVROT_GROUP_SIZE,
    }
    if metadata != expected:
        raise ValueError(
            f"unsupported ComfyUI quantization metadata for {name}: {metadata}"
        )
    return metadata


def _dequantize_convrot(
    weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    params = TensorWiseINT8Layout.Params(
        scale=scale,
        orig_dtype=torch.bfloat16,
        orig_shape=tuple(weight.shape),
        is_weight=True,
        convrot=True,
        convrot_groupsize=CONVROT_GROUP_SIZE,
    )
    return TensorWiseINT8Layout.dequantize(weight, params)


class ComfyPrunedMiniMaxH3DiTModel(MiniMaxH3DiTModel):
    """SGLang H3 model adapted to ComfyUI's pruned INT8 ConvRot export.

    ConvRot weights are restored once while streaming the checkpoint, then
    loaded into SGLang's ordinary TP-sharded BF16 linears. This prioritizes
    compatibility with the already-installed checkpoint; it intentionally does
    not claim native Comfy Kitchen W8A8 execution inside SGLang.
    """

    handles_checkpoint_quantization = True

    def __init__(self, config, hf_config, quant_config=None) -> None:
        if quant_config is not None:
            raise ValueError("Comfy H3 checkpoint adapter does not accept quant_config")
        config = deepcopy(config)
        config.arch_config.time_embed_dim = CURVE_WIDTH
        super().__init__(config=config, hf_config=hf_config, quant_config=None)

        # MiniMax's original checkpoint interleaves Q/K/V rows per head, and
        # SGLang normally reorders that layout while loading. Comfy-Org's
        # export has already converted every fused QKV tensor to contiguous
        # [q_all, k_all, v_all], matching ComfyUI's forward. Restore the base
        # merged-column loader so each logical matrix is only TP-sharded, not
        # reordered a second time. This includes token-refiner attention.
        for module in self.modules():
            if isinstance(module, MiniMaxH3Attention):
                _restore_contiguous_qkv_loader(module)

        self.time_embedder = _CurveTimeEmbedder()
        for index, block in enumerate(self.blocks):
            block.adaln_proj.linear = _fp32_adaln_linear(
                input_size=CURVE_WIDTH,
                output_size=self.arch.adaln_out_features,
                prefix=f"blocks.{index}.adaln_proj.linear",
            )
        self.final_layer.adaln_proj.linear = _fp32_adaln_linear(
            input_size=CURVE_WIDTH,
            output_size=self.arch.final_adaln_out_features,
            prefix="final_layer.adaln_proj.linear",
        )
        self._mark_missing_params_required()

    def preprocess_loaded_state_dict(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ):
        quantized_names: set[str] = set()
        pending_weights: dict[str, torch.Tensor] = {}
        pending_scales: dict[str, torch.Tensor] = {}

        def restored_weight(name: str):
            if (
                name not in quantized_names
                or name not in pending_weights
                or name not in pending_scales
            ):
                return None
            weight = pending_weights.pop(name)
            scale = pending_scales.pop(name)
            quantized_names.remove(name)
            return name, _dequantize_convrot(weight, scale)


        for name, value in weights:
            if name.endswith(".comfy_quant"):
                _decode_quant_metadata(value, name)
                weight_name = name.removesuffix(".comfy_quant") + ".weight"
                quantized_names.add(weight_name)
                restored = restored_weight(weight_name)
                if restored is not None:
                    yield restored
                continue

            if name.endswith(".weight_scale"):
                weight_name = name.removesuffix("_scale")
                pending_scales[weight_name] = value
                restored = restored_weight(weight_name)
                if restored is not None:
                    yield restored
                continue

            if value.dtype is torch.int8:
                pending_weights[name] = value
                restored = restored_weight(name)
                if restored is not None:
                    yield restored
                continue

            if name == "adaln_t_table":
                yield "time_embedder.table", value
            else:
                yield name, value

        if pending_weights or pending_scales or quantized_names:
            unresolved = set(pending_weights) | set(pending_scales) | quantized_names
            raise ValueError(
                "incomplete INT8 ConvRot tensor groups: "
                + ", ".join(sorted(unresolved)[:8])
            )

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


class ComfyBF16MiniMaxH3DiTModel(MiniMaxH3DiTModel):
    """Load Comfy-Org's full BF16 export with its contiguous Q/K/V rows."""

    def __init__(self, config, hf_config, quant_config=None) -> None:
        if quant_config is not None:
            raise ValueError("Comfy H3 BF16 checkpoint does not accept quant_config")
        super().__init__(config=config, hf_config=hf_config, quant_config=None)
        for module in self.modules():
            if isinstance(module, MiniMaxH3Attention):
                _restore_contiguous_qkv_loader(module)


__all__ = ["ComfyBF16MiniMaxH3DiTModel", "ComfyPrunedMiniMaxH3DiTModel"]
