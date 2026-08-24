from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


QUANT_METADATA_SUFFIX = ".comfy_quant"
WEIGHT_SUFFIX = ".weight"
QUANT_SIDECARS = (
    "weight_scale",
    "weight_scale_2",
    "weight_s_rel",
    "weight_s_channel",
    "weight_codebook",
    "weight_correction",
    "input_scale",
    "pre_quant_scale",
)
PACKED_LAST_DIM_FORMATS = frozenset(
    {"nvfp4", "convrot_w4a4", "asym_w4a8_int8"}
)
REQUIRED_QUANT_SIDECARS = {
    "float8_e4m3fn": frozenset({"weight_scale"}),
    "float8_e5m2": frozenset({"weight_scale"}),
    "mxfp8": frozenset({"weight_scale"}),
    "nvfp4": frozenset({"weight_scale", "weight_scale_2"}),
    "int8_tensorwise": frozenset({"weight_scale"}),
    "convrot_w4a4": frozenset({"weight_scale"}),
    "asym_w4a8_int8": frozenset({"weight_s_rel"}),
}


def decode_quant_metadata(value: torch.Tensor, name: str) -> dict[str, Any]:
    try:
        metadata = json.loads(value.detach().cpu().numpy().tobytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid ComfyUI quantization metadata: {name}") from error
    if not isinstance(metadata, dict) or not metadata.get("format"):
        raise ValueError(f"invalid ComfyUI quantization metadata: {name}")
    return metadata


def checkpoint_quant_metadata(
    checkpoint_path: str | Path,
) -> tuple[set[str], dict[str, torch.Tensor]]:
    """Return checkpoint keys and ComfyUI-normalized per-layer metadata.

    Current checkpoints carry ``*.comfy_quant`` tensors. Older checkpoints may
    carry the same layer map in the safetensors ``_quantization_metadata``
    header; this mirrors ``comfy.utils.convert_old_quants`` without loading the
    checkpoint tensors into the host process.
    """

    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        metadata = checkpoint.metadata() or {}
        inline = {
            name: checkpoint.get_tensor(name)
            for name in keys
            if name.endswith(QUANT_METADATA_SUFFIX)
        }

    legacy_scaled_fp8 = sorted(
        name
        for name in keys
        if name == "scaled_fp8"
        or name.endswith(".scaled_fp8")
        or name.endswith(".scale_weight")
        or name.endswith(".scale_input")
    )
    if legacy_scaled_fp8:
        raise ValueError(
            "legacy scaled_fp8 checkpoints are not supported by the MiniMax H3 "
            "SGLang loader: " + ", ".join(legacy_scaled_fp8[:8])
        )

    header = metadata.get("_quantization_metadata")
    if header is None:
        return keys, inline
    from comfy.utils import convert_old_quants

    try:
        legacy, _ = convert_old_quants({}, model_prefix="", metadata=metadata)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid legacy ComfyUI quantization metadata") from error
    normalized = dict(
        (name, value)
        for name, value in legacy.items()
        if name.endswith(QUANT_METADATA_SUFFIX)
    )
    # Tensor-form metadata is the current representation and wins if a
    # checkpoint happens to carry both it and the legacy header form.
    normalized.update(inline)
    return keys, normalized


def _sidecar(name: str) -> tuple[str, str] | None:
    for suffix in QUANT_SIDECARS:
        marker = f".{suffix}"
        if name.endswith(marker):
            return name[: -len(marker)] + WEIGHT_SUFFIX, suffix
    return None


def restore_weight_with_comfy(
    state_dict: dict[str, torch.Tensor],
    original_shape: tuple[int, ...],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Use stock ComfyUI's loader and Comfy-Kitchen layout to restore a weight."""

    if len(original_shape) != 2:
        raise ValueError(
            f"ComfyUI quantized diffusion weight must be a matrix: {original_shape}"
        )
    from comfy import ops
    from comfy.quant_ops import QuantizedTensor

    operations = ops.mixed_precision_ops({}, compute_dtype=output_dtype)
    layer = operations.Linear(
        original_shape[1],
        original_shape[0],
        bias=False,
        device="cpu",
        dtype=output_dtype,
    )
    layer.load_state_dict(dict(state_dict), strict=False)
    weight = layer.weight
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    weight = weight.detach().to(device="cpu", dtype=output_dtype)
    return fold_pre_quant_scale(
        weight, state_dict.get("pre_quant_scale")
    ).contiguous()


def fold_pre_quant_scale(
    weight: torch.Tensor, scale: torch.Tensor | None
) -> torch.Tensor:
    """Fold ComfyUI's AWQ-style input smoothing into an ordinary weight."""

    if scale is None:
        return weight
    values = scale.detach().to(device=weight.device, dtype=torch.float32).reshape(-1)
    if weight.ndim != 2 or values.numel() != weight.shape[1]:
        raise ValueError(
            f"pre_quant_scale shape {tuple(scale.shape)} cannot be folded into "
            f"weight shape {tuple(weight.shape)}"
        )
    return (weight.float() * values.unsqueeze(0)).to(dtype=weight.dtype)


def checkpoint_weight_shape(
    metadata: dict[str, Any], stored_weight: torch.Tensor
) -> tuple[int, ...]:
    """Recover the global matrix shape before SGLang applies TP sharding."""

    return checkpoint_weight_shape_from_shape(metadata, tuple(stored_weight.shape))


def checkpoint_weight_shape_from_shape(
    metadata: dict[str, Any], stored_shape: tuple[int, ...]
) -> tuple[int, ...]:
    """Recover a logical matrix shape without materializing its tensor."""

    shape = tuple(stored_shape)
    if len(shape) != 2:
        raise ValueError(f"ComfyUI quantized diffusion weight must be a matrix: {shape}")
    if metadata["format"] in PACKED_LAST_DIM_FORMATS:
        return shape[0], shape[1] * 2
    return shape


def validate_checkpoint_quantization(
    checkpoint_path: str | Path,
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    """Validate a checkpoint's quantized layers using installed ComfyUI support.

    This is intentionally header-only preflight. It keeps unsupported formats,
    missing sidecars, and orphaned quantization tensors out of the model picker
    instead of discovering them after the SGLang worker pool has started.
    """

    keys, encoded = checkpoint_quant_metadata(checkpoint_path)
    if not encoded:
        orphaned = sorted(name for name in keys if _sidecar(name) is not None)
        if orphaned:
            raise ValueError(
                "ComfyUI quantization sidecars have no layer metadata: "
                + ", ".join(orphaned[:8])
            )
        return keys, {}

    from comfy.quant_ops import QUANT_ALGOS

    layers: dict[str, dict[str, Any]] = {}
    for metadata_name, value in encoded.items():
        weight_name = (
            metadata_name.removesuffix(QUANT_METADATA_SUFFIX) + WEIGHT_SUFFIX
        )
        metadata = decode_quant_metadata(value, metadata_name)
        quant_format = metadata["format"]
        if quant_format not in QUANT_ALGOS:
            raise ValueError(
                f"installed ComfyUI cannot restore quantization format "
                f"{quant_format!r} for {weight_name}"
            )
        if weight_name not in keys:
            raise ValueError(
                f"ComfyUI quantization metadata has no weight tensor: {weight_name}"
            )
        prefix = weight_name.removesuffix(WEIGHT_SUFFIX)
        required = REQUIRED_QUANT_SIDECARS.get(quant_format, frozenset())
        missing = sorted(
            suffix
            for suffix in required
            if f"{prefix}.{suffix}" not in keys
        )
        if missing:
            raise ValueError(
                f"ComfyUI quantized layer {weight_name} is missing sidecars: "
                + ", ".join(missing)
            )
        layers[weight_name] = metadata

    orphaned = sorted(
        name
        for name in keys
        if (sidecar := _sidecar(name)) is not None and sidecar[0] not in layers
    )
    if orphaned:
        raise ValueError(
            "ComfyUI quantization sidecars have no layer metadata: "
            + ", ".join(orphaned[:8])
        )
    return keys, layers


class QuantizedWeightRestorer:
    """Restore ComfyUI quantized tensor groups incrementally.

    This class avoids retaining both compressed and restored copies of more than
    one layer. SGLang 0.5.17 subsequently collects the yielded tensors into its
    full-model loading dictionary before TP sharding, so callers must still plan
    for a full restored checkpoint in each worker process during cold start.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        target: Callable[[str], tuple[tuple[int, ...], torch.dtype]],
        restore: Callable[
            [dict[str, torch.Tensor], tuple[int, ...], torch.dtype], torch.Tensor
        ] = restore_weight_with_comfy,
    ) -> None:
        self.checkpoint_keys, legacy = checkpoint_quant_metadata(checkpoint_path)
        self.target = target
        self.restore = restore
        self.configs: dict[str, torch.Tensor] = {}
        self.pending: dict[str, dict[str, torch.Tensor]] = {}
        self.restored_names: set[str] = set()
        for metadata_name, value in legacy.items():
            self.configs[
                metadata_name.removesuffix(QUANT_METADATA_SUFFIX) + WEIGHT_SUFFIX
            ] = value

    def _expected_sidecars(self, weight_name: str) -> set[str]:
        prefix = weight_name.removesuffix(WEIGHT_SUFFIX)
        return {
            suffix
            for suffix in QUANT_SIDECARS
            if f"{prefix}.{suffix}" in self.checkpoint_keys
        }

    def _try_restore(self, weight_name: str):
        config_tensor = self.configs.get(weight_name)
        values = self.pending.get(weight_name)
        if config_tensor is None or values is None or "weight" not in values:
            return None
        expected = self._expected_sidecars(weight_name)
        if not expected.issubset(values):
            return None

        metadata = decode_quant_metadata(
            config_tensor,
            weight_name.removesuffix(WEIGHT_SUFFIX) + QUANT_METADATA_SUFFIX,
        )
        # Match core ComfyUI's failure mode for an unavailable/unknown layout
        # before discarding any checkpoint tensors.
        from comfy.quant_ops import QUANT_ALGOS

        if metadata["format"] not in QUANT_ALGOS:
            raise ValueError(
                f"ComfyUI cannot restore quantization format "
                f"{metadata['format']!r} for {weight_name}"
            )

        state_dict = {
            "weight": values["weight"],
            "comfy_quant": config_tensor,
            **{name: values[name] for name in expected},
        }
        _local_shape, dtype = self.target(weight_name)
        shape = checkpoint_weight_shape(metadata, values["weight"])
        restored = self.restore(state_dict, shape, dtype)
        del self.pending[weight_name]
        del self.configs[weight_name]
        self.restored_names.add(weight_name)
        return weight_name, restored

    def iter_restored(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        for name, value in weights:
            if name.endswith(QUANT_METADATA_SUFFIX):
                weight_name = name.removesuffix(QUANT_METADATA_SUFFIX) + WEIGHT_SUFFIX
                if weight_name in self.restored_names:
                    continue
                self.configs[weight_name] = value
                restored = self._try_restore(weight_name)
                if restored is not None:
                    yield restored
                continue

            sidecar = _sidecar(name)
            if sidecar is not None:
                weight_name, suffix = sidecar
                self.pending.setdefault(weight_name, {})[suffix] = value
                restored = self._try_restore(weight_name)
                if restored is not None:
                    yield restored
                continue

            if name in self.configs:
                self.pending.setdefault(name, {})["weight"] = value
                restored = self._try_restore(name)
                if restored is not None:
                    yield restored
                continue

            yield name, value

        if self.pending or self.configs:
            unresolved = set(self.pending) | set(self.configs)
            raise ValueError(
                "incomplete ComfyUI quantized tensor groups: "
                + ", ".join(sorted(unresolved)[:8])
            )


__all__ = [
    "QuantizedWeightRestorer",
    "checkpoint_quant_metadata",
    "checkpoint_weight_shape",
    "checkpoint_weight_shape_from_shape",
    "decode_quant_metadata",
    "fold_pre_quant_scale",
    "restore_weight_with_comfy",
    "validate_checkpoint_quantization",
]
