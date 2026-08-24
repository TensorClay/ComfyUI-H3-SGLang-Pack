from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import re

from safetensors import SafetensorError, safe_open

from .worker_pipeline.comfyui_quantized_weights import (
    QUANT_METADATA_SUFFIX,
    QUANT_SIDECARS,
    checkpoint_weight_shape_from_shape,
    validate_checkpoint_quantization,
)


H3_SIGNATURE_KEYS = frozenset(
    {
        "video_patch_proj.weight",
        "audio_patch_proj.weight",
        "condition_proj.weight",
        "final_layer.video_out.weight",
        "final_layer.audio_out.weight",
        "blocks.0.attn.q_norm.weight",
        "blocks.0.attn.k_norm.weight",
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.attn.out_proj.weight",
        "blocks.0.mlp.fc1.weight",
        "blocks.0.mlp.fc2.weight",
        "rope.inv_freq",
    }
)
FULL_ARCHITECTURE = "full"
PRUNED_ARCHITECTURE = "pruned_adaln"
SUPPORTED_ARCHITECTURES = frozenset({FULL_ARCHITECTURE, PRUNED_ARCHITECTURE})
SUPPORTED_VARIANTS = ("fl2va", "ref2va")
STORAGE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "F8_E5M2FN": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}


@dataclass(frozen=True)
class H3Checkpoint:
    name: str
    path: Path
    architecture: str
    size: int
    restored_size: int
    mtime_ns: int
    parameter_keys: tuple[str, ...]


def _folder_paths():
    import folder_paths

    return folder_paths


def _shape(
    checkpoint, name: str, quantized_layers: dict[str, dict]
) -> tuple[int, ...]:
    if name not in checkpoint.keys():
        raise ValueError(f"MiniMax H3 checkpoint is missing {name}")
    shape = tuple(checkpoint.get_slice(name).get_shape())
    metadata = quantized_layers.get(name)
    if metadata is not None:
        return checkpoint_weight_shape_from_shape(metadata, shape)
    return shape


@lru_cache(maxsize=1)
def _runtime_architecture_config() -> dict:
    path = (
        Path(__file__).resolve().parent
        / "runtime_config"
        / "transformer"
        / "config.json"
    )
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot read MiniMax H3 runtime configuration: {path}"
        ) from error
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid MiniMax H3 runtime configuration: {path}")
    return value


def _expect_shape(
    checkpoint,
    name: str,
    expected: tuple[int, ...],
    quantized_layers: dict[str, dict],
) -> None:
    actual = _shape(checkpoint, name, quantized_layers)
    if actual != expected:
        raise ValueError(
            f"MiniMax H3 tensor {name} has logical shape {actual}; "
            f"the bundled runtime requires {expected}"
        )


def _validate_block_coverage(keys: set[str], prefix: str, expected_count: int) -> None:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    actual = {
        int(match.group(1))
        for name in keys
        if (match := pattern.match(name)) is not None
    }
    expected = set(range(expected_count))
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {missing[:8]}")
        if extra:
            details.append(f"unexpected {extra[:8]}")
        raise ValueError(
            f"MiniMax H3 {prefix} coverage does not match the bundled runtime "
            f"({expected_count} blocks): " + ", ".join(details)
        )


def _validate_runtime_architecture(
    checkpoint, keys: set[str], quantized_layers: dict[str, dict]
) -> dict:
    config = _runtime_architecture_config()
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    head_dim = int(config["attention_head_dim"])
    ffn = int(config["ffn_hidden_size"])
    video_patch_dim = int(config["latents_dim"]) * math.prod(config["patch_size"])
    audio_dim = int(config["audio_latents_dim"])
    inner = heads * head_dim
    expected_shapes = {
        "video_patch_proj.weight": (hidden, video_patch_dim),
        "audio_patch_proj.weight": (hidden, audio_dim),
        "condition_proj.weight": (hidden, int(config["text_dim"])),
        "final_layer.video_out.weight": (video_patch_dim, hidden),
        "final_layer.audio_out.weight": (audio_dim, hidden),
        "blocks.0.attn.q_norm.weight": (head_dim,),
        "blocks.0.attn.k_norm.weight": (head_dim,),
        "blocks.0.attn.qkv_proj.weight": (3 * inner, hidden),
        "blocks.0.attn.out_proj.weight": (hidden, inner),
        "blocks.0.mlp.fc1.weight": (2 * ffn, hidden),
        "blocks.0.mlp.fc2.weight": (hidden, ffn),
        "rope.inv_freq": (int(config["rope_inv_freq_len"]),),
    }
    for name, expected in expected_shapes.items():
        _expect_shape(checkpoint, name, expected, quantized_layers)
    _validate_block_coverage(keys, "blocks", int(config["num_layers"]))
    _validate_block_coverage(
        keys,
        "token_refiner.blocks",
        int(config["token_refiner_num_layers"]),
    )
    return config


def _architecture(
    checkpoint, keys: set[str], quantized_layers: dict[str, dict]
) -> str:
    config = _validate_runtime_architecture(checkpoint, keys, quantized_layers)
    if "adaln_t_table" in keys:
        table_shape = _shape(checkpoint, "adaln_t_table", quantized_layers)
        if len(table_shape) != 2 or min(table_shape) <= 1:
            raise ValueError(
                f"invalid MiniMax H3 AdaLN curve shape: {table_shape}"
            )
        curve_width = table_shape[1]
        for name in (
            "blocks.0.adaln_proj.linear.weight",
            "final_layer.adaln_proj.linear.weight",
        ):
            shape = _shape(checkpoint, name, quantized_layers)
            expected_output = (
                int(config["adaln_out_features"])
                if name.startswith("blocks.")
                else int(config["final_adaln_out_features"])
            )
            if shape != (expected_output, curve_width):
                raise ValueError(
                    f"MiniMax H3 reduced AdaLN tensor {name} has shape "
                    f"{shape}; expected {(expected_output, curve_width)}"
                )
        return PRUNED_ARCHITECTURE

    proj_in = _shape(checkpoint, "time_embedder.proj_in.weight", quantized_layers)
    proj_out = _shape(checkpoint, "time_embedder.proj_out.weight", quantized_layers)
    block_adaln = _shape(
        checkpoint, "blocks.0.adaln_proj.linear.weight", quantized_layers
    )
    final_adaln = _shape(
        checkpoint, "final_layer.adaln_proj.linear.weight", quantized_layers
    )
    expected = (
        (int(config["time_embed_hidden_size"]), int(config["timestep_input_dim"])),
        (int(config["time_embed_dim"]), int(config["time_embed_hidden_size"])),
        (int(config["adaln_out_features"]), int(config["time_embed_dim"])),
        (int(config["final_adaln_out_features"]), int(config["time_embed_dim"])),
    )
    actual = (proj_in, proj_out, block_adaln, final_adaln)
    if actual != expected:
        raise ValueError(
            "MiniMax H3 full time/AdaLN architecture does not match the bundled "
            f"runtime: got {actual}, expected {expected}"
        )
    return FULL_ARCHITECTURE


def _is_quant_sidecar(name: str, quantized_names: set[str]) -> bool:
    for suffix in QUANT_SIDECARS:
        marker = f".{suffix}"
        if name.endswith(marker):
            weight_name = name[: -len(marker)] + ".weight"
            return weight_name in quantized_names
    return False


def _restored_checkpoint_size(
    path: Path, quantized_layers: dict[str, dict]
) -> int:
    """Estimate bytes resident after ComfyUI quantized weights become BF16."""

    quantized_names = set(quantized_layers)
    total = 0
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        for name in checkpoint.keys():
            if name.endswith(QUANT_METADATA_SUFFIX) or _is_quant_sidecar(
                name, quantized_names
            ):
                continue
            tensor = checkpoint.get_slice(name)
            shape = tuple(tensor.get_shape())
            if name in quantized_layers:
                shape = checkpoint_weight_shape_from_shape(
                    quantized_layers[name], shape
                )
                bytes_per_element = 2
            else:
                dtype = tensor.get_dtype()
                if name.endswith((".weight", ".bias")) and dtype in {
                    "I8",
                    "U8",
                    "I16",
                    "U16",
                    "I32",
                    "U32",
                    "I64",
                    "U64",
                }:
                    raise ValueError(
                        f"integer MiniMax H3 parameter lacks ComfyUI "
                        f"quantization metadata: {name}"
                    )
                try:
                    bytes_per_element = STORAGE_BYTES[dtype]
                except KeyError as error:
                    raise ValueError(
                        f"unsupported safetensors dtype {dtype!r} for {name}"
                    ) from error
                if name.endswith((".weight", ".bias")):
                    bytes_per_element = max(bytes_per_element, 2)
            total += math.prod(shape) * bytes_per_element
    return total


@lru_cache(maxsize=64)
def _inspect_cached(
    path_string: str, size: int, mtime_ns: int
) -> tuple[str, int, tuple[str, ...]]:
    del size, mtime_ns
    path = Path(path_string)
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        if not H3_SIGNATURE_KEYS.issubset(keys):
            raise ValueError("checkpoint does not contain a MiniMax H3 transformer")

    # Quantization metadata can require reading many small tensor sidecars.
    # Only do that work after the cheap structural H3 gate has succeeded.
    _, quantized_layers = validate_checkpoint_quantization(path)
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        architecture = _architecture(checkpoint, keys, quantized_layers)

    restored_size = _restored_checkpoint_size(path, quantized_layers)

    # The host-side proxy exposes the same model parameters to ComfyUI for
    # patch/LoRA discovery. Quantization sidecars are deliberately excluded.
    parameter_keys = tuple(
        sorted(name for name in keys if name.endswith((".weight", ".bias")))
    )
    return architecture, restored_size, parameter_keys


def inspect_checkpoint(name: str) -> H3Checkpoint:
    folders = _folder_paths()
    path = Path(folders.get_full_path_or_raise("diffusion_models", name)).resolve()
    stat = path.stat()
    architecture, restored_size, parameter_keys = _inspect_cached(
        str(path), stat.st_size, stat.st_mtime_ns
    )
    return H3Checkpoint(
        name=name,
        path=path,
        architecture=architecture,
        size=stat.st_size,
        restored_size=restored_size,
        mtime_ns=stat.st_mtime_ns,
        parameter_keys=parameter_keys,
    )


def compatible_model_names() -> list[str]:
    names = []
    for name in _folder_paths().get_filename_list("diffusion_models"):
        try:
            inspect_checkpoint(name)
        except (OSError, SafetensorError, ValueError):
            continue
        names.append(name)
    return names


def model_choices() -> list[str]:
    names = compatible_model_names()
    if not names:
        # ComfyUI requires at least one combo entry. Keep the error actionable
        # if the node is invoked before a compatible checkpoint is installed.
        return ["No compatible MiniMax H3 models found"]
    return names
