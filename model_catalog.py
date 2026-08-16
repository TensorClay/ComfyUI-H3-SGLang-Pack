from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

from safetensors import SafetensorError, safe_open


H3_SIGNATURE_KEYS = frozenset(
    {
        "video_patch_proj.weight",
        "audio_patch_proj.weight",
        "condition_proj.weight",
        "final_layer.video_out.weight",
        "final_layer.audio_out.weight",
    }
)
COMFY_INT8_CONVROT = "comfy_int8_convrot"
COMFY_BF16 = "comfy_bf16"
SUPPORTED_FORMATS = frozenset({COMFY_INT8_CONVROT, COMFY_BF16})

FULL_BF16_SIGNATURE = {
    "time_embedder.proj_in.weight": ((5376, 256), "F32"),
    "time_embedder.proj_out.weight": ((2688, 5376), "F32"),
    "blocks.0.adaln_proj.linear.weight": ((96768, 2688), "BF16"),
    "final_layer.adaln_proj.linear.weight": ((10752, 2688), "BF16"),
}


@dataclass(frozen=True)
class H3Checkpoint:
    name: str
    path: Path
    format: str
    variant: str
    size: int
    mtime_ns: int


def _folder_paths():
    import folder_paths

    return folder_paths


@lru_cache(maxsize=64)
def _inspect_cached(path_string: str, size: int, mtime_ns: int) -> tuple[str, str]:
    del size, mtime_ns
    path = Path(path_string)
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        if not H3_SIGNATURE_KEYS.issubset(keys):
            raise ValueError("checkpoint does not contain a MiniMax H3 transformer")

        quantized = [name for name in keys if name.endswith(".comfy_quant")]
        if quantized:
            table_key = "adaln_t_table"
            if table_key not in keys:
                raise ValueError(
                    "pruned INT8 MiniMax H3 checkpoint is missing adaln_t_table"
                )
            table_shape = tuple(checkpoint.get_slice(table_key).get_shape())
            if table_shape != (1025, 8):
                raise ValueError(
                    f"unsupported MiniMax H3 AdaLN curve shape: {table_shape}"
                )
            for name in quantized:
                raw = bytes(checkpoint.get_tensor(name).tolist())
                try:
                    metadata = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"invalid quantization metadata in {name}"
                    ) from error
                if metadata != {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": 256,
                }:
                    raise ValueError(
                        f"unsupported quantization metadata in {name}: {metadata}"
                    )
            checkpoint_format = COMFY_INT8_CONVROT
        else:
            if "adaln_t_table" in keys:
                raise ValueError(
                    "pruned non-INT8 MiniMax H3 checkpoints are not yet supported"
                )
            for name, (expected_shape, expected_dtype) in FULL_BF16_SIGNATURE.items():
                if name not in keys:
                    raise ValueError(f"full MiniMax H3 checkpoint is missing {name}")
                tensor = checkpoint.get_slice(name)
                actual_shape = tuple(tensor.get_shape())
                if actual_shape != expected_shape:
                    raise ValueError(
                        f"unsupported MiniMax H3 tensor shape for {name}: "
                        f"{actual_shape}"
                    )
                if tensor.get_dtype() != expected_dtype:
                    raise ValueError(f"{name} is not {expected_dtype}")
            checkpoint_format = COMFY_BF16

    # H3 checkpoint tensors do not encode the pipeline variant. The current
    # product is explicitly Ref2VA-only, so compatible local files are loaded
    # against the pinned Ref2VA pipeline configuration.
    return checkpoint_format, "ref2va"


def inspect_checkpoint(name: str) -> H3Checkpoint:
    folders = _folder_paths()
    path = Path(folders.get_full_path_or_raise("diffusion_models", name)).resolve()
    stat = path.stat()
    checkpoint_format, variant = _inspect_cached(
        str(path), stat.st_size, stat.st_mtime_ns
    )
    return H3Checkpoint(
        name=name,
        path=path,
        format=checkpoint_format,
        variant=variant,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def compatible_model_names() -> list[str]:
    names = []
    for name in _folder_paths().get_filename_list("diffusion_models"):
        try:
            checkpoint = inspect_checkpoint(name)
        except (OSError, SafetensorError, ValueError):
            continue
        if checkpoint.format in SUPPORTED_FORMATS:
            names.append(name)
    return names


def model_choices() -> list[str]:
    names = compatible_model_names()
    if not names:
        # ComfyUI requires at least one combo entry. Keep the error actionable
        # if the node is invoked before a compatible checkpoint is installed.
        return ["No compatible MiniMax H3 models found"]
    return names
