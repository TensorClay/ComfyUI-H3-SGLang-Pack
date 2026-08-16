from __future__ import annotations

from pathlib import Path


SOL_PROVIDER_NAME = "ComfyUI-SolAttn_triton"


def sol_attention_provider() -> Path | None:
    provider = Path(__file__).resolve().parents[2] / SOL_PROVIDER_NAME
    required = (
        "_tri_fwd.py",
        "_int8_fwd.py",
        "_preprocess.py",
        "_fused_prep.py",
        "_fused_quant.py",
        "_autotune_log.py",
    )
    if all((provider / name).is_file() for name in required):
        return provider
    return None
