from __future__ import annotations


NO_ACCELERATOR_TOPOLOGY = "No compatible CUDA/ROCm GPUs detected"
H3_TP_DIMENSIONS = (56, 5376, 14336, 96768, 10752, 96, 32)
H3_SEQUENCE_ALIGNMENT = 64


def detected_accelerator_count() -> int:
    """Return the accelerator count visible to ComfyUI's PyTorch process."""

    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def topology_map(device_count: int | None = None) -> dict[str, tuple[int, int]]:
    count = detected_accelerator_count() if device_count is None else device_count
    if count < 1:
        return {}
    return {
        f"TP{tp_size} / Ulysses{ulysses_degree}": (
            tp_size,
            ulysses_degree,
        )
        for tp_size in range(1, count + 1)
        if all(dimension % tp_size == 0 for dimension in H3_TP_DIMENSIONS)
        for ulysses_degree in range(1, count // tp_size + 1)
        if (56 // tp_size) % ulysses_degree == 0
        and H3_SEQUENCE_ALIGNMENT % ulysses_degree == 0
    }


def topology_choices(device_count: int | None = None) -> list[str]:
    choices = sorted(topology_map(device_count))
    return choices or [NO_ACCELERATOR_TOPOLOGY]


def default_topology(device_count: int | None = None) -> str:
    count = detected_accelerator_count() if device_count is None else device_count
    available = topology_map(count)
    if not available:
        return NO_ACCELERATOR_TOPOLOGY
    return max(
        available,
        key=lambda label: (
            available[label][0] * available[label][1],
            available[label][0] == 2,
            available[label][1],
        ),
    )


def parse_topology(
    topology: str,
    device_count: int | None = None,
) -> tuple[int, int]:
    count = detected_accelerator_count() if device_count is None else device_count
    available = topology_map(count)
    try:
        return available[topology]
    except KeyError as error:
        if count < 1:
            raise RuntimeError(
                "MiniMax H3 SGLang requires compatible CUDA or ROCm GPUs, but "
                "none are visible to ComfyUI"
            ) from error
        choices = ", ".join(sorted(available))
        raise ValueError(
            f"topology {topology!r} is not valid for the {count} GPU(s) visible "
            f"to ComfyUI; choose one of: {choices}"
        ) from error
