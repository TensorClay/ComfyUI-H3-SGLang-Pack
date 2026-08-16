from __future__ import annotations

from .model_catalog import inspect_checkpoint, model_choices
from .runtime.manager import RUNTIME_MANAGER, RuntimeKey
from .runtime.topology import default_topology, topology_choices


def _key(
    model_name: str,
    topology: str,
) -> RuntimeKey:
    checkpoint = inspect_checkpoint(model_name)
    return RuntimeKey(
        model_name=checkpoint.name,
        checkpoint_path=str(checkpoint.path),
        checkpoint_format=checkpoint.format,
        checkpoint_size=checkpoint.size,
        checkpoint_mtime_ns=checkpoint.mtime_ns,
        model_variant=checkpoint.variant,
        topology=topology,
    )


class LoadMiniMaxH3DiffusionModelSGLang:
    @classmethod
    def INPUT_TYPES(cls):
        choices = topology_choices()
        return {
            "required": {
                "model_name": (
                    model_choices(),
                ),
                "topology": (
                    choices,
                    {"default": default_topology()},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "SGLang/MiniMax H3"
    DESCRIPTION = (
        "Loads a local MiniMax H3 Ref2VA checkpoint behind ComfyUI's standard "
        "MODEL contract and distributes denoiser evaluation through SGLang."
    )

    def load_model(
        self,
        model_name: str,
        topology: str,
    ):
        bundle = RUNTIME_MANAGER.get(_key(model_name, topology))
        return (bundle.model,)


NODE_CLASS_MAPPINGS = {
    "LoadMiniMaxH3DiffusionModelSGLang": LoadMiniMaxH3DiffusionModelSGLang,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadMiniMaxH3DiffusionModelSGLang": (
        "Load MiniMax H3 Diffusion Model (SGLang)"
    ),
}
