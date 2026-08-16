from __future__ import annotations

import torch

from comfy import model_management, supported_models
from comfy.model_patcher import ModelPatcher


class SGLangH3ModelPatcher(ModelPatcher):
    def __init__(
        self,
        model,
        load_device=None,
        offload_device=None,
        size=0,
        weight_inplace_update=False,
        *,
        runtime=None,
        release_runtime=None,
    ):
        if load_device is None:
            load_device = model_management.get_torch_device()
        if offload_device is None:
            offload_device = model_management.unet_offload_device()
        super().__init__(
            model,
            load_device=load_device,
            offload_device=offload_device,
            size=size,
            weight_inplace_update=weight_inplace_update,
        )
        if runtime is not None:
            model._sglang_h3_runtime = runtime
            model._sglang_h3_release_runtime = release_runtime
        self.runtime = model._sglang_h3_runtime
        self._release_runtime = model._sglang_h3_release_runtime

    def model_size(self):
        return 0

    def load(
        self,
        device_to=None,
        lowvram_model_memory=0,
        force_patch_weights=False,
        full_load=False,
    ):
        self.runtime.start()

    def patch_model(
        self,
        device_to=None,
        lowvram_model_memory=0,
        load_weights=True,
        force_patch_weights=False,
    ):
        return None

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        return None

    def detach(self, unpatch_all=True):
        model = super().detach(unpatch_all)
        if unpatch_all:
            self._release_runtime(self.runtime)
        return model


def create_comfyui_model(
    executor,
    runtime,
    release_runtime,
) -> SGLangH3ModelPatcher:
    config = supported_models.MiniMaxH3(
        {
            "image_model": "minimax_h3",
            "disable_unet_model_creation": True,
            "dtype": torch.bfloat16,
        }
    )
    config.set_inference_dtype(torch.bfloat16, None)
    model = config.get_model({})
    model.diffusion_model = executor
    return SGLangH3ModelPatcher(
        model,
        runtime=runtime,
        release_runtime=release_runtime,
    )
