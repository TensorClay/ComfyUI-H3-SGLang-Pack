from __future__ import annotations

import torch

from comfy import model_management, supported_models
from comfy.weight_adapter import LoHaAdapter, LoKrAdapter, LoRAAdapter
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import CallbacksMP


ATTENTION_BACKEND_OPTION = "sglang_h3_attention_backend"
ATTENTION_OPTIONS_OPTION = "sglang_h3_attention_options"
H3_MEMORY_EFFICIENT_SAGE_COMPAT_OPTION = "sglang_h3_mem_eff_sage_compat"
CACHE_DIT_OPTION = "sglang_h3_cache_dit"
LORAS_OPTION = "sglang_h3_loras"


def _copy_matrix(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 2:
        raise TypeError("MiniMax H3 SGLang adapter tensors must be matrices")
    return tensor.detach().to(device="cpu", copy=True).contiguous()


def _lora_factors(adapter: LoRAAdapter) -> tuple[torch.Tensor, torch.Tensor]:
    up, down, alpha, mid, dora_scale, reshape = adapter.weights
    if mid is not None or reshape is not None:
        raise TypeError(
            "MiniMax H3 SGLang does not support convolutional or reshaped LoRA adapters"
        )
    if dora_scale is not None:
        raise TypeError("MiniMax H3 SGLang does not support DoRA adapters")
    up = _copy_matrix(up)
    down = _copy_matrix(down)
    if alpha is not None:
        scale = float(alpha) / down.shape[0]
        if scale != 1.0:
            up = up.float().mul_(scale)
    return up, down


def _loha_factors(adapter: LoHaAdapter) -> tuple[torch.Tensor, torch.Tensor]:
    w1a, w1b, alpha, w2a, w2b, t1, t2, dora_scale = adapter.weights
    if t1 is not None or t2 is not None:
        raise TypeError("MiniMax H3 SGLang does not support convolutional LoHa adapters")
    if dora_scale is not None:
        raise TypeError("MiniMax H3 SGLang does not support DoRA adapters")
    w1a = _copy_matrix(w1a).float()
    w1b = _copy_matrix(w1b).float()
    w2a = _copy_matrix(w2a).float()
    w2b = _copy_matrix(w2b).float()
    up = (w1a.unsqueeze(2) * w2a.unsqueeze(1)).flatten(1)
    down = (w1b.unsqueeze(1) * w2b.unsqueeze(0)).flatten(0, 1)
    if alpha is not None:
        up.mul_(float(alpha) / w1b.shape[0])
    return up.contiguous(), down.contiguous()


def _matrix_factors(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = _copy_matrix(matrix).float()
    rows, columns = matrix.shape
    if rows <= columns:
        return torch.eye(rows), matrix
    return matrix, torch.eye(columns)


def _lokr_factors(adapter: LoKrAdapter) -> tuple[torch.Tensor, torch.Tensor]:
    w1, w2, alpha, w1a, w1b, w2a, w2b, t2, dora_scale = adapter.weights
    if t2 is not None:
        raise TypeError("MiniMax H3 SGLang does not support convolutional LoKr adapters")
    if dora_scale is not None:
        raise TypeError("MiniMax H3 SGLang does not support DoRA adapters")
    rank = None
    if w1 is None:
        up1, down1 = _copy_matrix(w1a).float(), _copy_matrix(w1b).float()
        rank = down1.shape[0]
    else:
        up1, down1 = _matrix_factors(w1)
    if w2 is None:
        up2, down2 = _copy_matrix(w2a).float(), _copy_matrix(w2b).float()
        rank = down2.shape[0]
    else:
        up2, down2 = _matrix_factors(w2)
    up = torch.kron(up1, up2)
    down = torch.kron(down1, down2)
    if alpha is not None and rank is not None:
        up.mul_(float(alpha) / rank)
    return up.contiguous(), down.contiguous()


def _adapter_factors(adapter) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(adapter, LoRAAdapter):
        return _lora_factors(adapter)
    if isinstance(adapter, LoHaAdapter):
        return _loha_factors(adapter)
    if isinstance(adapter, LoKrAdapter):
        return _lokr_factors(adapter)
    raise TypeError(
        "MiniMax H3 SGLang supports linear LoRA, LoHa, and LoKr weight patches; "
        f"{type(adapter).__name__} is not supported by SGLang"
    )


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
        self._supported_patch_keys = frozenset(model.state_dict())

    def add_lora(self, path: str, strength: float) -> None:
        options = self.model_options["transformer_options"]
        loras = options.setdefault(LORAS_OPTION, [])
        loras.append({"path": path, "strength": float(strength)})

    def add_patches(
        self, patches, strength_patch=1.0, strength_model=1.0
    ):
        if float(strength_model) != 1.0:
            raise ValueError(
                "MiniMax H3 SGLang does not support scaling the base model "
                "inside an adapter patch"
            )
        tensors = {}
        loaded = []
        for key, patch in patches.items():
            if not isinstance(key, str) or not key.endswith(".weight"):
                raise TypeError("MiniMax H3 SGLang patches must target weights")
            if key not in self._supported_patch_keys:
                continue
            up, down = _adapter_factors(patch)
            target = key.removeprefix("diffusion_model.").removesuffix(".weight")
            prefix = f"diffusion_model.{target}"
            tensors[f"{prefix}.lora_A.weight"] = down
            tensors[f"{prefix}.lora_B.weight"] = up
            loaded.append(key)

        if not loaded:
            return loaded
        path = self.runtime.materialize_lora(tensors)
        self.add_lora(path, float(strength_patch))
        return loaded

    def get_key_patches(self, filter_prefix=None):
        del filter_prefix
        raise NotImplementedError(
            "MiniMax H3 SGLang cannot expose worker-resident weights for "
            "model merging"
        )

    def set_attention_backend(self, attention_backend: str, **options) -> None:
        transformer_options = self.model_options["transformer_options"]
        transformer_options[ATTENTION_BACKEND_OPTION] = attention_backend
        transformer_options[ATTENTION_OPTIONS_OPTION] = dict(options)

    def _attention_backend(self) -> str:
        return self.model_options["transformer_options"].get(
            ATTENTION_BACKEND_OPTION,
            "auto",
        )

    def _attention_options(self) -> dict:
        return self.model_options["transformer_options"].get(
            ATTENTION_OPTIONS_OPTION,
            {},
        )

    def enable_h3_memory_efficient_sage_compatibility(self) -> None:
        self.model_options["transformer_options"][
            H3_MEMORY_EFFICIENT_SAGE_COMPAT_OPTION
        ] = True

    def add_object_patch(self, name, obj):
        options = self.model_options["transformer_options"]
        prefix = "diffusion_model.blocks."
        suffix = ".attn.forward"
        if (
            options.get(H3_MEMORY_EFFICIENT_SAGE_COMPAT_OPTION)
            and self._attention_backend() == "sage_attn"
            and name.startswith(prefix)
            and name.endswith(suffix)
        ):
            block = name[len(prefix) : -len(suffix)]
            blocks = self.model.diffusion_model.blocks
            if block.isdigit() and int(block) < len(blocks):
                return
        super().add_object_patch(name, obj)

    def configure_cache_dit(
        self,
        *,
        max_warmup_steps: int,
        residual_diff_threshold: float,
        max_continuous_cached_steps: int,
    ) -> None:
        self.model_options["transformer_options"][CACHE_DIT_OPTION] = {
            "max_warmup_steps": int(max_warmup_steps),
            "residual_diff_threshold": float(residual_diff_threshold),
            "max_continuous_cached_steps": int(
                max_continuous_cached_steps
            ),
        }

    def load(
        self,
        device_to=None,
        lowvram_model_memory=0,
        force_patch_weights=False,
        full_load=False,
    ):
        attention_backend = self._attention_backend()
        self.runtime.set_attention_backend(
            attention_backend,
            self._attention_options(),
        )
        if self.runtime.is_running:
            return
        model_management.unload_all_models()
        self.runtime.start()

    def loaded_size(self):
        return self.size if self.runtime.is_running else 0

    def partially_unload(
        self,
        device_to,
        memory_to_free=0,
        force_patch_weights=False,
    ):
        if memory_to_free <= 0 or not self.runtime.is_running:
            return 0
        freed = self.loaded_size()
        self._release_runtime(self.runtime)
        return freed

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
        if (
            unpatch_all
            and self.runtime.attention_backend == self._attention_backend()
            and self.runtime.attention_options == self._attention_options()
        ):
            self._release_runtime(self.runtime)
        return model


def create_comfyui_model(
    executor,
    runtime,
    release_runtime,
    size,
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
    patcher = SGLangH3ModelPatcher(
        model,
        size=size,
        runtime=runtime,
        release_runtime=release_runtime,
    )
    patcher.add_callback(CallbacksMP.ON_CLEANUP, lambda _: executor.close())
    return patcher
