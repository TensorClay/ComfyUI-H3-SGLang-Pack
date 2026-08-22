from __future__ import annotations

import folder_paths

from .model_catalog import inspect_checkpoint, model_choices
from .runtime.manager import RUNTIME_MANAGER, RuntimeKey
from .runtime.model import SGLangH3ModelPatcher
from .runtime.topology import default_topology, topology_choices


HYBRID_MODES = ["ref2va", "fl2va"]


def _key(model_name: str, topology: str, hybrid_mode: str) -> RuntimeKey:
    if hybrid_mode not in HYBRID_MODES:
        raise ValueError(
            f"unsupported MiniMax H3 hybrid mode: {hybrid_mode!r}"
        )
    checkpoint = inspect_checkpoint(model_name)
    model_variant = (
        hybrid_mode if checkpoint.variant == "hybrid" else checkpoint.variant
    )
    return RuntimeKey(
        model_name=checkpoint.name,
        checkpoint_path=str(checkpoint.path),
        checkpoint_format=checkpoint.format,
        checkpoint_size=checkpoint.size,
        checkpoint_mtime_ns=checkpoint.mtime_ns,
        model_variant=model_variant,
        topology=topology,
        parameter_keys=checkpoint.parameter_keys,
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
                "hybrid_mode": (
                    HYBRID_MODES,
                    {
                        "default": "ref2va",
                        "tooltip": (
                            "Select how SGLang should initialize a hybrid "
                            "checkpoint. This control is shown only when the "
                            "selected filename identifies a hybrid model."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "SGLang/MiniMax H3"
    DESCRIPTION = (
        "Loads a local MiniMax H3 checkpoint behind ComfyUI's standard "
        "MODEL contract and distributes denoiser evaluation through SGLang."
    )

    def load_model(
        self,
        model_name: str,
        topology: str,
        hybrid_mode: str = "ref2va",
    ):
        bundle = RUNTIME_MANAGER.get(_key(model_name, topology, hybrid_mode))
        return (bundle.model,)


SAGE_ATTENTION_MODES = [
    "disabled",
    "auto",
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda++",
    "sageattn3",
    "sageattn3_per_block_mean",
]


def _sglang_model(model, node_name: str) -> SGLangH3ModelPatcher:
    if not isinstance(model, SGLangH3ModelPatcher):
        raise TypeError(
            f"{node_name} requires a model from "
            "Load MiniMax H3 Diffusion Model (SGLang)"
        )
    return model


class PatchSageAttentionKJSGLang:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sage_attention": (
                    SAGE_ATTENTION_MODES,
                    {
                        "default": False,
                        "tooltip": (
                            "Patch the attention of the model passing through "
                            "this node to use sageattn. To revert, run this "
                            "node again with the disabled option. Requires the "
                            "sageattention library to be installed."
                        ),
                    },
                ),
            },
            "optional": {
                "allow_compile": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Allow the use of torch.compile for the sage "
                            "attention function, requires latest sageattn "
                            "2.2.0 or higher."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "SGLang/MiniMax H3"
    DESCRIPTION = "Worker-native replacement for Patch Sage Attention KJ."
    EXPERIMENTAL = True

    def patch(self, model, sage_attention, allow_compile=False):
        model = _sglang_model(model, "Patch Sage Attention KJ (SGLang)")
        if sage_attention == "disabled":
            return (model,)
        patched = model.clone()
        patched.set_attention_backend(
            "sage_attn",
            sage_attention=str(sage_attention),
            allow_compile=bool(allow_compile),
        )
        patched.enable_h3_memory_efficient_sage_compatibility()
        return (patched,)


class PatchFlashAttentionDNSGLang:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Enable SGLang's Flash Attention backend. Set to False to "
                            "pass the model through unchanged."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "SGLang/MiniMax H3"
    DESCRIPTION = "Worker-native replacement for Patch Flash Attention DN."
    EXPERIMENTAL = True

    def patch(self, model, enabled):
        model = _sglang_model(model, "Patch Flash Attention DN (SGLang)")
        if not enabled:
            return (model,)
        patched = model.clone()
        patched.set_attention_backend("fa")
        return (patched,)


class PatchSolAttnSGLang:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "tau": (
                    "FLOAT",
                    {
                        "default": 1.3,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": (
                            "Threshold beta. Higher is sparser: 1.0 ~ 16% of "
                            "blocks kept exact, 1.5 ~ 7%, 2.0 ~ 2.7%."
                        ),
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Run dense before this point. The paper uses 0.2."
                        ),
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "min_tokens": (
                    "INT",
                    {
                        "default": 4096,
                        "min": 0,
                        "max": 1 << 20,
                        "step": 512,
                        "tooltip": "Sequences shorter than this stay dense.",
                    },
                ),
                "int8_qk": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "INT8 QK in the exact branch (Sage-style: smoothed "
                            "K, per-token scales). Measured free in quality; "
                            "helps at tau<=1.5, a net loss at tau>=2.0 where "
                            "the quantize pass outweighs the shrinking exact "
                            "branch."
                        ),
                    },
                ),
                "sink_conditioning": (
                    ["exact_kv", "exact_kv_and_rows", "off"],
                    {
                        "default": "exact_kv_and_rows",
                        "tooltip": (
                            "MiniMax-H3 only. exact_kv: every query sees the "
                            "packed text/audio/reference rows exactly (~3% "
                            "cost). exact_kv_and_rows: also runs those query "
                            "rows dense, making the generated audio stream "
                            "exact (~20% cost). No effect on other models."
                        ),
                    },
                ),
                "morton": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Reorder video tokens into Morton (Z-order) so "
                            "each 64-token block is a compact 3D neighbourhood "
                            "instead of a 2-row strip, which makes routing far "
                            "more accurate at a given density. Exactly neutral "
                            "for dense attention. Wan and MiniMax-H3 only; "
                            "logged and skipped elsewhere."
                        ),
                    },
                ),
                "morton_curve": (
                    ["3d", "2d_frame"],
                    {
                        "default": "2d_frame",
                        "tooltip": (
                            "3d interleaves t/h/w equally. 2d_frame Z-orders "
                            "within each frame and leaves frame order alone -- "
                            "use it when the temporal axis is not uniformly "
                            "spaced (MiniMax-H3's frame spacing is non-uniform; "
                            "try this if 3d degrades at some frame counts)."
                        ),
                    },
                ),
                "int8_pv": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Also run the exact branch's P@V in INT8, with a "
                            "per-row P scale and per-channel V scale. PV and QK "
                            "cost the same, so this is the other half of the "
                            "int8 win. Only applies when int8_qk is on."
                        ),
                    },
                ),
                "verbose": ("BOOLEAN", {"default": False}),
                "use_tma": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Use the TMA descriptor kernels instead of the "
                            "pointer ones. Descriptors address strided inputs "
                            "directly, so this no longer copies q/k/v and peak "
                            "VRAM matches the pointer path. Off by default "
                            "because it has not measured faster on any tested "
                            "GPU. Requires SM90+ and Triton 3.3+; ignored "
                            "otherwise. 'verbose' logs the path used."
                        ),
                    },
                ),
                "dense_blocks": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Transformer blocks to keep dense, e.g. '0-2,-1' "
                            "for the first three and the last. Negative indices "
                            "count from the end. The first and last blocks are "
                            "the most approximation-sensitive: their error "
                            "reaches the output with no later block to absorb "
                            "it. Empty means sparsify all."
                        ),
                    },
                ),
            },
            "optional": {
                "tau_profile": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "Per-block tau, overriding the base value. "
                            "'blocks=tau' entries separated by ';' or newlines, "
                            "so a multiline text node works: '0-30=2.0' then "
                            "'39-42=0.9'. '#' starts a comment. Block "
                            "sensitivity varies several-fold across depth, so "
                            "one tau either over-serves the insensitive blocks "
                            "or under-serves the fragile ones — use the Block "
                            "Probe to find them. Leave unconnected for a single "
                            "tau everywhere."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "SGLang/MiniMax H3"
    DESCRIPTION = "Worker-native replacement for Patch Sol-Attn."
    EXPERIMENTAL = True

    def patch(
        self,
        model,
        tau,
        start_percent,
        end_percent,
        min_tokens,
        int8_qk,
        sink_conditioning,
        morton,
        morton_curve,
        int8_pv,
        verbose,
        use_tma,
        dense_blocks,
        tau_profile=None,
    ):
        model = _sglang_model(model, "Patch Sol-Attn (SGLang)")
        if morton:
            raise NotImplementedError(
                "Patch Sol-Attn (SGLang) does not yet support Morton token "
                "reordering; disable morton"
            )
        model_sampling = model.get_model_object("model_sampling")
        patched = model.clone()
        patched.set_attention_backend(
            "sol_attn",
            tau=float(tau),
            start_percent=float(start_percent),
            end_percent=float(end_percent),
            sigma_start=float(model_sampling.percent_to_sigma(start_percent)),
            sigma_end=float(model_sampling.percent_to_sigma(end_percent)),
            min_tokens=int(min_tokens),
            int8_qk=bool(int8_qk),
            sink_conditioning=str(sink_conditioning),
            morton_curve=str(morton_curve),
            int8_pv=bool(int8_pv),
            verbose=bool(verbose),
            use_tma=bool(use_tma),
            dense_blocks=str(dense_blocks),
            tau_profile="" if tau_profile is None else str(tau_profile),
        )
        return (patched,)


class LoraLoaderModelOnlySGLang:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora_model_only"
    CATEGORY = "SGLang/MiniMax H3"
    DESCRIPTION = (
        "Loads a transformer LoRA from ComfyUI's models/loras directory "
        "and applies it on the SGLang workers."
    )

    def load_lora_model_only(self, model, lora_name: str, strength_model: float):
        if not isinstance(model, SGLangH3ModelPatcher):
            raise TypeError(
                "Load LoRA (SGLang) requires a model from Load MiniMax"
                "H3 Diffusion Model (SGLang)"
            )
        if strength_model == 0:
            return (model,)
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        patched = model.clone()
        patched.add_lora(path, strength_model)
        return (patched,)


class MiniMaxH3CacheDiTSGLang:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "max_warmup_steps": (
                    "INT",
                    {"default": 4, "min": 0, "max": 100, "step": 1},
                ),
                "residual_diff_threshold": (
                    "FLOAT",
                    {
                        "default": 0.04,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                    },
                ),
                "max_continuous_cached_steps": (
                    "INT",
                    {"default": 1, "min": 0, "max": 100, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch_model"
    CATEGORY = "SGLang/MiniMax H3"
    DESCRIPTION = (
        "Enables SGLang Cache-DiT for MiniMax H3 denoising. Caching can "
        "improve speed at the cost of approximate rather than lossless output."
    )

    def patch_model(
        self,
        model,
        max_warmup_steps: int,
        residual_diff_threshold: float,
        max_continuous_cached_steps: int,
    ):
        if not isinstance(model, SGLangH3ModelPatcher):
            raise TypeError(
                "MiniMax H3 Cache-DiT (SGLang) requires a model from "
                "Load MiniMax H3 Diffusion Model (SGLang)"
            )
        patched = model.clone()
        patched.configure_cache_dit(
            max_warmup_steps=max_warmup_steps,
            residual_diff_threshold=residual_diff_threshold,
            max_continuous_cached_steps=max_continuous_cached_steps,
        )
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "LoadMiniMaxH3DiffusionModelSGLang": LoadMiniMaxH3DiffusionModelSGLang,
    "LoraLoaderModelOnlySGLang": LoraLoaderModelOnlySGLang,
    "MiniMaxH3CacheDiTSGLang": MiniMaxH3CacheDiTSGLang,
    "PatchSageAttentionKJSGLang": PatchSageAttentionKJSGLang,
    "PatchSolAttnSGLang": PatchSolAttnSGLang,
    "PatchFlashAttentionDNSGLang": PatchFlashAttentionDNSGLang,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadMiniMaxH3DiffusionModelSGLang": (
        "Load MiniMax H3 Diffusion Model (SGLang)"
    ),
    "LoraLoaderModelOnlySGLang": "Load LoRA (SGLang)",
    "MiniMaxH3CacheDiTSGLang": "MiniMax H3 Cache-DiT (SGLang)",
    "PatchSageAttentionKJSGLang": "Patch Sage Attention KJ (SGLang)",
    "PatchSolAttnSGLang": "Patch Sol-Attn (SGLang)",
    "PatchFlashAttentionDNSGLang": "Patch Flash Attention DN (SGLang)",
}
