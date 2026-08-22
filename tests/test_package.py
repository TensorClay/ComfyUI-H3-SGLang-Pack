from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

from comfy.patcher_extension import CallbacksMP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_h3_sglang_pack_test"
NODE_ID = "LoadMiniMaxH3DiffusionModelSGLang"
NODE_IDS = {
    NODE_ID,
    "LoraLoaderModelOnlySGLang",
    "MiniMaxH3CacheDiTSGLang",
    "PatchSageAttentionKJSGLang",
    "PatchSolAttnSGLang",
    "PatchFlashAttentionDNSGLang",
}


def load_package():
    existing = sys.modules.get(PACKAGE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        PROJECT_ROOT / "__init__.py",
        submodule_search_locations=[str(PROJECT_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not construct package import specification")
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module


class PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = load_package()
        cls.loader = cls.package.NODE_CLASS_MAPPINGS[NODE_ID]
        cls.topology = sys.modules[f"{PACKAGE_NAME}.runtime.topology"]
        cls.model = sys.modules[f"{PACKAGE_NAME}.runtime.model"]
        cls.manager = sys.modules[f"{PACKAGE_NAME}.runtime.manager"]
        cls.nodes = sys.modules[f"{PACKAGE_NAME}.nodes"]
        cls.catalog = sys.modules[f"{PACKAGE_NAME}.model_catalog"]
        cls.protocol = sys.modules[f"{PACKAGE_NAME}.runtime.protocol"]
        cls.attention = sys.modules[f"{PACKAGE_NAME}.runtime.attention"]
        cls.lifecycle = sys.modules[f"{PACKAGE_NAME}.runtime.lifecycle"]

    def test_loader_uses_a_consistent_id_and_standard_model_contract(self):
        self.assertEqual(set(self.package.NODE_CLASS_MAPPINGS), NODE_IDS)
        self.assertEqual(self.loader.__name__, NODE_ID)
        self.assertEqual(self.loader.RETURN_TYPES, ("MODEL",))
        self.assertEqual(self.loader.RETURN_NAMES, ("model",))
        self.assertEqual(
            self.package.NODE_DISPLAY_NAME_MAPPINGS[NODE_ID],
            "Load MiniMax H3 Diffusion Model (SGLang)",
        )

    def test_schema_sorts_choices_without_changing_the_preferred_default(self):
        required = self.loader.INPUT_TYPES()["required"]
        choices = required["topology"][0]
        self.assertEqual(choices, sorted(choices))
        self.assertEqual(
            required["topology"][1]["default"],
            self.topology.default_topology(),
        )
        self.assertNotIn("performance_mode", required)
        self.assertNotIn("attention_backend", required)
        self.assertEqual(required["hybrid_mode"][0], ["ref2va", "fl2va"])
        self.assertEqual(required["hybrid_mode"][1]["default"], "ref2va")

    def test_attention_nodes_match_the_upstream_contracts(self):
        sage = self.package.NODE_CLASS_MAPPINGS[
            "PatchSageAttentionKJSGLang"
        ]
        sage_inputs = sage.INPUT_TYPES()
        self.assertEqual(
            list(sage_inputs["required"]),
            ["model", "sage_attention"],
        )
        self.assertEqual(
            sage_inputs["required"]["sage_attention"][0],
            self.package.nodes.SAGE_ATTENTION_MODES,
        )
        self.assertFalse(
            sage_inputs["optional"]["allow_compile"][1]["default"]
        )
        self.assertIn(
            "torch.compile",
            sage_inputs["optional"]["allow_compile"][1]["tooltip"],
        )
        self.assertIn(
            "sageattention library",
            sage_inputs["required"]["sage_attention"][1]["tooltip"],
        )
        self.assertTrue(sage.EXPERIMENTAL)
        self.assertEqual(sage.FUNCTION, "patch")
        self.assertEqual(sage.RETURN_TYPES, ("MODEL",))

        flash = self.package.NODE_CLASS_MAPPINGS[
            "PatchFlashAttentionDNSGLang"
        ]
        flash_inputs = flash.INPUT_TYPES()["required"]
        self.assertEqual(list(flash_inputs), ["model", "enabled"])
        self.assertTrue(flash_inputs["enabled"][1]["default"])
        self.assertIn(
            "SGLang's Flash Attention",
            flash_inputs["enabled"][1]["tooltip"],
        )
        self.assertTrue(flash.EXPERIMENTAL)
        self.assertEqual(flash.FUNCTION, "patch")
        self.assertEqual(flash.RETURN_TYPES, ("MODEL",))

        sol = self.package.NODE_CLASS_MAPPINGS["PatchSolAttnSGLang"]
        sol_inputs = sol.INPUT_TYPES()
        self.assertEqual(
            list(sol_inputs["required"]),
            [
                "model",
                "tau",
                "start_percent",
                "end_percent",
                "min_tokens",
                "int8_qk",
                "sink_conditioning",
                "morton",
                "morton_curve",
                "int8_pv",
                "verbose",
                "use_tma",
                "dense_blocks",
            ],
        )
        self.assertEqual(list(sol_inputs["optional"]), ["tau_profile"])
        self.assertIn("Threshold beta", sol_inputs["required"]["tau"][1]["tooltip"])
        self.assertIn(
            "packed text/audio/reference rows",
            sol_inputs["required"]["sink_conditioning"][1]["tooltip"],
        )
        self.assertIn(
            "Per-block tau",
            sol_inputs["optional"]["tau_profile"][1]["tooltip"],
        )
        self.assertTrue(sol.EXPERIMENTAL)
        self.assertEqual(sol.FUNCTION, "patch")
        self.assertEqual(sol.RETURN_TYPES, ("MODEL",))

    def test_worker_resolves_the_base_backend_before_attention_replacement(self):
        source = (PROJECT_ROOT / "worker_pipeline" / "comfyui_minimax_h3_external_pipeline.py").read_text()
        tree = ast.parse(source)
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        for name in ("_configure_sage_attention", "_configure_sol_attention"):
            function = functions[name]
            resolve_line = next(
                node.lineno
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_resolve_attention_backend_once"
            )
            replace_line = next(
                node.lineno
                for node in ast.walk(function)
                if isinstance(node, ast.Attribute)
                and node.attr == "_attention_impl"
                and isinstance(node.ctx, ast.Store)
            )
            self.assertLess(resolve_line, replace_line)

    def test_attention_nodes_store_worker_native_settings(self):
        runtime = mock.Mock()
        patcher = self.model.SGLangH3ModelPatcher(
            torch.nn.Module(),
            torch.device("cpu"),
            torch.device("cpu"),
            runtime=runtime,
            release_runtime=mock.Mock(),
        )
        sage_class = self.package.NODE_CLASS_MAPPINGS[
            "PatchSageAttentionKJSGLang"
        ]
        sage = sage_class().patch(patcher, "auto", False)[0]
        options = sage.model_options["transformer_options"]
        self.assertEqual(options[self.model.ATTENTION_BACKEND_OPTION], "sage_attn")
        self.assertEqual(
            options[self.model.ATTENTION_OPTIONS_OPTION],
            {"sage_attention": "auto", "allow_compile": False},
        )
        self.assertTrue(
            options[self.model.H3_MEMORY_EFFICIENT_SAGE_COMPAT_OPTION]
        )

        flash_class = self.package.NODE_CLASS_MAPPINGS[
            "PatchFlashAttentionDNSGLang"
        ]
        flash = flash_class().patch(patcher, True)[0]
        self.assertEqual(
            flash.model_options["transformer_options"][
                self.model.ATTENTION_BACKEND_OPTION
            ],
            "fa",
        )
        self.assertIs(flash_class().patch(patcher, False)[0], patcher)

        sampling = mock.Mock()
        sampling.percent_to_sigma.side_effect = lambda value: 1.0 - value
        with mock.patch.object(
            patcher,
            "get_model_object",
            return_value=sampling,
        ):
            sol = self.package.NODE_CLASS_MAPPINGS[
                "PatchSolAttnSGLang"
            ]().patch(
                patcher,
                1.3,
                0.2,
                0.9,
                4096,
                True,
                "exact_kv_and_rows",
                False,
                "2d_frame",
                True,
                False,
                False,
                "",
            )[0]
        sol_options = sol.model_options["transformer_options"][
            self.model.ATTENTION_OPTIONS_OPTION
        ]
        self.assertEqual(sol_options["sigma_start"], 0.8)
        self.assertAlmostEqual(sol_options["sigma_end"], 0.1)
        self.assertTrue(sol_options["int8_qk"])
        with self.assertRaisesRegex(NotImplementedError, "Morton"):
            with mock.patch.object(
                patcher,
                "get_model_object",
                return_value=sampling,
            ):
                self.package.NODE_CLASS_MAPPINGS[
                    "PatchSolAttnSGLang"
                ]().patch(
                    patcher,
                    1.3,
                    0.2,
                    0.9,
                    4096,
                    True,
                    "exact_kv_and_rows",
                    True,
                    "2d_frame",
                    True,
                    False,
                    False,
                    "",
                )

    def test_h3_memory_efficient_sage_object_patches_are_bridged(self):
        executor_class = sys.modules[
            f"{PACKAGE_NAME}.runtime.executor"
        ].H3SGLangExecutor
        executor = executor_class(
            mock.Mock(),
            ("blocks.0.attn.qkv_proj.weight",),
        )
        container = torch.nn.Module()
        container.diffusion_model = executor
        patcher = self.model.SGLangH3ModelPatcher(
            container,
            torch.device("cpu"),
            torch.device("cpu"),
            runtime=mock.Mock(),
            release_runtime=mock.Mock(),
        )
        patched = self.package.NODE_CLASS_MAPPINGS[
            "PatchSageAttentionKJSGLang"
        ]().patch(patcher, "auto")[0]

        patched.add_object_patch(
            "diffusion_model.blocks.0.attn.forward",
            object(),
        )

        self.assertNotIn(
            "diffusion_model.blocks.0.attn.forward",
            patched.object_patches,
        )

    def test_topologies_are_generated_and_sorted_for_visible_gpu_count(self):
        self.assertEqual(
            self.topology.topology_choices(2),
            ["TP1 / Ulysses1", "TP1 / Ulysses2", "TP2 / Ulysses1"],
        )
        self.assertEqual(
            self.topology.topology_choices(4),
            [
                "TP1 / Ulysses1",
                "TP1 / Ulysses2",
                "TP1 / Ulysses4",
                "TP2 / Ulysses1",
                "TP2 / Ulysses2",
                "TP4 / Ulysses1",
            ],
        )
        self.assertEqual(
            self.topology.topology_choices(8),
            [
                "TP1 / Ulysses1",
                "TP1 / Ulysses2",
                "TP1 / Ulysses4",
                "TP1 / Ulysses8",
                "TP2 / Ulysses1",
                "TP2 / Ulysses2",
                "TP2 / Ulysses4",
                "TP4 / Ulysses1",
                "TP4 / Ulysses2",
                "TP8 / Ulysses1",
            ],
        )

    def test_default_prefers_tp2_when_available(self):
        self.assertEqual(self.topology.default_topology(2), "TP2 / Ulysses1")
        self.assertEqual(self.topology.default_topology(4), "TP2 / Ulysses2")
        self.assertEqual(self.topology.default_topology(8), "TP2 / Ulysses4")

    def test_topologies_can_use_a_compatible_subset_of_visible_gpus(self):
        self.assertEqual(
            self.topology.topology_choices(3),
            ["TP1 / Ulysses1", "TP1 / Ulysses2", "TP2 / Ulysses1"],
        )
        self.assertEqual(self.topology.default_topology(3), "TP2 / Ulysses1")
        self.assertNotIn("TP1 / Ulysses3", self.topology.topology_choices(3))

    def test_topology_is_revalidated_at_runtime(self):
        self.assertEqual(
            self.topology.parse_topology("TP2 / Ulysses2", 4),
            (2, 2),
        )
        with self.assertRaisesRegex(ValueError, "not valid for the 4 GPU"):
            self.topology.parse_topology("TP2 / Ulysses4", 4)

    def test_runtime_manifests_select_the_external_pipeline(self):
        root_manifest = json.loads(
            (PROJECT_ROOT / "runtime_config" / "model_index.json").read_text()
        )
        self.assertEqual(
            root_manifest["_class_name"],
            "ComfyUIMiniMaxH3ExternalPipeline",
        )
        ref2va_manifest = json.loads(
            (
                PROJECT_ROOT
                / "runtime_config"
                / "Ref2VA"
                / "model_index.json"
            ).read_text()
        )
        self.assertEqual(ref2va_manifest["_minimax_h3"]["partition"], "ref2va")
        fl2va_manifest = json.loads(
            (
                PROJECT_ROOT
                / "runtime_config"
                / "FL2VA"
                / "model_index.json"
            ).read_text()
        )
        self.assertEqual(fl2va_manifest["_minimax_h3"]["partition"], "fl2va")
        self.assertEqual(fl2va_manifest["_minimax_h3"]["tasks"], ["t2va", "fl2va"])

    def test_checkpoint_variant_is_identified_by_filename(self):
        self.assertEqual(
            self.catalog._variant_from_path(
                Path("minimax_h3_fl2va_bf16.safetensors")
            ),
            "fl2va",
        )
        self.assertEqual(
            self.catalog._variant_from_path(
                Path("minimax_h3_ref2va_bf16.safetensors")
            ),
            "ref2va",
        )
        self.assertEqual(
            self.catalog._variant_from_path(
                Path("minimax_h3_hybrid_pruned_bf16.safetensors")
            ),
            "hybrid",
        )
        with self.assertRaisesRegex(ValueError, "filename must identify"):
            self.catalog._variant_from_path(Path("minimax_h3.safetensors"))

    def test_pruned_bf16_is_a_supported_checkpoint_format(self):
        self.assertIn(
            self.catalog.COMFY_PRUNED_BF16,
            self.catalog.SUPPORTED_FORMATS,
        )

    def test_hybrid_mode_selects_the_runtime_variant(self):
        checkpoint = mock.Mock()
        checkpoint.name = "minimax_h3_hybrid_pruned_bf16.safetensors"
        checkpoint.path = Path("/models/minimax_h3_hybrid_pruned_bf16.safetensors")
        checkpoint.format = "comfy_bf16"
        checkpoint.size = 123
        checkpoint.mtime_ns = 456
        checkpoint.variant = "hybrid"
        checkpoint.parameter_keys = ("blocks.0.weight",)

        with mock.patch.object(
            self.nodes,
            "inspect_checkpoint",
            return_value=checkpoint,
        ):
            ref_key = self.nodes._key(
                checkpoint.name,
                "TP1 / Ulysses1",
                "ref2va",
            )
            fl_key = self.nodes._key(
                checkpoint.name,
                "TP1 / Ulysses1",
                "fl2va",
            )

        self.assertEqual(ref_key.model_variant, "ref2va")
        self.assertEqual(fl_key.model_variant, "fl2va")
        self.assertNotEqual(ref_key, fl_key)

    def test_invalid_hybrid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "hybrid mode"):
            self.nodes._key(
                "minimax_h3_hybrid_pruned_bf16.safetensors",
                "TP1 / Ulysses1",
                "t2va",
            )

    def test_keyframes_are_lowered_to_sglang_first_last_signatures(self):
        self.assertEqual(
            self.protocol.sanitize_keyframes(
                [{"resolved_frame_index": 0}, {"resolved_frame_index": 123}],
                124,
            ),
            [{"frame_index": 0}, {"frame_index": -1}],
        )
        with self.assertRaisesRegex(ValueError, "first/last"):
            self.protocol.sanitize_keyframes(
                [{"resolved_frame_index": 42}],
                124,
            )

    def test_execution_signature_distinguishes_same_shape_payloads(self):
        context = torch.empty((1, 4, 5120))
        context_b = torch.empty((1, 4, 5120))
        video = torch.empty((1, 24, 2, 8, 8))
        audio = torch.empty((1, 32, 2, 4))
        ref_a = torch.empty((1, 24, 1, 8, 8))
        ref_b = torch.empty((1, 24, 1, 8, 8))
        tags_a = torch.empty((4,), dtype=torch.long)
        tags_b = torch.empty((4,), dtype=torch.long)
        payload_a = {
            "seed": 1,
            "text_token_tags": tags_a,
            "cond_video_latents": [ref_a],
        }
        payload_b = {
            "seed": 1,
            "text_token_tags": tags_a,
            "cond_video_latents": [ref_b],
        }
        payload_c = {
            "seed": 1,
            "text_token_tags": tags_b,
            "cond_video_latents": [ref_a],
        }

        signature_a = self.protocol.ExecutionSignature.from_inputs(
            context, video, audio, payload_a
        )
        self.assertNotEqual(
            signature_a,
            self.protocol.ExecutionSignature.from_inputs(
                context, video, audio, payload_b
            ),
        )
        self.assertNotEqual(
            signature_a,
            self.protocol.ExecutionSignature.from_inputs(
                context, video, audio, payload_c
            ),
        )
        self.assertNotEqual(
            signature_a,
            self.protocol.ExecutionSignature.from_inputs(
                context_b, video, audio, payload_a
            ),
        )
        self.assertNotEqual(
            signature_a,
            self.protocol.ExecutionSignature.from_inputs(
                context,
                video,
                audio,
                payload_a,
                {
                    "sglang_h3_attention_backend": "sage_attn",
                    "sglang_h3_attention_options": {
                        "sage_attention": "auto",
                    },
                },
            ),
        )

    def test_lora_node_uses_comfyui_resolution_and_stacks_adapters(self):
        lora_class = self.package.NODE_CLASS_MAPPINGS[
            "LoraLoaderModelOnlySGLang"
        ]
        required = lora_class.INPUT_TYPES()["required"]
        self.assertEqual(lora_class.RETURN_TYPES, ("MODEL",))
        self.assertEqual(lora_class.FUNCTION, "load_lora_model_only")
        self.assertEqual(required["model"], ("MODEL",))
        self.assertEqual(
            required["strength_model"],
            (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                },
            ),
        )
        lora_node = lora_class()
        patcher = self.model.SGLangH3ModelPatcher(
            torch.nn.Module(),
            torch.device("cpu"),
            torch.device("cpu"),
            runtime=mock.Mock(),
            release_runtime=mock.Mock(),
        )
        with mock.patch(
            f"{PACKAGE_NAME}.nodes.folder_paths.get_full_path_or_raise",
            return_value="/models/loras/h3.safetensors",
        ):
            patched = lora_node.load_lora_model_only(patcher, "h3.safetensors", 0.75)[0]
        self.assertEqual(
            patched.model_options["transformer_options"][
                self.model.LORAS_OPTION
            ],
            [{"path": "/models/loras/h3.safetensors", "strength": 0.75}],
        )

    def test_zero_strength_custom_lora_is_a_noop(self):
        lora_node = self.package.NODE_CLASS_MAPPINGS[
            "LoraLoaderModelOnlySGLang"
        ]()
        patcher = self.model.SGLangH3ModelPatcher(
            torch.nn.Module(),
            torch.device("cpu"),
            torch.device("cpu"),
            runtime=mock.Mock(),
            release_runtime=mock.Mock(),
        )
        with mock.patch(
            f"{PACKAGE_NAME}.nodes.folder_paths.get_full_path_or_raise"
        ) as resolver:
            result = lora_node.load_lora_model_only(patcher, "h3.safetensors", 0.0)[0]
        self.assertIs(result, patcher)
        resolver.assert_not_called()

    def test_executor_advertises_worker_weights_to_stock_lora_nodes(self):
        executor_class = sys.modules[
            f"{PACKAGE_NAME}.runtime.executor"
        ].H3SGLangExecutor
        executor = executor_class(
            mock.Mock(),
            ("blocks.0.attn.out_proj.weight",),
        )
        state = executor.state_dict(prefix="diffusion_model.")
        self.assertIn(
            "diffusion_model.blocks.0.attn.out_proj.weight",
            state,
        )

    def test_stock_comfy_lora_patch_is_materialized_for_workers(self):
        from comfy.weight_adapter import LoRAAdapter

        runtime = mock.Mock()
        runtime.materialize_lora.return_value = "/tmp/comfy_lora.safetensors"
        patcher = self.model.SGLangH3ModelPatcher(
            torch.nn.Module(),
            torch.device("cpu"),
            torch.device("cpu"),
            runtime=runtime,
            release_runtime=mock.Mock(),
        )
        key = "diffusion_model.blocks.0.attn.out_proj.weight"
        patcher._supported_patch_keys = frozenset({key})
        up = torch.randn(5376, 4)
        down = torch.randn(4, 7168)
        adapter = LoRAAdapter(
            set(),
            (up, down, 4, None, None, None),
        )

        loaded = patcher.add_patches({key: adapter}, 0.75)

        self.assertEqual(loaded, [key])
        tensors = runtime.materialize_lora.call_args.args[0]
        prefix = "diffusion_model.blocks.0.attn.out_proj"
        torch.testing.assert_close(tensors[f"{prefix}.lora_A.weight"], down)
        torch.testing.assert_close(tensors[f"{prefix}.lora_B.weight"], up)
        self.assertNotEqual(
            tensors[f"{prefix}.lora_A.weight"].data_ptr(), down.data_ptr()
        )
        self.assertNotEqual(
            tensors[f"{prefix}.lora_B.weight"].data_ptr(), up.data_ptr()
        )
        self.assertNotIn(f"{prefix}.alpha", tensors)
        self.assertEqual(
            patcher.model_options["transformer_options"][
                self.model.LORAS_OPTION
            ],
            [{"path": "/tmp/comfy_lora.safetensors", "strength": 0.75}],
        )

    def test_stock_lora_preserves_fractional_alpha(self):
        from comfy.weight_adapter import LoRAAdapter

        runtime = mock.Mock()
        runtime.materialize_lora.return_value = "/tmp/comfy_lora.safetensors"
        patcher = self.model.SGLangH3ModelPatcher(
            torch.nn.Module(),
            torch.device("cpu"),
            torch.device("cpu"),
            runtime=runtime,
            release_runtime=mock.Mock(),
        )
        key = "diffusion_model.blocks.0.attn.out_proj.weight"
        patcher._supported_patch_keys = frozenset({key})
        up = torch.randn(8, 4, dtype=torch.float16)
        down = torch.randn(4, 6, dtype=torch.float16)
        adapter = LoRAAdapter(
            set(),
            (up, down, 2.5, None, None, None),
        )

        patcher.add_patches({key: adapter})

        tensors = runtime.materialize_lora.call_args.args[0]
        prefix = "diffusion_model.blocks.0.attn.out_proj"
        torch.testing.assert_close(
            tensors[f"{prefix}.lora_B.weight"],
            up.float() * (2.5 / 4),
        )
        self.assertNotIn(f"{prefix}.alpha", tensors)

    def test_stock_loha_is_lowered_to_equivalent_lora_factors(self):
        from comfy.weight_adapter import LoHaAdapter

        runtime = mock.Mock()
        runtime.materialize_lora.return_value = "/tmp/comfy_loha.safetensors"
        patcher = self.model.SGLangH3ModelPatcher(
            torch.nn.Module(),
            torch.device("cpu"),
            torch.device("cpu"),
            runtime=runtime,
            release_runtime=mock.Mock(),
        )
        key = "diffusion_model.blocks.0.attn.out_proj.weight"
        patcher._supported_patch_keys = frozenset({key})
        w1a, w1b = torch.randn(6, 2), torch.randn(2, 4)
        w2a, w2b = torch.randn(6, 3), torch.randn(3, 4)
        adapter = LoHaAdapter(
            set(),
            (w1a, w1b, 1.5, w2a, w2b, None, None, None),
        )

        patcher.add_patches({key: adapter}, 0.8)

        tensors = runtime.materialize_lora.call_args.args[0]
        prefix = "diffusion_model.blocks.0.attn.out_proj"
        actual = (
            tensors[f"{prefix}.lora_B.weight"]
            @ tensors[f"{prefix}.lora_A.weight"]
        )
        expected = (w1a @ w1b) * (w2a @ w2b) * (1.5 / 2)
        torch.testing.assert_close(actual, expected)

    def test_stock_lokr_is_lowered_to_equivalent_lora_factors(self):
        from comfy.weight_adapter import LoKrAdapter

        runtime = mock.Mock()
        runtime.materialize_lora.return_value = "/tmp/comfy_lokr.safetensors"
        patcher = self.model.SGLangH3ModelPatcher(
            torch.nn.Module(),
            torch.device("cpu"),
            torch.device("cpu"),
            runtime=runtime,
            release_runtime=mock.Mock(),
        )
        key = "diffusion_model.blocks.0.attn.out_proj.weight"
        patcher._supported_patch_keys = frozenset({key})
        w1 = torch.randn(2, 3)
        w2a, w2b = torch.randn(4, 2), torch.randn(2, 5)
        adapter = LoKrAdapter(
            set(),
            (w1, None, 1.25, None, None, w2a, w2b, None, None),
        )

        patcher.add_patches({key: adapter}, 0.8)

        tensors = runtime.materialize_lora.call_args.args[0]
        prefix = "diffusion_model.blocks.0.attn.out_proj"
        actual = (
            tensors[f"{prefix}.lora_B.weight"]
            @ tensors[f"{prefix}.lora_A.weight"]
        )
        expected = torch.kron(w1, w2a @ w2b) * (1.25 / 2)
        torch.testing.assert_close(actual, expected)

    def test_worker_weights_fail_clearly_when_used_as_merge_source(self):
        patcher = self.model.SGLangH3ModelPatcher(
            torch.nn.Module(),
            torch.device("cpu"),
            torch.device("cpu"),
            runtime=mock.Mock(),
            release_runtime=mock.Mock(),
        )
        with self.assertRaisesRegex(NotImplementedError, "model merging"):
            patcher.get_key_patches("diffusion_model.")

    def test_stock_non_lora_weight_patch_fails_closed(self):
        runtime = mock.Mock()
        patcher = self.model.SGLangH3ModelPatcher(
            torch.nn.Module(),
            torch.device("cpu"),
            torch.device("cpu"),
            runtime=runtime,
            release_runtime=mock.Mock(),
        )
        key = "diffusion_model.blocks.0.attn.out_proj.weight"
        patcher._supported_patch_keys = frozenset({key})
        with self.assertRaisesRegex(TypeError, "LoRA, LoHa, and LoKr"):
            patcher.add_patches({key: ("diff", (torch.randn(1),))})
        runtime.materialize_lora.assert_not_called()

    def test_executor_rejects_batches_like_native_h3(self):
        executor_class = sys.modules[
            f"{PACKAGE_NAME}.runtime.executor"
        ].H3SGLangExecutor
        executor = executor_class(mock.Mock())
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            executor._forward(
                [
                    torch.zeros((2, 24, 1, 2, 2)),
                    torch.zeros((2, 32, 2, 1)),
                ],
                torch.tensor([1000.0]),
                torch.zeros((1, 2, 5120)),
                minimax_payload={"text_token_tags": torch.zeros(2)},
            )

    def test_executor_pads_and_crops_non_aligned_video_latents(self):
        executor_class = sys.modules[
            f"{PACKAGE_NAME}.runtime.executor"
        ].H3SGLangExecutor
        runtime = mock.Mock(model_variant="fl2va")
        video = torch.zeros((1, 24, 1, 3, 5))
        audio = torch.zeros((1, 32, 2, 1))
        padded_values = 24 * 1 * 4 * 6
        audio_values = 32 * 2 * 1
        result = mock.Mock(
            noise_pred=torch.ones((1, padded_values + audio_values))
        )
        runtime.send.side_effect = [mock.Mock(), result]
        executor = executor_class(runtime)

        output = executor._forward(
            [video, audio],
            torch.tensor([1000.0]),
            torch.zeros((1, 2, 5120)),
            transformer_options={"sample_sigmas": torch.tensor([1.0, 0.0])},
            minimax_payload={
                "seed": 1,
                "text_token_tags": torch.zeros(2, dtype=torch.long),
            },
        )

        self.assertEqual(tuple(output[0].shape), tuple(video.shape))
        self.assertEqual(tuple(output[1].shape), tuple(audio.shape))
        begin = runtime.send.call_args_list[0].args[0]
        evaluate = runtime.send.call_args_list[1].args[0]
        self.assertEqual(begin["video_shape"], (1, 24, 1, 4, 6))
        self.assertEqual(tuple(evaluate["video_x"].shape), (1, 24, 1, 4, 6))

    def test_diffusion_model_wrappers_run_around_the_remote_executor(self):
        executor = sys.modules[f"{PACKAGE_NAME}.runtime.executor"].H3SGLangExecutor(
            mock.Mock()
        )
        called = []

        def wrapper(next_executor, *args, **kwargs):
            called.append(next_executor.class_obj)
            return ["video", "audio"]

        options = {
            "wrappers": {
                "diffusion_model": {"test": [wrapper]},
            }
        }
        result = executor.forward(
            None, None, None, transformer_options=options
        )
        self.assertEqual(result, ["video", "audio"])
        self.assertEqual(called, [executor])

    def test_worker_local_model_patches_fail_closed(self):
        executor = sys.modules[f"{PACKAGE_NAME}.runtime.executor"].H3SGLangExecutor(
            mock.Mock()
        )
        cases = (
            {"optimized_attention_override": object()},
            {"patches": {"attn1_patch": [object()]}},
            {"patches_replace": {"dit": {("double_block", 0): object()}}},
        )
        for options in cases:
            with self.subTest(options=options):
                with self.assertRaises(NotImplementedError):
                    executor._reject_unsupported(
                        control=None,
                        transformer_options=options,
                    )

    def test_cache_dit_node_stores_serializable_execution_options(self):
        cache_node = self.package.NODE_CLASS_MAPPINGS[
            "MiniMaxH3CacheDiTSGLang"
        ]()
        patcher = self.model.SGLangH3ModelPatcher(
            torch.nn.Module(),
            torch.device("cpu"),
            torch.device("cpu"),
            runtime=mock.Mock(),
            release_runtime=mock.Mock(),
        )
        patched = cache_node.patch_model(patcher, 4, 0.04, 1)[0]
        self.assertIsNot(patched, patcher)
        self.assertEqual(
            patched.model_options["transformer_options"][
                self.model.CACHE_DIT_OPTION
            ],
            {
                "max_warmup_steps": 4,
                "residual_diff_threshold": 0.04,
                "max_continuous_cached_steps": 1,
            },
        )

    def test_model_clone_preserves_the_standard_patcher_contract(self):
        runtime = mock.Mock()
        runtime.is_running = False
        release_runtime = mock.Mock()
        patcher = self.model.SGLangH3ModelPatcher(
            torch.nn.Module(),
            torch.device("cpu"),
            torch.device("cpu"),
            size=1024,
            weight_inplace_update=True,
            runtime=runtime,
            release_runtime=release_runtime,
        )
        callback = mock.Mock()
        patcher.add_callback(CallbacksMP.ON_CLEANUP, callback)

        cloned = patcher.clone(disable_dynamic=True)

        self.assertIsInstance(cloned, self.model.SGLangH3ModelPatcher)
        self.assertIs(cloned.parent, patcher)
        self.assertEqual(cloned.clone_base_uuid, patcher.clone_base_uuid)
        self.assertIs(cloned.runtime, runtime)
        self.assertIs(cloned._release_runtime, release_runtime)
        self.assertEqual(
            cloned.get_all_callbacks(CallbacksMP.ON_CLEANUP),
            [callback],
        )
        self.assertTrue(cloned.weight_inplace_update)
        self.assertEqual(cloned.model_size(), 1024)
        self.assertEqual(cloned.loaded_size(), 0)

    def test_model_detach_releases_runtime_only_for_full_unload(self):
        patcher = object.__new__(self.model.SGLangH3ModelPatcher)
        patcher.runtime = mock.Mock()
        patcher.runtime.is_running = False
        patcher.runtime.attention_backend = "auto"
        patcher.runtime.attention_options = {}
        patcher.model_options = {"transformer_options": {}}
        patcher._release_runtime = mock.Mock()
        patcher.size = 1024
        patcher.pinned = set()
        detached_model = object()

        with mock.patch.object(
            self.model.model_management,
            "unload_all_models",
        ) as unload_all_models:
            patcher.load()
            unload_all_models.assert_called_once_with()
            patcher.runtime.start.assert_called_once_with()
            patcher.runtime.set_attention_backend.assert_called_once_with(
                "auto",
                {},
            )
            patcher.runtime.is_running = True
            patcher.load()
            unload_all_models.assert_called_once_with()
            patcher.runtime.start.assert_called_once_with()
            self.assertEqual(
                patcher.runtime.set_attention_backend.call_count,
                2,
            )
        patcher.runtime.attention_backend = "fa"
        patcher.runtime.attention_options = {"enabled": True}
        self.assertEqual(patcher.loaded_size(), 1024)
        self.assertEqual(patcher.partially_unload(None, 512), 1024)
        patcher._release_runtime.assert_called_once_with(patcher.runtime)
        patcher._release_runtime.reset_mock()

        with mock.patch.object(
            self.model.ModelPatcher,
            "detach",
            return_value=detached_model,
        ) as detach:
            self.assertIs(patcher.detach(unpatch_all=False), detached_model)
            patcher._release_runtime.assert_not_called()
            detach.assert_called_once_with(False)

            self.assertIs(patcher.detach(), detached_model)
            patcher._release_runtime.assert_not_called()

            patcher.runtime.attention_backend = "auto"
            patcher.runtime.attention_options = {}
            self.assertIs(patcher.detach(), detached_model)
            patcher._release_runtime.assert_called_once_with(patcher.runtime)
            detach.assert_called_with(True)
        with mock.patch.object(self.model.SGLangH3ModelPatcher, "__del__"):
            del patcher

    def test_attention_backend_change_stops_the_active_workers(self):
        runtime_class = sys.modules[
            f"{PACKAGE_NAME}.runtime.generator"
        ].H3SGLangRuntime
        runtime = runtime_class(
            model_path="model",
            transformer_weights_path="weights",
            checkpoint_format="comfy_bf16",
            model_variant="fl2va",
            tp_size=1,
            ulysses_degree=1,
            attention_backend="fa",
        )
        generator = mock.Mock()
        runtime.generator = generator
        runtime._templates["shape"] = object()
        runtime._active_loras = (("adapter", 1.0),)

        runtime.set_attention_backend(
            "sage_attn",
            {"sage_attention": "auto"},
        )

        generator.shutdown.assert_called_once_with()
        self.assertEqual(runtime.attention_backend, "sage_attn")
        self.assertEqual(
            runtime.attention_options,
            {"sage_attention": "auto"},
        )
        self.assertFalse(runtime.is_running)
        self.assertEqual(runtime._templates, {})
        self.assertEqual(runtime._active_loras, ())
        runtime._temporary_directory.cleanup()

    def test_worker_errors_discard_the_runtime(self):
        runtime_class = sys.modules[
            f"{PACKAGE_NAME}.runtime.generator"
        ].H3SGLangRuntime
        runtime = runtime_class(
            model_path="model",
            transformer_weights_path="weights",
            checkpoint_format="comfy_bf16",
            model_variant="fl2va",
            tp_size=1,
            ulysses_degree=1,
            attention_backend="auto",
        )
        generator = mock.Mock()
        generator._send_to_scheduler_and_wait_for_response.return_value.error = (
            "worker failed"
        )
        runtime.generator = generator
        runtime._template = mock.Mock(return_value=mock.Mock())

        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            runtime.send({}, (1, 24, 1, 8, 8))

        generator.shutdown.assert_called_once_with()
        self.assertFalse(runtime.is_running)
        runtime._temporary_directory.cleanup()

    def test_runtime_manager_reports_tp_sharded_residency(self):
        manager = self.manager.RuntimeManager()
        key = self.manager.RuntimeKey(
            model_name="model.safetensors",
            checkpoint_path="model.safetensors",
            checkpoint_format="comfy_bf16",
            checkpoint_size=101,
            checkpoint_mtime_ns=0,
            model_variant="ref2va",
            topology="TP2 / Ulysses1",
        )
        runtime = mock.Mock()
        executor = mock.Mock()
        model = mock.Mock()
        with (
            mock.patch.object(
                self.manager,
                "H3SGLangRuntime",
                return_value=runtime,
            ),
            mock.patch.object(
                self.manager,
                "H3SGLangExecutor",
                return_value=executor,
            ),
            mock.patch.object(
                self.manager,
                "create_comfyui_model",
                return_value=model,
            ) as create_model,
        ):
            bundle = manager.get(key)

        self.assertIs(bundle.model, model)
        create_model.assert_called_once_with(
            executor,
            runtime,
            manager.release,
            51,
        )

    def test_runtime_manager_releases_only_its_active_runtime(self):
        manager = self.manager.RuntimeManager()
        active_runtime = mock.Mock()
        key = self.manager.RuntimeKey(
            model_name="model.safetensors",
            checkpoint_path="model.safetensors",
            checkpoint_format="comfy_bf16",
            checkpoint_size=0,
            checkpoint_mtime_ns=0,
            model_variant="ref2va",
            topology="TP1 / Ulysses1",
        )
        bundle = mock.Mock(runtime=active_runtime, key=key)
        manager._bundle = bundle

        manager.release(object())
        bundle.close.assert_not_called()
        self.assertIs(manager._bundle, bundle)

        manager.release(active_runtime)
        bundle.close.assert_called_once_with()
        self.assertIs(manager._bundle, bundle)
        self.assertIs(manager.get(key), bundle)

        manager.unload()
        self.assertEqual(bundle.close.call_count, 2)
        self.assertIs(manager._bundle, bundle)

        manager.shutdown()
        self.assertEqual(bundle.close.call_count, 3)
        self.assertIsNone(manager._bundle)


    def test_runtime_lifecycle_unloads_only_non_sglang_prompts(self):
        manager = mock.Mock()
        lifecycle = self.lifecycle.RuntimeLifecycle(manager)
        native = {
            "prompt": {
                "1": {"class_type": "LoadDiffusionModel", "inputs": {}},
            },
        }
        result = lifecycle.on_prompt(native)
        native_id = result["prompt_id"]
        self.assertRegex(
            native_id,
            r"^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$",
        )

        lifecycle.on_prompt_start(native_id)
        manager.unload.assert_called_once_with()

        sglang_id = "12345678-1234-1234-1234-123456789abc"
        sglang = {
            "prompt_id": sglang_id,
            "prompt": {
                "1": {
                    "class_type": self.lifecycle.SGLANG_LOADER_NODE_ID,
                    "inputs": {},
                },
            },
        }
        self.assertEqual(lifecycle.on_prompt(sglang)["prompt_id"], sglang_id)
        lifecycle.on_prompt_start(sglang_id)
        manager.unload.assert_called_once_with()

        lifecycle.on_prompt({"prompt_id": sglang_id, "prompt": {}})
        lifecycle.on_prompt_end(sglang_id)
        self.assertNotIn(sglang_id, lifecycle._pending)
        self.assertFalse(lifecycle.should_cache(mock.Mock()))

if __name__ == "__main__":
    unittest.main()
