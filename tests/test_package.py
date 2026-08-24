from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import inspect
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

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
        cls.catalog = sys.modules[f"{PACKAGE_NAME}.model_catalog"]
        cls.quant_bridge = importlib.import_module(
            f"{PACKAGE_NAME}.worker_pipeline.comfyui_quantized_weights"
        )
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
        inputs = self.loader.INPUT_TYPES()
        required = inputs["required"]
        choices = required["topology"][0]
        self.assertEqual(choices, sorted(choices))
        self.assertEqual(
            required["topology"][1]["default"],
            self.topology.default_topology(),
        )
        self.assertEqual(
            list(required), ["model_name", "topology", "model_variant"]
        )
        self.assertNotIn("optional", inputs)
        self.assertEqual(
            required["model_variant"][0],
            ["fl2va", "ref2va"],
        )
        self.assertEqual(
            required["model_variant"][1]["default"], "fl2va"
        )
        self.assertIs(
            inspect.signature(self.loader.load_model)
            .parameters["model_variant"]
            .default,
            inspect.Parameter.empty,
        )
        self.assertIn(
            "filenames are not used",
            required["model_variant"][1]["tooltip"],
        )
        self.assertNotIn("performance_mode", required)
        self.assertNotIn("attention_backend", required)

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

    @staticmethod
    def _synthetic_h3(pruned=False):
        tensors = {
            "video_patch_proj.weight": torch.empty((4, 1)),
            "audio_patch_proj.weight": torch.empty((4, 1)),
            "condition_proj.weight": torch.empty((4, 1)),
            "final_layer.video_out.weight": torch.empty((1, 4)),
            "final_layer.audio_out.weight": torch.empty((1, 4)),
            "blocks.0.attn.q_norm.weight": torch.empty((1,)),
            "blocks.0.attn.k_norm.weight": torch.empty((1,)),
            "blocks.0.attn.qkv_proj.weight": torch.empty((3, 4)),
            "blocks.0.attn.out_proj.weight": torch.empty((4, 1)),
            "blocks.0.mlp.fc1.weight": torch.empty((2, 4)),
            "blocks.0.mlp.fc2.weight": torch.empty((4, 1)),
            "rope.inv_freq": torch.empty((1,)),
        }
        if pruned:
            tensors.update(
                {
                    "adaln_t_table": torch.empty((5, 2)),
                    "blocks.0.adaln_proj.linear.weight": torch.empty((4, 2)),
                    "final_layer.adaln_proj.linear.weight": torch.empty((4, 2)),
                }
            )
        else:
            tensors.update(
                {
                    "time_embedder.proj_in.weight": torch.empty((2, 1)),
                    "time_embedder.proj_out.weight": torch.empty((3, 2)),
                    "blocks.0.adaln_proj.linear.weight": torch.empty((4, 3)),
                    "final_layer.adaln_proj.linear.weight": torch.empty((4, 3)),
                }
            )
        return tensors

    @staticmethod
    def _synthetic_runtime_config(time_embed_dim=3):
        return {
            "hidden_size": 4,
            "num_layers": 1,
            "token_refiner_num_layers": 0,
            "num_attention_heads": 1,
            "attention_head_dim": 1,
            "ffn_hidden_size": 1,
            "latents_dim": 1,
            "audio_latents_dim": 1,
            "patch_size": [1, 1, 1],
            "text_dim": 1,
            "timestep_input_dim": 1,
            "time_embed_hidden_size": 2,
            "time_embed_dim": time_embed_dim,
            "adaln_out_features": 4,
            "final_adaln_out_features": 4,
            "rope_inv_freq_len": 1,
        }

    def test_checkpoint_architecture_is_structural_and_filename_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            full_path = Path(directory) / "anything.safetensors"
            pruned_path = Path(directory) / "also-anything.safetensors"
            save_file(self._synthetic_h3(), str(full_path))
            save_file(self._synthetic_h3(pruned=True), str(pruned_path))
            folders = mock.Mock()
            paths = {
                "anything.safetensors": str(full_path),
                "also-anything.safetensors": str(pruned_path),
            }
            folders.get_full_path_or_raise.side_effect = (
                lambda category, name: paths[name]
            )
            self.catalog._inspect_cached.cache_clear()
            with (
                mock.patch.object(
                    self.catalog, "_folder_paths", return_value=folders
                ),
                mock.patch.object(
                    self.catalog,
                    "_runtime_architecture_config",
                    return_value=self._synthetic_runtime_config(),
                ),
            ):
                full = self.catalog.inspect_checkpoint("anything.safetensors")
                pruned = self.catalog.inspect_checkpoint("also-anything.safetensors")

        self.assertEqual(full.architecture, "full")
        self.assertEqual(pruned.architecture, "pruned_adaln")
        self.assertFalse(hasattr(full, "variant"))

    def test_checkpoint_architecture_must_match_the_bundled_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "minimax_h3_fl2va_tiny.safetensors"
            save_file(self._synthetic_h3(), str(path))
            stat = path.stat()
            self.catalog._inspect_cached.cache_clear()
            with self.assertRaisesRegex(ValueError, "bundled runtime"):
                self.catalog._inspect_cached(
                    str(path), stat.st_size, stat.st_mtime_ns
                )

    def test_checkpoint_block_count_must_match_the_bundled_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "minimax_h3_fl2va_missing_block.safetensors"
            save_file(self._synthetic_h3(), str(path))
            stat = path.stat()
            config = self._synthetic_runtime_config()
            config["num_layers"] = 2
            self.catalog._inspect_cached.cache_clear()
            with (
                mock.patch.object(
                    self.catalog,
                    "_runtime_architecture_config",
                    return_value=config,
                ),
                self.assertRaisesRegex(ValueError, "blocks coverage"),
            ):
                self.catalog._inspect_cached(
                    str(path), stat.st_size, stat.st_mtime_ns
                )

    def test_real_checkpoint_cold_loads_in_sglang_when_configured(self):
        checkpoint_value = os.environ.get("H3_SGLANG_INTEGRATION_CHECKPOINT")
        if not checkpoint_value:
            self.skipTest("H3_SGLANG_INTEGRATION_CHECKPOINT is not set")

        checkpoint_path = Path(checkpoint_value).expanduser().resolve()
        if not checkpoint_path.is_file():
            self.fail(f"integration checkpoint does not exist: {checkpoint_path}")
        stat = checkpoint_path.stat()
        architecture, _restored_size, _parameter_keys = (
            self.catalog._inspect_cached(
                str(checkpoint_path), stat.st_size, stat.st_mtime_ns
            )
        )
        variant = os.environ.get("H3_SGLANG_INTEGRATION_VARIANT")
        if variant not in self.catalog.SUPPORTED_VARIANTS:
            self.fail(
                "H3_SGLANG_INTEGRATION_VARIANT must be set to fl2va or ref2va"
            )
        runtime_class = sys.modules[
            f"{PACKAGE_NAME}.runtime.generator"
        ].H3SGLangRuntime
        runtime = runtime_class(
            model_path=str(PROJECT_ROOT / "runtime_config"),
            transformer_weights_path=str(checkpoint_path),
            checkpoint_architecture=architecture,
            model_variant=variant,
            tp_size=int(os.environ.get("H3_SGLANG_INTEGRATION_TP", "1")),
            ulysses_degree=int(
                os.environ.get("H3_SGLANG_INTEGRATION_ULYSSES", "1")
            ),
            attention_backend="auto",
        )
        try:
            runtime.start()
            self.assertTrue(runtime.is_running)
        finally:
            runtime.shutdown()

    def test_explicit_variant_is_part_of_the_runtime_key(self):
        checkpoint = self.catalog.H3Checkpoint(
            name="minimax_h3_fl2va_hybrid.safetensors",
            path=Path("/models/minimax_h3_fl2va_hybrid.safetensors"),
            architecture="full",
            size=123,
            restored_size=789,
            mtime_ns=456,
            parameter_keys=("blocks.0.attn.qkv_proj.weight",),
        )
        with mock.patch.object(
            self.package.nodes, "inspect_checkpoint", return_value=checkpoint
        ):
            fl2va = self.package.nodes._key(
                checkpoint.name, "TP1 / Ulysses1", "fl2va"
            )
            ref2va = self.package.nodes._key(
                checkpoint.name, "TP1 / Ulysses1", "ref2va"
            )
        self.assertNotEqual(fl2va, ref2va)
        self.assertEqual(fl2va.model_variant, "fl2va")
        self.assertEqual(ref2va.model_variant, "ref2va")
        self.assertEqual(fl2va.checkpoint_restored_size, 789)
        with self.assertRaisesRegex(ValueError, "unsupported.*model variant"):
            self.package.nodes._key(
                checkpoint.name, "TP1 / Ulysses1", "hybrid"
            )

    def test_loader_passes_the_explicit_variant_unchanged(self):
        bundle = mock.Mock(model=object())
        with (
            mock.patch.object(self.package.nodes, "_key", return_value=object()) as key,
            mock.patch.object(
                self.package.nodes.RUNTIME_MANAGER, "get", return_value=bundle
            ),
        ):
            result = self.loader().load_model(
                "misleading_fl2va_hybrid.safetensors",
                "TP1 / Ulysses1",
                "ref2va",
            )
        key.assert_called_once_with(
            "misleading_fl2va_hybrid.safetensors",
            "TP1 / Ulysses1",
            "ref2va",
        )
        self.assertIs(result[0], bundle.model)

    def test_w4a8_quantized_group_is_restored_structurally(self):
        metadata = {
            "format": "asym_w4a8_int8",
            "group_size": 16,
            "convrot_groupsize": 256,
        }
        metadata_tensor = torch.tensor(
            list(json.dumps(metadata).encode("utf-8")), dtype=torch.uint8
        )
        tensors = {
            "layer.comfy_quant": metadata_tensor,
            "layer.weight": torch.zeros((2, 2), dtype=torch.int8),
            "layer.weight_codebook": torch.arange(16, dtype=torch.float32),
            "layer.weight_s_channel": torch.ones((2,), dtype=torch.float32),
            "layer.weight_s_rel": torch.ones((2, 1), dtype=torch.uint8),
        }
        seen = {}

        def restore(state_dict, shape, dtype):
            seen.update(state_dict)
            self.assertEqual(shape, (2, 4))
            self.assertEqual(dtype, torch.bfloat16)
            return torch.ones(shape, dtype=dtype)

        quant_ops = ModuleType("comfy.quant_ops")
        quant_ops.QUANT_ALGOS = {"asym_w4a8_int8": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "w4a8.safetensors"
            save_file(tensors, str(path))
            restorer = self.quant_bridge.QuantizedWeightRestorer(
                path,
                # Simulate a TP2 SGLang parameter. Restoration must still use
                # the checkpoint's global (2, 4) matrix shape.
                lambda name: ((1, 4), torch.bfloat16),
                restore=restore,
            )
            with mock.patch.dict(sys.modules, {"comfy.quant_ops": quant_ops}):
                restored = list(restorer.iter_restored(tensors.items()))

        self.assertEqual(restored[0][0], "layer.weight")
        self.assertEqual(tuple(restored[0][1].shape), (2, 4))
        self.assertEqual(
            set(seen),
            {
                "weight",
                "comfy_quant",
                "weight_codebook",
                "weight_s_channel",
                "weight_s_rel",
            },
        )

    def test_w4a8_restoration_matches_comfy_kitchen_dequantization(self):
        try:
            from comfy.quant_ops import QUANT_ALGOS
            from comfy_kitchen.tensor import QuantizedTensor
        except ImportError as error:
            self.skipTest(str(error))
        if "asym_w4a8_int8" not in QUANT_ALGOS:
            self.skipTest("installed ComfyUI does not provide W4A8")

        torch.manual_seed(7)
        source = torch.randn((16, 256), dtype=torch.bfloat16)
        quantized = QuantizedTensor.from_float(
            source,
            "AsymW4A8Int8Layout",
            group_size=16,
            convrot_groupsize=256,
        )
        state_dict = quantized.state_dict("weight")
        state_dict["comfy_quant"] = torch.tensor(
            list(
                json.dumps(
                    {
                        "format": "asym_w4a8_int8",
                        "group_size": 16,
                        "convrot_groupsize": 256,
                    }
                ).encode("utf-8")
            ),
            dtype=torch.uint8,
        )
        restored = self.quant_bridge.restore_weight_with_comfy(
            state_dict,
            tuple(source.shape),
            torch.bfloat16,
        )
        self.assertTrue(torch.equal(restored, quantized.dequantize()))

    def test_awq_pre_quant_scale_is_folded_into_restored_weight(self):
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        scale = torch.tensor([0.5, 2.0])
        restored = self.quant_bridge.fold_pre_quant_scale(weight, scale)
        self.assertTrue(
            torch.equal(restored, torch.tensor([[0.5, 4.0], [1.5, 8.0]]))
        )
        with self.assertRaisesRegex(ValueError, "cannot be folded"):
            self.quant_bridge.fold_pre_quant_scale(weight, torch.ones(3))

    def test_generic_bridge_restores_other_core_comfy_quant_layouts(self):
        try:
            from comfy.quant_ops import QUANT_ALGOS
            from comfy_kitchen.tensor import QuantizedTensor
        except ImportError as error:
            self.skipTest(str(error))

        cases = (
            ("float8_e4m3fn", "TensorCoreFP8E4M3Layout", {}, {}),
            ("float8_e5m2", "TensorCoreFP8E5M2Layout", {}, {}),
            ("mxfp8", "TensorCoreMXFP8Layout", {}, {}),
            ("nvfp4", "TensorCoreNVFP4Layout", {}, {}),
            (
                "int8_tensorwise",
                "TensorWiseINT8Layout",
                {"per_channel": True, "convrot": True},
                {"convrot": True, "convrot_groupsize": 256},
            ),
            ("convrot_w4a4", "TensorCoreConvRotW4A4Layout", {}, {}),
        )
        torch.manual_seed(11)
        source = torch.randn((32, 256), dtype=torch.bfloat16)
        tested = set()
        for quant_format, layout, quant_kwargs, metadata_kwargs in cases:
            if quant_format not in QUANT_ALGOS:
                continue
            quantized = QuantizedTensor.from_float(
                source, layout, **quant_kwargs
            )
            metadata = {"format": quant_format, **metadata_kwargs}
            state_dict = quantized.state_dict("weight")
            state_dict["comfy_quant"] = torch.tensor(
                list(json.dumps(metadata).encode("utf-8")),
                dtype=torch.uint8,
            )
            original_shape = self.quant_bridge.checkpoint_weight_shape(
                metadata, state_dict["weight"]
            )
            restored = self.quant_bridge.restore_weight_with_comfy(
                state_dict,
                original_shape,
                torch.bfloat16,
            )
            self.assertTrue(
                torch.equal(restored, quantized.dequantize()),
                quant_format,
            )
            tested.add(quant_format)
        self.assertTrue(
            {
                "float8_e4m3fn",
                "float8_e5m2",
                "nvfp4",
                "int8_tensorwise",
                "convrot_w4a4",
            }.issubset(tested)
        )

    def test_legacy_header_quantization_metadata_is_normalized(self):
        config = {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": 256,
        }
        tensors = {
            "layer.weight": torch.zeros((2, 4), dtype=torch.int8),
            "layer.weight_scale": torch.ones((2, 1)),
        }
        quant_ops = ModuleType("comfy.quant_ops")
        quant_ops.QUANT_ALGOS = {"int8_tensorwise": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.safetensors"
            save_file(
                tensors,
                str(path),
                metadata={
                    "_quantization_metadata": json.dumps(
                        {"layers": {"layer": config}}
                    )
                },
            )
            restorer = self.quant_bridge.QuantizedWeightRestorer(
                path,
                lambda name: ((2, 4), torch.bfloat16),
                restore=lambda state, shape, dtype: torch.ones(shape, dtype=dtype),
            )
            with mock.patch.dict(sys.modules, {"comfy.quant_ops": quant_ops}):
                restored = list(restorer.iter_restored(tensors.items()))
        self.assertEqual([name for name, _ in restored], ["layer.weight"])

    def test_header_w4a8_is_preflighted_and_restored_as_one_group(self):
        config = {
            "format": "asym_w4a8_int8",
            "group_size": 16,
            "convrot_groupsize": 256,
        }
        tensors = {
            "layer.weight": torch.zeros((2, 2), dtype=torch.int8),
            "layer.weight_codebook": torch.arange(16, dtype=torch.float32),
            "layer.weight_s_channel": torch.ones((2,), dtype=torch.float32),
            "layer.weight_s_rel": torch.ones((2, 1), dtype=torch.uint8),
        }
        quant_ops = ModuleType("comfy.quant_ops")
        quant_ops.QUANT_ALGOS = {"asym_w4a8_int8": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "header-w4a8.safetensors"
            save_file(
                tensors,
                str(path),
                metadata={
                    "_quantization_metadata": json.dumps(
                        {"layers": {"layer": config}}
                    )
                },
            )
            with mock.patch.dict(sys.modules, {"comfy.quant_ops": quant_ops}):
                keys, layers = self.quant_bridge.validate_checkpoint_quantization(
                    path
                )
                restorer = self.quant_bridge.QuantizedWeightRestorer(
                    path,
                    lambda name: ((1, 4), torch.bfloat16),
                    restore=lambda state, shape, dtype: torch.ones(
                        shape, dtype=dtype
                    ),
                )
                restored = list(restorer.iter_restored(tensors.items()))
        self.assertIn("layer.weight_s_rel", keys)
        self.assertEqual(layers, {"layer.weight": config})
        self.assertEqual(tuple(restored[0][1].shape), (2, 4))

    def test_quantization_preflight_rejects_unknown_and_incomplete_formats(self):
        quant_ops = ModuleType("comfy.quant_ops")
        quant_ops.QUANT_ALGOS = {
            "asym_w4a8_int8": {},
            "float8_e4m3fn": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            unknown = directory / "unknown.safetensors"
            missing = directory / "missing.safetensors"
            missing_fp8 = directory / "missing-fp8.safetensors"
            save_file(
                {
                    "layer.weight": torch.zeros((2, 2), dtype=torch.int8),
                    "layer.comfy_quant": torch.tensor(
                        list(json.dumps({"format": "private_w3"}).encode()),
                        dtype=torch.uint8,
                    ),
                },
                str(unknown),
            )
            save_file(
                {
                    "layer.weight": torch.zeros((2, 2), dtype=torch.int8),
                    "layer.comfy_quant": torch.tensor(
                        list(
                            json.dumps(
                                {"format": "asym_w4a8_int8"}
                            ).encode()
                        ),
                        dtype=torch.uint8,
                    ),
                },
                str(missing),
            )
            save_file(
                {
                    "layer.weight": torch.zeros((2, 2)),
                    "layer.comfy_quant": torch.tensor(
                        list(
                            json.dumps({"format": "float8_e4m3fn"}).encode()
                        ),
                        dtype=torch.uint8,
                    ),
                },
                str(missing_fp8),
            )
            with mock.patch.dict(sys.modules, {"comfy.quant_ops": quant_ops}):
                with self.assertRaisesRegex(ValueError, "private_w3"):
                    self.quant_bridge.validate_checkpoint_quantization(unknown)
                with self.assertRaisesRegex(ValueError, "weight_s_rel"):
                    self.quant_bridge.validate_checkpoint_quantization(missing)
                with self.assertRaisesRegex(ValueError, "weight_scale"):
                    self.quant_bridge.validate_checkpoint_quantization(missing_fp8)

    def test_legacy_scaled_fp8_is_rejected_before_model_discovery(self):
        tensors = self._synthetic_h3()
        tensors["scaled_fp8"] = torch.ones((1,))
        tensors["blocks.0.attn.qkv_proj.scale_weight"] = torch.ones((1,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "minimax_h3_fl2va_scaled_fp8.safetensors"
            save_file(tensors, str(path))
            folders = mock.Mock()
            folders.get_filename_list.return_value = [path.name]
            folders.get_full_path_or_raise.return_value = str(path)
            self.catalog._inspect_cached.cache_clear()
            with mock.patch.object(
                self.catalog, "_folder_paths", return_value=folders
            ):
                self.assertEqual(self.catalog.compatible_model_names(), [])
                with self.assertRaisesRegex(ValueError, "legacy scaled_fp8"):
                    self.catalog.inspect_checkpoint(path.name)

    def test_malformed_legacy_quantization_header_is_filtered(self):
        tensors = self._synthetic_h3()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "minimax_h3_fl2va_bad_header.safetensors"
            save_file(
                tensors,
                str(path),
                metadata={"_quantization_metadata": json.dumps({})},
            )
            folders = mock.Mock()
            folders.get_filename_list.return_value = [path.name]
            folders.get_full_path_or_raise.return_value = str(path)
            self.catalog._inspect_cached.cache_clear()
            with mock.patch.object(
                self.catalog, "_folder_paths", return_value=folders
            ):
                self.assertEqual(self.catalog.compatible_model_names(), [])

    def test_non_h3_checkpoint_is_rejected_before_quantization_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "other_quantized_model.safetensors"
            save_file(
                {
                    "unrelated.weight": torch.zeros((2, 2), dtype=torch.int8),
                    "unrelated.weight_scale": torch.ones((1,)),
                },
                str(path),
            )
            folders = mock.Mock()
            folders.get_filename_list.return_value = [path.name]
            folders.get_full_path_or_raise.return_value = str(path)
            self.catalog._inspect_cached.cache_clear()
            with (
                mock.patch.object(
                    self.catalog, "_folder_paths", return_value=folders
                ),
                mock.patch.object(
                    self.catalog, "validate_checkpoint_quantization"
                ) as validate,
            ):
                self.assertEqual(self.catalog.compatible_model_names(), [])
            validate.assert_not_called()

    def test_catalog_filters_structural_h3_with_unsupported_quantization(self):
        tensors = self._synthetic_h3(pruned=True)
        tensors["blocks.0.attn.qkv_proj.weight"] = torch.zeros(
            (3, 2), dtype=torch.int8
        )
        tensors["blocks.0.attn.qkv_proj.comfy_quant"] = torch.tensor(
            list(json.dumps({"format": "private_w3"}).encode()),
            dtype=torch.uint8,
        )
        quant_ops = ModuleType("comfy.quant_ops")
        quant_ops.QUANT_ALGOS = {}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "minimax_h3_ref2va_private.safetensors"
            save_file(tensors, str(path))
            folders = mock.Mock()
            folders.get_filename_list.return_value = [path.name]
            folders.get_full_path_or_raise.return_value = str(path)
            self.catalog._inspect_cached.cache_clear()
            with (
                mock.patch.object(
                    self.catalog, "_folder_paths", return_value=folders
                ),
                mock.patch.dict(sys.modules, {"comfy.quant_ops": quant_ops}),
            ):
                self.assertEqual(self.catalog.compatible_model_names(), [])
                with self.assertRaisesRegex(ValueError, "private_w3"):
                    self.catalog.inspect_checkpoint(path.name)

    def test_catalog_reports_restored_not_compressed_quantized_size(self):
        config = {
            "format": "asym_w4a8_int8",
            "group_size": 16,
            "convrot_groupsize": 256,
        }
        tensors = self._synthetic_h3(pruned=True)
        tensors["blocks.0.attn.qkv_proj.weight"] = torch.zeros(
            (3, 2), dtype=torch.int8
        )
        tensors["blocks.0.attn.qkv_proj.weight_s_rel"] = torch.ones(
            (3, 1), dtype=torch.uint8
        )
        quant_ops = ModuleType("comfy.quant_ops")
        quant_ops.QUANT_ALGOS = {"asym_w4a8_int8": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "minimax_h3_ref2va_w4a8.safetensors"
            save_file(
                tensors,
                str(path),
                metadata={
                    "_quantization_metadata": json.dumps(
                        {"layers": {"blocks.0.attn.qkv_proj": config}}
                    )
                },
            )
            folders = mock.Mock()
            folders.get_full_path_or_raise.return_value = str(path)
            self.catalog._inspect_cached.cache_clear()
            with (
                mock.patch.object(
                    self.catalog, "_folder_paths", return_value=folders
                ),
                mock.patch.dict(sys.modules, {"comfy.quant_ops": quant_ops}),
                mock.patch.object(
                    self.catalog,
                    "_runtime_architecture_config",
                    return_value=self._synthetic_runtime_config(),
                ),
            ):
                checkpoint = self.catalog.inspect_checkpoint(path.name)
        # The non-quantized tensors contain 65 FP32 elements after replacing
        # QKV; the logical packed QKV matrix is 3x4 BF16.
        self.assertEqual(checkpoint.restored_size, 65 * 4 + 3 * 4 * 2)
        self.assertNotIn(
            "blocks.0.attn.qkv_proj.weight_s_rel", checkpoint.parameter_keys
        )

    def test_packed_full_architecture_uses_logical_weight_dimensions(self):
        config = {"format": "nvfp4"}
        tensors = self._synthetic_h3()
        tensors["time_embedder.proj_out.weight"] = torch.empty((4, 2))
        for prefix in (
            "blocks.0.adaln_proj.linear",
            "final_layer.adaln_proj.linear",
        ):
            tensors[f"{prefix}.weight"] = torch.zeros((4, 2), dtype=torch.uint8)
            tensors[f"{prefix}.weight_scale"] = torch.ones((1,))
            tensors[f"{prefix}.weight_scale_2"] = torch.ones((1,))
            tensors[f"{prefix}.comfy_quant"] = torch.tensor(
                list(json.dumps(config).encode()), dtype=torch.uint8
            )
        quant_ops = ModuleType("comfy.quant_ops")
        quant_ops.QUANT_ALGOS = {"nvfp4": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "minimax_h3_ref2va_nvfp4.safetensors"
            save_file(tensors, str(path))
            folders = mock.Mock()
            folders.get_full_path_or_raise.return_value = str(path)
            self.catalog._inspect_cached.cache_clear()
            with (
                mock.patch.object(
                    self.catalog, "_folder_paths", return_value=folders
                ),
                mock.patch.dict(sys.modules, {"comfy.quant_ops": quant_ops}),
                mock.patch.object(
                    self.catalog,
                    "_runtime_architecture_config",
                    return_value=self._synthetic_runtime_config(time_embed_dim=4),
                ),
            ):
                checkpoint = self.catalog.inspect_checkpoint(path.name)
        self.assertEqual(checkpoint.architecture, "full")

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
            checkpoint_architecture="full",
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
            checkpoint_architecture="full",
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
            checkpoint_architecture="full",
            checkpoint_size=101,
            checkpoint_restored_size=301,
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
            mock.patch.object(
                self.manager,
                "parse_topology",
                return_value=(2, 1),
            ),
        ):
            bundle = manager.get(key)

        self.assertIs(bundle.model, model)
        create_model.assert_called_once_with(
            executor,
            runtime,
            manager.release,
            151,
        )

    def test_runtime_manager_releases_only_its_active_runtime(self):
        manager = self.manager.RuntimeManager()
        active_runtime = mock.Mock()
        key = self.manager.RuntimeKey(
            model_name="model.safetensors",
            checkpoint_path="model.safetensors",
            checkpoint_architecture="full",
            checkpoint_size=0,
            checkpoint_restored_size=0,
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
        with mock.patch.object(
            self.manager, "parse_topology", return_value=(1, 1)
        ):
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
