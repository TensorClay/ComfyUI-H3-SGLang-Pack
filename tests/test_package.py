from __future__ import annotations

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

    def test_loader_uses_a_consistent_id_and_standard_model_contract(self):
        self.assertEqual(set(self.package.NODE_CLASS_MAPPINGS), {NODE_ID})
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

    def test_model_clone_preserves_the_standard_patcher_contract(self):
        runtime = mock.Mock()
        release_runtime = mock.Mock()
        patcher = self.model.SGLangH3ModelPatcher(
            torch.nn.Module(),
            torch.device("cpu"),
            torch.device("cpu"),
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

    def test_model_detach_releases_runtime_only_for_full_unload(self):
        patcher = object.__new__(self.model.SGLangH3ModelPatcher)
        patcher.runtime = mock.Mock()
        patcher._release_runtime = mock.Mock()
        patcher.pinned = set()
        detached_model = object()

        patcher.load()
        patcher.runtime.start.assert_called_once_with()

        with mock.patch.object(
            self.model.ModelPatcher,
            "detach",
            return_value=detached_model,
        ) as detach:
            self.assertIs(patcher.detach(unpatch_all=False), detached_model)
            patcher._release_runtime.assert_not_called()
            detach.assert_called_once_with(False)

            self.assertIs(patcher.detach(), detached_model)
            patcher._release_runtime.assert_called_once_with(patcher.runtime)
            detach.assert_called_with(True)
        with mock.patch.object(self.model.SGLangH3ModelPatcher, "__del__"):
            del patcher

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
        active_runtime.start.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
