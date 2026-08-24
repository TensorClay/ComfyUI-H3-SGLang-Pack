from __future__ import annotations

import base64
from copy import deepcopy
import importlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import threading
import uuid

from safetensors.torch import save_file

from .attention import sol_attention_provider
from .protocol import CONTRACT_VERSION, PIPELINE_NAME


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = PROJECT_ROOT / "worker_pipeline"
BOOTSTRAP_DIR = PROJECT_ROOT / "worker_bootstrap"
ENVELOPE_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def configure_worker_discovery() -> None:
    os.environ["COMFYUI_SGLANG_H3_PIPELINE_DIR"] = str(PIPELINE_DIR)
    python_path = os.environ.get("PYTHONPATH", "")
    entries = [entry for entry in python_path.split(os.pathsep) if entry]
    bootstrap = str(BOOTSTRAP_DIR)
    if bootstrap not in entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([bootstrap, *entries])

    import sglang.multimodal_gen.runtime.pipelines as pipelines

    pipeline_dir = str(PIPELINE_DIR)
    if pipeline_dir not in pipelines.__path__:
        pipelines.__path__.append(pipeline_dir)

    module_name = (
        "sglang.multimodal_gen.runtime.pipelines."
        "comfyui_minimax_h3_external_pipeline"
    )
    already_loaded = module_name in sys.modules
    module = importlib.import_module(module_name)
    if already_loaded:
        module.register_checkpoint_model()
    from sglang.multimodal_gen import registry

    registry._PIPELINE_REGISTRY[PIPELINE_NAME] = module.EntryClass


class H3SGLangRuntime:
    def __init__(
        self,
        *,
        model_path: str,
        transformer_weights_path: str,
        checkpoint_architecture: str,
        model_variant: str,
        tp_size: int,
        ulysses_degree: int,
        attention_backend: str,
    ) -> None:
        self.model_path = model_path
        self.transformer_weights_path = transformer_weights_path
        self.checkpoint_architecture = checkpoint_architecture
        self.model_variant = model_variant
        self.tp_size = tp_size
        self.ulysses_degree = ulysses_degree
        self.attention_backend = attention_backend
        self.attention_options = {}
        self.generator = None
        self._active_loras: tuple[tuple[str, float], ...] = ()
        self._templates = {}
        self._lock = threading.RLock()
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="comfyui_sglang_h3_"
        )
        self._temp_dir = Path(self._temporary_directory.name)
        self._envelope_image = self._temp_dir / "envelope.png"

    @property
    def topology(self) -> str:
        return f"TP{self.tp_size}/Ulysses{self.ulysses_degree}"

    @property
    def is_running(self) -> bool:
        return self.generator is not None

    def _ensure_envelope_image(self) -> Path:
        expected = base64.b64decode(ENVELOPE_PNG)
        if (
            not self._envelope_image.is_file()
            or self._envelope_image.read_bytes() != expected
        ):
            self._envelope_image.write_bytes(expected)
        return self._envelope_image

    def set_attention_backend(
        self,
        attention_backend: str,
        attention_options: dict | None = None,
    ) -> None:
        if attention_backend not in {"auto", "fa", "sage_attn", "sol_attn"}:
            raise ValueError(
                f"unsupported SGLang attention backend: {attention_backend!r}"
            )
        options = dict(attention_options or {})
        with self._lock:
            if (
                attention_backend == self.attention_backend
                and options == self.attention_options
            ):
                return
            self.shutdown()
            self.attention_backend = attention_backend
            self.attention_options = options

    def start(self) -> None:
        with self._lock:
            if self.generator is not None:
                return
            os.environ["COMFYUI_SGLANG_H3_CHECKPOINT_ARCHITECTURE"] = (
                self.checkpoint_architecture
            )
            os.environ["COMFYUI_SGLANG_H3_CHECKPOINT_PATH"] = (
                self.transformer_weights_path
            )
            os.environ["COMFYUI_SGLANG_H3_ATTENTION_BACKEND"] = (
                self.attention_backend
            )
            os.environ["COMFYUI_SGLANG_H3_ATTENTION_OPTIONS"] = json.dumps(
                self.attention_options,
                sort_keys=True,
            )
            if self.attention_backend == "sol_attn":
                provider = sol_attention_provider()
                if provider is None:
                    raise RuntimeError(
                        "Sol Attention requires ComfyUI-SolAttn_triton in "
                        "ComfyUI/custom_nodes"
                    )
                os.environ["COMFYUI_SGLANG_H3_SOL_PROVIDER"] = str(provider)
            else:
                os.environ.pop("COMFYUI_SGLANG_H3_SOL_PROVIDER", None)
            configure_worker_discovery()
            from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
                DiffGenerator,
            )

            LOGGER.info(
                "Starting MiniMax H3 SGLang runtime: %s, %s",
                self.model_path,
                self.topology,
            )
            self.generator = DiffGenerator.from_pretrained(
                model_path=self.model_path,
                transformer_weights_path=self.transformer_weights_path,
                model_variant=self.model_variant,
                pipeline_class_name=PIPELINE_NAME,
                num_gpus=self.tp_size * self.ulysses_degree,
                tp_size=self.tp_size,
                ulysses_degree=self.ulysses_degree,
                ring_degree=1,
                performance_mode="speed",
                attention_backend=(
                    None
                    if self.attention_backend == "auto"
                    else (
                        "fa"
                        if self.attention_backend in {"sage_attn", "sol_attn"}
                        else self.attention_backend
                    )
                ),
                enable_torch_compile=False,
                warmup_mode="off",
                comfyui_mode=True,
            )
            LOGGER.info("MiniMax H3 SGLang runtime ready: %s", self.topology)

    def materialize_lora(self, tensors: dict[str, torch.Tensor]) -> str:
        path = self._temp_dir / f"comfy_lora_{uuid.uuid4().hex}.safetensors"
        save_file(tensors, path)
        return str(path)

    def configure_loras(self, loras: list[dict]) -> None:
        desired = tuple(
            (str(lora["path"]), float(lora["strength"]))
            for lora in loras
        )
        self.start()
        with self._lock:
            if desired == self._active_loras:
                return
            if not desired:
                if self._active_loras:
                    self.generator.unmerge_lora_weights(target="transformer")
                self._active_loras = ()
                return
            try:
                self.generator.set_lora(
                    lora_nickname=[
                        f"comfyui_h3_{index}" for index in range(len(desired))
                    ],
                    lora_path=[path for path, _ in desired],
                    target=["transformer"] * len(desired),
                    strength=[strength for _, strength in desired],
                    merge_mode="merge",
                )
            except Exception:
                self.shutdown()
                raise
            self._active_loras = desired

    def _template(self, video_shape: tuple[int, ...]):
        key = tuple(video_shape)
        template = self._templates.get(key)
        if template is not None:
            return template
        from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
        from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request

        is_ref2va = self.model_variant == "ref2va"
        conditions = []
        if is_ref2va:
            conditions.append(
                {
                    "type": "image",
                    "uri": self._ensure_envelope_image().as_uri(),
                    "role": "reference",
                }
            )
        sampling = SamplingParams.from_user_sampling_params_args(
            self.generator.server_args.model_path,
            server_args=self.generator.server_args,
            prompt="ComfyUI external MiniMax H3 denoiser",
            task="ref2va" if is_ref2va else "t2va",
            conditions=conditions,
            # SGLang validates transport metadata as a native request. The external
            # worker uses the exact ComfyUI latent shapes carried in external_h3.
            # This valid fixed envelope is scheduler plumbing only.
            target={
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
            num_inference_steps=20,
            flow_shift=12.0,
            audio_flow_shift=3.0,
            seed=0,
            save_output=True,
            output_path=str(self._temp_dir / "envelope"),
            suppress_logs=True,
        )
        template = prepare_request(self.generator.server_args, sampling)
        sampling.prepare_video_request_for_queue(template)
        self._templates[key] = template
        return template

    def send(self, payload: dict, video_shape: tuple[int, ...]):
        self.start()
        with self._lock:
            request = deepcopy(self._template(video_shape))
            request.request_id = str(uuid.uuid4())
            request.extra = {"external_h3": payload}
            output = self.generator._send_to_scheduler_and_wait_for_response(
                [request]
            )
            if output.error:
                error = output.error
                self.shutdown()
                raise RuntimeError(error)
            return output

    def end_execution(
        self, execution_id: str, video_shape: tuple[int, ...]
    ) -> None:
        if self.generator is None:
            return
        payload = {
            "contract_version": CONTRACT_VERSION,
            "action": "end_execution",
            "execution_id": execution_id,
        }
        self.send(payload, video_shape)

    def shutdown(self) -> None:
        with self._lock:
            if self.generator is None:
                return
            generator = self.generator
            self.generator = None
            self._templates.clear()
            self._active_loras = ()
        try:
            generator.shutdown()
        except Exception:
            LOGGER.exception("SGLang H3 runtime shutdown reported an error")
