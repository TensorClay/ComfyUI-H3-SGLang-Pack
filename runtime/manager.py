from __future__ import annotations

import atexit
from dataclasses import dataclass
import logging
from pathlib import Path
import threading

from .executor import H3SGLangExecutor
from .generator import H3SGLangRuntime
from .model import create_comfyui_model
from .topology import parse_topology


LOGGER = logging.getLogger(__name__)
BASE_MODEL_PATH = str(Path(__file__).resolve().parents[1] / "runtime_config")


@dataclass(frozen=True)
class RuntimeKey:
    model_name: str
    checkpoint_path: str
    checkpoint_format: str
    checkpoint_size: int
    checkpoint_mtime_ns: int
    model_variant: str
    topology: str
    parameter_keys: tuple[str, ...] = ()

    @property
    def parallelism(self) -> tuple[int, int]:
        return parse_topology(self.topology)


@dataclass
class RuntimeBundle:
    key: RuntimeKey
    runtime: H3SGLangRuntime
    executor: H3SGLangExecutor
    model: object

    def close(self) -> None:
        self.executor.close()
        self.runtime.shutdown()


class RuntimeManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._bundle: RuntimeBundle | None = None

    def get(self, key: RuntimeKey) -> RuntimeBundle:
        tp_size, ulysses_degree = key.parallelism
        with self._lock:
            if self._bundle is not None and self._bundle.key == key:
                return self._bundle
            self._shutdown_locked()
            runtime = H3SGLangRuntime(
                model_path=BASE_MODEL_PATH,
                transformer_weights_path=key.checkpoint_path,
                checkpoint_format=key.checkpoint_format,
                model_variant=key.model_variant,
                tp_size=tp_size,
                ulysses_degree=ulysses_degree,
                attention_backend="auto",
            )
            try:
                executor = H3SGLangExecutor(runtime, key.parameter_keys)
                model = create_comfyui_model(
                    executor,
                    runtime,
                    self.release,
                    (key.checkpoint_size + tp_size - 1) // tp_size,
                )
            except BaseException:
                runtime.shutdown()
                raise
            self._bundle = RuntimeBundle(key, runtime, executor, model)
            return self._bundle

    def release(self, runtime: H3SGLangRuntime) -> None:
        with self._lock:
            if self._bundle is None or self._bundle.runtime is not runtime:
                return
            try:
                self._bundle.close()
            except Exception:
                LOGGER.exception("MiniMax H3 SGLang runtime cleanup failed")

    def unload(self) -> None:
        with self._lock:
            if self._bundle is None:
                return
            try:
                self._bundle.close()
            except Exception:
                LOGGER.exception("MiniMax H3 SGLang runtime cleanup failed")

    def _shutdown_locked(self) -> None:
        if self._bundle is None:
            return
        bundle = self._bundle
        self._bundle = None
        try:
            bundle.close()
        except Exception:
            LOGGER.exception("MiniMax H3 SGLang runtime cleanup failed")

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown_locked()


RUNTIME_MANAGER = RuntimeManager()
atexit.register(RUNTIME_MANAGER.shutdown)
