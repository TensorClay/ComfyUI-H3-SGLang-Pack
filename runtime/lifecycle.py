from __future__ import annotations

from collections import OrderedDict
import threading
import uuid

from comfy_api.latest import Caching
from comfy_execution.cache_provider import register_cache_provider
from server import PromptServer

from .manager import RUNTIME_MANAGER, RuntimeManager


SGLANG_LOADER_NODE_ID = "LoadMiniMaxH3DiffusionModelSGLang"
MAX_PENDING_PROMPTS = 256


class RuntimeLifecycle(Caching.CacheProvider):
    def __init__(self, manager: RuntimeManager) -> None:
        self.manager = manager
        self._pending: OrderedDict[str, bool] = OrderedDict()
        self._lock = threading.Lock()

    def on_prompt(self, json_data: dict) -> dict:
        prompt = json_data.get("prompt")
        if not isinstance(prompt, dict):
            return json_data
        prompt_id = json_data.get("prompt_id")
        if prompt_id is None:
            prompt_id = str(uuid.uuid4())
            json_data["prompt_id"] = prompt_id
        uses_sglang = any(
            isinstance(node, dict)
            and node.get("class_type") == SGLANG_LOADER_NODE_ID
            for node in prompt.values()
        )
        with self._lock:
            self._pending[str(prompt_id)] = uses_sglang
            self._pending.move_to_end(str(prompt_id))
            while len(self._pending) > MAX_PENDING_PROMPTS:
                self._pending.popitem(last=False)
        return json_data

    def on_prompt_start(self, prompt_id: str) -> None:
        with self._lock:
            uses_sglang = self._pending.pop(prompt_id, None)
        if uses_sglang is False:
            self.manager.unload()

    def on_prompt_end(self, prompt_id: str) -> None:
        with self._lock:
            self._pending.pop(prompt_id, None)

    def should_cache(self, context, value=None) -> bool:
        return False

    async def on_lookup(self, context):
        return None

    async def on_store(self, context, value) -> None:
        return None


RUNTIME_LIFECYCLE = RuntimeLifecycle(RUNTIME_MANAGER)
_REGISTERED = False


def register_runtime_lifecycle() -> None:
    global _REGISTERED
    prompt_server = getattr(PromptServer, "instance", None)
    if _REGISTERED or prompt_server is None:
        return
    prompt_server.add_on_prompt_handler(RUNTIME_LIFECYCLE.on_prompt)
    register_cache_provider(RUNTIME_LIFECYCLE)
    _REGISTERED = True
