from __future__ import annotations

import os


pipeline_dir = os.environ.get("COMFYUI_SGLANG_H3_PIPELINE_DIR")
if pipeline_dir:
    try:
        import sglang.multimodal_gen.runtime.pipelines as pipelines

        if pipeline_dir not in pipelines.__path__:
            pipelines.__path__.append(pipeline_dir)
    except Exception:
        # SGLang may not be imported by every child process. The production
        # generator performs an explicit registry check and reports failures.
        pass
