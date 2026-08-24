from __future__ import annotations

from pathlib import Path
import sys
import unittest

from comfy.cli_args import args


# The test suite exercises loader behavior and never initializes a CUDA model.
# Set this before discovery because importing the package imports ComfyUI's
# server module, whose device setup otherwise probes CUDA on CPU-only CI hosts.
args.cpu = True

tests = Path(__file__).resolve().parent
suite = unittest.defaultTestLoader.discover(str(tests), pattern="test_*.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
