"""Make the repo root importable regardless of exactly how the test runner
was invoked (cwd, -t/--top-level-directory, etc). This runs once, before
any test_*.py module in this package is collected, because `unittest
discover` imports the package (this file) before its submodules.
"""
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
