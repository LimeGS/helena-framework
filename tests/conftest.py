"""Shared pytest bootstrap.

The project intentionally keeps reusable harness modules in the repository
root rather than installing them as a wheel.  Add that root explicitly so an
individual test file behaves the same as the full suite and does not depend on
another test mutating ``sys.path`` first.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
root = str(ROOT)
if root not in sys.path:
    sys.path.insert(0, root)
