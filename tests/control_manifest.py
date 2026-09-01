"""Which control manifest is in force, derived once.

Spelling the filename in each test is how the version drifts: bumping it moved
the runtime and left fixtures loading a superseded manifest, four times now.
Every one of those looked like a passing test comparing the wrong document.

Read from the panel's source rather than importing its loader, so a test using
this still compares two independent things rather than the panel against itself.
Tests that deliberately pin an older version -- comparing what changed between
them -- should name that version directly and not use this.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "framework/profiles/01-segmentation"

_VERSION = re.compile(r"first-letters-control-policy-(\d+\.\d+\.\d+)\.json")

IN_FORCE_VERSION = _VERSION.search(
    (ROOT / "panel/app.py").read_text(encoding="utf-8")).group(1)
IN_FORCE_PATH = PROFILES / f"first-letters-control-policy-{IN_FORCE_VERSION}.json"
IN_FORCE_ID = f"first-letters-control-policy@{IN_FORCE_VERSION}"


def load() -> dict:
    """The manifest the deployment validates against."""
    return json.loads(IN_FORCE_PATH.read_text(encoding="utf-8"))
