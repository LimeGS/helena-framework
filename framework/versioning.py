"""Small, dependency-free version and identity helpers for Helena Framework.

Scientific identity is deliberately separate from release identity: a framework
release may add a feature without changing a locked plan, while a new plan may
be created under the same release.  See the panel's developer reference.
"""

from __future__ import annotations

import re
from pathlib import Path


SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "VERSION"


def is_semver(value: str) -> bool:
    """Return whether *value* is a strict Semantic Version 2.0 identifier."""

    return bool(SEMVER_RE.fullmatch(value))


def framework_version() -> str:
    """Read the single release-version source of truth, fail closed if invalid."""

    value = VERSION_FILE.read_text(encoding="utf-8").strip()
    if not is_semver(value):
        raise RuntimeError(f"VERSION is not valid SemVer: {value!r}")
    return value
