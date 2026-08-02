"""The rename must not reach the frozen namespace.

This test exists because a blind rename of `campaignx.` to `helena.` broke 22
hash locks the first time it was tried. It is here so the next person -- or the
next model -- does not repeat it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.contracts import naming


def test_the_system_is_named_once_and_in_one_place():
    assert naming.NAME == "Helena Framework"


def test_the_schema_namespace_is_frozen_at_its_historical_value():
    """Renaming it would change the sha256 of every frozen artefact that
    declares it, breaking bindings in receipts this repository does not own."""
    assert naming.SCHEMA_NAMESPACE == "campaignx."
    assert naming.schema("mission.v1") == "campaignx.mission.v1"


def test_no_frozen_artefact_declares_the_product_name():
    """Profiles and registries must keep the historical namespace."""
    offenders = []
    for directory in ("framework/profiles", "framework/registries",
                      "framework/contracts/schemas"):
        for path in (ROOT / directory).rglob("*.json"):
            try:
                declared = json.loads(path.read_text()).get("schema")
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(declared, str) and declared.startswith("helena."):
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, (
        "these artefacts were renamed into the product namespace, which moves "
        f"their hash: {offenders}")


def test_the_old_product_name_is_gone_from_prose():
    """The identity rename should be complete in code and documentation."""
    out = subprocess.run(
        # Patches are exact-bytes artefacts: what they contain is what gets
        # applied to vendored source, so they keep whatever they were written
        # with. JSON is excluded for the hash reason above.
        ["git", "grep", "-Il", "Campaign X", "--",
         "framework", "panel", "tests", "docs", "scripts", "README.md",
         ":!*/vendored/*", ":!*.json", ":!*.patch",
         # These two document the rename; they are the one place the old name
         # is supposed to survive.
         ":!framework/contracts/naming.py", ":!tests/test_naming.py"],
        cwd=ROOT, capture_output=True, text=True)
    remaining = [line for line in out.stdout.split("\n") if line.strip()]
    assert not remaining, f"still calling it Campaign X: {remaining}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
