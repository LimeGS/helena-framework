"""Regression checks for the prospective naming standard."""

from __future__ import annotations

import json
from pathlib import Path
import runpy

from evidence import needs_campaign_evidence


ROOT = Path(__file__).resolve().parents[1]
VERSIONING = runpy.run_path(ROOT / "framework/versioning.py")
framework_version = VERSIONING["framework_version"]
is_semver = VERSIONING["is_semver"]


def test_framework_version_is_semver_and_single_source() -> None:
    assert framework_version() == (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    assert is_semver(framework_version())
    assert not is_semver("V17")
    assert not is_semver("0.1")


@needs_campaign_evidence
def test_active_historical_experiments_have_descriptive_aliases() -> None:
    registry = json.loads(
        (ROOT / "workspace/campaigns/campaign-x-2026/plans/EXPERIMENT_ID_REGISTRY.json").read_text(encoding="utf-8")
    )
    assert registry["schema"] == "campaignx.experiment_id_registry.v1"
    assert is_semver(registry["registry_version"])
    entries = {entry["legacy_id"]: entry for entry in registry["entries"]}
    assert {"V6", "V7", "V23"} <= set(entries)
    for entry in entries.values():
        assert entry["experiment_id"].startswith("EXP-")
        assert entry["plan_id"].startswith("PLN-EXP-")
        path = entry["historic_path"]
        assert path is None or (ROOT / path).exists() or entry["legacy_id"] == "V7"


def test_naming_standard_explains_all_identity_layers() -> None:
    """The standard is documentation the platform serves, not a file beside the
    code. It moved into the panel so that the people who need it -- somebody
    extending a phase -- meet it where they are working."""
    text = (ROOT / "panel/web/src/routes/DeveloperReference.tsx").read_text(encoding="utf-8")
    for term in ("Semantic Versioning", "Contract", "Scientific profile",
                 "Experiment", "Run", "Receipt"):
        assert term in text
