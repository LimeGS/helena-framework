"""P1 grows for a scroll the frozen catalog does not carry.

The catalog names the thirteen scrolls a campaign was frozen around. It is not
the list of scrolls that may be worked on: PHerc0139, the development control,
is deliberately absent from it, and its CT and m7 addresses come from the
control manifest instead.

`bootstrap_sources` read the catalog and nothing else, so a run queued for it
died with `unknown samples requested: ['PHerc0139']` -- raised after the panel
had resolved a snapshot carrying the very m7 volume the grow would read, and
surfaced in the browser as `the fleet bootstrap refused this request`.

The rule this restores is the one P3 already follows and
`test_a_phase_runs_for_any_scroll` states: a whitelist may decide what a
campaign certifies, never what can run. A name that resolves to no volume
anywhere is still refused.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.generator import bootstrap_sources  # noqa: E402


class _Store:
    """A control plane holding one snapshot the catalog never named."""

    def __init__(self, registered: dict[str, dict]):
        self.registered = registered
        self.written: list[dict] = []

    def snapshots(self, samples=None):
        return [row for name, row in self.registered.items()
                if samples is None or name in samples]

    def register_snapshot(self, payload):
        self.written.append(payload)
        return f"snap-{payload['sample_id']}"


def _catalog(tmp_path: Path, sample: str) -> Path:
    path = tmp_path / "eligible_volumes.json"
    path.write_text(json.dumps({"schema_version": 1, "entries": [{
        "sample_id": sample,
        "ct_uri": "https://example.invalid/ct.zarr",
        "surface_prediction_uri": "https://example.invalid/m7.zarr",
        "surface_prediction_threshold": 0.2,
        "eligible_scan_id": "20250821151825-9.362um-1.2m-113keV",
        "voxel_size_um": 9.362,
        "shape_zyx": [100, 200, 300],
    }]}))
    return path


def test_a_registered_scroll_outside_the_catalog_resolves(tmp_path) -> None:
    store = _Store({"PHerc0139": {
        "source_snapshot_id": "snap-control",
        "sample_id": "PHerc0139",
        "ct_uri": "https://example.invalid/control-ct.zarr",
        "m7_uri": "https://example.invalid/control-m7.zarr",
    }})
    resolved = bootstrap_sources(store, _catalog(tmp_path, "PHerc125"),
                                 {"PHerc0139"}, verify=False)
    # The snapshot already registered, not a new one invented from a catalog
    # entry that does not exist.
    assert resolved == {"PHerc0139": "snap-control"}
    assert store.written == []


def test_a_scroll_with_no_source_anywhere_is_still_refused(tmp_path) -> None:
    store = _Store({})
    with pytest.raises(ValueError) as refusal:
        bootstrap_sources(store, _catalog(tmp_path, "PHerc125"),
                          {"PHercUnknown"}, verify=False)
    # The name, and what it would have needed -- not just "unknown".
    assert "PHercUnknown" in str(refusal.value)
    assert "m7" in str(refusal.value)


def test_a_half_registered_source_does_not_count(tmp_path) -> None:
    """A snapshot with no m7 prediction cannot seed a grow.

    It is a CT address and nothing to look for in it, which is the same absence
    the refusal describes -- reported now rather than as an executor crash on a
    worker an hour later.
    """
    store = _Store({"PHerc0139": {
        "source_snapshot_id": "snap-partial",
        "sample_id": "PHerc0139",
        "ct_uri": "https://example.invalid/control-ct.zarr",
        "m7_uri": None,
    }})
    with pytest.raises(ValueError):
        bootstrap_sources(store, _catalog(tmp_path, "PHerc125"),
                          {"PHerc0139"}, verify=False)
