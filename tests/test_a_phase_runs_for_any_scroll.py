"""A catalog is a convenience, not a gate on which scrolls may be worked.

Every fleet phase resolved its scroll through the frozen catalog in strict mode,
so a name the catalog did not carry could not be queued at all. The control hit
it the first time one ever reached P3:

    POST /api/flattening/run -> 400
    "PHerc0139 is not a scroll the frozen catalog registers"

The reason given was that the name "has to resolve to a volume". For PHerc0139
it already did: the mission holds a registered source snapshot with its ct_uri
and m7_uri. The platform knew exactly where the volume was and refused anyway,
because the name was absent from a list.

That is backwards for an exploration platform. The thirteen evaluation scrolls
are a *mission's* selection of targets; the phases themselves have to work for
any scroll that exists, and for the ones not yet scanned. A whitelist may decide
what a campaign certifies. It must not decide what can run.

So the resolution falls back: the catalog first, and when it does not know the
name, whatever the control plane has registered for it. A refusal is kept for
the case the reason actually describes -- a name that resolves to no volume
anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")


class _Store:
    """A control plane that knows one scroll the catalog does not."""

    def __init__(self, known: set[str]):
        self.known = known
        self.enqueued: list[dict] = []

    def snapshots(self, samples=None):
        return [{"source_snapshot_id": f"snap-{name}", "sample_id": name,
                 "ct_uri": "https://example.invalid/ct.zarr",
                 "m7_uri": "https://example.invalid/m7.zarr"}
                for name in sorted(self.known)
                if samples is None or name in samples]


def test_a_scroll_the_catalog_carries_still_resolves(monkeypatch) -> None:
    """The ordinary path is unchanged: the catalog is asked first."""
    import panel.app as panel_app

    monkeypatch.setattr(panel_app, "fleet_store_read_only",
                        lambda: _Store({"PHerc0139"}))
    known = panel_app.catalog_sample_id("PHerc0826")
    assert known


def test_a_scroll_only_the_control_plane_knows_is_not_refused(monkeypatch) -> None:
    """The whole point. A registered source is a resolvable volume, whatever
    the catalog carries."""
    import panel.app as panel_app

    monkeypatch.setattr(panel_app, "fleet_store_read_only",
                        lambda: _Store({"PHerc9999"}))

    assert panel_app.catalog_sample_id("PHerc9999") == "PHerc9999"


def test_a_scroll_nobody_can_resolve_is_still_refused(monkeypatch) -> None:
    """The refusal keeps the case its own message describes: a name that
    resolves to no volume anywhere. Failing there is honest; failing on a list
    is not."""
    import panel.app as panel_app

    monkeypatch.setattr(panel_app, "fleet_store_read_only", lambda: _Store(set()))

    with pytest.raises(panel_app.HTTPException) as refused:
        panel_app.catalog_sample_id("PHercNoSuchScroll")
    assert refused.value.status_code == 400


def test_a_control_plane_that_cannot_be_asked_does_not_become_a_refusal(
        monkeypatch) -> None:
    """P0 runs before the segmentation store is configured on a fresh
    deployment, and a resolver that turns that into a 400 makes the platform
    unusable exactly when it is being set up."""
    import panel.app as panel_app

    def unavailable():
        raise RuntimeError("segmentation store is not configured")

    monkeypatch.setattr(panel_app, "fleet_store_read_only", unavailable)

    with pytest.raises(panel_app.HTTPException):
        panel_app.catalog_sample_id("PHercNoSuchScroll")


def test_reading_never_refused_before_and_still_does_not(monkeypatch) -> None:
    import panel.app as panel_app

    monkeypatch.setattr(panel_app, "fleet_store_read_only", lambda: _Store(set()))
    assert panel_app.catalog_sample_id("PHerc0841", strict=False) == "PHerc0841"
