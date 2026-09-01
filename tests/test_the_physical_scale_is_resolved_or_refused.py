"""The micron figure is settled at the queue, never defaulted by a component.

Two components meet the same missing datum and disagree about what to do, and
the disagreement had produced every PHerc826 render on this deployment.

The TimeSformer lane refuses without `--source-slice-um` and is right: this
campaign spans 8.64 and 9.362 um acquisitions, so assuming either rescales the
other by 8.4% in silence. But that refusal arrives an hour later, from inside a
worker, as a traceback.

volume-cartographer makes the opposite choice. Handed a volume whose zarr
carries no voxel metadata it prints `Voxel size: 1.0 (no metadata found;
override with --voxel-size)` and renders anyway. Nothing downstream can
distinguish that stack from a correctly scaled one -- same slice count, same
shape, same exit code -- and the figure is wrong by 9.362x.

So the question is answered once, at the queue. The frozen catalogue knows the
scale of every scan in the cohort; for those it is resolved rather than asked.
Where it cannot be resolved the job is refused with what the catalogue does
hold, because a person choosing between named numbers is the only honest
option left, and it costs a second rather than a run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(name="panel")
def _panel():
    sys.path.insert(0, str(ROOT / "panel"))
    import app  # noqa: PLC0415
    return app


def test_a_catalogued_scroll_needs_no_answer(panel):
    """PHerc826 is in the catalogue at 9.362; nobody should be asked."""
    parameters: dict = {}
    panel.resolve_source_scales(parameters, "PHerc826")
    assert parameters["source_pixel_um"] == 9.362
    assert parameters["source_slice_um"] == 9.362


def test_both_spellings_of_the_same_scroll_resolve(panel):
    """The catalogue calls it PHerc826 in `sample_id` and PHerc0826 in the
    bucket path of its ct_uri. Jobs are queued under either, and looking up one
    spelling is how a scale that is on record reads as missing."""
    assert panel.resolve_source_scale("PHerc826")[0] == 9.362
    assert panel.resolve_source_scale("PHerc0826")[0] == 9.362


def test_a_stated_scale_is_never_overridden(panel):
    """The catalogue settles what nobody stated. It does not overrule a caller
    who knows something it does not -- a different volume of the same scroll."""
    parameters = {"source_slice_um": 8.64}
    panel.resolve_source_scales(parameters, "PHerc826")
    assert parameters["source_slice_um"] == 8.64
    assert parameters["source_pixel_um"] == 9.362


def test_an_uncatalogued_scroll_is_refused_with_the_choices(panel):
    with pytest.raises(panel.HTTPException) as refusal:
        panel.resolve_source_scales({}, "PHercNotInTheCatalogue")
    detail = refusal.value.detail
    assert refusal.value.status_code == 409
    # The refusal has to be actionable: name the field, say why there is no
    # default, and give the figures the catalogue does hold.
    assert "source_pixel_um" in detail
    assert "8.64" in detail and "9.362" in detail
    # And it must not present that list as the only options -- PHerc0139's
    # control volumes are 2.399, which is in neither.
    assert "from the volume in use" in detail


def test_the_render_scale_is_settled_too(panel):
    parameters: dict = {}
    panel.resolve_source_voxel(parameters, "PHerc0826")
    assert parameters["source_voxel_um"] == 9.362


def test_the_render_refusal_names_the_silent_failure(panel):
    with pytest.raises(panel.HTTPException) as refusal:
        panel.resolve_source_voxel({}, "PHercNotInTheCatalogue")
    detail = refusal.value.detail
    assert "assumes 1.0" in detail, (
        "the refusal must say what happens without it, which is the whole "
        "reason this check exists")


def test_the_scale_is_settled_last_of_all_and_before_the_row_is_written():
    """Ordering, and it matters in both directions.

    Before the enqueue, or the row is written without it. After every other
    refusal, because those are about identity and provenance and are more
    specific than a missing number: a control job whose binding moved during
    the enqueue proof must say so, not "state the scale". Placed first, this
    check preempted exactly that and turned a race-detection test into an
    unrelated 409.
    """
    body = (ROOT / "panel/app.py").read_text(encoding="utf-8")
    resolves = body.index("resolve_source_voxel(parameters, request.sample_id)")
    enqueued = body.index("job_id = store.enqueue(")
    assert resolves < enqueued, "the scale is settled after the row is written"
    for earlier in ('409, "control P5 requires the exact bound P4 job"',
                    '409, "selected P0 control binding changed during enqueue proof"'):
        assert body.index(earlier) < resolves, (
            f"the scale check preempts a more specific refusal: {earlier}")
    assert body.index("resolve_source_scales(parameters, request.sample_id)") < enqueued


def test_every_catalogued_scroll_has_a_scale():
    """The resolution is only free while this holds; a catalogue entry without
    a voxel size would silently become a refusal."""
    catalogue = json.loads(
        (ROOT / "workspace/catalog/eligible_volumes.json").read_text())
    for entry in catalogue["entries"]:
        assert entry.get("voxel_size_um"), entry.get("sample_id")
