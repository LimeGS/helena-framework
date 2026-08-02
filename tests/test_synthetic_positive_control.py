"""FIX-10.1 — synthetic positive control for the strict text-like screen.

WHAT THIS COVERS
----------------
The decision surface that turns model probability maps into
``POTENTIAL_TEXT_LIKE_SIGNAL_REQUIRES_CT_REVIEW``:

* ``analyze_ink_stability.load_maps``            (map discovery / stacking)
* ``analyze_ink_stability.group_depth_maps_by_offset`` (the ``min``
  aggregation across depth centres — audit mechanism M2)
* ``analyze_ink_stability.glyph_like_support``   (the strict screen)
* ``annotate_glyph_candidates.discover_glyph_candidates``
  (threshold, 2x2 morphological opening, row-band detection, per-shape
  geometry filters, two-offset IoU agreement)
* ``analyze_ink_stability.manual_review_routing`` (human-review routing)

A stack with known text morphology (bounded shapes of known contrast arranged
in separated horizontal rows) must be recovered with ``qualifies is True``,
>= 10 shapes and >= 2 populated rows.  Diffuse blobs, white noise and blank
fields must not qualify.  Mutating the contrast below the screen threshold, the
shape size below the morphological opening, or the depth persistence must all
destroy the recovery -- so the test fails if anyone weakens those three.

WHAT THIS DOES *NOT* COVER
--------------------------
This is **not** the full end-to-end positive control demanded by FIX-10.1.
The real runner ``framework/stages/03-ink/scripts/run_ink_timesformer.py``
requires a TimeSformer checkpoint (``--checkpoint``).  No ``.ckpt`` / ``.pth`` /
``.safetensors`` file exists anywhere in this repository and every
``known_checkpoint_sha256`` in ``framework/registries/method-capabilities-0.1.0.json``
is ``null``, so the CT-volume -> ``interpolate_depth`` -> ``resize_stack`` ->
network -> probability-map half of the chain cannot be exercised here.  This
test therefore starts from synthetic probability maps and proves only the
*decision* half.  Until a checkpoint is available, "zero verified letters"
remains partially untested end to end.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
STABILITY_SCRIPT = (
    ROOT / "framework/stages/04-validation/scripts/analyze_ink_stability.py"
)
INK_RUNNER = ROOT / "framework/stages/03-ink/scripts/run_ink_timesformer.py"

MAP_HEIGHT = 300
MAP_WIDTH = 400
DEPTH_CENTERS = (25, 32, 39)
TILING_OFFSETS = (0, 1)
SCREEN_THRESHOLD = 0.55
GLYPH_PROBABILITY = 0.90
BACKGROUND_PROBABILITY = 0.08


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STABILITY = _load("helena_ink_stability_positive_control", STABILITY_SCRIPT)


def synthetic_text_mask(
    *,
    row_centers: tuple[int, ...] = (60, 140, 220),
    glyphs_per_row: int = 5,
    glyph_width: int = 8,
    glyph_height: int = 12,
    glyph_gap: int = 12,
    left_margin: int = 60,
) -> np.ndarray:
    """Bounded shapes of known size arranged in separated horizontal rows."""

    mask = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=bool)
    for center_y in row_centers:
        top = center_y - glyph_height // 2
        for index in range(glyphs_per_row):
            left = left_margin + index * (glyph_width + glyph_gap)
            mask[top : top + glyph_height, left : left + glyph_width] = True
    return mask


def probability_from_mask(
    mask: np.ndarray,
    *,
    rng: np.random.Generator,
    foreground: float = GLYPH_PROBABILITY,
    background: float = BACKGROUND_PROBABILITY,
) -> np.ndarray:
    array = np.where(mask, foreground, background).astype(np.float32)
    array = array + rng.normal(0.0, 0.01, array.shape).astype(np.float32)
    return np.clip(array, 0.01, 0.99).astype(np.float32)


def write_screening_stack(
    directory: Path,
    per_depth_masks: dict[int, np.ndarray],
    *,
    seed: int = 7,
) -> Path:
    """Write ``center-*_offset-*.npy`` maps exactly as the runner emits them."""

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for center in DEPTH_CENTERS:
        for offset in TILING_OFFSETS:
            array = probability_from_mask(per_depth_masks[center], rng=rng)
            np.save(directory / f"center-{center}_offset-{offset}.npy", array)
    return directory


def screen(directory: Path, *, threshold: float = SCREEN_THRESHOLD) -> dict:
    maps, paths = STABILITY.load_maps(directory)
    return STABILITY.glyph_like_support(maps, paths, threshold=threshold)


def qualifies(result: dict) -> bool:
    """Read ``qualifies`` from whichever shape the strict screen currently has.

    FIX-08 is consolidating the two divergent copies of the criterion, so the
    flag may live under ``strict_screen`` or only be visible through the
    outcome label.  Both spellings are accepted; the semantics are not.
    """

    strict = result.get("strict_screen")
    if isinstance(strict, dict) and "qualifies" in strict:
        return bool(strict["qualifies"])
    return (
        result["screening_outcome"] == "POTENTIAL_TEXT_LIKE_SIGNAL_REQUIRES_CT_REVIEW"
    )


# ---------------------------------------------------------------------------
# Checkpoint availability — the reason this control is not yet end to end.
# ---------------------------------------------------------------------------


def test_timesformer_checkpoint_absence_is_explicit_not_silent():
    """Document *why* the model half of the positive control is not exercised.

    If a checkpoint ever lands in the tree this test fails, which is the
    trigger to extend this file into the real end-to-end control.
    """

    assert INK_RUNNER.is_file(), "the ink runner itself must exist"
    weights = [
        path
        for suffix in ("*.ckpt", "*.pth", "*.pt", "*.safetensors")
        for path in ROOT.rglob(suffix)
        if ".git" not in path.parts
        # site-packages ships *.pth path-configuration files, which are not
        # model weights; a virtualenv inside the tree must not read as one.
        and not any(part in (".venv", "venv", "site-packages", "node_modules")
                    for part in path.parts)
    ]
    assert weights == [], (
        "a model checkpoint is now present: replace the synthetic-probability "
        f"control with the real end-to-end run. found={weights}"
    )


# ---------------------------------------------------------------------------
# Positive control
# ---------------------------------------------------------------------------


def test_synthetic_text_is_recovered_by_the_strict_screen(tmp_path: Path):
    mask = synthetic_text_mask()
    directory = write_screening_stack(
        tmp_path / "screening", {center: mask for center in DEPTH_CENTERS}
    )

    result = screen(directory)

    assert qualifies(result) is True
    assert result["glyph_like_candidate_count"] >= 10
    assert result["rows_with_at_least_four_candidates"] >= 2
    assert result["row_band_count"] >= 2
    assert (
        result["screening_outcome"]
        == "POTENTIAL_TEXT_LIKE_SIGNAL_REQUIRES_CT_REVIEW"
    )
    # 3 rows x 5 shapes were synthesised; every one must survive the pipeline.
    assert result["glyph_like_candidate_count"] == 15
    assert result["candidate_count_by_row"] == {"1": 5, "2": 5, "3": 5}


def test_recovered_candidate_geometry_matches_the_synthesised_shapes(tmp_path: Path):
    directory = write_screening_stack(
        tmp_path / "screening",
        {center: synthetic_text_mask() for center in DEPTH_CENTERS},
    )

    candidates = screen(directory)["candidates"]

    assert len(candidates) == 15
    for item in candidates:
        x0, y0, x1, y1 = item["bbox_xyxy"]
        assert (x1 - x0) == 8, "synthesised glyph width must survive the screen"
        assert (y1 - y0) == 12, "synthesised glyph height must survive the screen"
        assert item["agreement_iou"] == pytest.approx(1.0, abs=1e-9)
    # The three synthesised rows must come back as three distinct bands.
    assert sorted({item["row"] for item in candidates}) == [1, 2, 3]


def test_a_recovered_positive_control_is_routed_to_human_ct_review(tmp_path: Path):
    directory = write_screening_stack(
        tmp_path / "screening",
        {center: synthetic_text_mask() for center in DEPTH_CENTERS},
    )

    routing = STABILITY.manual_review_routing(screen(directory))

    assert routing["human_review_required"] is True
    assert routing["route"] == "QUEUE_FOR_RAW_CT_FIBER_CONFOUND_REVIEW"


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


def _uniform_stack(array: np.ndarray, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for center in DEPTH_CENTERS:
        for offset in TILING_OFFSETS:
            np.save(directory / f"center-{center}_offset-{offset}.npy", array)
    return directory


def test_a_large_diffuse_blob_does_not_qualify(tmp_path: Path):
    yy, xx = np.mgrid[0:MAP_HEIGHT, 0:MAP_WIDTH]
    blob = np.exp(-(((yy - 150) / 70.0) ** 2 + ((xx - 200) / 90.0) ** 2))
    array = np.clip(0.05 + 0.90 * blob, 0.01, 0.99).astype(np.float32)

    result = screen(_uniform_stack(array, tmp_path / "blob"))

    assert qualifies(result) is False
    assert result["glyph_like_candidate_count"] == 0
    assert result["screening_outcome"] == "INSUFFICIENT_TEXT_LIKE_SUPPORT"
    assert (
        STABILITY.manual_review_routing(result)["human_review_required"] is False
    )


def test_white_noise_does_not_qualify(tmp_path: Path):
    rng = np.random.default_rng(11)
    array = np.clip(
        rng.normal(0.30, 0.16, (MAP_HEIGHT, MAP_WIDTH)), 0.01, 0.99
    ).astype(np.float32)

    result = screen(_uniform_stack(array, tmp_path / "noise"))

    assert qualifies(result) is False
    assert result["rows_with_at_least_four_candidates"] == 0


def test_a_blank_field_does_not_qualify(tmp_path: Path):
    array = np.full((MAP_HEIGHT, MAP_WIDTH), 0.05, dtype=np.float32)

    result = screen(_uniform_stack(array, tmp_path / "blank"))

    assert qualifies(result) is False
    assert result["persistent_pixels"] == 0


def test_a_single_populated_row_does_not_qualify(tmp_path: Path):
    """>= 10 shapes are not enough: the screen also demands two populated rows."""

    mask = synthetic_text_mask(row_centers=(140,), glyphs_per_row=12, glyph_gap=10)
    directory = write_screening_stack(
        tmp_path / "one-row", {center: mask for center in DEPTH_CENTERS}
    )

    result = screen(directory)

    assert result["glyph_like_candidate_count"] >= 10
    assert result["rows_with_at_least_four_candidates"] == 1
    assert qualifies(result) is False


# ---------------------------------------------------------------------------
# Mutation sensitivity — the screen's own knobs must be load bearing.
# ---------------------------------------------------------------------------


def test_raising_the_screen_threshold_above_the_glyph_contrast_destroys_recovery(
    tmp_path: Path,
):
    directory = write_screening_stack(
        tmp_path / "screening",
        {center: synthetic_text_mask() for center in DEPTH_CENTERS},
    )

    assert qualifies(screen(directory, threshold=SCREEN_THRESHOLD)) is True
    assert qualifies(screen(directory, threshold=0.95)) is False


def test_shapes_thinner_than_the_morphological_opening_are_destroyed(tmp_path: Path):
    """A 1 px stroke cannot survive the 2x2 binary opening."""

    thin = synthetic_text_mask(glyph_width=1, glyph_height=12)
    directory = write_screening_stack(
        tmp_path / "thin", {center: thin for center in DEPTH_CENTERS}
    )

    result = screen(directory)

    assert result["persistent_pixels"] == 0
    assert qualifies(result) is False


def test_single_depth_signal_is_erased_by_the_minimum_across_depth_centres(
    tmp_path: Path,
):
    """Characterises audit mechanism M2 on a *known* positive.

    Identical text morphology, present at every depth centre, qualifies.  The
    same text present at only one of the three depth centres is annihilated by
    ``group_depth_maps_by_offset``'s ``min``.  This is the mechanism that
    removes ink confined to a single papyrus lamina; it is pinned here so that
    any change to the aggregation is visible as a test change.
    """

    text = synthetic_text_mask()
    blank = np.zeros_like(text)

    all_depths = write_screening_stack(
        tmp_path / "all", {center: text for center in DEPTH_CENTERS}
    )
    one_depth = write_screening_stack(
        tmp_path / "one",
        {25: blank, 32: text, 39: blank},
    )

    assert qualifies(screen(all_depths)) is True

    localized = screen(one_depth)
    assert qualifies(localized) is False
    assert localized["persistent_pixels"] == 0
    assert localized["glyph_like_candidate_count"] == 0
