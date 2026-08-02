"""FIX-08 — one canonical strict text-like screen, shared by both call sites.

The criterion used to exist twice with divergent terms: Stage 04 required two
rows holding at least four shapes each, Stage 03 required only two populated
rows.  The second was strictly more permissive and carried the First Letters
label.  The strict form is canonical -- it produced the PHerc0139 positive
control -- and the per-row minimum is not relaxed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/scripts"))
sys.path.insert(0, str(ROOT / "framework/stages/04-validation/scripts"))

from annotate_glyph_candidates import (  # noqa: E402
    STRICT_SCREEN_MINIMUM_CANDIDATES,
    STRICT_SCREEN_MINIMUM_CANDIDATES_PER_ROW,
    STRICT_SCREEN_MINIMUM_QUALIFYING_ROWS,
    strict_text_like_screen,
)

STAGE03 = ROOT / "framework/stages/03-ink/scripts/annotate_glyph_candidates.py"
STAGE04 = (
    ROOT / "framework/stages/04-validation/scripts/analyze_ink_stability.py"
)


def candidates(rows: list[int]) -> list[dict[str, int]]:
    """Build a candidate list with the given number of shapes per row."""

    out: list[dict[str, int]] = []
    for row_index, count in enumerate(rows, start=1):
        out.extend({"row": row_index} for _ in range(count))
    return out


def test_thresholds_are_the_strict_ones() -> None:
    assert STRICT_SCREEN_MINIMUM_CANDIDATES == 10
    assert STRICT_SCREEN_MINIMUM_CANDIDATES_PER_ROW == 4
    assert STRICT_SCREEN_MINIMUM_QUALIFYING_ROWS == 2


@pytest.mark.parametrize(
    ("rows", "expected", "why"),
    [
        ([5, 4], False, "9 shapes over 2 qualifying rows is under the count floor"),
        ([10], False, "10 shapes in 1 row is under the row floor"),
        ([6, 4], True, "10 shapes over 2 rows of >=4 qualifies"),
        ([5, 5], True, "10 shapes over 2 rows of >=4 qualifies"),
        ([7, 3], False, "the second row has only 3 shapes"),
        ([9, 3, 3], False, "no second row reaches 4 shapes"),
        ([4, 4, 4], True, "12 shapes over 3 qualifying rows"),
        ([], False, "no candidates at all"),
    ],
)
def test_strict_screen_boundary(rows: list[int], expected: bool, why: str) -> None:
    assert strict_text_like_screen(candidates(rows))["qualifies"] is expected, why


def test_nine_shapes_two_rows_is_false() -> None:
    assert strict_text_like_screen(candidates([5, 4]))["qualifies"] is False


def test_ten_shapes_one_row_is_false() -> None:
    assert strict_text_like_screen(candidates([10]))["qualifies"] is False


def test_ten_shapes_two_rows_is_true() -> None:
    assert strict_text_like_screen(candidates([5, 5]))["qualifies"] is True


def test_pherc0139_positive_control_still_qualifies() -> None:
    """15 shapes over 2 rows -- the out-of-cohort positive control."""

    screen = strict_text_like_screen(candidates([8, 7]))
    assert screen["qualifies"] is True
    assert screen["glyph_like_candidate_count"] == 15
    assert screen["row_band_count"] == 2


def test_row_histogram_is_reported() -> None:
    screen = strict_text_like_screen(candidates([6, 4, 2]))

    assert screen["candidate_count_by_row"] == {"1": 6, "2": 4, "3": 2}
    assert screen["row_band_count"] == 3
    assert screen["rows_with_at_least_four_candidates"] == 2


def test_both_call_sites_agree_on_the_same_input() -> None:
    """Identity between the two routes that used to diverge."""

    import analyze_ink_stability as stage04

    # A case where the old Stage 03 rule said yes and Stage 04 said no.
    divergent = candidates([9, 1, 1])
    screen = strict_text_like_screen(divergent)
    assert screen["qualifies"] is False
    assert screen["row_band_count"] >= 2  # the permissive rule would have passed

    persistent = np.zeros((8, 8), dtype=bool)
    maps = np.stack([np.zeros((8, 8), dtype=np.float32)] * 4)
    paths = [
        Path(f"center-{center:03d}_offset-{offset:02d}.npy")
        for center in (25, 32)
        for offset in (0, 8)
    ]

    captured: dict[str, object] = {}

    def fake_discover(first, second, *, threshold):
        captured["threshold"] = threshold
        return persistent, divergent, [(0, 1, 0), (2, 3, 2)]

    stage04.ensure_stage03_helper_on_path()
    import annotate_glyph_candidates as stage03

    original = stage03.discover_glyph_candidates
    stage03.discover_glyph_candidates = fake_discover
    try:
        result = stage04.glyph_like_support(maps, paths, threshold=0.5)
    finally:
        stage03.discover_glyph_candidates = original

    assert result["screening_outcome"] == "INSUFFICIENT_TEXT_LIKE_SUPPORT"
    assert result["glyph_like_candidate_count"] == screen["glyph_like_candidate_count"]
    assert result["row_band_count"] == screen["row_band_count"]
    assert (
        result["rows_with_at_least_four_candidates"]
        == screen["rows_with_at_least_four_candidates"]
    )
    assert result["strict_screen"] == screen


def test_criterion_is_implemented_exactly_once() -> None:
    """No call site may re-derive the criterion inline."""

    pattern = re.compile(r"qualifies\s*=\s*len\(candidates\)")
    for path in (STAGE03, STAGE04):
        assert not pattern.search(path.read_text(encoding="utf-8")), path

    source = STAGE03.read_text(encoding="utf-8")
    assert source.count("def strict_text_like_screen") == 1
    assert "strict_text_like_screen" in STAGE04.read_text(encoding="utf-8")


def test_threshold_is_not_weakened_in_source() -> None:
    """Guard against a future relaxation of the per-row minimum."""

    source = STAGE03.read_text(encoding="utf-8")
    assert "STRICT_SCREEN_MINIMUM_CANDIDATES_PER_ROW = 4" in source
    assert "STRICT_SCREEN_MINIMUM_CANDIDATES = 10" in source
    assert "STRICT_SCREEN_MINIMUM_QUALIFYING_ROWS = 2" in source
