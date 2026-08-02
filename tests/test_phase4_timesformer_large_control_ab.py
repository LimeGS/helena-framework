import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/scripts"))

from build_timesformer_large_control_ct_spec import build_spec
from finalize_timesformer_large_control_ab import finalize


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def analysis(outcome: str, candidates: int) -> dict:
    return {
        "text_like_screening": {
            "screening_outcome": outcome,
            "glyph_like_candidate_count": candidates,
            "rows_with_at_least_four_candidates": 2 if candidates else 0,
            "candidates": [
                {
                    "candidate_id": f"G{index:02d}",
                    "bbox_xyxy": [0, 0, 1, 1],
                }
                for index in range(candidates)
            ],
        }
    }


def fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "root"
    controls = [
        {
            "id": "positive-pherc0139",
            "expected_role": "POSITIVE_SCROLL1_TRANSFER",
            "depth_centers": [11, 14, 16],
            "source_slice_um": 9.362,
            "tiff_directory": "positive-0139",
        },
        {
            "id": "positive-pherc172",
            "expected_role": "POSITIVE_CROSS_SCROLL_TRANSFER",
            "depth_centers": [16, 17, 18],
            "source_slice_um": 7.91,
            "tiff_directory": "positive-172",
        },
        {
            "id": "negative-a",
            "expected_role": "HARD_NEGATIVE_CT_DEPTH_DIFFUSE",
            "depth_centers": [25, 32, 39],
            "source_slice_um": 8.64,
            "tiff_directory": "negative-a/tiffs",
        },
        {
            "id": "negative-b",
            "expected_role": "HARD_NEGATIVE_CT_DEPTH_DIFFUSE",
            "depth_centers": [25, 32, 39],
            "source_slice_um": 8.64,
            "tiff_directory": "negative-b/tiffs",
        },
    ]
    manifest = write_json(root / "manifest.json", {"controls": controls})
    amendment = write_json(root / "amendment.json", {"status": "FROZEN"})
    gate = write_json(root / "gate.json", {"status": "FROZEN"})
    runtime = root / "runtime"
    write_json(
        runtime
        / "positive-pherc0139"
        / "analysis"
        / "INK_STABILITY_ANALYSIS.json",
        analysis("INSUFFICIENT_TEXT_LIKE_SUPPORT", 4),
    )
    write_json(
        runtime
        / "positive-pherc172"
        / "analysis"
        / "INK_STABILITY_ANALYSIS.json",
        analysis("POTENTIAL_TEXT_LIKE_SIGNAL_REQUIRES_CT_REVIEW", 12),
    )
    write_json(
        runtime
        / "negative-a"
        / "analysis"
        / "INK_STABILITY_ANALYSIS.json",
        analysis("INSUFFICIENT_TEXT_LIKE_SUPPORT", 0),
    )
    write_json(
        runtime
        / "negative-b"
        / "analysis"
        / "INK_STABILITY_ANALYSIS.json",
        analysis("POTENTIAL_TEXT_LIKE_SIGNAL_REQUIRES_CT_REVIEW", 2),
    )
    return {
        "root": root,
        "manifest": manifest,
        "amendment": amendment,
        "gate": gate,
        "runtime": runtime,
    }


def test_ct_spec_only_routes_hard_negatives_with_components(
    tmp_path: Path,
) -> None:
    paths = fixture(tmp_path)

    spec = build_spec(
        root=paths["root"],
        manifest_path=paths["manifest"],
        amendment_path=paths["amendment"],
        runtime=paths["runtime"],
        gate_freeze=paths["gate"],
    )

    assert [group["group_id"] for group in spec["groups"]] == ["negative-b"]
    assert spec["groups"][0]["central_slice"] == 32
    assert [
        item["candidate_count"] for item in spec["morphology_observations"]
    ] == [0, 2]


def test_finalizer_applies_frozen_positive_failure_even_if_ct_is_clean(
    tmp_path: Path,
) -> None:
    paths = fixture(tmp_path)
    spec = write_json(paths["root"] / "ct-spec.json", {"groups": []})
    decisions = write_json(
        paths["root"] / "decisions.json",
        [
            {
                "group_id": "negative-b",
                "candidate_id": "G00",
                "retained": False,
            },
            {
                "group_id": "negative-b",
                "candidate_id": "G01",
                "retained": False,
            },
        ],
    )

    result = finalize(
        root=paths["root"],
        manifest_path=paths["manifest"],
        amendment_path=paths["amendment"],
        runtime=paths["runtime"],
        ct_spec_path=spec,
        ct_gate_path=decisions,
    )

    assert result["status"] == "REJECT_LARGE_MODEL"
    assert result["controls"][0]["positive_morphology_pass"] is False
    assert result["controls"][3]["hard_negative_specificity_pass"] is True


def test_finalizer_fails_closed_when_ct_decisions_are_missing(
    tmp_path: Path,
) -> None:
    paths = fixture(tmp_path)
    spec = write_json(paths["root"] / "ct-spec.json", {"groups": []})

    result = finalize(
        root=paths["root"],
        manifest_path=paths["manifest"],
        amendment_path=paths["amendment"],
        runtime=paths["runtime"],
        ct_spec_path=spec,
        ct_gate_path=None,
    )

    assert result["status"] == "INCOMPLETE_FAIL_CLOSED"
    assert result["controls"][3]["status"] == "CT_DECISION_COUNT_MISMATCH"
