from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ROOT / "framework/stages/01-segmentation/backends"
import sys

sys.path.insert(0, str(BACKENDS))

from scrollfiesta.cube_stage import (  # noqa: E402
    CubePlan,
    CubeResult,
    CubeStageError,
    cube_environment,
    plan_cubes,
    require_complete,
    resolve_threshold,
)


def _result(tmp_path: Path, plan: CubePlan, *, rc: int = 0, obj: bool = True) -> CubeResult:
    expected = tmp_path / f"{plan.cube_id}.obj"
    if obj:
        expected.write_text("v 0 0 0\n", encoding="utf-8")
    stdout = tmp_path / f"{plan.cube_id}.stdout"
    stderr = tmp_path / f"{plan.cube_id}.stderr"
    stdout.write_text("", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    return CubeResult(plan.cube_id, plan.origin_zyx, rc, obj, expected, stdout, stderr)


def test_plan_cubes_uses_canonical_roi_and_scrollfiesta_cube_names() -> None:
    plans = plan_cubes((5120, 4992, 5376, 5376, 5248, 5632), level=0)
    assert len(plans) == 8
    assert plans[0].origin_zyx == (5120, 4992, 5376)
    assert plans[-1].origin_zyx == (5248, 5120, 5504)
    assert plans[0].cube_id == "z05120_y04992_x05376"


@pytest.mark.parametrize(
    "roi",
    [
        (1, 0, 0, 129, 128, 128),
        (0, 0, 0, 0, 128, 128),
        (0, 0, 0, 128, 128, 127),
    ],
)
def test_plan_cubes_rejects_unaligned_or_nonpositive_roi(roi: tuple[int, ...]) -> None:
    with pytest.raises(CubeStageError):
        plan_cubes(roi, level=0)


def test_require_complete_accepts_only_one_successful_obj_per_plan(tmp_path: Path) -> None:
    plans = plan_cubes((0, 0, 0, 128, 128, 256), level=0)
    require_complete([_result(tmp_path, plan) for plan in plans], plans)


def test_require_complete_rejects_partial_or_nonzero_cube(tmp_path: Path) -> None:
    plans = plan_cubes((0, 0, 0, 128, 128, 256), level=0)
    with pytest.raises(CubeStageError, match="one-to-one"):
        require_complete([_result(tmp_path, plans[0])], plans)
    with pytest.raises(CubeStageError, match="refusing partial weld"):
        require_complete(
            [_result(tmp_path, plans[0]), _result(tmp_path, plans[1], rc=9, obj=True)],
            plans,
        )


def test_nonzero_profile_alias_maps_to_upstream_threshold_without_semantic_change() -> None:
    assert resolve_threshold("nonzero") == ">=1"
    assert resolve_threshold(">128") == ">128"
    assert resolve_threshold(None) is None


def test_cube_environment_sets_scrollfiesta_and_openmp_thread_budgets() -> None:
    environment = cube_environment(4)
    assert environment["VESUVIUS_THREADS"] == "4"
    assert environment["OMP_NUM_THREADS"] == "4"
    with pytest.raises(CubeStageError, match="positive"):
        cube_environment(0)
