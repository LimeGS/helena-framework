from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.harness.stage_script_registry import resolve_stage_script


def test_resolves_scripts_from_their_semantic_stage() -> None:
    assert resolve_stage_script(
        ROOT, "helena_segment_search_fleet.py"
    ).relative_to(ROOT) == Path(
        "framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py"
    )
    assert resolve_stage_script(
        ROOT, "run_ink_timesformer.py"
    ).relative_to(ROOT) == Path(
        "framework/stages/03-ink/scripts/run_ink_timesformer.py"
    )
    assert resolve_stage_script(
        ROOT, "analyze_ink_stability.py"
    ).relative_to(ROOT) == Path(
        "framework/stages/04-validation/scripts/analyze_ink_stability.py"
    )
    assert resolve_stage_script(
        ROOT, "helena_develop_ct_priority_router_v42.py"
    ).relative_to(ROOT) == Path(
        "framework/stages/04-validation/scripts/helena_develop_ct_priority_router_v42.py"
    )
    assert resolve_stage_script(
        ROOT, "helena_apply_ct_priority_router_v42.py"
    ).relative_to(ROOT) == Path(
        "framework/stages/04-validation/scripts/helena_apply_ct_priority_router_v42.py"
    )
    assert resolve_stage_script(
        ROOT, "helena_build_v43_source_split.py"
    ).relative_to(ROOT) == Path(
        "framework/stages/04-validation/scripts/helena_build_v43_source_split.py"
    )
    assert resolve_stage_script(
        ROOT, "helena_develop_ct_priority_router_v43.py"
    ).relative_to(ROOT) == Path(
        "framework/stages/04-validation/scripts/helena_develop_ct_priority_router_v43.py"
    )
    assert resolve_stage_script(
        ROOT, "helena_execute_ct_priority_transfer_v4_once.py"
    ).relative_to(ROOT) == Path(
        "framework/stages/04-validation/scripts/helena_execute_ct_priority_transfer_v4_once.py"
    )


def test_resolves_shared_harnesses() -> None:
    assert resolve_stage_script(
        ROOT, "run_geometry_recovery_screen.py"
    ).relative_to(ROOT) == Path(
        "scripts/harness/run_geometry_recovery_screen.py"
    )


def test_rejects_missing_or_traversal_names() -> None:
    with pytest.raises(FileNotFoundError):
        resolve_stage_script(ROOT, "does-not-exist.py")
    with pytest.raises(ValueError):
        resolve_stage_script(ROOT, "../README.md")
