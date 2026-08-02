"""The panel's copy of TERMINAL_STATES has to match the fleet's.

The stage lives under "01-segmentation", which is not an importable module name,
so the panel keeps its own tuple. A copy that drifts is exactly how the queue
tile came to read "162 waiting" on a queue with nothing claimable in it: the
list named SUCCEEDED, FAILED and CANCELLED -- three states this fleet never
writes -- and omitted the nine it does, so every finished task counted as
waiting.

Parsed with ast rather than imported, so this needs neither the fleet package
nor a database.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "framework/stages/01-segmentation/fleet/store.py"
PANEL = ROOT / "panel/app.py"


def tuple_named(path: Path, name: str) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"{name} not found in {path}")


def test_the_panel_and_the_fleet_agree_on_what_terminal_means():
    fleet = tuple_named(STORE, "TERMINAL_STATES")
    panel = tuple_named(PANEL, "SEGMENT_TERMINAL_STATES")
    assert set(panel) == set(fleet), (
        "the panel counts a task as waiting or stalled by excluding these; "
        f"only in the fleet: {sorted(set(fleet) - set(panel))}; "
        f"only in the panel: {sorted(set(panel) - set(fleet))}")


def test_no_state_the_fleet_never_writes_is_listed():
    """The states that caused it. They belong to no fleet code path."""
    panel = tuple_named(PANEL, "SEGMENT_TERMINAL_STATES")
    for invented in ("SUCCEEDED", "FAILED", "CANCELLED"):
        assert invented not in panel, (
            f"{invented} is not a state this fleet writes; listing it hides "
            "nothing and its absence from the fleet's own tuple is the point")
