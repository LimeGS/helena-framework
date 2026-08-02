"""Free-space guards must not block the paths that write nothing.

A guard exists so a long job does not die halfway and leave a truncated
artefact. Applied to a dry run -- a plan being validated, a manifest being
built -- it turns "this machine is low on disk" into what reads as a rejected
plan, and it does it intermittently, because whether it fires depends on what
else was on the disk that hour. That is the worst kind of red: the next run
clears it and nobody looks again.

`run_geometry_recovery_v1.py` had exactly that bug. This file is the
audit of every other call site, kept as a test so the answer stays true.

The inventory is explicit rather than discovered. A new `shutil.disk_usage`
call fails the last test here until somebody classifies it, which is the point:
the decision is whether the guard protects work, and that is not something a
grep can settle.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Every site that measures free space, and why it is correct.
#
#   "guards-work"  -- refuses, and only on a path that is about to write.
#   "records"      -- measures into a receipt; refuses nothing.
GUARDS = {
    "framework/stages/01-segmentation/scripts/run_geometry_recovery_v1.py": (
        "guards-work",
        "Behind `if execute:`. A dry run writes a profile and a receipt.",
    ),
    "framework/stages/01-segmentation/fleet/executor.py": (
        "guards-work",
        "Inside execute(), which has no dry path: reaching it means growing.",
    ),
    "framework/stages/01-segmentation/scripts/run_gpu_tier_supervisor.py": (
        "guards-work",
        "preflight() runs before --dry-run returns, and that is correct here: "
        "this dry run *is* the host check, so disk is what it is checking.",
    ),
    "scripts/harness/run_geometry_recovery_screen.py": (
        "guards-work",
        "After the dry-run `continue`, so a dry run never reaches it.",
    ),
    "framework/stages/05-reconstruction/scripts/ingest_paris4.py": (
        "guards-work",
        "After `if args.plan: return 0`, and sized from the actual download.",
    ),
    "framework/stages/03-ink/scripts/run_official_gp_scroll1_robust_chain.py": (
        "guards-work",
        "Only inside execute(); --dry-run calls build_plan() instead. Sized "
        "from an estimate rather than a constant, and it names free and "
        "required when it refuses -- the model the others should follow.",
    ),
    "framework/stages/02-flattening/scripts/run.py": (
        "records",
        "local_environment() puts disk in a receipt. Refuses nothing.",
    ),
    "framework/stages/05-reconstruction/scripts/build_relation_v2_local_holdout_v1_source_lock_request.py": (
        "records",
        "free_bytes_at_lock in the request. Refuses nothing.",
    ),
}


def call_sites() -> set[str]:
    found = set()
    for directory in ("framework", "scripts", "panel"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            if ".venv" in path.parts or "vendored" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "disk_usage" in text:
                found.add(str(path.relative_to(ROOT)))
    return found


def test_the_geometry_runner_checks_free_space_only_when_it_will_grow():
    """The bug that started this. Guarded structurally, not just by behaviour.

    The regression test in test_historical_growth_exclusions_v1 drives the
    function; this one pins the shape, because the failure mode is somebody
    hoisting the check back out of the branch during a refactor.
    """
    path = ROOT / "framework/stages/01-segmentation/scripts/run_geometry_recovery_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    run = next(node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == "run")

    guarded = []
    for node in ast.walk(run):
        if not isinstance(node, ast.If):
            continue
        if not (isinstance(node.test, ast.Name) and node.test.id == "execute"):
            continue
        guarded.extend(
            child for child in ast.walk(node)
            if isinstance(child, ast.Attribute) and child.attr == "disk_usage"
        )
    every = [node for node in ast.walk(run)
             if isinstance(node, ast.Attribute) and node.attr == "disk_usage"]
    assert every, "the guard disappeared entirely; a real grow needs it"
    assert len(guarded) == len(every), (
        "a disk_usage call in run() sits outside `if execute:` — a plan being "
        "validated would fail on a machine that is merely low on disk"
    )


def disk_refusals(tree: ast.AST) -> list[ast.Raise]:
    """Raises whose message mentions free space, however it is built.

    AST rather than a regex: the ink chain's message is four adjacent string
    literals and only the later ones interpolate, so matching text found the
    bare fragment and concluded the message reported nothing.
    """
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        rendered = ast.dump(node)
        if re.search(r"GiB|free bytes|disk guard|free disk", rendered):
            out.append(node)
    return out


def test_a_refusing_guard_reports_what_it_measured():
    """"less than 4 GiB free" without the measurement gives nobody anything to
    check. Every guard that raises must put the number in the message."""
    for relative, (kind, _why) in GUARDS.items():
        if kind != "guards-work":
            continue
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        for raised in disk_refusals(tree):
            interpolates = any(isinstance(child, ast.FormattedValue)
                               for child in ast.walk(raised))
            assert interpolates, (
                f"{relative}:{raised.lineno} refuses on disk without reporting "
                "what it found"
            )


def test_every_call_site_is_classified():
    """A new one fails here until somebody says which kind it is.

    Whether a guard belongs on a path is a judgement about what that path
    writes. Discovering the list automatically would let a new unconditional
    guard land silently, which is how the last one did.
    """
    found = call_sites()
    unclassified = sorted(found - set(GUARDS))
    stale = sorted(set(GUARDS) - found)
    assert not unclassified, (
        "new free-space call sites, classify them in GUARDS: " + ", ".join(unclassified)
    )
    assert not stale, "GUARDS names files that no longer measure disk: " + ", ".join(stale)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
