"""A runner refuses a result it would clobber, not the directory it was given.

The fleet worker makes every job's run directory before it starts the runner --
some lanes write a file into it and expect it to be there, and vet_map died on
FileNotFoundError for exactly that. A runner that refuses an output directory
because it *exists* therefore refuses every queued run of itself.

The canonical 2 um lane did. Since the guard was written it failed in two and a
half seconds every time, with `refusing to overwrite`, while the launcher went
on offering it: one success in the queue's whole history, and it predates the
guard. The rule it wanted is the one run_ink.py already states -- an empty
directory is not a map.

Read statically because what went wrong is one predicate, and a test that needs
a GPU, a layer stack and a checkpoint to notice it is a test nobody runs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "framework/stages/03-ink/scripts"

# The lanes the queue routes to, which are the ones the worker hands a
# directory it has already created.
QUEUED_LANES = ("run_ink.py", "run_ink_canonical2um.py", "run_ink_9um.py")


@pytest.mark.parametrize("script", QUEUED_LANES)
def test_a_queued_lane_does_not_refuse_its_own_run_directory(script: str) -> None:
    source = (SCRIPTS / script).read_text()
    for match in re.finditer(r"if\s+args\.output\.exists\(\)([^\n:]*):", source):
        # `.exists()` alone is the refusal that cannot be satisfied. Paired with
        # a contents test it is the rule that was meant.
        assert "iterdir" in match.group(1), (
            f"{script}: refuses an output directory for existing, and the "
            "worker creates it before the runner starts")


def test_the_worker_still_creates_the_directory() -> None:
    """The other half of the pair, so this cannot be 'fixed' from the wrong end.

    Removing the mkdir would satisfy the lanes above and break the ones that
    write into a directory they were promised.
    """
    worker = (ROOT / "framework/stages/03-ink/fleet/ink_worker.py").read_text()
    assert "output.mkdir(parents=True, exist_ok=True)" in worker
