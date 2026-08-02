"""P3 reported ModuleNotFoundError once anybody opened the Hosts page.

Two stages ship a package directory called `fleet`:

    framework/stages/01-segmentation/fleet    store.py, and much else
    framework/stages/03-ink/fleet             job_store.py, ink_worker.py

Neither has an __init__.py, so each is importable as the top-level name `fleet`
depending only on which parent directory reached sys.path first -- and the first
one to win is cached in sys.modules for the life of the process.

/api/hosts inserted `03-ink` and imported `fleet.ink_worker`. That bound `fleet`
to the ink directory, which has no store.py, so every later
`from fleet.store import ...` raised `No module named 'fleet.store'`. P3
Flattening reads that module for its admissibility policy and went unavailable
until the panel restarted.

Nothing caught it because neither path is broken. Open P3 first and it works;
open Hosts first and P3 does not. The failure lives in the order, in one
process, and no test of either endpoint alone can see it -- which is what this
one is for. It walks the two imports in the order that used to break, in the
process the test runs in.

The fix is that the panel imports the ink modules flat, the way ink_worker
itself does (`from job_store import ...`), so the name `fleet` is never bound to
that directory at all.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(body: str) -> subprocess.CompletedProcess[str]:
    """A fresh interpreter: sys.modules is the thing under test."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )


def test_the_ink_import_does_not_capture_the_name_fleet() -> None:
    """What /api/hosts does must leave `fleet.store` importable afterwards."""
    done = _run(f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        # Exactly what the panel does for the Hosts page, flat and with no
        # insert of 03-ink itself.
        sys.path.insert(0, {str(ROOT / "framework/stages/03-ink/fleet")!r})
        import ink_worker
        assert "fleet" not in sys.modules, (
            "importing the ink worker bound the name 'fleet'; a later "
            "fleet.store will resolve into the ink directory"
        )

        # Then what P3 does.
        sys.path.insert(0, {str(ROOT / "framework/stages/01-segmentation")!r})
        from fleet.store import ADMISSIBLE_PHYSICAL_QC_STATES
        assert ADMISSIBLE_PHYSICAL_QC_STATES
        print("ok")
    """)
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout


def test_importing_ink_by_package_name_is_what_broke_it() -> None:
    """The old form, kept as the reason the new one is written the way it is.

    If this ever stops failing -- because one of the two directories grew an
    __init__.py, or store.py appeared in the ink one -- then the constraint this
    file protects has changed and the test above is worth re-reading rather than
    trusted.
    """
    done = _run(f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        sys.path.insert(0, {str(ROOT / "framework" / "stages" / "03-ink")!r})
        import fleet.ink_worker
        sys.path.insert(0, {str(ROOT / "framework/stages/01-segmentation")!r})
        try:
            from fleet.store import ADMISSIBLE_PHYSICAL_QC_STATES
        except ModuleNotFoundError as exc:
            print("shadowed:", exc)
        else:
            print("no longer shadowed")
    """)
    assert done.returncode == 0, done.stderr
    assert "shadowed: No module named 'fleet.store'" in done.stdout, done.stdout
