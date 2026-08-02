"""When one profile names another by hash, the hash has to be that file's.

A surface-QC profile pins the ink lane it runs by path *and* by SHA-256, which
is the right way round: the path says which file and the hash says which
version, so a lane cannot be swapped underneath a frozen QC profile without the
change being visible.

It is only worth anything if the two agree. Nothing checked that, and the cost
of the general failure is known: on 2026-07-29 a rename refactor changed the ink
adapter's filename, which changed the ink profile, which changed its hash. That
one was updated correctly. The deployment's own pin of the *outer* profile was
not, and both GPUs on gpu-1 spent two days claiming QC jobs, failing on
"surface-QC profile hash differs", requeuing, and claiming them again.

That particular pin lives in a host env file the repository cannot see. This is
the half that is checkable: every hash a profile states about another profile.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = sorted((ROOT / "framework/profiles").rglob("*.json"))


def _quoted_hashes() -> list[tuple[Path, str, str]]:
    """Every (profile, path it names, hash it claims) in the tree."""
    found = []
    for path in PROFILES:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(document, dict):
            continue
        for block in document.values():
            if not isinstance(block, dict):
                continue
            named, digest = block.get("profile"), block.get("profile_sha256")
            if isinstance(named, str) and isinstance(digest, str):
                found.append((path, named, digest))
    return found


def test_a_profile_that_quotes_a_hash_quotes_the_right_one() -> None:
    for source, named, claimed in _quoted_hashes():
        target = ROOT / named
        assert target.exists(), (
            f"{source.name} pins {named}, which is not in the repository"
        )
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        assert actual == claimed, (
            f"{source.name} says {named} hashes to {claimed[:16]}… and it hashes "
            f"to {actual[:16]}…. Either the lane changed and this profile was not "
            "re-approved, or the hash was copied from the wrong file."
        )


def test_there_is_something_to_check() -> None:
    """The loop above passes trivially over an empty list, and a rename that
    moved these keys would empty it silently."""
    quoted = _quoted_hashes()
    assert quoted, (
        "no profile pins another by profile_sha256 any more, so the assertion "
        "above is checking nothing"
    )
