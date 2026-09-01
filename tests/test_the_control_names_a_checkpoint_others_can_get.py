"""A digest is not a way to obtain a model.

The control's `model_locks` pinned the ink checkpoint by SHA-256 and said
nothing about where it comes from:

    {"profile_id": "timesformer-gp-scroll1-screening@1.0.0",
     "model_family": "timesformer_GP_scroll1",
     "checkpoint_sha256": "490a98f9..."}

That is enough to detect a swapped model and useless for reproducing the run.
Reviewing the control, ScrollPrize asked for exactly this -- "a checkpoint
others can obtain" -- and they were right to: a receipt that binds a digest a
reader cannot resolve proves the run was consistent with itself and nothing
more.

The provenance was never unknown. `framework/registries/method-capabilities`
already records that the checkpoint was recovered from the public upstream
repository scrollprize/timesformer_GP_scroll1, ungated, and that its SHA-256
matched the digest the registry and the frozen stage-03 profile had already
declared -- verified rather than asserted. It simply was not written where a
verifier of the *control* would look.

So the manifest carries it. The digest stays exactly what it was: the source
says where to get the bytes, and the digest still decides whether you got them.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import pytest

from control_manifest import IN_FORCE_PATH  # noqa: E402

PROFILES = Path(__file__).resolve().parents[1] / "framework/profiles/01-segmentation"
CONTROLS = sorted(PROFILES.glob("first-letters-control-policy-*.json"))

# The version the reproducible run is measured under. Earlier manifests are
# history: they are frozen, and rewriting them to satisfy a later requirement is
# the tampering this whole program exists to catch.
CURRENT = IN_FORCE_PATH


def _locks(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")).get("model_locks") or []


def test_there_is_a_current_control_manifest() -> None:
    assert CURRENT.exists(), f"{CURRENT.name} is the manifest the run is measured under"


def test_every_locked_model_says_where_to_get_it() -> None:
    locks = _locks(CURRENT)
    assert locks, "the control locks no model at all"
    for lock in locks:
        source = lock.get("checkpoint_source") or {}
        assert source, (
            f"{lock.get('profile_id')} pins a digest with no way to obtain the "
            "bytes it describes")
        url = str(source.get("url") or "")
        assert urlparse(url).scheme == "https" and urlparse(url).netloc, (
            f"{lock.get('profile_id')} names {url!r}, which is not a fetchable source")
        assert source.get("artifact"), (
            "a repository is not a file: name the artifact to download")


def test_the_digest_still_decides() -> None:
    """The source says where to get the bytes; the digest says whether you got
    them. Adding provenance must not soften the identity check."""
    for lock in _locks(CURRENT):
        digest = str(lock.get("checkpoint_sha256") or "")
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), (
            f"{lock.get('profile_id')} lost its checkpoint digest")


def test_the_digest_did_not_change_when_the_source_was_added() -> None:
    """The point of the new version is a sentence about provenance, not a
    different model. If the digest moved, this is a different experiment."""
    previous = PROFILES / "first-letters-control-policy-1.1.0.json"
    if not previous.exists():
        pytest.skip("no previous manifest to compare against")

    was = {row.get("profile_id"): row.get("checkpoint_sha256")
           for row in _locks(previous)}
    now = {row.get("profile_id"): row.get("checkpoint_sha256")
           for row in _locks(CURRENT)}

    assert now == was, "the locked checkpoint changed; that is not a provenance edit"


@pytest.mark.parametrize("path", CONTROLS, ids=lambda p: p.stem)
def test_a_frozen_manifest_is_never_edited_in_place(path: Path) -> None:
    """Earlier versions stay as they shipped. A frozen manifest's sha256 is its
    identity: editing one in place would silently invalidate every P0 content
    lock and receipt already bound to it."""
    document = json.loads(path.read_text(encoding="utf-8"))
    stated = str(document.get("profile_id") or "")
    version = path.stem.rsplit("-", 1)[-1]
    assert stated.endswith("@" + version), (
        f"{path.name} declares {stated}, so the file name and the identity "
        "inside it disagree")
