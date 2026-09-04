"""Two ways outside data reached further than it should, from an audit.

Both were in code written the same day, and neither had been reviewed. A
third -- the spiral fitter's dataset-layout rewrite emitting an f-string a
crafted `lasagna_volume_name` could put an expression inside -- was fixed the
same way and is covered where the mechanism that used to make it possible now
lives: see test_the_spiral_scroll_is_named_in_a_manifest_not_rebound_in_source.py.
At 23adee04 that rewrite is gone entirely (the scroll's layout is JSON now,
never source), so this file no longer carries a test for it.

**An S3 key was trusted as a path.** `os.path.relpath(key, source)` on a key
from a public bucket returns `../..` segments happily, and joining that under
the cache leaves it. The bucket is explicitly untrusted in this design.

**Two artifact-store methods skipped the check the third one had.**
`publish_probe` sanitises every identifier it puts in a path;
`stage` and `promote` joined theirs raw. Every caller passes a derived id today,
which is exactly the kind of thing that stops being true quietly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation/backends/spiral"))
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation/scripts"))
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))


# -- the key that was a path ----------------------------------------------

def test_an_s3_key_shaped_like_a_traversal_is_refused(tmp_path):
    import stage_spiral_dataset as staging

    source = "bucket/PHerc0826/lasagna/4"
    target = tmp_path / "cache"
    target.mkdir()

    class Escapes:
        def get_file(self, key, path):  # pragma: no cover - must never run
            raise AssertionError(f"the fetch was attempted for {key}")

    escaping = f"{source}/../../../../../../{tmp_path.name}/evil"
    with pytest.raises(staging.StagingRefused, match="escapes the cache"):
        staging._one_object(Escapes(), escaping, 3, source, target)


def test_an_ordinary_key_still_lands(tmp_path):
    import stage_spiral_dataset as staging

    source = "bucket/PHerc0826/lasagna/4"
    target = tmp_path / "cache"

    class Bucket:
        def get_file(self, key, path):
            Path(path).write_bytes(b"abc")

    assert staging._one_object(Bucket(), f"{source}/0/0/0", 3, source, target) is None
    assert (target / "0/0/0").read_bytes() == b"abc"


# -- the identifiers that were path components ----------------------------

@pytest.mark.parametrize("escape", [
    "../../../../tmp/evil", "..", "/etc", "a/../../../../etc", "",
])
def test_an_artifact_identifier_may_not_leave_the_store(escape, tmp_path):
    from fleet.artifact_store import LocalArtifactStore

    store = LocalArtifactStore(tmp_path)
    with pytest.raises(ValueError):
        store.promote({"staging_uri": str(tmp_path)}, "PHerc0826", escape,
                      {"files": {}})
    with pytest.raises(ValueError):
        store.promote({"staging_uri": str(tmp_path)}, escape, "s-1",
                      {"files": {}})
    with pytest.raises(ValueError):
        store.stage(tmp_path, escape, {"files": {}})


def test_a_layer_stack_is_still_published_under_its_prefix(tmp_path):
    """The rule is containment, not "one component". P4 publishes a layer stack
    as `layers/<id>` on purpose, and a check that forbade the slash would have
    been a correct-looking rule that broke the caller it was protecting."""
    from fleet.artifact_store import LocalArtifactStore

    store = LocalArtifactStore(tmp_path / "store")
    staging = tmp_path / "in"
    staging.mkdir()
    (staging / "00.tif").write_bytes(b"x")
    manifest = {"files": {"00.tif": {"size_bytes": 1,
                                     "sha256": __import__("hashlib")
                                     .sha256(b"x").hexdigest()}}}

    promoted = store.promote({"staging_uri": str(staging)}, "PHerc0139",
                             "layers/p4-run", manifest)

    assert promoted["artifact_uri"].endswith("surfaces/PHerc0139/layers/p4-run")
    assert (Path(promoted["artifact_uri"]) / "00.tif").is_file()


def test_every_path_a_caller_names_goes_through_one_check():
    """`publish_probe` had a sanitiser and `stage`/`promote` did not, in one
    file. One rule with two implementations is the shape of the bug, so the
    check is that neither joins a caller value onto the root directly."""
    source = (ROOT / "framework/stages/01-segmentation/fleet"
              / "artifact_store.py").read_text(encoding="utf-8")
    body = source[source.index("class LocalArtifactStore"):
                  source.index("class S3ArtifactStore")]

    assert "_contained(self.root" in body
    assert body.count("_contained(") >= 2, "stage and promote both need it"
    assert "_probe_component(" in body, "the probe paths keep the stricter rule"
    # And the shape the fix removed must not come back.
    assert 'self.root / "staging" / attempt_id' not in body
    assert '"surfaces" / sample_id / surface_id' not in body
