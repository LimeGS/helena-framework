"""Three ways outside data reached further than it should, from an audit.

All three were in code written the same day, and none had been reviewed.

**The rewrite emitted an f-string.** `rebind_layout` built each templated path
by `.format()`-ing a caller value and wrapping the result as `f"{value!r}"`.
`repr()` quotes a string; it does not neutralise braces that the surrounding `f`
prefix then reads as expressions. `{{` and `}}` are escapes rather than fields,
so they survived a validator that only looked for unresolved field names, and
`.format()` turned them back into single braces. The emitted line ran arbitrary
code at import, on a GPU worker, before any fitting started. The emitted form is
a concatenation now, which has no interpolation context at all, and the value is
refused before it gets there.

**An S3 key was trusted as a path.** `os.path.relpath(key, source)` on a key
from a public bucket returns `../..` segments happily, and joining that under
the cache leaves it. The bucket is explicitly untrusted in this design.

**Two artifact-store methods skipped the check the third one had.**
`publish_probe` sanitises every identifier it puts in a path;
`stage` and `promote` joined theirs raw. Every caller passes a derived id today,
which is exactly the kind of thing that stops being true quietly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation/backends/spiral"))
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation/scripts"))
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

import repin  # noqa: E402

PINNED = ROOT / "vendor/villa/volume-cartographer/scripts/spiral/fit_spiral.py"


# -- the injection ---------------------------------------------------------

@pytest.mark.parametrize("payload", [
    # The auditor's own proof-of-concept, kept verbatim.
    "las_{array}_{{__import__('os').system('id')}}",
    "las_{array}_{{open('/tmp/x','w')}}",
    "{array}}{",
    "x{array}{oops}",
    "las_{array}/../../etc",
])
def test_a_volume_name_that_could_carry_an_expression_is_refused(payload):
    with pytest.raises(repin.ScrollNotRebindable):
        repin.validate_layout({"lasagna_volume_name": payload})


def test_a_tracks_name_carries_no_braces_either():
    with pytest.raises(repin.ScrollNotRebindable, match="no braces"):
        repin.validate_layout({"tracks_file": "{__import__('os')}.dbm"})


def test_the_rewrite_emits_a_concatenation_and_never_an_f_string():
    """The structural half of the fix, and the one that holds even if the
    validator above is ever loosened: there is no interpolation context to
    escape into, so an injected value is data inside a literal."""
    import ast

    if not PINNED.is_file():
        pytest.skip("upstream fit_spiral.py is not vendored here")
    rewritten = repin.rebind_layout(
        PINNED.read_text(encoding="utf-8"),
        {"lasagna_volume_name": "PHerc0826_{array}.ome.zarr",
         "normal_zarr_group": "3"})

    for statement in ast.parse(rewritten).body:
        if not isinstance(statement, ast.Assign):
            continue
        for target in statement.targets:
            if isinstance(target, ast.Name) and target.id in repin.TEMPLATE_CONSTANTS:
                node = statement.value
                assert isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
                assert isinstance(node.left, ast.Name)
                assert node.left.id == "dataset_path"
                assert isinstance(node.right, ast.Constant)
                assert not isinstance(node, ast.JoinedStr)


def test_the_readback_checks_a_shape_and_not_a_substring():
    """`wanted in unparse(node)` passes for a node that also carries something
    else. The accepted shape is `dataset_path + <the exact literal>`, which
    nothing but the intended rewrite produces."""
    import ast

    source = (ROOT / "framework/stages/01-segmentation/backends/spiral"
              / "repin.py").read_text(encoding="utf-8")
    assert "isinstance(node, _ast.BinOp)" in source
    assert "node.right.value == wanted" in source
    # And the emitter must not reintroduce the f-string it used to write.
    assert 'f"f{' not in source and "f'f{" not in source


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
