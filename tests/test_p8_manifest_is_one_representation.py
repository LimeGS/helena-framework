"""P8 hashes a manifest exactly one way, and rechecks that the hash binds.

The reconstruction lane carried two JSON canonicalizations. `publish_set` used
the repository's `fleet.common.content_sha256` -- sorted keys, tight separators,
`ensure_ascii=False` -- to compute the `artifact_sha256` that becomes a merged
surface's catalogue identity. `materialize_parents` used a second `json.dumps`
declared in the lane itself, which defaulted `ensure_ascii` to True, over the
whole manifest document rather than its file inventory.

Two canonicalizations of the same document are two digests of the same document,
and a content digest whose value depends on which function you happened to call
identifies nothing. Worse, no one ever recomputed the binding: a parent manifest
could name any `artifact_sha256` it liked, because materialization compared that
field to the catalogue instead of deriving it from the inventory it sits next to.

The canonical representation is the one the control plane already enforces at
artifact-set upload (`fleet/store.py`, `fleet/postgres_store.py`): the file
inventory hashed with `content_sha256` is `artifact_sha256`, and the whole
manifest hashed with `content_sha256` is `manifest_sha256`. These tests hold P8
to it and prove the incompatible form is now refused rather than accepted.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "framework/stages/05-reconstruction/scripts/run_vc3d_tifxyz_merge.py"
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.common import canonical_bytes, content_sha256  # noqa: E402


def lane_module():
    spec = importlib.util.spec_from_file_location("vc3d_merge_lane", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ascii_escaped_sha256(value: object) -> str:
    """The other representation: the lane's own dumps, with ensure_ascii on.

    This is what `materialize_parents` computed before this change, and what a
    reimplementation reaching for `json.dumps` defaults would compute again.
    """
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def inventory(names: list[str], seed: int = 1) -> dict[str, dict[str, object]]:
    return {
        name: {"sha256": format(seed + offset, "064x"),
               "size_bytes": 4096 + seed + offset}
        for offset, name in enumerate(sorted(names))
    }


def manifest(files: dict[str, object] | None = None, **changed: object) -> dict:
    files = inventory(["x.tif", "y.tif", "z.tif"]) if files is None else files
    value = {
        "schema": "campaignx.merged_tifxyz_artifact_set.v1",
        "files": files,
        "artifact_sha256": content_sha256(files),
    }
    value.update(changed)
    return value


# --- one representation -----------------------------------------------------


def test_the_lane_canonicalizes_json_exactly_as_the_control_plane_does():
    """A second `json.dumps` in the lane is a second digest of the same bytes."""
    module = lane_module()
    for value in (
        {"note": "papyrus — PHerc0826", "n": 1},
        {"files": {"café.txt": {"sha256": "0" * 64, "size_bytes": 1}}},
        ["é", {"b": 2, "a": 1}],
    ):
        assert module.canonical(value).encode("utf-8") == canonical_bytes(value)


def test_non_finite_numbers_are_refused_rather_than_written_as_invalid_json():
    """`NaN` is not JSON. Emitting it makes a digest nothing can reproduce."""
    module = lane_module()
    with pytest.raises(ValueError):
        module.canonical({"voxel_size_um": float("nan")})


def test_the_manifest_digest_is_the_control_plane_manifest_digest():
    module = lane_module()
    document = manifest()
    assert module.manifest_sha256(document) == content_sha256(document)


# --- hashed one way, revalidated the other ----------------------------------


def test_an_inventory_digest_computed_the_other_way_is_refused():
    """The exact failure this class prevents: same document, two digests.

    The inventory carries one non-ASCII name, so the ASCII-escaped form and the
    canonical form disagree on bytes. Before this change the lane would accept
    the manifest, record a `manifest_sha256` from the second form, and hand the
    surface downstream with a digest nothing else could reproduce.
    """
    files = inventory(["x.tif", "y.tif", "café.txt"])
    assert ascii_escaped_sha256(files) != content_sha256(files)
    module = lane_module()
    forged = manifest(files, artifact_sha256=ascii_escaped_sha256(files))
    with pytest.raises(module.MergeRefused, match="artifact_sha256"):
        module.require_canonical_manifest(forged)


def test_a_digest_over_the_whole_document_is_not_the_inventory_digest():
    """The other half of the divergence: the same function, the wrong scope."""
    module = lane_module()
    document = manifest()
    wrong_scope = dict(document)
    wrong_scope["artifact_sha256"] = content_sha256(
        {key: value for key, value in document.items() if key != "artifact_sha256"})
    with pytest.raises(module.MergeRefused, match="artifact_sha256"):
        module.require_canonical_manifest(wrong_scope)


def test_a_rewritten_inventory_keeping_the_published_digest_is_refused():
    """A digest that is never recomputed from what it names verifies nothing."""
    module = lane_module()
    published = manifest()
    swapped = manifest(inventory(["x.tif", "y.tif", "z.tif"], seed=9))
    replayed = {**swapped, "artifact_sha256": published["artifact_sha256"]}
    assert swapped["artifact_sha256"] != published["artifact_sha256"]
    with pytest.raises(module.MergeRefused, match="artifact_sha256"):
        module.require_canonical_manifest(replayed)


def test_a_canonical_manifest_is_accepted_and_returned_unchanged():
    module = lane_module()
    document = manifest()
    assert module.require_canonical_manifest(document) == document
    assert module.require_canonical_manifest(
        document, expected_artifact_sha256=document["artifact_sha256"],
        schema="campaignx.merged_tifxyz_artifact_set.v1") == document


def test_a_manifest_that_is_not_the_catalogued_artifact_is_refused():
    module = lane_module()
    with pytest.raises(module.MergeRefused, match="catalogue"):
        module.require_canonical_manifest(manifest(), expected_artifact_sha256="a" * 64)


def test_a_manifest_of_the_wrong_schema_is_refused():
    module = lane_module()
    with pytest.raises(module.MergeRefused, match="schema"):
        module.require_canonical_manifest(
            manifest(), schema="campaignx.vc3d_merge_evidence_set.v1")


@pytest.mark.parametrize("broken", [
    "not a mapping",
    {"schema": "campaignx.merged_tifxyz_artifact_set.v1", "files": {}},
    {"schema": "campaignx.merged_tifxyz_artifact_set.v1",
     "files": {"x.tif": {"sha256": "0" * 64}}, "artifact_sha256": "b" * 64},
    {"schema": "campaignx.merged_tifxyz_artifact_set.v1",
     "files": {"x.tif": {"sha256": "NOTHEX", "size_bytes": 1}},
     "artifact_sha256": "b" * 64},
    {"schema": "campaignx.merged_tifxyz_artifact_set.v1",
     "files": {"x.tif": {"sha256": "0" * 64, "size_bytes": True}},
     "artifact_sha256": "b" * 64},
    manifest(artifact_sha256="NOT A DIGEST"),
    manifest(schema=None),
])
def test_a_manifest_that_cannot_be_hashed_at_all_is_refused(broken):
    module = lane_module()
    with pytest.raises(module.MergeRefused):
        module.require_canonical_manifest(broken)


# --- both paths use it ------------------------------------------------------


def test_publication_stores_the_digest_its_own_revalidation_recomputes(tmp_path):
    """The stored/publication hash and the revalidation must agree exactly."""
    module = lane_module()
    source = tmp_path / "out"
    source.mkdir()
    names = ["x.tif", "y.tif", "z.tif", "meta.json"]
    for index, name in enumerate(names):
        (source / name).write_bytes(b"tifxyz-" + bytes([index]))
    published = module.publish_set(
        source, names, schema="campaignx.merged_tifxyz_artifact_set.v1",
        store_spec=str(tmp_path / "store"), sample_id="PHerc826",
        key="surface-1", attempt="surface-1-scientific")

    stored = json.loads(
        (Path(published["artifact_uri"]) / "ARTIFACT_SET.json").read_text())
    assert stored == published["manifest"]
    assert module.require_canonical_manifest(
        stored, expected_artifact_sha256=published["artifact_sha256"]) == stored
    assert published["artifact_sha256"] == content_sha256(stored["files"])


def test_a_parent_manifest_hashed_the_other_way_refuses_the_merge(tmp_path,
                                                                 monkeypatch):
    """Materialization is the second path, and it must refuse what publication would."""
    module = lane_module()
    files = inventory(["x.tif", "y.tif", "café.txt"])
    forged = manifest(files, artifact_sha256=ascii_escaped_sha256(files))

    import fleet.certifier as certifier

    class Adapter:
        @staticmethod
        def materialize_surface(uri, digest, destination, **_kwargs):
            Path(destination).mkdir(parents=True, exist_ok=True)
            return forged

    monkeypatch.setattr(certifier, "load_qc_adapter", lambda: Adapter)
    parents = [{"surface_id": "surface-a", "artifact_uri": "s3://b/surface-a",
                "artifact_sha256": forged["artifact_sha256"]}]
    with pytest.raises(module.MergeRefused, match="artifact_sha256"):
        module.materialize_parents(parents, {"surface-a": "s000"}, tmp_path)


def test_a_materialized_parent_records_the_one_canonical_manifest_digest(
        tmp_path, monkeypatch):
    module = lane_module()
    # Non-ASCII somewhere in the document, so the two forms provably differ and
    # the assertion below is about which one was used, not about a coincidence.
    document = manifest(inventory(["x.tif", "y.tif", "café.txt"]))
    assert ascii_escaped_sha256(document) != content_sha256(document)

    import fleet.certifier as certifier

    class Adapter:
        @staticmethod
        def materialize_surface(uri, digest, destination, **_kwargs):
            Path(destination).mkdir(parents=True, exist_ok=True)
            return document

    monkeypatch.setattr(certifier, "load_qc_adapter", lambda: Adapter)
    parents = [{"surface_id": "surface-a", "artifact_uri": "s3://b/surface-a",
                "artifact_sha256": document["artifact_sha256"]}]
    materialized = module.materialize_parents(
        parents, {"surface-a": "s000"}, tmp_path)
    assert materialized[0]["manifest_sha256"] == content_sha256(document)
    assert materialized[0]["manifest_sha256"] != ascii_escaped_sha256(document)


def test_a_parent_whose_manifest_is_not_the_catalogued_artifact_is_refused(
        tmp_path, monkeypatch):
    module = lane_module()
    document = manifest()

    import fleet.certifier as certifier

    class Adapter:
        @staticmethod
        def materialize_surface(uri, digest, destination, **_kwargs):
            Path(destination).mkdir(parents=True, exist_ok=True)
            return document

    monkeypatch.setattr(certifier, "load_qc_adapter", lambda: Adapter)
    parents = [{"surface_id": "surface-a", "artifact_uri": "s3://b/surface-a",
                "artifact_sha256": "c" * 64}]
    with pytest.raises(module.MergeRefused, match="catalogue"):
        module.materialize_parents(parents, {"surface-a": "s000"}, tmp_path)


def test_the_lane_does_not_restate_the_canonical_form():
    """A textual guard: the divergence returns the moment someone restates it.

    canonical() must call fleet.common.canonical_bytes, not reproduce its flags.
    The only json.dumps left may be write_json, which formats evidence files for
    people to read and is hashed as file bytes, never as a document.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "from fleet.common import canonical_bytes" in source
    assert "ensure_ascii=" not in source, (
        "passing ensure_ascii here means the canonical form is being restated "
        "rather than called, which is the bug this module is the fix for")
    assert source.count("json.dumps(") == 1, (
        "run_vc3d_tifxyz_merge.py serializes JSON in exactly one place -- "
        "write_json(), for human-readable evidence. Anything hashed goes "
        "through canonical().")
