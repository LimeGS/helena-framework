"""Artifacts are immutable, selections move, and lineage survives both.

The workflow these exist for: run a phase, notice later that an earlier phase
was wrong, redo the earlier phase, and carry on -- without the earlier results
quietly becoming lies and without anybody having to remember which run used
which input.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.contracts import artifact  # noqa: E402


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_identity_is_the_content(tmp_path: Path):
    """The same bytes are the same artifact; different bytes are not."""
    first = write(tmp_path / "a" / "catalog.json", '{"um": 9.362}')
    same = write(tmp_path / "b" / "catalog.json", '{"um": 9.362}')
    other = write(tmp_path / "c" / "catalog.json", '{"um": 2.399}')

    a = artifact.register(tmp_path, phase="P0", sample_id="PHerc0139",
                          kind="catalog-entry", path=first)
    b = artifact.register(tmp_path, phase="P0", sample_id="PHerc0139",
                          kind="catalog-entry", path=same)
    c = artifact.register(tmp_path, phase="P0", sample_id="PHerc0139",
                          kind="catalog-entry", path=other)

    assert a["artifact_id"] == b["artifact_id"]
    assert a["artifact_id"] != c["artifact_id"]
    # Registering the same content twice records it once.
    assert len(artifact.artifacts(tmp_path)) == 2


def test_a_correction_does_not_replace_what_it_corrects(tmp_path: Path):
    """Both versions exist afterwards. That is what makes going back safe."""
    old = artifact.register(tmp_path, phase="P0", sample_id="PHerc0139",
                            kind="catalog-entry",
                            path=write(tmp_path / "v1" / "c.json", '{"um": 9.362}'))
    new = artifact.register(tmp_path, phase="P0", sample_id="PHerc0139",
                            kind="catalog-entry",
                            path=write(tmp_path / "v2" / "c.json", '{"um": 2.399}'),
                            note="the 9.362 entry named the wrong rescan")
    ids = {a["artifact_id"] for a in artifact.artifacts(tmp_path, phase="P0")}
    assert {old["artifact_id"], new["artifact_id"]} <= ids


def test_a_directory_hashes_by_its_whole_tree(tmp_path: Path):
    """Surfaces are directories, and a file changing inside one is a different
    surface even though the directory name did not move."""
    surface = tmp_path / "surface"
    write(surface / "x.tif", "x")
    write(surface / "y.tif", "y")
    write(surface / "meta.json", "{}")
    before = artifact.register(tmp_path, phase="P1", sample_id="PHerc0139",
                               kind="surface", path=surface)

    write(surface / "y.tif", "y changed")
    after = artifact.register(tmp_path, phase="P1", sample_id="PHerc0139",
                              kind="surface", path=surface)
    assert before["artifact_id"] != after["artifact_id"]
    assert after["file_count"] == 3


def test_an_empty_directory_is_not_an_artifact(tmp_path: Path):
    (tmp_path / "nothing").mkdir()
    with pytest.raises(artifact.ArtifactError, match="empty directory"):
        artifact.register(tmp_path, phase="P1", sample_id="X", kind="surface",
                          path=tmp_path / "nothing")


def test_lineage_finds_everything_a_correction_affects(tmp_path: Path):
    """The question after redoing P0: what did the old one feed?"""
    p0 = artifact.register(tmp_path, phase="P0", sample_id="S", kind="catalog-entry",
                           path=write(tmp_path / "p0" / "c.json", "1"))
    p1 = artifact.register(tmp_path, phase="P1", sample_id="S", kind="surface",
                           path=write(tmp_path / "p1" / "x.tif", "surface"),
                           inputs=[p0["artifact_id"]])
    p4 = artifact.register(tmp_path, phase="P4", sample_id="S", kind="layers",
                           path=write(tmp_path / "p4" / "00.tif", "layers"),
                           inputs=[p1["artifact_id"]])
    unrelated = artifact.register(tmp_path, phase="P1", sample_id="OTHER",
                                  kind="surface",
                                  path=write(tmp_path / "other" / "x.tif", "elsewhere"))

    affected = {a["artifact_id"] for a in artifact.descendants(tmp_path, p0["artifact_id"])}
    assert affected == {p1["artifact_id"], p4["artifact_id"]}
    assert unrelated["artifact_id"] not in affected


def test_the_selection_is_versioned_as_a_whole(tmp_path: Path):
    a = artifact.register(tmp_path, phase="P0", sample_id="S", kind="catalog-entry",
                          path=write(tmp_path / "a" / "c.json", "1"))
    b = artifact.register(tmp_path, phase="P0", sample_id="S", kind="catalog-entry",
                          path=write(tmp_path / "b" / "c.json", "2"))

    first = artifact.select(tmp_path, choices={"P0/S": a["artifact_id"]},
                            reason="the original")
    second = artifact.select(tmp_path, choices={"P0/S": b["artifact_id"]},
                             reason="the 9.362 entry named the wrong rescan")

    assert first["version_id"] != second["version_id"]
    assert first["content_sha256"] != second["content_sha256"]
    assert artifact.current_selection(tmp_path)["version_id"] == second["version_id"]
    assert len(artifact.selections(tmp_path)) == 2


def test_selecting_the_same_thing_twice_is_refused(tmp_path: Path):
    a = artifact.register(tmp_path, phase="P0", sample_id="S", kind="catalog-entry",
                          path=write(tmp_path / "a" / "c.json", "1"))
    artifact.select(tmp_path, choices={"P0/S": a["artifact_id"]})
    with pytest.raises(artifact.ArtifactError, match="already current"):
        artifact.select(tmp_path, choices={"P0/S": a["artifact_id"]})


def test_a_selection_cannot_name_an_unregistered_or_mismatched_artifact(tmp_path: Path):
    a = artifact.register(tmp_path, phase="P0", sample_id="S", kind="catalog-entry",
                          path=write(tmp_path / "a" / "c.json", "1"))
    with pytest.raises(artifact.ArtifactError, match="not registered"):
        artifact.select(tmp_path, choices={"P0/S": "p0:S:deadbeefcafe"})
    # Right artifact, wrong slot: this is the mistake that would otherwise make
    # a phase read another phase's output and never say so.
    with pytest.raises(artifact.ArtifactError, match="selected under"):
        artifact.select(tmp_path, choices={"P4/S": a["artifact_id"]})


def test_going_back_is_a_new_version_naming_the_old_one(tmp_path: Path):
    """History is never rewritten, so the round trip stays visible."""
    a = artifact.register(tmp_path, phase="P0", sample_id="S", kind="catalog-entry",
                          path=write(tmp_path / "a" / "c.json", "1"))
    b = artifact.register(tmp_path, phase="P0", sample_id="S", kind="catalog-entry",
                          path=write(tmp_path / "b" / "c.json", "2"))
    first = artifact.select(tmp_path, choices={"P0/S": a["artifact_id"]})
    artifact.select(tmp_path, choices={"P0/S": b["artifact_id"]})

    back = artifact.restore_selection(tmp_path, first["version_id"])
    assert back["restored_from"] == first["version_id"]
    assert back["content_sha256"] == first["content_sha256"]
    assert back["version_id"] != first["version_id"]
    # Three decisions, not one that never happened.
    assert len(artifact.selections(tmp_path)) == 3
    assert artifact.resolve(tmp_path, "P0", "S")["artifact_id"] == a["artifact_id"]


def test_resolve_says_whether_anybody_chose(tmp_path: Path):
    """"selected" and "the only one there was" are different claims."""
    a = artifact.register(tmp_path, phase="P0", sample_id="S", kind="catalog-entry",
                          path=write(tmp_path / "a" / "c.json", "1"))
    assert artifact.resolve(tmp_path, "P0", "S")["resolved_by"].startswith("newest")
    artifact.select(tmp_path, choices={"P0/S": a["artifact_id"]})
    assert artifact.resolve(tmp_path, "P0", "S")["resolved_by"] == "selection"
    assert artifact.resolve(tmp_path, "P9", "S") is None


def test_one_phase_can_export_one_artifact_per_scroll(tmp_path: Path):
    """P0's shape: many artifacts of the same kind, one per scroll, and the
    selection addresses them independently."""
    for index, sample in enumerate(("PHerc0139", "PHerc0172", "PHerc1667")):
        artifact.register(tmp_path, phase="P0", sample_id=sample, kind="catalog-entry",
                          path=write(tmp_path / sample / "c.json", str(index)))
    exported = artifact.artifacts(tmp_path, phase="P0")
    assert len({a["sample_id"] for a in exported}) == 3

    choices = {artifact.selection_key("P0", a["sample_id"]): a["artifact_id"]
               for a in exported}
    version = artifact.select(tmp_path, choices=choices)
    assert len(version["choices"]) == 3
    for sample in ("PHerc0139", "PHerc0172", "PHerc1667"):
        assert artifact.resolve(tmp_path, "P0", sample)["resolved_by"] == "selection"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


PIPELINE = ROOT / "framework" / "contracts" / "pipeline_phases.json"


def test_a_run_records_what_it_read_without_being_told(tmp_path: Path):
    """The point of the whole thing: lineage as a by-product of running.

    P5 declares it needs what P4 produces, in pipeline_phases.json. Nothing
    passes that edge in -- the run resolves it.
    """
    layers = tmp_path / "p4-out"
    layers.mkdir()
    (layers / "00.tif").write_text("layer", encoding="utf-8")
    upstream = artifact.record_run(tmp_path, PIPELINE, phase="P4", sample_id="S",
                                   output=layers, produced_by="job:1")

    maps = tmp_path / "p5-out"
    maps.mkdir()
    (maps / "probability.npy").write_text("map", encoding="utf-8")
    downstream = artifact.record_run(tmp_path, PIPELINE, phase="P5", sample_id="S",
                                     output=maps, produced_by="job:2")

    assert upstream["kind"] == "surface-layers"
    assert downstream["kind"] == "probability-map"
    assert downstream["inputs"] == [upstream["artifact_id"]]
    assert downstream["produced_by"] == "job:2"


def test_a_run_reads_the_selection_not_the_newest(tmp_path: Path):
    """A run inherits the mission's choice, not whatever landed most recently.

    Otherwise correcting a phase would silently redirect every job already in
    flight, which is the behaviour this whole design exists to prevent.
    """
    first = tmp_path / "p4-a"
    first.mkdir()
    (first / "00.tif").write_text("a", encoding="utf-8")
    old = artifact.record_run(tmp_path, PIPELINE, phase="P4", sample_id="S",
                              output=first, produced_by="job:1")

    second = tmp_path / "p4-b"
    second.mkdir()
    (second / "00.tif").write_text("b", encoding="utf-8")
    artifact.record_run(tmp_path, PIPELINE, phase="P4", sample_id="S",
                        output=second, produced_by="job:2")

    # Pin the older one, then run P5.
    artifact.select(tmp_path, choices={"P4/S": old["artifact_id"]},
                    reason="the newer render has a seam")
    maps = tmp_path / "p5-out"
    maps.mkdir()
    (maps / "probability.npy").write_text("map", encoding="utf-8")
    run = artifact.record_run(tmp_path, PIPELINE, phase="P5", sample_id="S",
                              output=maps, produced_by="job:3")
    assert run["inputs"] == [old["artifact_id"]]


def test_bookkeeping_never_fails_the_run(tmp_path: Path):
    """A job that produced a real map must not be marked failed because the
    note about it could not be written."""
    missing = tmp_path / "never-written"
    assert artifact.record_run(tmp_path, PIPELINE, phase="P5", sample_id="S",
                               output=missing, produced_by="job:1") is None

    empty = tmp_path / "empty"
    empty.mkdir()
    assert artifact.record_run(tmp_path, PIPELINE, phase="P5", sample_id="S",
                               output=empty, produced_by="job:1") is None
