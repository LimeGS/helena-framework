"""A P1 task can be asked what it read.

An audit asked for every task to carry the P0 artifact and selection version, the
CT URI and hash, the m7 URI and hash, the catalogue hash and the coordinate/scale
contract. Most of that was already there, one indirection away: the task names a
source_snapshot_id, and bootstrap_sources writes ct_uri, m7_uri, coordinate_frame,
voxel_size_um, shape_xyz and eligible_manifest_sha256 onto that snapshot.

Two things were genuinely missing, and one of them is not a defect.

The gap was the P0 selection. The launcher records one and then queued a run that
did not know which was current, so a mission that reselected between two runs
produced two tasks that look identical and read different inputs. That is now on
the task, resolved by the panel from the control plane rather than taken from the
browser's word for it.

The byte hashes are the other, and they are not coming. A CT volume in this
catalogue is 20840x8387x8387 voxels behind an HTTP URL; hashing it to queue a task
is not a thing anybody will do. The snapshot already marks itself
URI_LOCKED_HASH_UNAVAILABLE, which is the honest ceiling rather than a gap, and
this test pins that label so it cannot quietly become an empty string that reads
like "verified".
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "panel/app.py").read_text()
GENERATOR = (ROOT / "framework/stages/01-segmentation/fleet/generator.py").read_text()
CLI = (ROOT / "framework/stages/01-segmentation/fleet/cli.py").read_text()
LAUNCHER = (ROOT / "panel/web/src/routes/Segmentation.tsx").read_text()


def test_the_snapshot_carries_the_source_contract() -> None:
    """What the task reads, one indirection from the task itself."""
    snapshot = GENERATOR.split("def bootstrap_sources")[1]
    snapshot = snapshot[: snapshot.index("\ndef ")]
    for field in ("ct_uri", "m7_uri", "coordinate_frame", "voxel_size_um",
                  "shape_xyz", "eligible_manifest_sha256"):
        assert f'"{field}"' in snapshot, f"the snapshot no longer records {field}"


def test_an_unhashed_source_says_so_rather_than_saying_nothing() -> None:
    """The difference between "not verified" and "nothing to report".

    An absent hash with no status beside it reads as though the field simply did
    not apply. It applies; it could not be met.
    """
    snapshot = GENERATOR.split("def bootstrap_sources")[1]
    snapshot = snapshot[: snapshot.index("\ndef ")]
    assert "URI_LOCKED_HASH_UNAVAILABLE" in snapshot
    assert "HASH_LOCKED" in snapshot, (
        "there is no longer a state for a source whose bytes were verified, so "
        "the unavailable one has nothing to be distinguished from"
    )


def test_the_task_carries_the_p0_selection_it_read() -> None:
    # On the task dict, not merely in a signature it is threaded through: the
    # first version of this test passed with the field removed from the task,
    # because the parameter name still appeared in three function definitions.
    # Both builders, counted rather than found: two functions create tasks here,
    # and an earlier version of this test passed with the field stripped from one
    # of them because the other still carried it.
    assert GENERATOR.count('**({"p0_selection_version": p0_selection_version}') == 2, (
        "not every task builder records which P0 selection it read"
    )
    assert GENERATOR.count('**({"p0_artifact_id": p0_artifact_id}') == 2
    # Through the CLI, so a run queued from a terminal records it too.
    assert '"--p0-selection-version"' in CLI
    assert '"--p0-artifact-id"' in CLI


def test_the_panel_resolves_the_selection_itself() -> None:
    """Not from the request body.

    A browser that names its own selection version is describing a per-run
    override nothing else in the control plane knows about. The panel reads the
    mission's current selection and records that.
    """
    handler = APP[APP.index('@app.post("/api/segmentation/runs")'):]
    handler = handler[: handler.index("\n@app.", 1)]
    assert "artifact_contract.current_selection" in handler, (
        "the run handler does not read the mission's selection, so whatever it "
        "records about P0 came from somewhere it should not have"
    )
    assert "mission_id: str | None = Field(None" in APP, (
        "the run request no longer names a mission, so there is no selection to "
        "resolve"
    )
    # The browser sends the mission and nothing more about the selection.
    assert "mission_id: missionId" in LAUNCHER
    assert "p0_selection_version" not in LAUNCHER, (
        "the browser is sending a selection version, which is the override this "
        "was written to avoid"
    )


def test_provenance_no_longer_fails_open() -> None:
    """The opposite of what this test used to assert, and an audit was right.

    It said a failure reading the selection should not block the run. That is
    wrong when the operator named a mission: the run then proceeds with blank
    provenance and looks provenanced, which is worse than not running. A mission
    that cannot be read, or a scroll with no P0 artifact, is a reason to stop.

    A run queued with no mission at all is untouched -- that is the CLI's case and
    it never claimed a selection.
    """
    handler = APP[APP.index('@app.post("/api/segmentation/runs")'):]
    handler = handler[: handler.index("\n@app.", 1)]
    assert "cannot say what" in handler, (
        "a mission whose selection cannot be read no longer stops the run"
    )
    assert "no P0 artifact is registered" in handler
    assert "provenance must not block a run" not in handler, (
        "the fail-open comment is still there, so the behaviour probably is too"
    )


def test_a_reselection_cannot_silently_insert_nothing() -> None:
    """The defect the audit named, and the sharpest one.

    Task identity is (source_snapshot, grid_version, cell, policy_version) with ON
    CONFLICT DO NOTHING, and the P0 artifact is not in it. So requeueing after a
    reselection inserted nothing while the reply said queued, and the tasks that
    already existed kept the older artifact id -- the control plane reading one
    thing and telling anyone who asked the task that it read another.
    """
    handler = APP[APP.index('@app.post("/api/segmentation/runs")'):]
    handler = handler[: handler.index("\n@app.", 1)]
    assert "p0_artifact_id' IS NOT NULL" in handler, (
        "nothing checks whether tasks already exist against another artifact"
    )
    assert "already_queued_against" in handler
    assert "409" in handler
    # And the check must come before the queue, or it is a report and not a gate.
    assert handler.index("already_queued_against") < handler.index("argv = [")


def test_the_task_records_the_digest_and_how_it_resolved() -> None:
    """An id cannot show the artifact changed underneath it."""
    assert '"--p0-artifact-sha256"' in CLI
    assert '"--p0-resolved-by"' in CLI
    assert GENERATOR.count('**({"p0_artifact_sha256": p0_artifact_sha256}') == 2
    assert GENERATOR.count('**({"p0_resolved_by": p0_resolved_by}') == 2
    # resolve() is what the rest of the platform uses, and it distinguishes a
    # selection from the newest registered artifact.
    assert "artifact_contract.resolve(" in APP
