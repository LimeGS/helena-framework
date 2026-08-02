"""S3 is optional; the panel host is where data lives by default.

The worker refuses a local --artifact-root, and it is right to: a surface
written to a worker's own scratch carries that path downstream, so QC on
another host requeues it forever and it dies with the machine.

But that refusal assumed object storage was the only alternative, and it is
not. The default deployment keeps everything in a volume on the panel host, and
a worker on that same machine publishing into that volume is not stranding
anything -- the surface is exactly where every later phase looks for it.

Nothing inside the container can tell the two cases apart. Both are a local
path. So the deployment says which it is, and these tests pin that down.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = (ROOT / "containers/images/worker-entrypoint.sh").read_text()
SEGMENT = yaml.safe_load((ROOT / "containers/compose/segment.compose.yaml").read_text())
INK = yaml.safe_load((ROOT / "containers/compose/ink.compose.yaml").read_text())
PLATFORM = yaml.safe_load((ROOT / "containers/compose/platform.compose.yaml").read_text())


def test_the_panel_owns_a_volume_for_artifacts() -> None:
    """The default store has to be a named volume, not a path on the host.

    A bind mount is somebody's directory; a named volume is the deployment's
    own and survives a recreate without anyone having prepared a path.
    """
    assert "helena-artifacts" in PLATFORM["volumes"], (
        "the panel has no artifact volume, so there is nowhere for the default "
        "storage to be"
    )
    mounts = " ".join(PLATFORM["services"]["panel"]["volumes"])
    assert ":/artifacts" in mounts, "the panel does not mount its artifact volume"


def test_the_worker_attaches_the_panel_volume_rather_than_making_its_own() -> None:
    """external: true is the whole point.

    Without it compose creates `helena-segment_helena-artifacts` -- a second,
    empty volume with a similar name -- and the worker publishes into a
    directory the panel has never heard of. The surfaces would be written,
    recorded in the control plane, and invisible.
    """
    declared = SEGMENT.get("volumes", {})
    assert "helena-artifacts" in declared, (
        "segment.compose.yaml does not declare the panel's artifact volume"
    )
    assert declared["helena-artifacts"].get("external") is True, (
        "the artifact volume is not external, so compose will create a second "
        "one scoped to this stack and the panel will never see what is written"
    )


def test_both_services_write_to_the_same_place() -> None:
    """segment grows the surface and fleet-runner flattens it. A phase that
    reads from a different volume than the one written to fails as a missing
    file, which reads like a bug in the science."""
    for service in ("segment", "fleet-runner"):
        mounts = " ".join(SEGMENT["services"][service]["volumes"])
        assert "HELENA_SEGMENT_ARTIFACTS" in mounts and ":/artifacts" in mounts, (
            f"{service} does not mount the artifact store at /artifacts"
        )


def test_the_ink_worker_attaches_the_same_artifact_volume() -> None:
    """P4 and P5 publish there too, and their worker was the only one detached.

    With the panel's default local stores, the old compose wrote beneath the
    container's ephemeral ``/artifacts`` and advertised those bytes as durable.
    """
    declared = INK.get("volumes", {})
    assert declared.get("helena-artifacts", {}).get("external") is True
    mounts = " ".join(INK["services"]["ink"]["volumes"])
    assert "HELENA_INK_ARTIFACTS" in mounts and ":/artifacts" in mounts


def test_a_local_artifact_root_is_still_refused_unless_declared() -> None:
    """The guard has to keep guarding.

    The entrypoint may only pass --allow-local-artifacts when the deployment
    has explicitly said this mount is the panel's storage. Passing it whenever
    the root happens to be local would silently restore the exact failure the
    refusal exists to prevent -- on a multi-host deployment, where it matters.
    """
    assert "--allow-local-artifacts" in ENTRYPOINT
    flag = ENTRYPOINT.index("artifact_local_ok=\"--allow-local-artifacts\"")
    guard = ENTRYPOINT.index("HELENA_ARTIFACTS_ON_PANEL")
    assert guard < flag, (
        "the entrypoint allows local artifacts without the deployment having "
        "declared that the mount is the panel's volume"
    )


def test_object_storage_still_takes_precedence_when_configured() -> None:
    """S3 is optional, not forbidden. A deployment that sets an s3:// root must
    keep using it, and must not be handed the local escape hatch."""
    case = ENTRYPOINT[ENTRYPOINT.index('case "$ARTIFACT_ROOT" in'):]
    case = case[: case.index("esac")]
    remote = case[: case.index(")")]
    for scheme in ("s3://", "http://", "https://"):
        assert scheme in remote, f"{scheme} is not treated as remote storage"
    # And the remote branch does nothing but continue -- no flag.
    assert "--allow-local-artifacts" not in case[: case.index("*)")]
