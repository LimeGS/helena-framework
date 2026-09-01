"""P2 over surfaces that already exist.

The geometry gate runs inside the finalizer, so a surface grown after the gate
landed gets a verdict on its way out. Every surface grown before it did not, and
nothing ever went back for them: all 43 surfaces in the control plane read
GEOMETRY_UNMEASURED, which is not a verdict but the absence of one. P2 was a
gate with no way to be applied to the corpus it was written for.

This is that way. It reads the surfaces carrying no verdict, materialises each
one from wherever it was published, runs the same gate the finalizer runs, and
records the result through the same store method -- so a surface certified here
and a surface certified on its way out are indistinguishable afterwards, which
is the only way the verdict means one thing.

Non-claims
----------
* Certifying an old surface does not re-examine the segmentation that produced
  it. The verdict describes the artifact, and the artifact is what downstream
  phases consume.
* GEOMETRY_UNMEASURED remains a possible outcome here. A surface whose artifact
  cannot be fetched, or whose grid is too coarse to measure, is still unmeasured
  afterwards -- recorded as such, rather than quietly counted as certified.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

from .common import content_sha256, utc_now
from .finalizer import certify_surface_geometry

QC_ADAPTER_RELATIVE = (
    "framework/stages/04-validation/scripts/campaignx_surface_qc_adapter.py"
)
GEOMETRY_PROFILE_ID = "tifxyz-geometry-certification@1.0.0"
LAMINA_PROFILE_PATH = (
    "framework/profiles/01-segmentation/lamina-gate-1.0.0.json"
)


def load_lamina_profile(root: Path | None = None) -> dict[str, Any]:
    """The frozen bands, read from the repository rather than typed here.

    A calibration in code is a calibration nobody can cite. This one came from
    somebody else's measurements and the file says so, including what has not
    been checked about it.
    """
    import json  # noqa: PLC0415

    base = Path(root) if root else Path(__file__).resolve().parents[4]
    return json.loads((base / LAMINA_PROFILE_PATH).read_text(encoding="utf-8"))


def assess_surface_lamina(
    store: Any,
    surface: dict[str, Any],
    staging: Path,
    *,
    requested_by_job_id: str,
    ct_sampler: Any | None = None,
) -> dict[str, Any]:
    """Read the CT along this surface's normal and record what it found.

    Never raises. Every way this can fail -- no registered source for the
    scroll, no voxel size, a volume that will not answer, a runtime without
    zarr -- ends in LAMINA_UNMEASURED carrying the reason, because an absent
    measurement is not a verdict and must not be able to fail a geometry
    certification that already succeeded.
    """
    from .lamina_columns import sample_columns  # noqa: PLC0415

    surface_id = str(surface["surface_id"])
    unmeasured = {"state": "LAMINA_UNMEASURED", "surface_id": surface_id}
    try:
        profile = load_lamina_profile()
        snapshots = [row for row in store.snapshots({str(surface["sample_id"])})
                     if row.get("ct_uri")]
        if not snapshots:
            return {**unmeasured,
                    "reason": (f"no registered source for {surface['sample_id']} "
                               "names a CT volume to read")}
        snapshot = snapshots[0]
        voxel_um = float(snapshot.get("voxel_size_um") or 0.0)
        if voxel_um <= 0:
            return {**unmeasured,
                    "reason": "the registered source states no voxel size in microns"}
        if ct_sampler is None:
            from .ct_support import OmeZarrCtSupportSampler  # noqa: PLC0415

            ct_sampler = OmeZarrCtSupportSampler()
        outcome = sample_columns(
            staging, ct_uri=str(snapshot["ct_uri"]), voxel_size_um=voxel_um,
            profile=profile, sampler=ct_sampler)
    except Exception as failure:  # noqa: BLE001 -- see the docstring
        return {**unmeasured,
                "reason": f"{type(failure).__name__}: {failure}"}

    recorded = store.record_lamina_assessment(
        surface_id, str(outcome["state"]), outcome,
        requested_by_job_id=requested_by_job_id,
        profile_id=str(profile["profile_id"]),
        profile_sha256=content_sha256(profile),
    )
    return {**recorded, "reason": outcome.get("reason"),
            "median_thickness_um": outcome.get("median_thickness_um"),
            "clean_fraction": outcome.get("clean_fraction"),
            "bimodality": outcome.get("bimodality")}


def load_qc_adapter() -> ModuleType:
    """The adapter, for its surface materialiser and nothing else.

    Fetching a published surface -- S3 or a local mirror, manifest read, digest
    verified -- is solved there and has been running against this bucket since
    July. A second copy of that logic is a second thing that can disagree about
    whether an artifact is intact.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / QC_ADAPTER_RELATIVE
        if candidate.is_file():
            spec = spec_from_file_location("campaignx_surface_qc_adapter", candidate)
            if spec is None or spec.loader is None:
                break
            module = module_from_spec(spec)
            # The adapter imports scripts.harness at module scope, which resolves
            # only from the repository root. Normally it is run as a subprocess
            # with that on the path; loading it in-process is not, so put the
            # root there rather than not importing it -- the alternative is a
            # second implementation of "fetch a published surface intact".
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            spec.loader.exec_module(module)
            return module
    raise RuntimeError(f"surface QC adapter not found at {QC_ADAPTER_RELATIVE}")


def _registered_voxel_um(store: Any, surface: dict[str, Any]) -> float | None:
    """This scroll's scale, or None when the control plane does not say.

    The geometry gate reads three of its thresholds as lengths when it is given
    one and as counts of cells and voxels when it is not, and it records which.
    So this never raises: a scroll with no registered source still gets a
    verdict, made against the frozen counts and saying so.
    """
    try:
        for row in store.snapshots({str(surface["sample_id"])}):
            voxel = float(row.get("voxel_size_um") or 0.0)
            if voxel > 0:
                return voxel
    except Exception:  # noqa: BLE001 -- see the docstring
        return None
    return None


def certify_one(
    store: Any,
    surface: dict[str, Any],
    workspace: Path,
    *,
    adapter: ModuleType | None = None,
    s3_client: Any | None = None,
    ct_sampler: Any | None = None,
    requested_by_job_id: str,
) -> dict[str, Any]:
    """Fetch one surface, measure it, and record what the gate said."""

    adapter = adapter or load_qc_adapter()
    surface_id = str(surface["surface_id"])
    # An absence until something measures it. A surface whose artifact could not
    # be fetched never reaches the gate, and saying so is the whole point of the
    # word: unmeasured is not a pass and not a failure.
    lamina: dict[str, Any] = {"state": "LAMINA_UNMEASURED", "surface_id": surface_id,
                              "reason": "the surface artifact was not fetched"}
    staging = Path(workspace) / surface_id
    try:
        adapter.materialize_surface(
            str(surface["artifact_uri"]),
            str(surface["artifact_sha256"] or ""),
            staging,
            s3_client=s3_client,
        )
    except Exception as failure:  # noqa: BLE001
        # An artifact that cannot be fetched is unmeasured, not rejected. The
        # difference matters: rejected blocks the surface from the ink model
        # for a reason found in its geometry, and a network error is not one.
        receipt = {
            "schema": "campaignx.tifxyz_geometry_certification.v1",
            "geometry_qc_state": "GEOMETRY_UNMEASURED",
            "reason": "ARTIFACT_UNAVAILABLE",
            "status": "FAIL",
            "measurement_complete": False,
            "error": f"{type(failure).__name__}: {failure}",
            "generated_at_utc": utc_now(),
            "non_claims": ["an unmeasured surface is not a certified surface"],
        }
    else:
        receipt = certify_surface_geometry(
            staging, voxel_um=_registered_voxel_um(store, surface))
        # The third axis, measured off the same staged copy while it is still
        # here. Never fatal: a volume that cannot be read leaves the lamina
        # unmeasured, which is an absence of a verdict, and must not take the
        # geometry verdict down with it.
        lamina = assess_surface_lamina(
            store, surface, staging,
            requested_by_job_id=requested_by_job_id, ct_sampler=ct_sampler)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    state = str(receipt.get("geometry_qc_state") or "GEOMETRY_UNMEASURED")
    profile_sha256 = content_sha256(receipt.get("policy") or {})
    recorded = store.record_geometry_certification(
        surface_id, state, receipt,
        requested_by_job_id=requested_by_job_id,
        profile_id=GEOMETRY_PROFILE_ID,
        profile_sha256=profile_sha256,
    )
    return {**recorded, "reason": receipt.get("reason"),
            "result": receipt, "result_sha256": content_sha256(receipt),
            "resolution_limited": receipt.get("resolution_limited"),
            "median_edge_voxels": receipt.get("median_edge_voxels"),
            "lamina": lamina}


def certify_pending(
    store: Any,
    *,
    workspace: Path | None = None,
    limit: int = 25,
    sample_id: str | None = None,
    surface_id: str | None = None,
    dry_run: bool = False,
    s3_client: Any | None = None,
    mission_id: str | None = None,
    requested_by_job_id: str,
) -> dict[str, Any]:
    """Give a verdict to surfaces that have none. Safe to run again.

    `mission_id` scopes the backlog to one mission's own surfaces. Without it
    this is the whole control plane, which is right for a fleet-wide sweep from
    the CLI and wrong for a run somebody started inside a mission.
    """

    pending = store.surfaces_without_geometry_verdict(
        limit=limit, sample_id=sample_id, surface_id=surface_id,
        mission_id=mission_id)
    if dry_run:
        return {
            "schema": "campaignx.segment_geometry_certification_run.v1",
            "dry_run": True,
            "considered": len(pending),
            "surfaces": [s["surface_id"] for s in pending],
            "generated_at_utc": utc_now(),
        }

    adapter = load_qc_adapter()
    outcomes: list[dict[str, Any]] = []
    root = Path(workspace) if workspace else Path(tempfile.mkdtemp(prefix="p2-certify-"))
    root.mkdir(parents=True, exist_ok=True)
    for surface in pending:
        outcomes.append(certify_one(
            store, surface, root, adapter=adapter, s3_client=s3_client,
            requested_by_job_id=requested_by_job_id))
    verdicts: dict[str, int] = {}
    for outcome in outcomes:
        state = outcome["geometry_qc_state"]
        verdicts[state] = verdicts.get(state, 0) + 1
    return {
        "schema": "campaignx.segment_geometry_certification_run.v1",
        "considered": len(pending),
        "certified": verdicts,
        "blocked_qc_jobs": sum(int(o.get("blocked_qc_jobs") or 0) for o in outcomes),
        "surfaces": outcomes,
        "requested_by_job_id": requested_by_job_id,
        "generated_at_utc": utc_now(),
    }
