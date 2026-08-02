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

from .common import utc_now
from .finalizer import certify_surface_geometry

QC_ADAPTER_RELATIVE = (
    "framework/stages/04-validation/scripts/campaignx_surface_qc_adapter.py"
)


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


def certify_one(
    store: Any,
    surface: dict[str, Any],
    workspace: Path,
    *,
    adapter: ModuleType | None = None,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Fetch one surface, measure it, and record what the gate said."""

    adapter = adapter or load_qc_adapter()
    surface_id = str(surface["surface_id"])
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
        receipt = certify_surface_geometry(staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    state = str(receipt.get("geometry_qc_state") or "GEOMETRY_UNMEASURED")
    recorded = store.record_geometry_certification(surface_id, state, receipt)
    return {**recorded, "reason": receipt.get("reason"),
            "resolution_limited": receipt.get("resolution_limited"),
            "median_edge_voxels": receipt.get("median_edge_voxels")}


def certify_pending(
    store: Any,
    *,
    workspace: Path | None = None,
    limit: int = 25,
    sample_id: str | None = None,
    dry_run: bool = False,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Give a verdict to surfaces that have none. Safe to run again."""

    pending = store.surfaces_without_geometry_verdict(limit=limit, sample_id=sample_id)
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
        outcomes.append(certify_one(store, surface, root,
                                    adapter=adapter, s3_client=s3_client))
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
        "generated_at_utc": utc_now(),
    }
