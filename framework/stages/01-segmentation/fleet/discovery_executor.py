from __future__ import annotations

import hashlib
import inspect
import json
import os
import socket
import sys
from typing import Any

from .common import content_sha256, stable_id
from .ct_support import OmeZarrCtSupportSampler


DISCOVERY_EXECUTOR_CAPABILITY = "FIRST_LETTERS_DISCOVERY_CT_PROBE_V1"
PRODUCTION_DISCOVERY_EXECUTOR_ID = "helena-first-letters-discovery-v1"


def runtime_discovery_executor_sha256(executor: Any) -> str:
    """Hash executable and sampler class source, never object metadata."""

    try:
        sources = [inspect.getsource(type(executor))]
        sampler = getattr(executor, "_ct_sampler", None)
        if sampler is not None:
            sources.append(inspect.getsource(type(sampler)))
    except (OSError, TypeError) as error:
        raise ValueError("DISCOVERY_EXECUTOR_CODE_UNAVAILABLE") from error
    return hashlib.sha256("\0".join(sources).encode("utf-8")).hexdigest()


def production_discovery_worker_id(*, executor: Any | None = None) -> str:
    """Who is registering, including which executor they are.

    The registry is immutable per worker_id, which is the control that stops a
    worker swapping its executor and keeping its identity. In a container the
    other three ingredients are `gpu-1`, pid 1 and /usr/local/bin/python3 on
    every deploy, so the id was stable while the executor was not -- and there
    is no update path in either store. Adding a retry to the CT sampler moved
    `executor_sha256`, the init container raised
    DISCOVERY_EXECUTOR_REGISTRATION_CONFLICT, and the panel never started.

    So the executor's digest is part of the identity. A different executor is a
    different worker: a new row, the old one untouched, nothing mutated and
    nothing superseded. The control is not weakened -- a swap is still
    impossible, because the id moves with the code and the old registration
    stays exactly as it was.

    A pinned `HELENA_FIRST_LETTERS_DISCOVERY_WORKER_ID` still wins, and an
    operator who pins one is choosing to hold the id still across code changes,
    which is the case the conflict is there to catch.
    """
    configured = os.environ.get("HELENA_FIRST_LETTERS_DISCOVERY_WORKER_ID")
    if configured:
        return configured
    return stable_id("first-letters-discovery-worker", {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python": sys.executable,
        **({"executor_sha256": runtime_discovery_executor_sha256(executor)}
           if executor is not None else {}),
    })


def production_discovery_executor_registration(
    executor: Any, *, worker_id: str,
) -> dict[str, Any]:
    core = {
        "schema":
            "campaignx.first_letters_discovery_executor_registration.v1",
        "worker_id": worker_id,
        "executor_id": PRODUCTION_DISCOVERY_EXECUTOR_ID,
        "executor_sha256": runtime_discovery_executor_sha256(executor),
        "capabilities": [DISCOVERY_EXECUTOR_CAPABILITY],
        "enabled": True,
        "allow_unvalidated": False,
    }
    return {**core, "registration_sha256": content_sha256(core)}


class ProductionFirstLettersDiscoveryExecutor:
    """Ink-blind CT sampler and fail-closed noncanonical probe executor.

    The v1 production path performs the registered CT read itself. Until a
    full probe grow backend is configured, a CT-supported candidate is retained
    only as ``GEOMETRY_UNMEASURED`` and therefore can never become a winner.
    """

    def __init__(self, *, ct_sampler: Any | None = None):
        self._ct_sampler = ct_sampler or OmeZarrCtSupportSampler()
        self._claim_tokens: dict[str, str] = {}

    def accept_first_letters_discovery_claim(
        self, *, run_id: str, claim_token: str,
    ) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("discovery executor run ID is invalid")
        if not isinstance(claim_token, str) or len(claim_token) < 32:
            raise ValueError("discovery executor claim token is invalid")
        self._claim_tokens[run_id] = claim_token

    def first_letters_discovery_claim_token(self, *, run_id: str) -> str | None:
        return self._claim_tokens.get(run_id)

    @staticmethod
    def _json_bytes(value: dict[str, Any]) -> bytes:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    def measure_first_letters_discovery_run(
        self, *, executor_claim: dict[str, Any],
        run_authority: dict[str, Any], provider_request: dict[str, Any],
        provider_response_bytes: bytes, source_snapshot: dict[str, Any],
    ) -> tuple[Any, ...]:
        from .seed_probe import (
            FirstLettersDiscoveryCandidateMeasurement,
            PROBE_PROFILE_SHA256,
            project_provider_response_v1,
        )

        if (executor_claim.get("run_id") != run_authority.get("run_id")
                or executor_claim.get("capability") !=
                    DISCOVERY_EXECUTOR_CAPABILITY):
            raise ValueError("discovery executor claim differs from run")
        projected = project_provider_response_v1(provider_response_bytes)
        dependencies = {
            row["role"]: row for row in run_authority["dependencies"]
        }
        ct_dependency = dependencies["CT_VOLUME"]
        ct_uri = source_snapshot.get("ct_uri")
        if projected["candidates"] and (
            not isinstance(ct_uri, str) or not ct_uri
        ):
            raise ValueError("production discovery CT source is unavailable")
        sampling = (
            source_snapshot.get("first_letters_discovery_authority") or {}
        ).get("executor_ct_sampling") or {}
        level = int(sampling.get("level", 5))
        radius = int(sampling.get("radius_l0_voxels", 1))
        measurements = []
        for candidate in projected["candidates"]:
            coordinate = candidate["promotion_coordinate_ct_l0_xyz"]
            if coordinate is None:
                measurements.append(FirstLettersDiscoveryCandidateMeasurement(
                    candidate_id=candidate["candidate_id"],
                    ct_read_evidence_bytes=None,
                    probe_evidence_bytes=None,
                ))
                continue
            sample = self._ct_sampler.sample(
                ct_uri,
                {axis: coordinate[index] for index, axis in enumerate("xyz")},
                level=level, radius_l0_voxels=radius,
            )
            sampled = sample.get("voxel_count")
            nonzero = sample.get("nonzero_voxel_count")
            ct_read = {
                "schema":
                    "campaignx.first_letters_ct_material_read_evidence.v1",
                "candidate_id": candidate["candidate_id"],
                "source_snapshot_id": run_authority["source_snapshot_id"],
                "raw_coordinate_sha256": candidate["raw_coordinate_sha256"],
                "ct_metadata_sha256": ct_dependency["artifact_sha256"],
                "ct_read_set_manifest_sha256":
                    ct_dependency["read_set_manifest_sha256"],
                "sampled_voxel_count": sampled,
                "nonzero_voxel_count": nonzero,
                "allow_unvalidated": False,
            }
            probe = None
            region = provider_request["ct_l0_region"]
            cell_clearance = min(
                *(coordinate[index] - region["minimum"][index]
                  for index in range(3)),
                *(region["maximum"][index] - 1 - coordinate[index]
                  for index in range(3)),
            )
            shape = run_authority["source_shape_xyz"]
            volume_clearance = min(
                *coordinate,
                *(shape[index] - 1 - coordinate[index]
                  for index in range(3)),
            )
            clearance_supported = (
                cell_clearance >=
                    run_authority["minimum_cell_clearance_voxels"]
                and volume_clearance >=
                    run_authority["minimum_volume_clearance_voxels"]
            )
            if (isinstance(nonzero, int) and not isinstance(nonzero, bool)
                    and nonzero > 0 and clearance_supported):
                probe = self._json_bytes({
                    "schema":
                        "campaignx.first_letters_probe_geometry_read_evidence.v1",
                    "candidate_id": candidate["candidate_id"],
                    "raw_coordinate_sha256":
                        candidate["raw_coordinate_sha256"],
                    "probe_execution_profile_sha256": PROBE_PROFILE_SHA256,
                    "measurement_complete": False,
                    "geometry_qc_state": "GEOMETRY_UNMEASURED",
                    "allow_unvalidated": False,
                })
            measurements.append(FirstLettersDiscoveryCandidateMeasurement(
                candidate_id=candidate["candidate_id"],
                ct_read_evidence_bytes=self._json_bytes(ct_read),
                probe_evidence_bytes=probe,
            ))
        return tuple(measurements)
