from __future__ import annotations

import errno
import hashlib
import json
import inspect
import os
import threading
import time
from itertools import product
from pathlib import Path
from typing import Any, Callable, Protocol
import traceback
from urllib.error import HTTPError, URLError

from .common import canonical_bytes, content_sha256, utc_now, write_json_atomic
from .ct_support import CtSupportSampler, CtSupportSourceUnavailable, apply_ct_material_support_gate
from .executor import GrowExecutor, InsufficientGpuMemoryError
from .finalizer import finalize_surface
from .planner import (
    DeterministicPlanner,
    Planner,
    PlannerOutputInvalid,
    PlannerProviderUnavailable,
    PlannerScientificViolation,
    screen_candidates,
    task_packet_for_planner,
    validate_and_lock,
)
from .seed_probe import (
    ProbeWinnerMaterializationError,
    SeedProbeCoordinator,
    validate_seed_probe_benchmark_execution_task,
    validate_seed_probe_task_contract,
)
from .store import FleetStore, normalize_worker_capabilities


class SeedProvider(Protocol):
    def discover(self, task: dict[str, Any]) -> dict[str, Any]: ...


class SourceProviderUnavailable(RuntimeError):
    """A transient CT/m7 source outage, not a geometric assessment."""


def task6_recenter_candidates(candidates: Any) -> dict[str, Any]:
    """Build recenter evidence from integral Task 6 candidates only."""

    from .seed_probe import coordinate_sha256_v1, validate_task6_coordinate

    if not isinstance(candidates, list):
        raise ValueError("Task 6 candidates must be a list")
    eligible: list[tuple[str, list[int]]] = []
    rejected: list[str] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id"))
        promotion = candidate.get("promotion_coordinate_ct_l0_xyz")
        if promotion is None:
            rejected.append(candidate_id)
            continue
        coordinate = validate_task6_coordinate(promotion, require_integral=True)
        if (candidate.get("promotion_coordinate_sha256") !=
                coordinate_sha256_v1(coordinate)):
            raise ValueError("Task 6 recenter coordinate hash drift")
        eligible.append((candidate_id, coordinate))
    if not eligible:
        return {
            "eligible_candidate_ids": [],
            "rejected_candidate_ids": rejected,
            "median_coordinate_ct_l0_xyz": None,
        }
    axes = [sorted(coordinate[index] for _, coordinate in eligible) for index in range(3)]
    middle = len(eligible) // 2
    return {
        "eligible_candidate_ids": [candidate_id for candidate_id, _ in eligible],
        "rejected_candidate_ids": rejected,
        "median_coordinate_ct_l0_xyz": [axis[middle] for axis in axes],
    }


def _is_transient_operational_error(error: BaseException) -> bool:
    """Return true only for explicit transport, storage, or DB outages.

    Integrity, authorization, schema, and policy failures stay terminal.  The
    cause chain is inspected because boto/urllib/psycopg commonly wrap the
    underlying timeout or connection error.
    """
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, HTTPError):
            if current.code in {429, 500, 502, 503, 504}:
                return True
        elif isinstance(current, URLError):
            return True
        elif isinstance(current, (TimeoutError, ConnectionError)):
            return True
        elif isinstance(current, OSError) and current.errno in {
            errno.EAGAIN,
            errno.EBUSY,
            errno.ECONNABORTED,
            errno.ECONNREFUSED,
            errno.ECONNRESET,
            errno.EHOSTUNREACH,
            errno.ENETDOWN,
            errno.ENETUNREACH,
            errno.ENOBUFS,
            errno.ETIMEDOUT,
        }:
            return True

        response = getattr(current, "response", None)
        if isinstance(response, dict):
            metadata = response.get("ResponseMetadata")
            status = (
                metadata.get("HTTPStatusCode")
                if isinstance(metadata, dict)
                else None
            )
            code = (
                response.get("Error", {}).get("Code")
                if isinstance(response.get("Error"), dict)
                else None
            )
            if status == 429 or isinstance(status, int) and status >= 500:
                return True
            if str(code) in {
                "InternalError",
                "RequestTimeout",
                "ServiceUnavailable",
                "SlowDown",
            }:
                return True

        class_name = type(current).__name__
        module_name = type(current).__module__
        if class_name in {
            "ConnectTimeoutError",
            "ConnectionClosedError",
            "EndpointConnectionError",
            "ReadTimeoutError",
        }:
            return True
        message = str(current).lower()
        if (
            class_name in {"InterfaceError", "OperationalError"}
            and module_name.startswith(("psycopg", "psycopg2"))
        ):
            return True
        if class_name == "OperationalError" and module_name == "sqlite3":
            if "database is locked" in message or "database is busy" in message:
                return True
        if any(
            token in message
            for token in (
                "http error 429",
                "http error 500",
                "http error 502",
                "http error 503",
                "http error 504",
                "connection refused",
                "connection reset",
                "connection closed",
                "database is locked",
                "database is busy",
                "read timeout",
                "request timeout",
                "server closed the connection",
                "service unavailable",
                "slow down",
                "timed out",
                "temporarily unavailable",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_transient_source_error(error: BaseException) -> bool:
    """Compatibility name for the source adapter's narrower call site."""

    return _is_transient_operational_error(error)


class RecordedSeedProvider:
    def discover(self, task: dict[str, Any]) -> dict[str, Any]:
        candidates = task.get("recorded_candidates")
        if not isinstance(candidates, list):
            raise RuntimeError("task has no recorded_candidates fixture")
        return {
            "schema": "campaignx.recorded_seed_candidates.v1",
            "candidates": candidates,
            "fixture": True,
            "ink_used": False,
        }


class TaskRoutedSeedProvider:
    """Routes each task to the provider it names, instead of one per process.

    A task already declares `candidate_discovery.provider`, so the worker does not
    need to be started differently to run manual seeds -- which matters because a
    fleet has one worker per host and manual and automatic work arrive in the same
    queue. Restarting a host to grow somebody's point would make the feature
    unusable in the way that counts.

    No schema change: the field is on every task the generator has ever written.
    An unrecognised provider falls through to the default rather than failing, so
    a task written by a newer generator does not strand on an older worker.
    """

    def __init__(self, default: SeedProvider):
        self.default = default
        self.manual = ManualSeedProvider()

    def discover(self, task: dict[str, Any]) -> dict[str, Any]:
        provider = str((task.get("candidate_discovery") or {}).get("provider") or "")
        if provider == "manual":
            return self.manual.discover(task)
        return self.default.discover(task)


class ManualSeedProvider:
    """Seeds a person supplied, instead of seeds a model proposed.

    The same envelope every other provider returns, so the CT-material gate,
    the planner contract and the validator all apply unchanged -- a manual seed
    is screened by exactly what screens a proposed one. That matters more here,
    not less: a manual seed skips the m7 prediction, so the raw scan agreeing
    that there is material at the point is the only remaining check between a
    person's guess and hours of growing.

    `fixture` is false. RecordedSeedProvider marks its candidates as fixtures
    because they are test material; these are real coordinates on a real scan
    and their surfaces belong in the catalogue. What keeps a surface out of it
    is `fixture_only` on the task, which this does not set.

    `seed_origin` is what a receipt needs to stay honest. A surface grown from a
    human seed and one the fleet found alone are the same object and not the
    same claim, and "the fleet found forty surfaces" is only true if the two can
    be told apart afterwards.
    """

    def discover(self, task: dict[str, Any]) -> dict[str, Any]:
        candidates = task.get("manual_candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError(
                "task has no manual_candidates; a manual run needs at least one "
                "uploaded point, and an empty list is a mistake rather than a "
                "finding of nothing")
        return {
            "schema": "campaignx.manual_seed_candidates.v1",
            "candidates": candidates,
            "fixture": False,
            "seed_origin": "human",
            "ink_used": False,
        }


class McpSeedProvider:
    def __init__(self, url: str | None = None, token: str | None = None):
        self.url = url or os.environ.get("VC_MCP_URL")
        self.token = token or os.environ.get("VC_MCP_AUTH_TOKEN")

    def discover(self, task: dict[str, Any]) -> dict[str, Any]:
        if not self.url or not self.token:
            raise RuntimeError("VC_MCP_URL and VC_MCP_AUTH_TOKEN are required")
        from campaign_x import McpClient, structured

        discovery = task["candidate_discovery"]
        request = {
            "prediction_uri": discovery["prediction_uri"],
            "prediction_space": discovery.get("prediction_space", "ct_l0_xyz"),
            "region": discovery["region"],
            "max_candidates": int(discovery.get("max_candidates", 8)),
            "minimum_separation_voxels": int(discovery.get("minimum_separation_voxels", 16)),
            # The published threshold, when the task carries one. Absent means the
            # server keeps its own default, which is what every task queued before
            # this existed relies on.
            **({"threshold": float(discovery["m7_threshold"])}
               if discovery.get("m7_threshold") is not None else {}),
        }
        try:
            client = McpClient(self.url, self.token)
            client.initialize()
            request_suffix = str(task.get("attempt_id") or "survey")
            exchanges: list[dict[str, Any]] = []
            read_objects: dict[str, dict[str, Any]] = {}

            def call(arguments: dict[str, Any], suffix: str) -> dict[str, Any]:
                response = structured(client.call(
                    "vc_find_seed_candidates", arguments,
                    f"segment-fleet-{task['task_id']}-{request_suffix}-{suffix}"))
                exchanges.append({"request": arguments, "response": response})
                source_read_set = response.get("source_read_set") if isinstance(response, dict) else None
                objects = (source_read_set or {}).get("objects")
                if (not isinstance(source_read_set, dict)
                        or source_read_set.get("schema") != "campaignx.first_letters_source_read_set.v1"
                        or not isinstance(objects, list) or not objects
                        or source_read_set.get("canonical_manifest_sha256") != content_sha256(objects)):
                    raise SourceProviderUnavailable(
                        "MCP returned no valid production source read evidence")
                for item in objects:
                    if (not isinstance(item, dict) or not item.get("object_key")
                            or not isinstance(item.get("bytes"), int)
                            or not isinstance(item.get("sha256"), str)
                            or len(item["sha256"]) != 64):
                        raise SourceProviderUnavailable(
                            "MCP returned malformed production source read evidence")
                    key = str(item["object_key"])
                    existing = read_objects.get(key)
                    if existing is not None and existing != item:
                        raise SourceProviderUnavailable(
                            f"MCP returned contradictory read evidence for {key}")
                    read_objects[key] = dict(item)
                return response

            def candidate_array(response: dict[str, Any], phase: str) -> list[dict[str, Any]]:
                """Validate the MCP's required candidate-array output contract.

                The current Streamable HTTP wrapper can encode a tool-side
                exception as an empty structured object.  That is neither an
                empty m7 result nor a safe basis for a NO_SEED decision.  It
                is explicitly a retryable source failure.
                """

                raw = response.get("candidates") if isinstance(response, dict) else None
                if not isinstance(raw, list):
                    raise SourceProviderUnavailable(
                        f"MCP returned no candidate array during {phase}; source response is unusable"
                    )
                if not all(isinstance(candidate, dict) for candidate in raw):
                    raise SourceProviderUnavailable(
                        f"MCP returned a malformed candidate array during {phase}"
                    )
                return raw

            policy = str(discovery.get("seed_region_policy", "fixed-v1"))
            if policy == "fixed-v1":
                result = call(request, "fixed")
                effective_region = request["region"]
                initial_probe: dict[str, Any] | None = None
            elif policy in {"m7-recenter-z-v1", "m7-recenter-xyz-v1"}:
                if request["prediction_space"] != "ct_l0_xyz":
                    raise RuntimeError(f"{policy} requires ct_l0_xyz prediction space")
                probe_limit = int(discovery.get("recenter_probe_max_candidates", 100))
                if probe_limit < 1 or probe_limit > 100:
                    raise RuntimeError("recenter_probe_max_candidates must be 1..100")
                first_request = {**request, "max_candidates": probe_limit}
                initial = call(first_request, "initial")
                raw = candidate_array(initial, "initial recenter probe")
                coordinates: dict[str, list[int]] = {axis: [] for axis in "xyz"}
                for candidate in raw:
                    coordinate = candidate.get("ct_l0_coordinate") if isinstance(candidate, dict) else None
                    if not isinstance(coordinate, dict):
                        continue
                    for axis in "xyz":
                        if axis in coordinate:
                            coordinates[axis].append(int(float(coordinate[axis])))
                if not coordinates["z"]:
                    result = initial
                    effective_region = request["region"]
                    initial_probe = {
                        "policy": policy,
                        "request": first_request,
                        "candidate_count": 0,
                        "recentered": False,
                    }
                else:
                    for values in coordinates.values():
                        values.sort()
                    raw_radius = discovery.get("recenter_radius_xyz", {"x": 64, "y": 64, "z": 64})
                    if not isinstance(raw_radius, dict) or any(axis not in raw_radius for axis in "xyz"):
                        raise RuntimeError("recenter_radius_xyz must define x/y/z")
                    radius = {axis: int(raw_radius[axis]) for axis in "xyz"}
                    if any(value < 1 or value > 192 for value in radius.values()):
                        raise RuntimeError("recenter_radius_xyz values must be 1..192")
                    if policy == "m7-recenter-xyz-v1":
                        if not all(coordinates[axis] for axis in "xyz"):
                            raise RuntimeError("m7-recenter-xyz-v1 requires x/y/z coordinates from the initial probe")
                        center = {axis: coordinates[axis][len(coordinates[axis]) // 2] for axis in "xyz"}
                    else:
                        center = {axis: int(request["region"]["center"][axis]) for axis in "xy"}
                        center["z"] = coordinates["z"][len(coordinates["z"]) // 2]
                    effective_region = {"center": center, "radius": radius}
                    second_request = {**request, "region": effective_region}
                    result = call(second_request, "recentered")
                    candidate_array(result, "recentered query")
                    initial_probe = {
                        "policy": policy,
                        "request": first_request,
                        "candidate_count": len(coordinates["z"]),
                        "median_z_ct_l0": coordinates["z"][len(coordinates["z"]) // 2],
                        "median_coordinate_ct_l0": center if policy == "m7-recenter-xyz-v1" else None,
                        "recentered": True,
                    }
            elif policy in {
                "m7-recenter-z-chunk-safe-v1",
                "m7-chunk-safe-merge-interior-v2",
            }:
                # A 128-voxel broad cube can cross 3x3x3 m7 chunks depending
                # on its alignment. VC3D refuses requests spanning >8 chunks.
                # Eight deterministic 64-voxel subregions cover that cube and
                # each can touch at most 2x2x2 chunks for the frozen 192-voxel
                # m7 chunk geometry. This is a new policy, never an implicit
                # fallback for the historical single-query policy.
                if request["prediction_space"] != "ct_l0_xyz":
                    raise RuntimeError(f"{policy} requires ct_l0_xyz prediction space")
                probe_limit = int(discovery.get("recenter_probe_max_candidates", 100))
                if probe_limit < 1 or probe_limit > 100:
                    raise RuntimeError("recenter_probe_max_candidates must be 1..100")
                raw_radius = discovery.get("recenter_radius_xyz", {"x": 64, "y": 64, "z": 64})
                if raw_radius != {"x": 64, "y": 64, "z": 64}:
                    raise RuntimeError(f"{policy} requires recenter_radius_xyz exactly 64x64x64")
                initial_candidates: dict[tuple[int, int, int, str], dict[str, Any]] = {}
                subquery_receipts: list[dict[str, Any]] = []
                for index, offsets in enumerate(product((-64, 64), repeat=3), start=1):
                    center = {
                        axis: int(request["region"]["center"][axis]) + int(offset)
                        for axis, offset in zip("xyz", offsets, strict=True)
                    }
                    subregion = {"center": center, "radius": {"x": 64, "y": 64, "z": 64}}
                    subrequest = {**request, "region": subregion, "max_candidates": probe_limit}
                    response = call(subrequest, f"chunk-safe-{index:02d}")
                    candidates = candidate_array(response, f"chunk-safe initial subquery {index}")
                    subquery_receipts.append({
                        "region": subregion,
                        "candidate_count": len(candidates),
                        "response_sha256": content_sha256(response),
                    })
                    for candidate in candidates:
                        coordinate = candidate.get("ct_l0_coordinate")
                        if not isinstance(coordinate, dict):
                            continue
                        try:
                            key = (
                                int(float(coordinate["x"])),
                                int(float(coordinate["y"])),
                                int(float(coordinate["z"])),
                                str(candidate.get("candidate_id", "")),
                            )
                        except (KeyError, TypeError, ValueError):
                            continue
                        # Identical candidates from the inclusive subregion
                        # boundary are intentionally coalesced byte-for-byte.
                        initial_candidates.setdefault(key, candidate)
                combined = [initial_candidates[key] for key in sorted(initial_candidates)]
                depths = [
                    int(float(candidate["ct_l0_coordinate"]["z"]))
                    for candidate in combined
                    if isinstance(candidate.get("ct_l0_coordinate"), dict)
                    and "z" in candidate["ct_l0_coordinate"]
                ]
                if policy == "m7-chunk-safe-merge-interior-v2":
                    # The historical recenter policy used the merged probe only
                    # to estimate depth, then discarded it and requested eight
                    # final candidates from a 64-voxel cube.  On real m7 grids
                    # those top candidates can all lie on a subquery face, so a
                    # scientifically useful interior-clearance gate rejects
                    # every one despite the broad probe having valid interior
                    # candidates.  V2 preserves the deterministic union and
                    # lets the existing geometry-only screen rank candidates
                    # by score and clearance relative to the original broad
                    # task region.  It does not read ink or previous outcomes.
                    result = {
                        "candidates": combined,
                        "ink_used": False,
                    }
                    effective_region = request["region"]
                    initial_probe = {
                        "policy": policy,
                        "subquery_count": len(subquery_receipts),
                        "subqueries": subquery_receipts,
                        "candidate_count": len(combined),
                        "merged_candidate_count": len(combined),
                        "recentered": False,
                        "merge_used_as_final_candidate_set": True,
                    }
                elif not depths:
                    result = {
                        "candidates": [],
                        "ink_used": False,
                    }
                    effective_region = request["region"]
                    initial_probe = {
                        "policy": policy,
                        "subquery_count": len(subquery_receipts),
                        "subqueries": subquery_receipts,
                        "candidate_count": 0,
                        "recentered": False,
                    }
                else:
                    depths.sort()
                    median_z = depths[len(depths) // 2]
                    center = {axis: int(request["region"]["center"][axis]) for axis in "xy"}
                    center["z"] = median_z
                    effective_region = {"center": center, "radius": {"x": 64, "y": 64, "z": 64}}
                    result = call({**request, "region": effective_region}, "recentered")
                    candidate_array(result, "chunk-safe recentered query")
                    initial_probe = {
                        "policy": policy,
                        "subquery_count": len(subquery_receipts),
                        "subqueries": subquery_receipts,
                        "candidate_count": len(depths),
                        "median_z_ct_l0": median_z,
                        "recentered": True,
                    }
            else:
                raise RuntimeError(f"unsupported seed_region_policy: {policy}")
        except BaseException as error:
            if _is_transient_source_error(error):
                raise SourceProviderUnavailable(f"transient MCP source failure: {error}") from error
            raise
        candidate_array(result, "final candidate query")
        request_document = [entry["request"] for entry in exchanges]
        response_document = [entry["response"] for entry in exchanges]
        request_bytes = canonical_bytes(request_document)
        response_bytes = canonical_bytes(response_document)
        objects = [read_objects[key] for key in sorted(read_objects)]
        source_read_set = {
            "schema": "campaignx.first_letters_source_read_set.v1",
            "objects": objects,
            "canonical_manifest_sha256": content_sha256(objects),
        } if objects else None
        # Credentials are never present in the returned receipt.
        return {
            **result,
            "request": request,
            "effective_candidate_region": effective_region,
            "initial_probe": initial_probe,
            "source_read_set": source_read_set,
            "provider_exchange": {
                "encoding": "canonical-json-utf8",
                "call_count": len(exchanges),
                "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
                "request_bytes": len(request_bytes),
                "response_sha256": hashlib.sha256(response_bytes).hexdigest(),
                "response_bytes": len(response_bytes),
            },
            "ink_used": False,
        }


class LeaseHeartbeat:
    def __init__(self, store: FleetStore, task: dict[str, Any], lease_seconds: int):
        self.store = store
        self.task = task
        self.lease_seconds = lease_seconds
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name=f"fleet-heartbeat-{task['attempt_id']}", daemon=True)

    def _run(self) -> None:
        interval = max(10.0, self.lease_seconds / 3.0)
        while not self.stop_event.wait(interval):
            try:
                self.store.heartbeat(self.task["task_id"], self.task["attempt_id"], self.task["lease_token"], self.lease_seconds)
            except BaseException as error:
                self.error = error
                self.stop_event.set()
                return

    def __enter__(self) -> "LeaseHeartbeat":
        self.thread.start()
        return self

    def ensure(self) -> None:
        if self.error is not None:
            raise RuntimeError("worker lost its lease heartbeat") from self.error

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.ensure()


class SegmentWorker:
    def __init__(
        self,
        store: FleetStore,
        worker_id: str,
        seed_provider: SeedProvider,
        planner: Planner,
        grow_executor: GrowExecutor,
        run_root: Path,
        artifact_root: Path | str,
        qc_profile_id: str,
        lease_seconds: int = 900,
        provider_retry_delay_seconds: int = 300,
        source_retry_delay_seconds: int = 300,
        task_id: str | None = None,
        ct_support_sampler: CtSupportSampler | None = None,
        worker_capabilities: dict[str, Any] | None = None,
        planner_factory: Callable[..., Planner] | None = None,
        seed_probe_support: bool = False,
        finalization_retry_delay_seconds: int = 300,
        probe_artifact_max_requeues: int = 5,
        finalization_max_requeues: int = 2,
    ):
        self.store = store
        self.worker_id = worker_id
        self.seed_provider = seed_provider
        self.planner = planner
        self.grow_executor = grow_executor
        self.run_root = run_root
        self.artifact_root = artifact_root
        if not qc_profile_id or "@" not in qc_profile_id:
            raise ValueError("qc_profile_id must be a versioned semantic profile ID")
        self.qc_profile_id = qc_profile_id
        self.lease_seconds = lease_seconds
        self.provider_retry_delay_seconds = provider_retry_delay_seconds
        self.source_retry_delay_seconds = source_retry_delay_seconds
        self.task_id = task_id
        self.ct_support_sampler = ct_support_sampler
        capabilities = dict(worker_capabilities or {})
        capabilities["seed_probe_v1"] = bool(seed_probe_support)
        self.worker_capabilities = normalize_worker_capabilities(capabilities)
        self.planner_factory = planner_factory
        self.seed_probe_support = bool(seed_probe_support)
        self.finalization_retry_delay_seconds = int(
            finalization_retry_delay_seconds
        )
        self.probe_artifact_max_requeues = int(
            probe_artifact_max_requeues
        )
        self.finalization_max_requeues = int(finalization_max_requeues)
        if (
            self.probe_artifact_max_requeues < 0
            or self.finalization_max_requeues < 0
        ):
            raise ValueError(
                "operational retry caps must be non-negative"
            )

    def planner_for(self, task: dict[str, Any]) -> Planner:
        """The planner this task asked for, or the one the host was started with.

        A run queued as "Panel of LLM experts" that grows with the deterministic
        planner because that is what the host happened to be started with is not
        a choice, it is a caption. The name rides on the task.

        A host that cannot build the named planner -- no API key, no opencode --
        says so as a provider outage rather than a policy rejection, so the task
        goes back to the queue for a host that can, instead of being consumed by
        a machine that was never equipped to run it.
        """
        wanted = str(task.get("planner") or "")
        if not wanted or self.planner_factory is None:
            return self.planner
        try:
            built = self.planner_factory(wanted, task.get("planner_model") or None)
            # Per-task planner configuration, which is what the panel refuses to
            # accept while it cannot reach a host. candidate_rank is the first
            # field to travel: it says which rung of m7's ordering to grow, and a
            # rank accepted at the API and dropped here would produce a surface
            # attributed to a configuration that never ran.
            rank = task.get("candidate_rank")
            if rank is not None:
                setattr(built, "candidate_rank", int(rank))
            return built
        except Exception as error:  # noqa: BLE001
            raise PlannerProviderUnavailable(
                f"this host cannot run the {wanted} planner: "
                f"{type(error).__name__}: {error}") from error

    def run_one(self) -> dict[str, Any] | None:
        task = self.store.claim(
            self.worker_id,
            self.lease_seconds,
            task_id=self.task_id,
            capabilities=self.worker_capabilities,
        )
        if task is None:
            return None
        attempt_dir = self.run_root / task["task_id"] / task["attempt_id"]
        attempt_dir.mkdir(parents=True, exist_ok=False)
        write_json_atomic(attempt_dir / "CLAIMED_TASK.json", {key: value for key, value in task.items() if key != "lease_token"})
        try:
            benchmark_execution = (
                validate_seed_probe_benchmark_execution_task(task)
            )
            expected_spec = (
                benchmark_execution["benchmark_spec_sha256"]
                if benchmark_execution is not None
                else None
            )
            if (
                self.worker_capabilities.get("benchmark_spec_sha256")
                != expected_spec
            ):
                raise ValueError(
                    "worker benchmark capability does not match the task "
                    "execution scope"
                )
        except ValueError as error:
            receipt = {
                "status": "POLICY_REJECTED",
                "failure_class": "CONFIGURATION_BLOCK",
                "reason": "INVALID_BENCHMARK_EXECUTION_CONTRACT",
                "error": f"{type(error).__name__}: {error}",
                "generated_at_utc": utc_now(),
                "ink_used": False,
                "non_claim": (
                    "No candidate source was read and no grow was executed."
                ),
            }
            write_json_atomic(attempt_dir / "TERMINAL_RECEIPT.json", receipt)
            self.store.mark_terminal(
                task["task_id"],
                task["attempt_id"],
                task["lease_token"],
                "POLICY_REJECTED",
                receipt,
            )
            return receipt
        with LeaseHeartbeat(self.store, task, self.lease_seconds) as heartbeat:
            self.store.transition(task["task_id"], task["attempt_id"], task["lease_token"], "PLANNING")
            if task.get("seed_probe") is not None:
                try:
                    if not self.seed_probe_support:
                        raise ValueError(
                            "worker has no seed-probe-v1 capability"
                        )
                    validate_seed_probe_task_contract(task)
                except (RuntimeError, ValueError) as error:
                    receipt = {
                        "status": "POLICY_REJECTED",
                        "failure_class": "CONFIGURATION_BLOCK",
                        "reason": "INVALID_SEED_PROBE_TASK_CONTRACT",
                        "error": f"{type(error).__name__}: {error}",
                        "generated_at_utc": utc_now(),
                        "ink_used": False,
                        "non_claim": (
                            "No candidate source was read and no seed was assessed."
                        ),
                    }
                    write_json_atomic(
                        attempt_dir / "TERMINAL_RECEIPT.json", receipt
                    )
                    self.store.mark_terminal(
                        task["task_id"],
                        task["attempt_id"],
                        task["lease_token"],
                        "POLICY_REJECTED",
                        receipt,
                    )
                    return receipt
            try:
                raw_candidates = self.seed_provider.discover(task)
                gated_candidates, ct_support_screen = apply_ct_material_support_gate(
                    raw_candidates,
                    task,
                    self.ct_support_sampler,
                )
            except (SourceProviderUnavailable, CtSupportSourceUnavailable) as error:
                receipt = {
                    "status": "RETRYABLE_SOURCE_UNAVAILABLE",
                    "error": f"{type(error).__name__}: {error}",
                    "retry_delay_seconds": self.source_retry_delay_seconds,
                    "generated_at_utc": utc_now(),
                    "ink_used": False,
                    "non_claim": "The CT/m7 source was temporarily unavailable; this does not assess geometry.",
                }
                write_json_atomic(attempt_dir / "RETRYABLE_SOURCE_RECEIPT.json", receipt)
                self.store.requeue_source_unavailable(
                    task["task_id"],
                    task["attempt_id"],
                    task["lease_token"],
                    receipt,
                    retry_delay_seconds=self.source_retry_delay_seconds,
                )
                return receipt
            except PlannerProviderUnavailable as error:
                receipt = {
                    "status": "RETRYABLE_PROVIDER_UNAVAILABLE",
                    "error": f"{type(error).__name__}: {error}",
                    "retry_delay_seconds": self.provider_retry_delay_seconds,
                    "generated_at_utc": utc_now(),
                    "ink_used": False,
                    "non_claim": "No model proposal was received; this does not assess geometry.",
                }
                write_json_atomic(attempt_dir / "RETRYABLE_PROVIDER_RECEIPT.json", receipt)
                self.store.requeue_provider_unavailable(
                    task["task_id"],
                    task["attempt_id"],
                    task["lease_token"],
                    receipt,
                    retry_delay_seconds=self.provider_retry_delay_seconds,
                )
                return receipt
            except BaseException as error:
                receipt = {
                    "status": "BLOCKED_SOURCE_UNAVAILABLE",
                    "failure_class": "SOURCE_FAILURE",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "generated_at_utc": utc_now(),
                }
                write_json_atomic(attempt_dir / "TERMINAL_RECEIPT.json", receipt)
                self.store.mark_terminal(task["task_id"], task["attempt_id"], task["lease_token"], "BLOCKED_SOURCE_UNAVAILABLE", receipt)
                return receipt
            write_json_atomic(attempt_dir / "SEED_CANDIDATES.json", raw_candidates)
            write_json_atomic(attempt_dir / "CT_MATERIAL_SUPPORT_SCREEN.json", ct_support_screen)
            candidate_screen = screen_candidates(gated_candidates, task)
            write_json_atomic(attempt_dir / "SEED_SCREEN.json", candidate_screen)
            candidates = candidate_screen["usable_candidates"]
            if not candidates:
                raw_m7_rows = raw_candidates.get("candidates", [])
                raw_m7_count = (
                    len(raw_m7_rows) if isinstance(raw_m7_rows, list) else 0
                )
                ct_input_count = int(
                    ct_support_screen.get("input_candidate_count", raw_m7_count)
                )
                ct_retained_count = int(
                    ct_support_screen.get("retained_candidate_count", ct_input_count)
                )
                ct_rejected_count = int(
                    ct_support_screen.get(
                        "rejected_candidate_count",
                        max(0, ct_input_count - ct_retained_count),
                    )
                )
                clearance = candidate_screen["rejection_diagnostics"]
                cause_counts = {
                    "NO_M7_CANDIDATES": 1 if raw_m7_count == 0 else 0,
                    "CT_MATERIAL_SUPPORT_REJECTED": ct_rejected_count,
                    **clearance["rejection_counts"],
                }
                primary_causes = sorted([
                    cause
                    for cause, count in cause_counts.items()
                    if int(count) > 0
                ])
                diagnosis = {
                    "schema": "campaignx.no_seed_causal_diagnosis.v1",
                    "status": "NO_SEED",
                    "task_id": task["task_id"],
                    "attempt_id": task["attempt_id"],
                    "m7_raw_candidate_count": raw_m7_count,
                    "ct_support_input_candidate_count": ct_input_count,
                    "ct_support_retained_candidate_count": ct_retained_count,
                    "ct_support_rejected_candidate_count": ct_rejected_count,
                    "post_ct_candidate_count": candidate_screen[
                        "raw_candidate_count"
                    ],
                    "eligible_after_clearance_count": candidate_screen[
                        "eligible_candidate_count"
                    ],
                    "cause_counts": cause_counts,
                    "primary_causes": primary_causes,
                    "clearance_policy": clearance["clearance_policy"],
                    "clearance_rejection_examples_first_32": clearance[
                        "rejection_examples_first_32"
                    ],
                    "generated_at_utc": utc_now(),
                    "ink_used": False,
                    "non_claim": (
                        "NO_SEED identifies the stage that removed proposals; "
                        "it does not establish absence of a physical surface."
                    ),
                }
                diagnosis["diagnosis_sha256"] = content_sha256({
                    key: value for key, value in diagnosis.items()
                    if key != "generated_at_utc"
                })
                write_json_atomic(
                    attempt_dir / "NO_SEED_CAUSAL_DIAGNOSIS.json", diagnosis
                )
                receipt = {
                    "status": "NO_SEED",
                    "raw_candidate_count": raw_m7_count,
                    "post_ct_candidate_count": candidate_screen[
                        "raw_candidate_count"
                    ],
                    "usable_candidate_count": 0,
                    "no_seed_cause_counts": cause_counts,
                    "primary_causes": primary_causes,
                    "no_seed_causal_diagnosis": diagnosis,
                    "no_seed_causal_diagnosis_sha256": diagnosis[
                        "diagnosis_sha256"],
                    "clearance_policy": clearance["clearance_policy"],
                    "reason": (
                        "No MCP candidate met the frozen interior-clearance and CT-material-support policies."
                        if ct_support_screen["status"] == "COMPLETED_INK_BLIND"
                        else "No MCP candidate met this task's frozen interior-clearance policy."
                    ),
                    "generated_at_utc": utc_now(),
                    "ink_used": False,
                    "non_claim": (
                        "This terminal records candidate availability and frozen "
                        "screening causes; it is not evidence that no surface exists."
                    ),
                }
                write_json_atomic(attempt_dir / "TERMINAL_RECEIPT.json", receipt)
                self.store.mark_terminal(task["task_id"], task["attempt_id"], task["lease_token"], "NO_SEED", receipt)
                return receipt
            planner_task = task
            planner_candidates = candidates
            probe_result: dict[str, Any] | None = None
            if task.get("seed_probe") is not None:
                if not self.seed_probe_support:
                    # Admission normally prevents this claim. Keep the runtime
                    # guard because a manually edited/legacy control plane must
                    # still fail closed instead of silently skipping a required
                    # experiment.
                    receipt = {
                        "status": "POLICY_REJECTED",
                        "failure_class": "CONFIGURATION_BLOCK",
                        "reason": "WORKER_HAS_NO_SEED_PROBE_V1_CAPABILITY",
                        "generated_at_utc": utc_now(),
                        "ink_used": False,
                        "non_claim": "No probe ran and no seed was assessed.",
                    }
                    write_json_atomic(
                        attempt_dir / "TERMINAL_RECEIPT.json", receipt
                    )
                    self.store.mark_terminal(
                        task["task_id"],
                        task["attempt_id"],
                        task["lease_token"],
                        "POLICY_REJECTED",
                        receipt,
                    )
                    return receipt
                self.store.transition(
                    task["task_id"],
                    task["attempt_id"],
                    task["lease_token"],
                    "PROBING",
                )
                heartbeat.ensure()
                coordinator = SeedProbeCoordinator(
                    self.store,
                    self.grow_executor,
                    self.artifact_root,
                    self.worker_id,
                    self.lease_seconds,
                    self.worker_capabilities,
                )
                try:
                    probe_result = coordinator.run(
                        task, candidates, attempt_dir
                    )
                except InsufficientGpuMemoryError as error:
                    observed_vram_gb = float(
                        self.worker_capabilities["gpu_vram_gb"]
                    )
                    current_floor = float(
                        task["resource_requirements"]["minimum_vram_gb"]
                    )
                    next_floor = max(
                        current_floor,
                        observed_vram_gb + 1.0
                        if observed_vram_gb > 0
                        else 12.0,
                    )
                    receipt = {
                        **error.receipt,
                        "status": "RETRY_ON_LARGER_GPU",
                        "phase": "SEED_PROBE",
                        "error": f"{type(error).__name__}: {error}",
                        "worker_capabilities": self.worker_capabilities,
                        "prior_minimum_vram_gb": current_floor,
                        "minimum_vram_gb": next_floor,
                        "generated_at_utc": utc_now(),
                        "ink_used": False,
                        "non_claim": (
                            "Probe GPU capacity is operational metadata, "
                            "not an assessment of a seed or geometry."
                        ),
                    }
                    write_json_atomic(
                        attempt_dir / "RETRY_ON_LARGER_GPU_RECEIPT.json",
                        receipt,
                    )
                    self.store.requeue_for_larger_gpu(
                        task["task_id"],
                        task["attempt_id"],
                        task["lease_token"],
                        receipt,
                        minimum_vram_gb=next_floor,
                    )
                    return receipt
                except ProbeWinnerMaterializationError as error:
                    cause = error.__cause__ or error
                    review_reason = (
                        "WINNER_ARTIFACT_MATERIALIZATION_FAILED"
                    )
                    if _is_transient_operational_error(cause):
                        retry_receipt = {
                            "status": "RETRYABLE_PROBE_ARTIFACT_UNAVAILABLE",
                            "phase": "PROBE_WINNER_MATERIALIZATION",
                            "probe_run_id": error.probe_run_id,
                            "winner_trial_id": error.winner_trial_id,
                            "artifact_uri": error.artifact_uri,
                            "error": (
                                f"{type(cause).__name__}: {str(cause)[:2000]}"
                            ),
                            "retry_delay_seconds": (
                                self.source_retry_delay_seconds
                            ),
                            "maximum_requeues": (
                                self.probe_artifact_max_requeues
                            ),
                            "generated_at_utc": utc_now(),
                            "ink_used": False,
                            "non_claim": (
                                "The retained winner could not be fetched due "
                                "to a transient infrastructure outage; no full "
                                "surface was grown or catalogued."
                            ),
                        }
                        write_json_atomic(
                            attempt_dir
                            / "RETRYABLE_PROBE_ARTIFACT_RECEIPT.json",
                            retry_receipt,
                        )
                        requeued = (
                            self.store.requeue_probe_artifact_unavailable(
                            task["task_id"],
                            task["attempt_id"],
                            task["lease_token"],
                            retry_receipt,
                            retry_delay_seconds=(
                                self.source_retry_delay_seconds
                            ),
                            maximum_requeues=(
                                self.probe_artifact_max_requeues
                            ),
                        )
                        )
                        if requeued:
                            return retry_receipt
                        review_reason = (
                            "WINNER_ARTIFACT_RETRY_BUDGET_EXHAUSTED"
                        )
                    receipt = {
                        "schema": (
                            "campaignx.seed_probe_terminal_receipt.v1"
                        ),
                        "status": "BLOCKED_PROBE_ARTIFACT_UNAVAILABLE",
                        "failure_class": "SOURCE_FAILURE",
                        "reason": review_reason,
                        "probe_run_id": error.probe_run_id,
                        "winner_trial_id": error.winner_trial_id,
                        "artifact_uri": error.artifact_uri,
                        "decision": error.decision,
                        "error": (
                            f"{type(cause).__name__}: {str(cause)[:2000]}"
                        ),
                        "generated_at_utc": utc_now(),
                        "ink_used": False,
                        "non_claim": (
                            "The selected probe evidence was not readable and "
                            "hash-verifiable, so it was retained for human "
                            "review and no canonical surface was created."
                        ),
                    }
                    write_json_atomic(
                        attempt_dir / "TERMINAL_RECEIPT.json", receipt
                    )
                    self.store.mark_probe_continuation_review(
                        task["task_id"],
                        task["attempt_id"],
                        task["lease_token"],
                        error.probe_run_id,
                        receipt,
                    )
                    return receipt
                write_json_atomic(
                    attempt_dir / "SEED_PROBE_RECEIPT.json", probe_result
                )
                if probe_result["status"] in {
                    "PROBE_REVIEW_PENDING",
                    "PROBE_REJECTED_ALL",
                }:
                    trial_outcomes = probe_result["decision"].get(
                        "trial_outcomes") or []
                    technical_failure = (
                        probe_result["status"] == "PROBE_REVIEW_PENDING"
                        and any(
                            isinstance(outcome, dict)
                            and outcome.get("state") == "FAILED"
                            for outcome in trial_outcomes
                        )
                    )
                    terminal_status = (
                        "PROBE_TECHNICAL_FAILURE"
                        if technical_failure else probe_result["status"]
                    )
                    receipt = {
                        "schema": "campaignx.seed_probe_terminal_receipt.v1",
                        "status": terminal_status,
                        "probe_run_id": probe_result["probe_run_id"],
                        "decision": probe_result["decision"],
                        "generated_at_utc": utc_now(),
                        "ink_used": False,
                        "non_claim": (
                            "This terminal applies to the bounded candidates "
                            "probed; it is not evidence that no physical sheet "
                            "exists in the cell."
                        ),
                    }
                    if technical_failure:
                        receipt["failure_class"] = "WORKER_FAILURE"
                    write_json_atomic(
                        attempt_dir / "TERMINAL_RECEIPT.json", receipt
                    )
                    self.store.mark_terminal(
                        task["task_id"],
                        task["attempt_id"],
                        task["lease_token"],
                        terminal_status,
                        receipt,
                    )
                    return receipt
                planner_task = probe_result["planner_task"]
                planner_candidates = probe_result["planner_candidates"]
                self.store.transition(
                    task["task_id"],
                    task["attempt_id"],
                    task["lease_token"],
                    "PLANNING",
                    result={
                        "probe_run_id": probe_result["probe_run_id"],
                        "probe_status": probe_result["status"],
                        "probe_decision_sha256": content_sha256(
                            probe_result["decision"]
                        ),
                    },
                )
            try:
                planner = self.planner_for(planner_task)
                task_contract = str(
                    planner_task.get("planner_contract_version", "v1")
                )
                planner_capability = str(
                    getattr(planner, "contract_version", "v1")
                )
                if task_contract == "v2" and planner_capability != "v2":
                    raise ValueError(
                        "task requires segmentation planner v2 but this worker only supports v1"
                    )
                regional_history = None
                if task_contract == "v2":
                    regional_history = self.store.regional_attempt_history(
                        planner_task, limit=12
                    )
                    write_json_atomic(
                        attempt_dir / "REGIONAL_ATTEMPT_HISTORY.json", regional_history
                    )
                packet = task_packet_for_planner(
                    planner_task,
                    planner_candidates,
                    gated_candidates,
                    contract_version=task_contract,
                    regional_attempt_history=regional_history,
                )
                write_json_atomic(attempt_dir / "PLANNER_PACKET.json", packet)
                repair_feedback: str | None = None
                proposal: dict[str, Any] | None = None
                locked_plan: dict[str, Any] | None = None
                operational_errors: list[dict[str, Any]] = []
                for planner_attempt in range(1, 4):
                    planner_call_dir = attempt_dir / f"planner-call-{planner_attempt:02d}"
                    planner_call_dir.mkdir()
                    try:
                        planner_parameters = inspect.signature(
                            planner.propose
                        ).parameters
                        if "repair_feedback" in planner_parameters:
                            candidate_proposal = planner.propose(
                                packet,
                                planner_call_dir,
                                repair_feedback=repair_feedback,
                            )
                        else:
                            # Compatibility path for frozen fixture planners
                            # in historical receipts and tests.
                            candidate_proposal = planner.propose(
                                packet,
                                planner_call_dir,
                            )
                        write_json_atomic(
                            planner_call_dir / "SEGMENTATION_PROPOSAL.json",
                            candidate_proposal,
                        )
                        candidate_locked_plan = validate_and_lock(packet, candidate_proposal)
                    except PlannerOutputInvalid as error:
                        repair_feedback = f"{type(error).__name__}: {error}"
                        operational_error = {
                            "planner_attempt": planner_attempt,
                            "error": repair_feedback,
                            "generated_at_utc": utc_now(),
                            "retryable_operational_error": True,
                            "ink_used": False,
                        }
                        operational_errors.append(operational_error)
                        write_json_atomic(
                            planner_call_dir / "OPERATIONAL_RETRY_RECEIPT.json",
                            operational_error,
                        )
                        continue
                    proposal = candidate_proposal
                    locked_plan = candidate_locked_plan
                    break
                if proposal is None or locked_plan is None:
                    fallback_dir = attempt_dir / "deterministic-fallback"
                    fallback_dir.mkdir()
                    fallback = DeterministicPlanner(contract_version=task_contract)
                    proposal = fallback.propose(
                        packet,
                        fallback_dir,
                        repair_feedback=repair_feedback,
                    )
                    write_json_atomic(
                        fallback_dir / "SEGMENTATION_PROPOSAL.json",
                        proposal,
                    )
                    locked_plan = validate_and_lock(packet, proposal)
                    write_json_atomic(
                        fallback_dir / "DETERMINISTIC_FALLBACK_RECEIPT.json",
                        {
                            "schema": "campaignx.segmentation_planner_fallback_receipt.v1",
                            "status": "DETERMINISTIC_FALLBACK_LOCKED",
                            "operational_attempt_count": len(operational_errors),
                            "operational_errors": operational_errors,
                            "proposal_sha256": content_sha256(proposal),
                            "generated_at_utc": utc_now(),
                            "ink_used": False,
                        },
                    )
                write_json_atomic(attempt_dir / "SEGMENTATION_PROPOSAL.json", proposal)
                if (
                    probe_result is not None
                    and probe_result.get("status") == "PROBE_WINNER"
                ):
                    promotion = self.store.begin_probe_promotion(
                        task["task_id"],
                        task["attempt_id"],
                        task["lease_token"],
                        probe_result["probe_run_id"],
                        locked_plan,
                    )
                    write_json_atomic(
                        attempt_dir / "SEED_PROBE_PROMOTION.json",
                        promotion,
                    )
            except PlannerProviderUnavailable as error:
                receipt = {
                    "status": "RETRYABLE_PROVIDER_UNAVAILABLE",
                    "error": f"{type(error).__name__}: {error}",
                    "retry_delay_seconds": self.provider_retry_delay_seconds,
                    "generated_at_utc": utc_now(),
                    "ink_used": False,
                    "non_claim": "No model proposal was received; this does not assess geometry.",
                }
                write_json_atomic(attempt_dir / "RETRYABLE_PROVIDER_RECEIPT.json", receipt)
                self.store.requeue_provider_unavailable(
                    task["task_id"],
                    task["attempt_id"],
                    task["lease_token"],
                    receipt,
                    retry_delay_seconds=self.provider_retry_delay_seconds,
                )
                return receipt
            except PlannerScientificViolation as error:
                receipt = {"status": "POLICY_REJECTED", "failure_class": "CONFIGURATION_BLOCK", "error": f"{type(error).__name__}: {error}", "generated_at_utc": utc_now(), "ink_used": False}
                write_json_atomic(attempt_dir / "TERMINAL_RECEIPT.json", receipt)
                self.store.mark_terminal(task["task_id"], task["attempt_id"], task["lease_token"], "POLICY_REJECTED", receipt)
                return receipt
            except BaseException as error:
                receipt = {
                    "status": "POLICY_REJECTED",
                    "failure_class": "CONFIGURATION_BLOCK",
                    "error": f"{type(error).__name__}: {error}",
                    "generated_at_utc": utc_now(),
                    "ink_used": False,
                    "non_claim": "An internal planner/contract error prevented execution; this does not assess geometry.",
                }
                write_json_atomic(attempt_dir / "TERMINAL_RECEIPT.json", receipt)
                self.store.mark_terminal(task["task_id"], task["attempt_id"], task["lease_token"], "POLICY_REJECTED", receipt)
                return receipt
            write_json_atomic(attempt_dir / "SEGMENTATION_PLAN.json", locked_plan)
            self.store.transition(task["task_id"], task["attempt_id"], task["lease_token"], "LOCKED_READY", proposal=proposal, locked_plan=locked_plan)
            heartbeat.ensure()
            self.store.transition(task["task_id"], task["attempt_id"], task["lease_token"], "RUNNING")
            try:
                grown = self.grow_executor.execute(locked_plan, attempt_dir)
            except InsufficientGpuMemoryError as error:
                observed_vram_gb = float(self.worker_capabilities["gpu_vram_gb"])
                current_floor = float(task["resource_requirements"]["minimum_vram_gb"])
                next_floor = max(current_floor, observed_vram_gb + 1.0 if observed_vram_gb > 0 else 12.0)
                receipt = {
                    **error.receipt,
                    "status": "RETRY_ON_LARGER_GPU",
                    "error": f"{type(error).__name__}: {error}",
                    "worker_capabilities": self.worker_capabilities,
                    "prior_minimum_vram_gb": current_floor,
                    "minimum_vram_gb": next_floor,
                    "generated_at_utc": utc_now(),
                    "ink_used": False,
                    "non_claim": "GPU capacity is operational metadata, not an assessment of geometry.",
                }
                write_json_atomic(attempt_dir / "RETRY_ON_LARGER_GPU_RECEIPT.json", receipt)
                self.store.requeue_for_larger_gpu(
                    task["task_id"],
                    task["attempt_id"],
                    task["lease_token"],
                    receipt,
                    minimum_vram_gb=next_floor,
                )
                return receipt
            except BaseException as error:
                receipt = {"status": "GROW_FAILED", "failure_class": "WORKER_FAILURE", "error": f"{type(error).__name__}: {error}", "generated_at_utc": utc_now(), "ink_used": False}
                write_json_atomic(attempt_dir / "TERMINAL_RECEIPT.json", receipt)
                self.store.mark_terminal(task["task_id"], task["attempt_id"], task["lease_token"], "GROW_FAILED", receipt)
                return receipt
            heartbeat.ensure()
            try:
                return finalize_surface(
                    self.store,
                    task,
                    locked_plan,
                    Path(grown["surface_dir"]),
                    self.artifact_root,
                    attempt_dir,
                    self.qc_profile_id,
                )
            except BaseException as error:
                if _is_transient_operational_error(error):
                    receipt = {
                        "status": "RETRYABLE_FINALIZATION_UNAVAILABLE",
                        "phase": "FINALIZATION",
                        "error": (
                            f"{type(error).__name__}: {str(error)[:2000]}"
                        ),
                        "retry_delay_seconds": (
                            self.finalization_retry_delay_seconds
                        ),
                        "maximum_requeues": (
                            self.finalization_max_requeues
                        ),
                        "generated_at_utc": utc_now(),
                        "ink_used": False,
                        "non_claim": (
                            "A transient storage or database outage interrupted "
                            "finalization. The grow and any selected probe "
                            "remain noncanonical and the task is eligible for "
                            "a replacement attempt."
                        ),
                    }
                    write_json_atomic(
                        attempt_dir
                        / "RETRYABLE_FINALIZATION_RECEIPT.json",
                        receipt,
                    )
                    requeued = self.store.requeue_finalization_unavailable(
                        task["task_id"],
                        task["attempt_id"],
                        task["lease_token"],
                        receipt,
                        retry_delay_seconds=(
                            self.finalization_retry_delay_seconds
                        ),
                        maximum_requeues=(
                            self.finalization_max_requeues
                        ),
                    )
                    if requeued:
                        return receipt
                    exhausted = {
                        **receipt,
                        "status": "FINALIZATION_FAILED",
                        "reason": (
                            "TRANSIENT_FINALIZATION_RETRY_BUDGET_EXHAUSTED"
                        ),
                        "retry_budget_exhausted": True,
                        "failure_class": "PUBLICATION_FAILURE",
                        "non_claim": (
                            "The bounded infrastructure retry budget was "
                            "exhausted. Probe evidence and any staged bytes are "
                            "retained for operator review; no scientific "
                            "conclusion was made."
                        ),
                    }
                    write_json_atomic(
                        attempt_dir / "TERMINAL_RECEIPT.json", exhausted
                    )
                    self.store.mark_terminal(
                        task["task_id"],
                        task["attempt_id"],
                        task["lease_token"],
                        "FINALIZATION_FAILED",
                        exhausted,
                    )
                    return exhausted
                receipt = {"status": "FINALIZATION_FAILED", "failure_class": "PUBLICATION_FAILURE", "error": f"{type(error).__name__}: {error}", "generated_at_utc": utc_now(), "ink_used": False}
                write_json_atomic(attempt_dir / "TERMINAL_RECEIPT.json", receipt)
                self.store.mark_terminal(task["task_id"], task["attempt_id"], task["lease_token"], "FINALIZATION_FAILED", receipt)
                return receipt

    def run(self, max_jobs: int | None = None, idle_exit: bool = True, poll_seconds: float = 10.0) -> list[dict[str, Any]]:
        if self.task_id is not None and max_jobs not in (None, 1):
            raise ValueError("a task-specific worker may execute exactly one task")
        completed: list[dict[str, Any]] = []
        while max_jobs is None or len(completed) < max_jobs:
            result = self.run_one()
            if result is None:
                if idle_exit:
                    break
                time.sleep(poll_seconds)
                continue
            completed.append(result)
        return completed
