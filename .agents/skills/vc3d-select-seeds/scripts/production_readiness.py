#!/usr/bin/env python3
"""Fail-closed readiness checks for the Helena VC3D seed skill."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


TRUE_VALUES = {"1", "true", "yes", "on"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_RUNTIME_FILES = (
    "framework/stages/01-segmentation/stage.json",
    "framework/stages/01-segmentation/SEED_PROBE_RUNBOOK.md",
    "framework/stages/01-segmentation/fleet/seed_probe.py",
    "framework/stages/01-segmentation/fleet/worker.py",
    "framework/stages/01-segmentation/fleet/store.py",
    "framework/stages/01-segmentation/fleet/postgres_store.py",
    "framework/stages/01-segmentation/fleet/migrations/001_postgresql.sql",
    "framework/stages/01-segmentation/fleet/profiles/vc3d-m7-probe-v1.json",
    "framework/stages/01-segmentation/fleet/profiles/tifxyz-geometry-gate-probe-v1.json",
    "framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py",
    "framework/contracts/schemas/m7-seed-candidate-evidence-v1.schema.json",
    "framework/contracts/schemas/seed-probe-artifact-set-v1.schema.json",
    "framework/contracts/schemas/seed-probe-benchmark-execution-v1.schema.json",
    "framework/contracts/schemas/seed-probe-decision-v1.schema.json",
    "framework/contracts/schemas/seed-probe-evaluation-v1.schema.json",
    "framework/contracts/schemas/seed-probe-locked-plan-v1.schema.json",
    "framework/contracts/schemas/seed-probe-policy-v1.schema.json",
    "framework/contracts/schemas/seed-probe-promotion-v1.schema.json",
    "framework/contracts/schemas/source-content-lock-v1.schema.json",
    ".agents/skills/vc3d-select-seeds/SKILL.md",
    ".agents/skills/vc3d-select-seeds/references/capability-matrix.md",
    ".agents/skills/vc3d-select-seeds/references/operations.md",
    ".agents/skills/vc3d-select-seeds/references/benchmark-contract.md",
    ".agents/skills/vc3d-select-seeds/scripts/production_readiness.py",
    ".agents/skills/vc3d-select-seeds/scripts/compare_strategies.py",
)
PRODUCTION_TRACKED_PATHS = (
    "framework/stages/01-segmentation",
    "framework/stages/04-validation/scripts/helena_tifxyz_geometry_gate.py",
    "framework/stages/04-validation/scripts/helena_audit_mesh_integrity.py",
    "framework/contracts/schemas",
    ".agents/skills/vc3d-select-seeds",
)


class ReadinessError(RuntimeError):
    """The readiness input itself is malformed."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReadinessError(f"cannot read JSON {path}: {error}") from error


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def discover_root(start: Path) -> Path:
    candidate = start.resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (
            directory
            .joinpath("framework/stages/01-segmentation/stage.json")
            .is_file()
            and directory
            .joinpath(
                "framework/stages/01-segmentation/scripts/"
                "helena_segment_search_fleet.py"
            )
            .is_file()
        ):
            return directory
    raise ReadinessError(
        f"no Helena root found from {start}; stage.json and fleet CLI are required"
    )


def resolve_project_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else root.joinpath(path).resolve()


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in TRUE_VALUES


def utc_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.utcoffset() == timezone.utc.utcoffset(parsed) else None


def valid_source_lock(entry: dict[str, Any]) -> bool:
    lock = entry.get("source_content_lock")
    expected = {
        "schema",
        "status",
        "verification_method",
        "verified_at_utc",
        "ct_uri",
        "ct_sha256",
        "ct_version_id",
        "m7_uri",
        "m7_sha256",
        "m7_version_id",
    }
    if not isinstance(lock, dict) or set(lock) != expected:
        return False
    ct_uri = str(entry.get("ct_uri") or "")
    m7_uri = str(entry.get("surface_prediction_uri") or "")
    ct_sha = str(entry.get("ct_sha256") or "")
    m7_sha = str(entry.get("surface_prediction_sha256") or "")
    return bool(
        lock["schema"] == "campaignx.source_content_lock.v1"
        and lock["status"] == "VERIFIED_IMMUTABLE"
        and lock["verification_method"] == "immutable-uri-manifest-sha256-v1"
        and utc_timestamp(lock["verified_at_utc"]) is not None
        and SHA256.fullmatch(ct_sha)
        and SHA256.fullmatch(m7_sha)
        and lock["ct_uri"] == ct_uri
        and lock["m7_uri"] == m7_uri
        and lock["ct_sha256"] == ct_sha
        and lock["m7_sha256"] == m7_sha
        and isinstance(lock["ct_version_id"], str)
        and len(lock["ct_version_id"]) >= 8
        and lock["ct_version_id"] in ct_uri
        and isinstance(lock["m7_version_id"], str)
        and len(lock["m7_version_id"]) >= 8
        and lock["m7_version_id"] in m7_uri
    )


def add_check(
    checks: list[dict[str, Any]],
    check_id: str,
    status: str,
    detail: str,
    evidence: Any = None,
) -> None:
    row = {"check_id": check_id, "status": status, "detail": detail}
    if evidence is not None:
        row["evidence"] = evidence
    checks.append(row)


def check_required_files(root: Path, checks: list[dict[str, Any]]) -> None:
    missing = [relative for relative in REQUIRED_RUNTIME_FILES if not root.joinpath(relative).is_file()]
    add_check(
        checks,
        "CODE_REQUIRED_FILES",
        "PASS" if not missing else "FAIL",
        "all required runtime, contract, and skill files exist"
        if not missing
        else "required runtime files are missing",
        {"missing": missing, "required_count": len(REQUIRED_RUNTIME_FILES)},
    )


def check_stage_contract(root: Path, checks: list[dict[str, Any]]) -> None:
    try:
        stage = load_json(root / "framework/stages/01-segmentation/stage.json")
        probe = stage["seed_probe"]
        valid = bool(
            probe["active_profile"] == "seed-probe-v1"
            and probe["production_mode"] == "SHADOW_ONLY"
            and probe["relationship_to_router"]
            == "PRE_PLANNER_DETERMINISTIC_EVIDENCE_BENEATH_COST_AWARE_V2"
            and probe["select_rollout_gate"]
            == (
                "SAMPLE_SCOPED_ISOLATED_PAIRED_SELECT_APPROVAL_RECEIPT_PLUS_"
                "VERIFIED_IMMUTABLE_SOURCE_LOCK"
            )
            and probe["replaces_cost_aware"] is False
            and probe["replaces_fusion"] is False
        )
        evidence = {
            key: probe.get(key)
            for key in (
                "active_profile",
                "production_mode",
                "relationship_to_router",
                "select_rollout_gate",
                "replaces_cost_aware",
                "replaces_fusion",
            )
        }
    except (KeyError, TypeError, ReadinessError) as error:
        valid = False
        evidence = {"error": str(error)}
    add_check(
        checks,
        "CODE_STAGE_BOUNDARY",
        "PASS" if valid else "FAIL",
        "stage declares the audited additive shadow-only boundary"
        if valid
        else "stage seed-probe boundary is missing or contradictory",
        evidence,
    )


def check_profile_integrity(root: Path, checks: list[dict[str, Any]]) -> None:
    profile_path = root / (
        "framework/stages/01-segmentation/fleet/profiles/"
        "tifxyz-geometry-gate-probe-v1.json"
    )
    evidence: list[dict[str, Any]] = []
    valid = True
    try:
        profile = load_json(profile_path)
        for key in ("gate", "integrity_dependency"):
            dependency = profile[key]
            path = root / str(dependency["path"])
            expected = str(dependency["sha256"])
            actual = file_sha256(path) if path.is_file() else None
            matches = actual == expected
            evidence.append(
                {
                    "kind": key,
                    "path": str(dependency["path"]),
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "matches": matches,
                }
            )
            valid = valid and matches
    except (KeyError, TypeError, ReadinessError, OSError) as error:
        valid = False
        evidence.append({"error": str(error)})
    add_check(
        checks,
        "CODE_PROFILE_INTEGRITY",
        "PASS" if valid else "FAIL",
        "probe geometry profile matches its pinned gate code"
        if valid
        else "probe geometry profile dependency hash mismatch",
        evidence,
    )


def catalogue_entries(path: Path) -> list[dict[str, Any]]:
    document = load_json(path)
    raw = document.get("entries") if isinstance(document, dict) else document
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise ReadinessError(f"{path} does not contain an entry list")
    return raw


def check_catalogue(
    root: Path,
    eligible: str,
    sample: str | None,
    mode: str,
    checks: list[dict[str, Any]],
) -> None:
    path = resolve_project_path(root, eligible)
    try:
        # Three states, not two. `row.get("target_allowed", True)` reads a
        # present `null` as False, so a catalogue that records "nobody
        # established the rights for this scroll" -- which is what the bucket
        # supports, since it publishes one licence at its root and none per
        # scroll -- dropped every row and this check reported "no eligible
        # source matches the requested scope". That is a different sentence from
        # the truth, which is that the sources are there and their rights are
        # unstated.
        #
        # Only an explicit False excludes. Unstated is carried and counted, so
        # the check says what it actually knows.
        all_rows = list(catalogue_entries(path))
        entries = [row for row in all_rows if row.get("target_allowed", True) is not False]
        unstated = [row for row in entries if row.get("target_allowed") is None]
        scoped = [
            row
            for row in entries
            if sample is None or str(row.get("sample_id")) == sample
        ]
        locked = [row for row in scoped if valid_source_lock(row)]
        readable = bool(scoped)
        add_check(
            checks,
            "SOURCE_CATALOGUE",
            "PASS" if readable else "FAIL",
            "eligible source catalogue is readable"
            if readable
            else "no eligible source matches the requested scope",
            {
                "path": str(path),
                "eligible_entries": len(entries),
                "scoped_entries": len(scoped),
                "rights_unstated": len(unstated),
                "sample": sample,
            },
        )
        if mode == "select":
            all_locked = bool(scoped) and len(locked) == len(scoped)
            add_check(
                checks,
                "SELECT_IMMUTABLE_SOURCES",
                "PASS" if all_locked else "FAIL",
                "every selected source has an exact verified immutable CT/m7 lock"
                if all_locked
                else "select scope contains unlocked or mutable CT/m7 sources",
                {
                    "scoped_entries": len(scoped),
                    "locked_entries": len(locked),
                    "unlocked_samples": sorted(
                        {
                            str(row.get("sample_id") or "UNKNOWN")
                            for row in scoped
                            if not valid_source_lock(row)
                        }
                    ),
                },
            )
    except ReadinessError as error:
        add_check(
            checks,
            "SOURCE_CATALOGUE",
            "FAIL",
            "eligible source catalogue is unreadable",
            {"path": str(path), "error": str(error)},
        )
        if mode == "select":
            add_check(
                checks,
                "SELECT_IMMUTABLE_SOURCES",
                "FAIL",
                "immutable source locks cannot be verified",
            )


def validate_benchmark_receipt(
    root: Path, path: Path
) -> tuple[bool, dict[str, Any], dict[str, Any] | None]:
    try:
        stage_root = root / "framework/stages/01-segmentation"
        if str(stage_root) not in sys.path:
            sys.path.insert(0, str(stage_root))
        from fleet.seed_probe import (  # noqa: PLC0415
            load_seed_probe_benchmark_receipt,
        )

        authorization = load_seed_probe_benchmark_receipt(path)
        return True, {
            "status": "APPROVED_SELECT",
            "benchmark_id": authorization["benchmark_id"],
            "paired_cell_count": authorization["paired_cell_count"],
            "scroll_count": authorization["scroll_count"],
            "receipt_sha256": authorization["decision_receipt_sha256"],
            "authorized_sample_count": len(
                authorization["authorized_sample_ids"]
            ),
            "hash_valid": True,
        }, authorization
    except (ImportError, ReadinessError, TypeError, ValueError) as error:
        return False, {
            "error": (
                "benchmark receipt failed the runtime validator "
                f"({type(error).__name__})"
            )
        }, None


def parse_status_stdout(stdout: str) -> dict[str, Any]:
    candidate = stdout.strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ReadinessError(f"fleet status returned non-JSON output: {error}") from error
    if not isinstance(value, dict):
        raise ReadinessError("fleet status did not return an object")
    return value


def fresh_probe_worker_count(db: str) -> int:
    """Count capable PostgreSQL workers seen in the last 20 minutes.

    Fleet ``status`` intentionally exposes capabilities but not heartbeat
    timestamps.  Production readiness must query the authoritative timestamp
    instead of treating every historical registration as live.
    """

    if db.startswith("postgres-env://"):
        variable = db.removeprefix("postgres-env://")
        if not variable or not re.fullmatch(r"[A-Za-z0-9_]+", variable):
            raise ReadinessError(
                "postgres-env database spec must name one environment variable"
            )
        database_url = os.environ.get(variable)
        if not database_url:
            raise ReadinessError(
                f"database environment variable is not set: {variable}"
            )
    else:
        database_url = db
    if not str(database_url).startswith(("postgresql://", "postgres://")):
        raise ReadinessError("resolved production database is not PostgreSQL")

    try:
        import psycopg2
    except ImportError as error:
        raise ReadinessError(
            "PostgreSQL worker freshness requires psycopg2"
        ) from error

    try:
        with psycopg2.connect(str(database_url), connect_timeout=5) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT count(*)::bigint
                         FROM segment_worker_capabilities
                        WHERE lower(coalesce(
                                capabilities->>'seed_probe_v1','false'))
                              IN ('true','1','yes','on')
                          AND updated_at >= now() - interval '20 minutes'"""
                )
                row = cursor.fetchone()
    except Exception as error:  # noqa: BLE001 - readiness must fail closed
        raise ReadinessError(
            "cannot verify the PostgreSQL worker heartbeat "
            f"({type(error).__name__})"
        ) from error
    return int(row[0]) if row else 0


def check_database(
    root: Path,
    db: str | None,
    checks: list[dict[str, Any]],
) -> None:
    if not db:
        add_check(
            checks,
            "LIVE_POSTGRESQL",
            "FAIL",
            "production readiness requires --db",
        )
        add_check(
            checks,
            "LIVE_PROBE_WORKER",
            "FAIL",
            "probe-capable worker freshness cannot be checked without PostgreSQL",
        )
        return
    postgres_alias = db.startswith("postgres-env://")
    if not postgres_alias:
        add_check(
            checks,
            "LIVE_POSTGRESQL",
            "FAIL",
            "production readiness requires postgres-env://ENV_NAME so credentials "
            "never enter commands or receipts",
            {
                "database_kind": (
                    "postgresql"
                    if db.startswith(("postgresql://", "postgres://"))
                    else db.split(":", 1)[0]
                )
            },
        )
        add_check(
            checks,
            "LIVE_PROBE_WORKER",
            "FAIL",
            "worker freshness requires an environment-backed PostgreSQL control plane",
        )
        return
    cli = root / (
        "framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py"
    )
    completed = subprocess.run(
        [sys.executable, str(cli), "status", "--db", db],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    try:
        status = parse_status_stdout(completed.stdout) if completed.returncode == 0 else {}
        valid = bool(
            completed.returncode == 0
            and status.get("schema") == "campaignx.segment_fleet_status.v1"
        )
    except ReadinessError:
        status = {}
        valid = False
    add_check(
        checks,
        "LIVE_POSTGRESQL",
        "PASS" if valid else "FAIL",
        "PostgreSQL control plane answered with the fleet status contract"
        if valid
        else "PostgreSQL control plane status check failed",
        {
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip()[:500] or None,
            "status_schema": status.get("schema"),
        },
    )
    try:
        capable_count = fresh_probe_worker_count(db) if valid else 0
        freshness_error = None
    except ReadinessError as error:
        capable_count = 0
        freshness_error = str(error)
    add_check(
        checks,
        "LIVE_PROBE_WORKER",
        "PASS" if capable_count else "FAIL",
        "at least one seed-probe-v1 worker checked in during the last 20 minutes"
        if capable_count
        else "no recent seed-probe-v1 worker heartbeat could be verified",
        {
            "freshness_window_minutes": 20,
            "capable_worker_count": capable_count,
            "error": freshness_error,
        },
    )


def check_artifact_store(
    artifact_root: str | None, checks: list[dict[str, Any]]
) -> None:
    parsed = urlparse(artifact_root or "")
    valid = bool(parsed.scheme == "s3" and parsed.netloc and parsed.path.strip("/"))
    add_check(
        checks,
        "LIVE_S3_NAMESPACE",
        "PASS" if valid else "FAIL",
        "production artifact root is a scoped S3 prefix"
        if valid
        else "production readiness requires --artifact-root s3://BUCKET/PREFIX",
        (
            {
                "scheme": parsed.scheme or None,
                "bucket": parsed.netloc or None,
                "prefix_present": bool(parsed.path.strip("/")),
            }
            if artifact_root
            else None
        ),
    )


def check_release_state(root: Path, checks: list[dict[str, Any]]) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", *PRODUCTION_TRACKED_PATHS],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    dirty = [line for line in status.stdout.splitlines() if line.strip()]
    valid = bool(head.returncode == 0 and status.returncode == 0 and not dirty)
    add_check(
        checks,
        "RELEASE_REPRODUCIBLE",
        "PASS" if valid else "FAIL",
        "runtime and skill paths are committed and clean"
        if valid
        else "production runtime or skill paths contain uncommitted changes",
        {
            "commit_sha": head.stdout.strip() if head.returncode == 0 else None,
            "dirty_path_count": len(dirty),
            "dirty_paths": dirty[:30],
        },
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Fail-closed VC3D seed-probe code/deployment readiness."
    )
    result.add_argument("--root", default=".")
    result.add_argument("--mode", choices=("shadow", "select"), required=True)
    result.add_argument(
        "--eligible", default="workspace/catalog/eligible_volumes.json"
    )
    result.add_argument("--sample")
    result.add_argument("--production", action="store_true")
    result.add_argument("--db")
    result.add_argument("--artifact-root")
    result.add_argument("--benchmark-receipt", type=Path)
    result.add_argument("--review-owner")
    result.add_argument("--output", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        root = discover_root(Path(arguments.root))
    except ReadinessError as error:
        print(json.dumps({"status": "NOT_READY", "error": str(error)}, indent=2))
        return 2

    checks: list[dict[str, Any]] = []
    check_required_files(root, checks)
    check_stage_contract(root, checks)
    check_profile_integrity(root, checks)
    check_catalogue(
        root, arguments.eligible, arguments.sample, arguments.mode, checks
    )

    disabled = truthy(os.environ.get("HELENA_DISABLE_SEED_PROBE_V1"))
    add_check(
        checks,
        "ROLLOUT_NOT_DISABLED",
        "FAIL" if disabled else "PASS",
        "seed-probe-v1 is not disabled"
        if not disabled
        else "HELENA_DISABLE_SEED_PROBE_V1 is truthy",
    )

    if arguments.mode == "select":
        rollout = truthy(os.environ.get("HELENA_ENABLE_SEED_PROBE_SELECT"))
        add_check(
            checks,
            "SELECT_ROLLOUT_FLAG",
            "PASS" if rollout else "FAIL",
            "select rollout flag is explicitly enabled"
            if rollout
            else "HELENA_ENABLE_SEED_PROBE_SELECT is not enabled",
        )
        if arguments.benchmark_receipt:
            valid, evidence, authorization = validate_benchmark_receipt(
                root,
                arguments.benchmark_receipt.resolve()
            )
        else:
            valid, evidence, authorization = False, None, None
        add_check(
            checks,
            "SELECT_BENCHMARK_APPROVAL",
            "PASS" if valid else "FAIL",
            "an immutable matched benchmark approves select"
            if valid
            else "select requires an approved benchmark decision receipt",
            evidence,
        )
        sample_authorized = bool(
            valid
            and arguments.sample
            and authorization is not None
            and arguments.sample in authorization["authorized_sample_ids"]
        )
        add_check(
            checks,
            "SELECT_BENCHMARK_SCOPE",
            "PASS" if sample_authorized else "FAIL",
            "the selected sample is explicitly authorized by the benchmark"
            if sample_authorized
            else (
                "select requires --sample within the approved benchmark scope"
                if not arguments.sample
                else "the selected sample is outside the approved benchmark scope"
            ),
            {
                "sample": arguments.sample,
                "authorized_sample_count": (
                    len(authorization["authorized_sample_ids"])
                    if authorization is not None
                    else 0
                ),
            },
        )
        owner = str(arguments.review_owner or "").strip()
        add_check(
            checks,
            "SELECT_REVIEW_OWNER",
            "PASS" if owner else "FAIL",
            "a review-resolution owner is declared"
            if owner
            else "select requires --review-owner",
            {"owner": owner} if owner else None,
        )

    if arguments.production:
        check_database(root, arguments.db, checks)
        check_artifact_store(arguments.artifact_root, checks)
        check_release_state(root, checks)
    else:
        for check_id, detail in (
            (
                "LIVE_POSTGRESQL",
                "not checked without --production",
            ),
            (
                "LIVE_PROBE_WORKER",
                "not checked without --production",
            ),
            (
                "LIVE_S3_NAMESPACE",
                "not checked without --production",
            ),
            (
                "RELEASE_REPRODUCIBLE",
                "not checked without --production",
            ),
        ):
            add_check(checks, check_id, "SKIP", detail)

    failed = [row for row in checks if row["status"] == "FAIL"]
    receipt: dict[str, Any] = {
        "schema": "campaignx.vc3d_seed_skill_readiness.v1",
        "status": "READY" if not failed else "NOT_READY",
        "scope": "PRODUCTION" if arguments.production else "CODE",
        "mode": arguments.mode,
        "campaign_root": str(root),
        "sample": arguments.sample,
        "checks": checks,
        "failed_check_ids": [row["check_id"] for row in failed],
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "non_claims": [
            "READY does not prove a seed follows the correct physical lamina",
            "code readiness does not establish live infrastructure readiness",
            "shadow readiness does not authorize select",
        ],
    }
    receipt["receipt_sha256"] = content_sha256(receipt)
    if arguments.output:
        write_json_atomic(arguments.output.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
