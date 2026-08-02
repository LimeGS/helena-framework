#!/usr/bin/env python3
"""Build the result-blind source-lock request from verified support and CT."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
sys.path.insert(0, str(ROOT / "phase2/src"))

from campaign_x_phase2.local_holdout_v1_authorization import (  # noqa: E402
    REPOSITORY_ROOT,
    LocalHoldoutV1Error,
    require_local_holdout_v1_authorized,
)
from campaign_x_phase2.local_holdout_v1_ct_acquisition import (  # noqa: E402
    FINAL_STAGE,
    validated_plan_summary,
)
from campaign_x_phase2.local_holdout_v1_freeze import (  # noqa: E402
    _require_timestamp,
)
from campaign_x_phase2.local_holdout_v1_io import (  # noqa: E402
    exclusive_json_create,
    read_strict_json,
)
from campaign_x_phase2.local_holdout_v1_source_lock import (  # noqa: E402
    LocalSourceLockError,
    _source_lock_inputs_from_support,
)


OUTPUT_FILENAME = "LOCAL_HOLDOUT_V1_SOURCE_LOCK_REQUEST.json"
PROJECTED_STAGING_AND_ARTIFACT_BYTES = 32 * 1024**3
PROJECTED_WALL_CLOCK_HOURS = 12.0
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")


def _object(path: Path, *, maximum_bytes: int) -> dict:
    value = read_strict_json(path, maximum_bytes=maximum_bytes)
    if not isinstance(value, dict):
        raise LocalSourceLockError(f"{path} must contain one JSON object")
    return value


def _private_output(path: Path) -> Path:
    candidate = Path(os.path.abspath(os.fspath(path)))
    if candidate.name != OUTPUT_FILENAME or candidate.exists() or candidate.is_symlink():
        raise LocalSourceLockError(
            f"source-lock request must be a new {OUTPUT_FILENAME}"
        )
    try:
        parent_info = candidate.parent.lstat()
        parent = candidate.parent.resolve(strict=True)
        repository = REPOSITORY_ROOT.resolve(strict=True)
    except OSError as exc:
        raise LocalSourceLockError(
            "source-lock request parent is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(parent_info.st_mode)
        or not stat.S_ISDIR(parent_info.st_mode)
        or stat.S_IMODE(parent_info.st_mode) != 0o700
        or parent != candidate.parent
        or parent == repository
        or repository in parent.parents
    ):
        raise LocalSourceLockError(
            "source-lock request requires a real mode-0700 parent outside the repository"
        )
    return candidate


def _capacity_root(path: Path) -> Path:
    candidate = Path(os.path.abspath(os.fspath(path)))
    try:
        info = candidate.lstat()
        resolved = candidate.resolve(strict=True)
        repository = REPOSITORY_ROOT.resolve(strict=True)
    except OSError as exc:
        raise LocalSourceLockError("capacity root is unavailable") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or resolved == repository
        or repository in resolved.parents
    ):
        raise LocalSourceLockError(
            "capacity root must be a real directory outside the repository"
        )
    return resolved


def _head_commit() -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise LocalSourceLockError(
            "cannot bind the source request to the authorization commit"
        ) from exc
    if _COMMIT_RE.fullmatch(value) is None:
        raise LocalSourceLockError("authorization commit is not canonical")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-support", type=Path, required=True)
    parser.add_argument("--ct-download-plan", type=Path, required=True)
    parser.add_argument("--capacity-root", type=Path, required=True)
    parser.add_argument("--generated-at-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        # Authorization is deliberately first after parsing: no input path,
        # disk capacity or Git state is inspected while activation is off.
        require_local_holdout_v1_authorized()
        support = _object(args.source_support, maximum_bytes=512 * 1024**2)
        plan = _object(args.ct_download_plan, maximum_bytes=256 * 1024**2)
        summary = validated_plan_summary(plan)
        if summary["stage"] != FINAL_STAGE:
            raise LocalSourceLockError(
                "source lock requires the final exact CT chunk plan"
            )
        bindings, regions, roundtrip, _inventory_sha = (
            _source_lock_inputs_from_support(support)
        )
        capacity_root = _capacity_root(args.capacity_root)
        output = _private_output(args.output)
        generated = _require_timestamp(args.generated_at_utc)
        request = {
            "generated_at_utc": generated,
            "authorization_commit": _head_commit(),
            "support_bindings": bindings,
            "canonical_region_ids": list(regions),
            "coordinate_roundtrip_max_error_voxels": roundtrip,
            "free_bytes_at_lock": shutil.disk_usage(capacity_root).free,
            "projected_selective_ct_bytes": summary["total_bytes"],
            "projected_staging_and_artifact_bytes": (
                PROJECTED_STAGING_AND_ARTIFACT_BYTES
            ),
            "projected_wall_clock_hours": PROJECTED_WALL_CLOCK_HOURS,
        }
        exclusive_json_create(output, request, mode=0o600)
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        LocalHoldoutV1Error,
        LocalSourceLockError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        f"wrote result-blind source-lock request for {len(regions)} regions "
        f"and {summary['required_chunk_count']} CT chunks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
