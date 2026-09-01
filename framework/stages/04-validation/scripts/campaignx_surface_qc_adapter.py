#!/usr/bin/env python3
"""Scientific adapter for one Segment Search Fleet surface-QC lease.

The adapter renders one preserved TIFXYZ surface and runs the frozen
six-replica TimeSformer screen. Unusable liveness short-circuits before later
scientific stages; otherwise the adapter adds the existing high-recall router
and CT/fiber gate. It publishes a compact immutable evidence bundle, has
exactly four closed routing outcomes, and never accepts ink, text, letters, or
a submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def discover_repo_root() -> Path:
    configured = os.environ.get("HELENA_REPO_ROOT", "").strip()
    candidates = (
        [Path(configured).expanduser().resolve()] if configured else []
    ) + list(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (
            (candidate / "framework" / "profiles").is_dir()
            and (candidate / "framework" / "stages").is_dir()
        ):
            return candidate
    raise RuntimeError(
        "cannot discover Helena Framework repository root; set HELENA_REPO_ROOT"
    )


ROOT = discover_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.contracts.slice_order import ordered_tiff_files
from scripts.harness.stage_script_registry import resolve_stage_script


REQUIRED_SURFACE_FILES = ("x.tif", "y.tif", "z.tif", "meta.json")
REQUIRED_TIFF_COUNT = 65
NO_COMMON_VALID_ERROR = "RuntimeError: screening maps have no common valid pixels"
OUTCOME_INSUFFICIENT = "CT_INSUFFICIENT_NO_COMMON_VALID_PIXELS"
OUTCOME_INK_SCREEN_INSUFFICIENT = (
    "INK_SCREEN_INSUFFICIENT_DEGENERATE_OR_EMPTY"
)
OUTCOME_NO_RETAINED = "CT_SUPPORTED_NO_RETAINED_INK_SIGNAL"
OUTCOME_RETAINED = "CT_SUPPORTED_RETAINED_FOR_REVIEW"
RENDER_MAX_ATTEMPTS = 3
RENDER_RETRY_BACKOFF_SECONDS = 10
TRANSIENT_RENDER_ABORT = "terminate called without an active exception"
MAX_EVIDENCE_PUBLISH_WORKERS = 8


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# sysexits.h EX_CONFIG. The worker sees an exit code and nothing else, so this
# is how "somebody has to change a setting" crosses the process boundary and
# stops being indistinguishable from "S3 was down for a minute".
EX_CONFIG = 78


class ConfigurationError(RuntimeError):
    """A failure that will still be a failure the next time it is tried.

    A frozen hash that does not match, a file that is not there, a profile that
    names a schema nobody implements: retrying these is not patience, it is a
    loop. Two GPUs once spent two days on one -- 3118 attempts, no surfaces
    measured -- because the worker could not tell it from a transient outage
    and requeued it every time.

    A RuntimeError subclass so that everything already catching RuntimeError
    still does.
    """


def require_env_path(name: str) -> Path:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise ConfigurationError(f"required environment variable is missing: {name}")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        # A path that was set and points at nothing. The renderer and the
        # checkpoint arrive this way, and neither appears by waiting.
        raise ConfigurationError(f"{name} points at a file that is not there: {path}")
    return path


def require_sha256(value: str, label: str) -> str:
    digest = value.strip()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ConfigurationError(f"{label} must be a frozen lowercase SHA-256")
    return digest


def resolve_profile_reference(relative: str, expected_sha256: str, label: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ConfigurationError(f"{label} escapes the repository: {relative}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256(path) != require_sha256(expected_sha256, f"{label} SHA-256"):
        raise ConfigurationError(f"{label} hash differs from the frozen QC profile")
    return path


def load_qc_profile(qc_job: dict[str, Any]) -> tuple[dict[str, Any], Path, str, Path]:
    profile_path = require_env_path("HELENA_QC_PROFILE")
    try:
        profile_path.relative_to(ROOT / "framework" / "profiles" / "04-validation")
    except ValueError as error:
        raise ConfigurationError("surface-QC profile must be a committed Stage 04 profile") from error
    expected_profile_sha256 = require_sha256(
        os.environ.get("HELENA_QC_PROFILE_SHA256", ""),
        "HELENA_QC_PROFILE_SHA256",
    )
    actual_profile_sha256 = sha256(profile_path)
    if actual_profile_sha256 != expected_profile_sha256:
        raise ConfigurationError(
            "surface-QC profile hash differs from HELENA_QC_PROFILE_SHA256: the "
            f"profile is {actual_profile_sha256}, the deployment pins "
            f"{expected_profile_sha256}")
    profile = load(profile_path)
    if profile.get("schema") != "campaignx.surface_qc_profile.v1":
        raise ConfigurationError("unexpected surface-QC profile schema")
    profile_id = str(profile.get("profile_id", ""))
    if not profile_id or "@" not in profile_id:
        raise ConfigurationError("surface-QC profile ID is not semantic and versioned")
    if qc_job.get("profile_id") != profile_id:
        raise ConfigurationError("claimed QC job profile differs from the runtime QC profile")

    ink_lane = profile.get("ink_lane")
    if not isinstance(ink_lane, dict):
        raise RuntimeError("surface-QC profile omitted ink_lane")
    ink_profile_path = resolve_profile_reference(
        str(ink_lane.get("profile", "")),
        str(ink_lane.get("profile_sha256", "")),
        "ink lane profile",
    )
    ink_profile = load(ink_profile_path)
    if ink_profile.get("profile_id") is None or ink_profile.get("method_id") != ink_lane.get("method_id"):
        raise RuntimeError("surface-QC and ink-lane profiles disagree about method identity")
    if ink_profile.get("checkpoint_sha256") != ink_lane.get("checkpoint_sha256"):
        raise ConfigurationError("surface-QC and ink-lane profiles disagree about checkpoint")
    input_contract = ink_profile.get("input_contract")
    if not isinstance(input_contract, dict) or input_contract.get("model_family") != ink_lane.get("model_family"):
        raise RuntimeError("surface-QC and ink-lane profiles disagree about model family")

    render = profile.get("render")
    screening = profile.get("screening")
    if not isinstance(render, dict) or not isinstance(screening, dict):
        raise RuntimeError("surface-QC profile omitted render or screening settings")
    if int(render.get("slice_count", 0)) != REQUIRED_TIFF_COUNT:
        raise RuntimeError("surface-QC adapter requires exactly 65 registered CT slices")
    depth_centers = screening.get("depth_centers")
    tiling_offsets = screening.get("tiling_offsets")
    if not isinstance(depth_centers, list) or not isinstance(tiling_offsets, list):
        raise RuntimeError("surface-QC profile omitted replica coordinates")
    if len(depth_centers) * len(tiling_offsets) != 6 or int(
        screening.get("replica_count", 0)
    ) != 6:
        raise RuntimeError("surface-QC adapter requires the frozen six-replica screen")

    gate = profile.get("ct_fiber_gate")
    if not isinstance(gate, dict):
        raise RuntimeError("surface-QC profile omitted ct_fiber_gate")
    gate_path = resolve_profile_reference(
        str(gate.get("profile", "")),
        str(gate.get("profile_sha256", "")),
        "CT/fiber gate profile",
    )
    shadow = profile.get("ct_fiber_shadow_router")
    if shadow is not None:
        if not isinstance(shadow, dict):
            raise RuntimeError("surface-QC shadow router must be an object")
        shadow_path = resolve_profile_reference(
            str(shadow.get("profile", "")),
            str(shadow.get("profile_sha256", "")),
            "CT/fiber shadow router profile",
        )
        shadow_profile = load(shadow_path)
        if shadow_profile.get("status") != "SHADOW_ONLY_NOT_SCIENTIFICALLY_FROZEN":
            raise RuntimeError("candidate shadow router must remain explicitly shadow-only")
    safety = profile.get("safety")
    required_safety = {
        "screening_only": True,
        "no_automatic_ink_acceptance": True,
        "no_automatic_letter_acceptance": True,
        "silence_proves_absence": False,
        "retained_means_human_review_only": True,
    }
    if not isinstance(safety, dict) or any(
        safety.get(key) is not value for key, value in required_safety.items()
    ):
        raise RuntimeError("surface-QC profile weakens mandatory scientific safety")
    return profile, profile_path, actual_profile_sha256, gate_path


def resolve_code_commit() -> str:
    configured = os.environ.get("HELENA_QC_CODE_COMMIT", "").strip().lower()
    if configured:
        if len(configured) != 40 or any(
            character not in "0123456789abcdef" for character in configured
        ):
            raise RuntimeError("HELENA_QC_CODE_COMMIT must be a full Git SHA")
        return configured
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    commit = completed.stdout.strip().lower()
    if completed.returncode or len(commit) != 40:
        raise RuntimeError("QC evidence requires HELENA_QC_CODE_COMMIT or a Git checkout")
    return commit


def run_logged(command: list[str], log: Path) -> None:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {command[0]}"
        )


def run_renderer_with_retries(
    command: list[str], *, output: Path, tiff_dir: Path
) -> dict[str, Any]:
    """Retry only the observed transient renderer SIGABRT, byte-for-byte.

    VC3D's renderer has occasionally aborted before writing any CT output with
    ``terminate called without an active exception``.  The exact same command
    succeeds immediately afterwards.  This helper never changes scientific
    parameters: it removes partial TIFFs and retries that one fail signature a
    bounded number of times while preserving every attempt log and a receipt.
    """

    rows: list[dict[str, Any]] = []
    renderer_env = {
        key: value for key, value in os.environ.items() if not key.startswith("AWS_")
    }
    removed_environment_names = sorted(
        key for key in os.environ if key.startswith("AWS_")
    )
    command_digest = hashlib.sha256(
        json.dumps(command, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    final_log = output / "ct-render.stdout.log"
    renderer_log = output / "ct-render.log"
    for attempt in range(1, RENDER_MAX_ATTEMPTS + 1):
        if tiff_dir.exists():
            shutil.rmtree(tiff_dir)
        renderer_log.unlink(missing_ok=True)
        attempt_log = output / f"ct-render.attempt-{attempt:02d}.stdout.log"
        # This VC3D build aborts inside ABF++ when its combined output is a
        # Python pipe. A regular file descriptor is stable with the identical
        # command, so stream directly to the durable per-attempt log.
        with attempt_log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                env=renderer_env,
            )
        stdout = attempt_log.read_text(encoding="utf-8", errors="replace")
        transient_abort = (
            completed.returncode == -6
            and TRANSIENT_RENDER_ABORT in stdout
        )
        rows.append(
            {
                "attempt": attempt,
                "returncode": completed.returncode,
                "stdout_log": attempt_log.name,
                "stdout_sha256": sha256(attempt_log),
                "transient_abort": transient_abort,
            }
        )
        final_log.write_text(stdout, encoding="utf-8")
        if completed.returncode == 0:
            receipt = {
                "schema": "campaignx.vc3d_renderer_retry_receipt.v1",
                "generated_at_utc": utc_now(),
                "status": "COMPLETED",
                "attempt_count": attempt,
                "max_attempts": RENDER_MAX_ATTEMPTS,
                "retry_backoff_seconds": RENDER_RETRY_BACKOFF_SECONDS,
                "command_sha256": command_digest,
                "scientific_parameters_unchanged": True,
                "removed_environment_names": removed_environment_names,
                "attempts": rows,
            }
            write(output / "CT_RENDER_RETRY_RECEIPT.json", receipt)
            return receipt
        if not transient_abort:
            break
        if attempt < RENDER_MAX_ATTEMPTS:
            time.sleep(RENDER_RETRY_BACKOFF_SECONDS)
    receipt = {
        "schema": "campaignx.vc3d_renderer_retry_receipt.v1",
        "generated_at_utc": utc_now(),
        "status": "FAILED",
        "attempt_count": len(rows),
        "max_attempts": RENDER_MAX_ATTEMPTS,
        "retry_backoff_seconds": RENDER_RETRY_BACKOFF_SECONDS,
        "command_sha256": command_digest,
        "scientific_parameters_unchanged": True,
        "removed_environment_names": removed_environment_names,
        "attempts": rows,
    }
    write(output / "CT_RENDER_RETRY_RECEIPT.json", receipt)
    raise RuntimeError(
        f"renderer failed after {len(rows)} attempt(s) with exit code "
        f"{rows[-1]['returncode']}: {command[0]}"
    )


# A grown surface and a flattened sheet are both immutable TIFXYZ artifact sets
# and are verified identically. They keep separate schema names on purpose: the
# provenance of a layer stack depends on which of the two it was rendered from,
# and one name for both would make that unanswerable from the artifact.
TIFXYZ_MANIFEST_SCHEMAS = (
    "campaignx.segmentation_artifact_set.v1",
    "campaignx.flattened_artifact_set.v1",
    "campaignx.merged_tifxyz_artifact_set.v1",
)


def _verify_surface(directory: Path, manifest: dict[str, Any], expected_digest: str) -> None:
    if manifest.get("schema") not in TIFXYZ_MANIFEST_SCHEMAS:
        raise RuntimeError(
            f"unexpected TIFXYZ artifact manifest schema: "
            f"{manifest.get('schema')!r}; expected one of {TIFXYZ_MANIFEST_SCHEMAS}")
    if manifest.get("artifact_sha256") != expected_digest:
        raise RuntimeError("TIFXYZ artifact digest differs from the catalogue")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("TIFXYZ artifact manifest has no file inventory")
    for name in REQUIRED_SURFACE_FILES:
        expected = files.get(name)
        path = directory / name
        if not isinstance(expected, dict) or not path.is_file():
            raise RuntimeError(f"TIFXYZ artifact is incomplete: {name}")
        if path.stat().st_size != int(expected.get("size_bytes", -1)):
            raise RuntimeError(f"TIFXYZ artifact size mismatch: {name}")
        if sha256(path) != expected.get("sha256"):
            raise RuntimeError(f"TIFXYZ artifact hash mismatch: {name}")


def _copy_local_surface(
    source: Path,
    destination: Path,
    expected_digest: str,
) -> dict[str, Any]:
    """Copy and verify one immutable TIFXYZ artifact set from local storage."""

    if not source.is_dir():
        raise FileNotFoundError(source)
    manifest = load(source / "ARTIFACT_SET.json")
    for name in (*REQUIRED_SURFACE_FILES, "ARTIFACT_SET.json"):
        shutil.copy2(source / name, destination / name)
    _verify_surface(destination, manifest, expected_digest)
    return manifest


def _s3_mirror_source(parsed: Any) -> Path | None:
    """Resolve an optional credential-free local mirror for an S3 prefix.

    The mirror preserves the bucket/key hierarchy so the catalogue URI stays
    authoritative.  Files can be delivered with narrowly scoped presigned
    URLs; all scientific inputs are still checked against ARTIFACT_SET.json
    and the catalogue digest before rendering.
    """

    configured = os.environ.get("HELENA_QC_SURFACE_MIRROR_ROOT", "").strip()
    if not configured:
        return None
    return (
        Path(configured).expanduser().resolve()
        / parsed.netloc
        / parsed.path.strip("/")
    )


SURFACE_HTTP_TIMEOUT_SECONDS = 300


def _http_get(url: str, *, timeout: float = SURFACE_HTTP_TIMEOUT_SECONDS) -> bytes:
    """Read one published object, with a deadline.

    A read without one is how a flatten job burns its whole lease and dies with
    nothing recorded: no CPU, no output, no reason. The deadline is per object,
    not per surface, because a stalled connection is what needs bounding.
    """

    request = urllib.request.Request(
        url, headers={"User-Agent": "Campaign-X-surface-qc-adapter/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _download_http_surface(
    base_uri: str,
    destination: Path,
    expected_digest: str,
) -> dict[str, Any]:
    """Materialize one TIFXYZ artifact set published over HTTP(S).

    A public host is not a trusted one: the bytes are checked against the
    manifest and the manifest against the catalogue digest by the same
    `_verify_surface` the S3 and local branches use.
    """

    base = base_uri.rstrip("/")
    manifest = json.loads(_http_get(f"{base}/ARTIFACT_SET.json").decode("utf-8"))
    write(destination / "ARTIFACT_SET.json", manifest)
    for name in REQUIRED_SURFACE_FILES:
        (destination / name).write_bytes(_http_get(f"{base}/{name}"))
    _verify_surface(destination, manifest, expected_digest)
    return manifest


def _locked_inventory(expected_files: Any) -> dict[str, dict[str, Any]]:
    """Normalize the catalogue's file inventory, and require it to be complete.

    An inventory that names fewer files than the artifact set would quietly
    narrow what gets verified, which is worse than having none: it reads as a
    check while leaving whatever it omits unguarded.
    """

    entries: dict[str, dict[str, Any]] = {}
    for entry in expected_files:
        name = str(entry.get("path") or entry.get("name") or "")
        if not name or not entry.get("sha256"):
            raise ValueError(
                f"catalogue inventory entry names no file or no digest: {entry!r}")
        entries[name] = entry
    missing = [name for name in REQUIRED_SURFACE_FILES if name not in entries]
    if missing:
        raise ValueError(
            f"the catalogue inventory does not cover {', '.join(missing)}: an "
            "inventory that omits a file would narrow what is verified")
    return entries


def _verify_against_inventory(
    directory: Path, entries: dict[str, dict[str, Any]]
) -> None:
    """Check what arrived against what the catalogue locked.

    Distinct from `_verify_surface`, which checks an artifact set against a
    manifest that travelled with it. Where the two disagree this one wins: a
    manifest from the same place as the bytes can only confirm itself, and for
    a surface somebody else published that is the whole of the difference.
    """

    for name in REQUIRED_SURFACE_FILES:
        entry = entries[name]
        path = directory / name
        if not path.is_file():
            raise RuntimeError(f"TIFXYZ artifact is incomplete: {name}")
        size = entry.get("size_bytes")
        if size is not None and path.stat().st_size != int(size):
            raise RuntimeError(f"TIFXYZ artifact size mismatch: {name}")
        if sha256(path) != entry.get("sha256"):
            raise RuntimeError(
                f"TIFXYZ artifact hash mismatch against the catalogue: {name}")


def _download_listed_surface(
    base_uri: str,
    destination: Path,
    expected_digest: str,
    locked: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Fetch exactly the files the catalogue locked, and ask the host for nothing else.

    No `ARTIFACT_SET.json` is requested. A community host does not publish one
    -- the name is our convention, and expecting third parties to adopt it is an
    obstacle no community segmentation will clear -- and one fetched from the
    same place as the bytes could only vouch for itself.

    The manifest written here is therefore ours, built from what arrived after
    it matched the inventory. Its `artifact_sha256` is the catalogue's digest
    carried over rather than anything derived: the verification that happened is
    the per-file one above it.
    """

    base = base_uri.rstrip("/")
    for name in REQUIRED_SURFACE_FILES:
        (destination / name).write_bytes(_http_get(f"{base}/{name}"))
    _verify_against_inventory(destination, locked)
    manifest = {
        "schema": "campaignx.segmentation_artifact_set.v1",
        "files": {
            name: {"sha256": sha256(destination / name),
                   "size_bytes": (destination / name).stat().st_size}
            for name in REQUIRED_SURFACE_FILES},
        "artifact_sha256": expected_digest,
    }
    write(destination / "ARTIFACT_SET.json", manifest)
    return manifest


def materialize_surface(
    artifact_uri: str,
    expected_digest: str,
    destination: Path,
    *,
    s3_client: Any | None = None,
    expected_files: Any | None = None,
) -> dict[str, Any]:
    if destination.exists():
        raise RuntimeError(f"refusing to reuse surface staging: {destination}")
    destination.mkdir(parents=True)
    locked = _locked_inventory(expected_files) if expected_files is not None else None
    parsed = urlparse(artifact_uri)
    if parsed.scheme in ("http", "https") and locked is not None:
        return _download_listed_surface(
            artifact_uri, destination, expected_digest, locked)
    if parsed.scheme == "s3":
        mirror_source = _s3_mirror_source(parsed)
        if mirror_source is not None and mirror_source.is_dir():
            # Falls through rather than returning: a mirror is a convenience
            # copy, and the inventory check at the end applies to it exactly as
            # it does to the bucket it stands in for.
            manifest = _copy_local_surface(
                mirror_source,
                destination,
                expected_digest,
            )
        else:
            if s3_client is None:
                try:
                    import boto3
                except ImportError as error:  # pragma: no cover - runtime dependency
                    raise RuntimeError(
                        "S3 surface materialization requires boto3") from error
                s3_client = boto3.client("s3")
            base = parsed.path.strip("/")
            manifest_key = f"{base}/ARTIFACT_SET.json"
            response = s3_client.get_object(Bucket=parsed.netloc, Key=manifest_key)
            body = response["Body"].read()
            manifest = json.loads(body.decode("utf-8"))
            write(destination / "ARTIFACT_SET.json", manifest)
            for name in REQUIRED_SURFACE_FILES:
                s3_client.download_file(
                    parsed.netloc, f"{base}/{name}", str(destination / name)
                )
            _verify_surface(destination, manifest, expected_digest)
    elif parsed.scheme in ("http", "https"):
        manifest = _download_http_surface(artifact_uri, destination, expected_digest)
    elif parsed.scheme == "file":
        source = Path(urllib.request.url2pathname(parsed.path))
        manifest = _copy_local_surface(source, destination, expected_digest)
    elif "://" in artifact_uri:
        # Anything else must say so by name. Falling through to the local branch
        # is what made an `https://` surface resolve to `<cwd>/https:/host/key`:
        # a URI the lane cannot read became a filesystem accident that depended
        # on where the worker happened to be standing. Keyed off `://` rather
        # than a parsed scheme so a relative path containing a colon stays a
        # path, which is what `urlparse` would otherwise call a scheme.
        raise ValueError(
            f"surface artifact URI scheme {parsed.scheme!r} cannot be read by "
            f"this lane: {artifact_uri}")
    else:
        source = Path(artifact_uri).expanduser().resolve()
        manifest = _copy_local_surface(source, destination, expected_digest)
    # Every branch above has now checked its artifact set against the manifest
    # that travelled with it.
    if locked is not None:
        # Last and decisive: whatever manifest travelled with the bytes, the
        # catalogue's inventory is the one the lane locked.
        _verify_against_inventory(destination, locked)
    return manifest


def fetch_zarr_metadata(ct_uri: str, cache: Path) -> None:
    for suffix in (".zgroup", ".zattrs", "0/.zarray"):
        destination = cache / suffix
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            f"{ct_uri.rstrip('/')}/{suffix}",
            headers={"User-Agent": "Campaign-X-surface-qc-adapter/1.0"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            destination.write_bytes(response.read())


def render_and_screen(
    *,
    output: Path,
    sample_id: str,
    surface_id: str,
    source: dict[str, Any],
    renderer: Path,
    checkpoint: Path,
    qc_profile: dict[str, Any],
    qc_profile_sha256: str,
    input_sha256: str,
) -> tuple[dict[str, Any], Path]:
    render_profile = qc_profile["render"]
    screening_profile = qc_profile["screening"]
    ink_lane = qc_profile["ink_lane"]
    expected_checkpoint_sha256 = require_sha256(
        str(ink_lane["checkpoint_sha256"]), "QC profile checkpoint SHA-256"
    )
    actual_checkpoint_sha256 = sha256(checkpoint)
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ConfigurationError(
            "checkpoint hash differs from the frozen surface-QC profile: the file "
            f"is {actual_checkpoint_sha256}, the profile names "
            f"{expected_checkpoint_sha256}")
    inference = resolve_stage_script(ROOT, "run_ink_timesformer.py")
    analysis = resolve_stage_script(ROOT, "analyze_ink_stability.py")
    surface = output / "surface"
    tiff_dir = output / "tiffs"
    screening_label = str(screening_profile["label"])
    screening = output / screening_label
    analysis_dir = screening / "analysis"
    ct_cache = output / "ct_metadata_cache.zarr"
    ct_uri = str(source["ct_uri"])
    voxel_um = float(source["voxel_size_um"])
    fetch_zarr_metadata(ct_uri, ct_cache)
    render_command = [
            str(renderer),
            "--segmentation",
            str(surface),
            "--volume",
            str(ct_cache),
            "--remote-url",
            ct_uri,
            "--prefetch-remote",
            "--scale",
            "1",
            "--group-idx",
            "0",
            "--auto-crop",
            "--flatten",
            "--flatten-iterations",
            str(int(render_profile["flatten_iterations"])),
            "--flatten-downsample",
            str(int(render_profile["flatten_downsample"])),
            "--num-slices",
            str(int(render_profile["slice_count"])),
            "--slice-step",
            str(int(render_profile["slice_step"])),
            "--cache-gb",
            os.environ.get("HELENA_QC_CACHE_GB", "4"),
            "--timeout",
            "90",
            "--voxel-size",
            str(voxel_um),
            "--voxel-unit",
            "micrometer",
            "--tif-output",
            str(tiff_dir),
            "--log-path",
            str(output / "ct-render.log"),
        ]
    run_renderer_with_retries(render_command, output=output, tiff_dir=tiff_dir)
    rendered, slice_ordering = ordered_tiff_files(tiff_dir)
    if len(rendered) != REQUIRED_TIFF_COUNT:
        raise RuntimeError(
            f"renderer produced {len(rendered)} TIFFs instead of {REQUIRED_TIFF_COUNT}"
        )
    run_logged(
        [
            sys.executable,
            str(inference),
            "--sample-id",
            sample_id,
            "--tiff-dir",
            str(tiff_dir),
            "--checkpoint",
            str(checkpoint),
            "--model-family",
            str(ink_lane["model_family"]),
            "--output",
            str(screening),
            "--depth-centers",
            ",".join(str(value) for value in screening_profile["depth_centers"]),
            "--tiling-offsets",
            ",".join(str(value) for value in screening_profile["tiling_offsets"]),
            "--frames",
            str(int(screening_profile["frames"])),
            "--source-pixel-um",
            str(voxel_um),
            "--training-pixel-um",
            str(float(screening_profile["training_pixel_um"])),
            "--source-slice-um",
            str(voxel_um),
            "--training-slice-um",
            str(float(screening_profile["training_slice_um"])),
            "--tile-size",
            str(int(screening_profile["tile_size"])),
            "--stride",
            str(int(screening_profile["stride"])),
            "--batch-size",
            os.environ.get("HELENA_QC_BATCH_SIZE", "128"),
            "--min-valid-ratio",
            str(float(screening_profile["minimum_valid_ratio"])),
            "--device",
            os.environ.get("HELENA_QC_DEVICE", "cuda"),
            "--on-degenerate",
            "warn",
        ],
        output / "robust-inference.stdout.log",
    )
    screening_receipt = screening / "INK_SCREENING_RECEIPT.json"
    if not screening_receipt.is_file():
        raise RuntimeError("six-replica inference receipt is absent")
    screening_payload = load(screening_receipt)
    liveness = screening_payload.get("liveness")
    if not isinstance(liveness, dict):
        raise RuntimeError("six-replica inference receipt omitted liveness")
    verdict = str(liveness.get("verdict", ""))
    if verdict not in {"ALIVE", "DEGENERATE", "EMPTY"}:
        raise RuntimeError("six-replica inference receipt has an unknown liveness verdict")
    if verdict != "ALIVE":
        reason = liveness.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise RuntimeError(
                "six-replica inference receipt has an invalid liveness reason"
            )
        row = {
            "sample_id": sample_id,
            "seed_id": surface_id,
            "state": OUTCOME_INK_SCREEN_INSUFFICIENT,
            "screening_outcome": OUTCOME_INK_SCREEN_INSUFFICIENT,
            "manual_review_route": "NO_USABLE_INK_MAP",
            "tiff_count": REQUIRED_TIFF_COUNT,
            "slice_ordering": slice_ordering,
            "screening_receipt": str(screening_receipt),
            "screening_receipt_sha256": sha256(screening_receipt),
            "liveness": liveness,
        }
        summary = _write_screen_summary(
            output,
            row,
            renderer,
            checkpoint,
            inference,
            analysis,
            actual_checkpoint_sha256,
            input_sha256,
            qc_profile,
            qc_profile_sha256,
        )
        return summary, output / "FLEET_SURFACE_SCREEN_EXECUTION.json"
    analysis_log = output / "robust-analysis.stdout.log"
    try:
        run_logged(
            [
                sys.executable,
                str(analysis),
                "--sample-id",
                sample_id,
                "--screening-dir",
                str(screening),
                "--tiff-dir",
                str(tiff_dir),
                "--output",
                str(analysis_dir),
                "--source-center",
                "32",
                # FIX-09: source scale is cross-checked against the frozen
                # catalog by the analysis script and the training scale comes
                # from the ink lane profile, never a literal.
                "--source-pixel-um",
                str(voxel_um),
                "--training-pixel-um",
                str(float(screening_profile["training_pixel_um"])),
                "--glyph-threshold",
                "0.5",
                "--hotspots",
                "12",
                "--crop-size",
                "384",
            ],
            analysis_log,
        )
    except RuntimeError:
        if NO_COMMON_VALID_ERROR not in analysis_log.read_text(
            encoding="utf-8", errors="replace"
        ):
            raise
        row = {
            "sample_id": sample_id,
            "seed_id": surface_id,
            "state": OUTCOME_INSUFFICIENT,
            "screening_outcome": OUTCOME_INSUFFICIENT,
            "manual_review_route": "NO_CROSS_REPLICA_CT_SUPPORT",
            "tiff_count": REQUIRED_TIFF_COUNT,
            "slice_ordering": slice_ordering,
            "analysis_log": str(analysis_log),
            "analysis_log_sha256": sha256(analysis_log),
            "error": NO_COMMON_VALID_ERROR,
        }
        summary = _write_screen_summary(
            output,
            row,
            renderer,
            checkpoint,
            inference,
            analysis,
            actual_checkpoint_sha256,
            input_sha256,
            qc_profile,
            qc_profile_sha256,
        )
        return summary, output / "FLEET_SURFACE_SCREEN_EXECUTION.json"
    analysis_path = analysis_dir / "INK_STABILITY_ANALYSIS.json"
    if not analysis_path.is_file():
        raise RuntimeError("stability analysis is absent")
    analysis_payload = load(analysis_path)
    text_like = analysis_payload.get("text_like_screening", {})
    row = {
        "sample_id": sample_id,
        "seed_id": surface_id,
        "state": "COMPLETED_DIAGNOSTIC_ONLY",
        "tiff_count": REQUIRED_TIFF_COUNT,
        "slice_ordering": slice_ordering,
        "analysis": str(analysis_path),
        "analysis_sha256": sha256(analysis_path),
        "screening_receipt": str(screening_receipt),
        "screening_receipt_sha256": sha256(screening_receipt),
        "screening_outcome": text_like.get("screening_outcome"),
        "glyph_like_candidate_count": text_like.get("glyph_like_candidate_count"),
        "rows_with_at_least_four_candidates": text_like.get(
            "rows_with_at_least_four_candidates"
        ),
        "manual_review_route": analysis_payload.get("manual_review_routing", {}).get(
            "route"
        ),
    }
    summary = _write_screen_summary(
        output,
        row,
        renderer,
        checkpoint,
        inference,
        analysis,
        actual_checkpoint_sha256,
        input_sha256,
        qc_profile,
        qc_profile_sha256,
    )
    return summary, output / "FLEET_SURFACE_SCREEN_EXECUTION.json"


def _write_screen_summary(
    output: Path,
    row: dict[str, Any],
    renderer: Path,
    checkpoint: Path,
    inference: Path,
    analysis: Path,
    checkpoint_sha256: str,
    input_sha256: str,
    qc_profile: dict[str, Any],
    qc_profile_sha256: str,
) -> dict[str, Any]:
    insufficient = row["state"] in {
        OUTCOME_INSUFFICIENT,
        OUTCOME_INK_SCREEN_INSUFFICIENT,
    }
    summary = {
        "kind": "campaign_x_phase4_geometry_recovery_v1_screen_execution",
        "generated_at_utc": utc_now(),
        "status": "COMPLETED_DIAGNOSTIC_ONLY",
        "plan_sha256": input_sha256,
        "planned_selected_count": 1,
        "executed_sample_ids": [row["sample_id"]],
        "selected_count": 1,
        "completed_count": 0 if insufficient else 1,
        "renderer_sha256": sha256(renderer),
        "checkpoint_sha256": checkpoint_sha256,
        "inference_sha256": sha256(inference),
        "analysis_sha256": sha256(analysis),
        "model_provenance": {
            "checkpoint_sha256": checkpoint_sha256,
            "expected_checkpoint_sha256": checkpoint_sha256,
            "surface_qc_profile_id": qc_profile["profile_id"],
            "surface_qc_profile_sha256": qc_profile_sha256,
            "ink_method_id": qc_profile["ink_lane"]["method_id"],
            "model_family": qc_profile["ink_lane"]["model_family"],
            "screening_label": qc_profile["screening"]["label"],
            "comparability_classification": "NONCOMPARABLE_ABF_DOWNSAMPLE_2_SERIES",
            "flatten_downsample": qc_profile["render"]["flatten_downsample"],
        },
        "receipts": [row],
        "policy": [
            "six replica maps are a router only",
            "no activation is accepted as ink or letters",
            "no-common-valid CT is insufficiency, not a negative",
            "all retained signals still require visual interpretation",
        ],
    }
    write(output / "FLEET_SURFACE_SCREEN_EXECUTION.json", summary)
    return summary


def run_high_recall(
    output: Path,
    summary_path: Path,
    gate_profile_path: Path,
    voxel_um: float | None = None,
    shadow_profile_path: Path | None = None,
) -> dict[str, Any]:
    high_recall = output / "high-recall"
    high_recall.mkdir()
    scripts = {
        name: resolve_stage_script(ROOT, filename)
        for name, filename in {
            "manifest": "build_geometry_high_recall_manifest.py",
            "router": "route_high_recall_ct_candidates.py",
            "adapter": "build_high_recall_ct_application.py",
            "features": "extract_ct_fiber_features.py",
            "gate": "apply_ct_fiber_gate.py",
            "review": "build_high_recall_retained_review.py",
            "near_miss": "helena_build_ct_gate_near_miss_review.py",
            "physical_features": "extract_ct_fiber_features_physical.py",
            "shadow_router": "helena_route_ct_gate_shadow_v4.py",
        }.items()
    }
    logs = high_recall / "logs"
    manifest = high_recall / "HIGH_RECALL_CT_ROUTER_MANIFEST.json"
    router_dir = high_recall / "router"
    router_path = router_dir / "HIGH_RECALL_CT_ROUTER_RECEIPT.json"
    run_logged(
        [
            sys.executable,
            str(scripts["manifest"]),
            "--root",
            str(output),
            "--screen-summary",
            str(summary_path),
            "--output",
            str(manifest),
        ],
        logs / "manifest.stdout.log",
    )
    run_logged(
        [
            sys.executable,
            str(scripts["router"]),
            "--manifest",
            str(manifest),
            "--output",
            str(router_dir),
        ],
        logs / "router.stdout.log",
    )
    router = load(router_path)
    queued = int(router.get("ct_review_queue_count", -1))
    if queued < 0:
        raise RuntimeError("high-recall router omitted its queue count")
    if queued == 0:
        result = {
            "kind": "campaignx.segment_fleet_high_recall_result.v1",
            "status": "COMPLETED_NO_CT_REVIEW_COMPONENTS",
            "router_component_count": 0,
            "ct_retained_count": 0,
            "ct_downranked_count": 0,
        }
        write(high_recall / "FOLLOWUP_RESULT.json", result)
        return result
    application = high_recall / "ct_application"
    freeze = gate_profile_path
    if not freeze.is_file():
        raise RuntimeError(f"frozen CT/fiber gate profile is absent: {freeze}")
    spec = application / "HIGH_RECALL_CT_FIBER_SPEC.json"
    adapter = application / "HIGH_RECALL_CT_ADAPTER_RECEIPT.json"
    features_dir = application / "features"
    features = features_dir / "CT_FIBER_FEATURES.csv"
    gate_dir = application / "gate"
    gate_path = gate_dir / "CT_FIBER_GATE_EVALUATION.json"
    review_dir = application / "review_v1"
    review_path = review_dir / "HIGH_RECALL_RETAINED_REVIEW_RECEIPT.json"
    near_miss_dir = application / "near_miss_audit_v1"
    near_miss_path = near_miss_dir / "CT_GATE_NEAR_MISS_AUDIT.json"
    physical_features_dir = application / "physical_features_v1"
    physical_features = physical_features_dir / "CT_FIBER_PHYSICAL_FEATURES.csv"
    shadow_dir = application / "shadow_router_v4"
    shadow_receipt_path = shadow_dir / "CT_GATE_V4_SHADOW_RECEIPT.json"
    adapter_command = [
            sys.executable,
            str(scripts["adapter"]),
            "--root",
            str(output),
            "--metadata-root",
            str(ROOT),
            "--router-receipt",
            str(router_path),
            "--output",
            str(application),
            "--gate-freeze",
            str(freeze),
            "--patch-radius-um",
            "200",
            "--central-slice",
            "32",
        ]
    if voxel_um is not None:
        if voxel_um <= 0:
            raise RuntimeError("surface-QC voxel size must be positive")
        adapter_command.extend(["--voxel-um", str(voxel_um)])
    run_logged(
        adapter_command,
        logs / "adapter.stdout.log",
    )
    run_logged(
        [
            sys.executable,
            str(scripts["features"]),
            "--root",
            str(output),
            "--spec",
            str(spec),
            "--output",
            str(features_dir),
        ],
        logs / "features.stdout.log",
    )
    run_logged(
        [
            sys.executable,
            str(scripts["gate"]),
            "--features",
            str(features),
            "--rule",
            str(freeze),
            "--output",
            str(gate_dir),
        ],
        logs / "gate.stdout.log",
    )
    run_logged(
        [
            sys.executable,
            str(scripts["review"]),
            "--root",
            str(output),
            "--adapter-receipt",
            str(adapter),
            "--gate-evaluation",
            str(gate_path),
            "--output",
            str(review_dir),
            "--half-size",
            "128",
        ],
        logs / "review.stdout.log",
    )
    run_logged(
        [
            sys.executable,
            str(scripts["near_miss"]),
            "--root",
            str(output),
            "--adapter-receipt",
            str(adapter),
            "--gate-evaluation",
            str(gate_path),
            "--output",
            str(near_miss_dir),
            "--half-size",
            "128",
            "--max-failed-checks",
            "1",
            "--limit",
            "6",
        ],
        logs / "near-miss.stdout.log",
    )
    gate = load(gate_path)
    review = load(review_path)
    near_miss = load(near_miss_path)
    retained = int(gate.get("retained_count", -1))
    if retained < 0 or retained != int(review.get("candidate_count", -2)):
        raise RuntimeError("CT gate and review bundle retained counts differ")
    shadow_result: dict[str, Any] | None = None
    if shadow_profile_path is not None:
        run_logged(
            [
                sys.executable,
                str(scripts["physical_features"]),
                "--root",
                str(output),
                "--spec",
                str(spec),
                "--profile",
                str(shadow_profile_path),
                "--output",
                str(physical_features_dir),
            ],
            logs / "physical-features.stdout.log",
        )
        run_logged(
            [
                sys.executable,
                str(scripts["shadow_router"]),
                "--gate-evaluation",
                str(gate_path),
                "--profile",
                str(shadow_profile_path),
                "--physical-features",
                str(physical_features),
                "--output",
                str(shadow_dir),
            ],
            logs / "shadow-router.stdout.log",
        )
        shadow_result = load(shadow_receipt_path)
    result = {
        "kind": "campaignx.segment_fleet_high_recall_result.v1",
        "status": "COMPLETED_REVIEW_EVIDENCE_ONLY",
        "router_component_count": queued,
        "ct_retained_count": retained,
        "ct_downranked_count": int(gate.get("downranked_count", -1)),
        "ct_near_miss_audited_count": int(near_miss.get("candidate_count", -1)),
        "ct_fiber_gate_profile": str(freeze.relative_to(ROOT)),
        "ct_fiber_gate_profile_sha256": sha256(freeze),
        "ct_fiber_shadow_router": (
            {
                "status": shadow_result["status"],
                "profile": str(shadow_profile_path.relative_to(ROOT)),
                "profile_sha256": sha256(shadow_profile_path),
                "tier_counts": shadow_result["tier_counts"],
                "not_discarded_count": shadow_result["not_discarded_count"],
                "extension_request_count": shadow_result["extension_request_count"],
            }
            if shadow_result is not None and shadow_profile_path is not None
            else None
        ),
    }
    write(high_recall / "FOLLOWUP_RESULT.json", result)
    return result


def evidence_files(output: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output)
        if relative.parts[0] in {"surface", "tiffs", "ct_metadata_cache.zarr"}:
            continue
        if path.suffix.lower() in {".npy", ".tif", ".tiff"}:
            continue
        if path.name in {"EVIDENCE_MANIFEST.json", "QC_EXECUTOR_RESULT.json"}:
            continue
        files.append(path)
    return files


def build_evidence_manifest(
    *,
    output: Path,
    qc_job: dict[str, Any],
    outcome: str,
    ink_used: bool,
    followup: dict[str, Any] | None,
    qc_profile: dict[str, Any],
    qc_profile_sha256: str,
    qc_profile_path: Path,
    gate_profile_path: Path,
) -> Path:
    rows = [
        {
            "path": str(path.relative_to(output)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in evidence_files(output)
    ]
    manifest = output / "EVIDENCE_MANIFEST.json"
    write(
        manifest,
        {
            "schema": "campaignx.segment_qc_evidence_manifest.v1",
            "generated_at_utc": utc_now(),
            "code_commit": resolve_code_commit(),
            "qc_job_id": qc_job["qc_job_id"],
            "surface_id": qc_job["surface_id"],
            "sample_id": qc_job["surface"].get("sample_id")
            or qc_job["source"]["sample_id"],
            "source_snapshot_id": qc_job["source"]["source_snapshot_id"],
            "surface_artifact_uri": qc_job["surface"]["artifact_uri"],
            "surface_artifact_sha256": qc_job["surface"]["artifact_sha256"],
            "surface_qc_profile": {
                "profile_id": qc_profile["profile_id"],
                "profile_path": str(qc_profile_path.relative_to(ROOT)),
                "profile_sha256": qc_profile_sha256,
                "ink_method_id": qc_profile["ink_lane"]["method_id"],
                "model_family": qc_profile["ink_lane"]["model_family"],
                "checkpoint_sha256": qc_profile["ink_lane"]["checkpoint_sha256"],
                "ct_fiber_gate_path": str(gate_profile_path.relative_to(ROOT)),
                "ct_fiber_gate_sha256": qc_profile["ct_fiber_gate"]["profile_sha256"],
                "adapter_sha256": sha256(Path(__file__).resolve()),
            },
            "render_profile": {
                **qc_profile["render"],
            },
            "outcome": outcome,
            "ink_used": ink_used,
            "followup": followup,
            "files": rows,
            "excluded_regenerable_payloads": [
                "65 rendered TIFF slices",
                "six raw replica NPY maps",
                "source TIFXYZ already preserved by surface artifact URI",
            ],
            "non_claims": [
                "not accepted ink",
                "not accepted text or letters",
                "not a First Letters submission",
            ],
        },
    )
    return manifest


def publish_evidence(
    output: Path,
    manifest: Path,
    root_uri: str,
    qc_job: dict[str, Any],
    *,
    s3_client: Any | None = None,
    attempt_id: str | None = None,
) -> str:
    """Publish one attempt's evidence under a key that names that attempt.

    `attempt_id` is what keeps a retry from colliding with the attempt before
    it. Not everything an attempt publishes is the same across attempts:
    CT_RENDER_RETRY_RECEIPT.json is the attempt's own account of itself and
    carries its timestamp and count, so its bytes differ every time.

    Publication skips a key that already exists and verifies the object that
    is there. Scoped to the job alone, a second attempt therefore met an
    object it could never match -- measured on gpu-1 on 2026-08-23, after 57
    minutes of completed work: local 9ae5b2a2..., in the bucket 6cea728a...
    from 18 days earlier, and no number of retries could clear it.

    The verification is not what was wrong; it is what keeps published
    evidence from being replaced unnoticed, and it stays. What was wrong is
    that one key was being asked to hold two things that legitimately differ.

    Optional, so every existing caller keeps working: a direct CLI run without
    one publishes exactly where it used to.
    """
    sample_id = str(qc_job["source"]["sample_id"])
    suffix = f"{sample_id}/{qc_job['surface_id']}/{qc_job['qc_job_id']}"
    if attempt_id:
        suffix = f"{suffix}/{attempt_id}"
    paths = [*evidence_files(output), manifest]
    parsed = urlparse(root_uri)
    if parsed.scheme == "s3":
        if s3_client is None:
            try:
                import boto3
            except ImportError as error:  # pragma: no cover
                raise RuntimeError("S3 QC evidence publication requires boto3") from error
            s3_client = boto3.client("s3")
        base = "/".join(part for part in (parsed.path.strip("/"), suffix) if part)

        def publish_one(path: Path) -> None:
            relative = path.relative_to(output).as_posix()
            key = f"{base}/{relative}"
            digest = sha256(path)
            try:
                head = s3_client.head_object(Bucket=parsed.netloc, Key=key)
            except Exception as error:
                response = getattr(error, "response", {})
                code = str(response.get("Error", {}).get("Code", ""))
                if not isinstance(error, (FileNotFoundError, KeyError)) and code not in {
                    "404",
                    "NoSuchKey",
                    "NotFound",
                }:
                    raise
                s3_client.upload_file(
                    str(path),
                    parsed.netloc,
                    key,
                    ExtraArgs={"Metadata": {"sha256": digest}},
                )
                head = s3_client.head_object(Bucket=parsed.netloc, Key=key)
            metadata = {str(k).lower(): str(v) for k, v in head.get("Metadata", {}).items()}
            if int(head["ContentLength"]) != path.stat().st_size or metadata.get("sha256") != digest:
                raise RuntimeError(f"published QC evidence failed verification: s3://{parsed.netloc}/{key}")

        workers = min(MAX_EVIDENCE_PUBLISH_WORKERS, max(1, len(paths)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(publish_one, paths))
        return f"s3://{parsed.netloc}/{base}/EVIDENCE_MANIFEST.json"
    destination = Path(root_uri).expanduser().resolve() / suffix
    if destination.exists():
        for path in paths:
            target = destination / path.relative_to(output)
            if not target.is_file() or sha256(target) != sha256(path):
                raise RuntimeError(f"existing QC evidence differs: {target}")
        return (destination / "EVIDENCE_MANIFEST.json").as_uri()
    for path in paths:
        target = destination / path.relative_to(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha256(target) != sha256(path):
            raise RuntimeError(f"local QC evidence hash mismatch: {target}")
    return (destination / "EVIDENCE_MANIFEST.json").as_uri()


def publish_evidence_with_failover(
    output: Path,
    manifest: Path,
    primary_root: str,
    fallback_root: str,
    qc_job: dict[str, Any],
    *,
    attempt_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Publish durably, using an explicit local fallback on primary outage.

    The fallback is never implicit: operators must configure it.  Scientific
    outputs and hashes are identical regardless of which storage endpoint is
    reachable, and the primary failure remains recorded for later relay.
    """

    try:
        uri = publish_evidence(output, manifest, primary_root, qc_job,
                               attempt_id=attempt_id)
        return uri, {
            "primary_root": primary_root,
            "fallback_root": fallback_root or None,
            "fallback_used": False,
            "primary_error": None,
        }
    except Exception as error:
        if not fallback_root:
            raise
        uri = publish_evidence(output, manifest, fallback_root, qc_job,
                               attempt_id=attempt_id)
        return uri, {
            "primary_root": primary_root,
            "fallback_root": fallback_root,
            "fallback_used": True,
            "primary_error": {
                "class": type(error).__name__,
                "message": str(error)[:1024],
            },
        }


def cleanup_regenerable_payloads(output: Path) -> dict[str, Any]:
    """Remove only payloads excluded by the durable evidence manifest."""

    removed: list[dict[str, Any]] = []
    roots = [output / "surface", output / "tiffs", output / "ct_metadata_cache.zarr"]
    for path in roots:
        if not path.exists():
            continue
        size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        shutil.rmtree(path)
        removed.append({"path": str(path.relative_to(output)), "bytes": size})
    for path in sorted(output.rglob("*.npy")):
        size = path.stat().st_size
        relative = str(path.relative_to(output))
        path.unlink()
        removed.append({"path": relative, "bytes": size})
    return {
        "enabled": True,
        "removed_file_or_tree_count": len(removed),
        "removed_bytes": sum(row["bytes"] for row in removed),
        "removed": removed,
        "durable_evidence_unchanged": True,
    }


def execute(input_path: Path, output: Path) -> dict[str, Any]:
    if not output.is_dir() or any(output.iterdir()):
        raise RuntimeError("QC adapter output must be an existing empty directory")
    payload = load(input_path)
    if payload.get("schema") != "campaignx.segment_qc_input.v1":
        raise RuntimeError("unexpected QC input schema")
    qc_job = payload.get("qc_job")
    if not isinstance(qc_job, dict):
        raise RuntimeError("QC input omitted its job")
    if "lease_token" in qc_job:
        raise RuntimeError("QC input must never contain a lease token")
    qc_profile, qc_profile_path, qc_profile_sha256, gate_profile_path = (
        load_qc_profile(qc_job)
    )
    source = qc_job["source"]
    surface = qc_job["surface"]
    sample_id = str(source["sample_id"])
    surface_id = str(qc_job["surface_id"])
    materialize_surface(
        str(surface["artifact_uri"]),
        str(surface["artifact_sha256"]),
        output / "surface",
    )
    renderer = require_env_path("HELENA_QC_RENDERER")
    checkpoint = require_env_path("HELENA_QC_CHECKPOINT")
    checkpoint_digest = str(qc_profile["ink_lane"]["checkpoint_sha256"])
    summary, summary_path = render_and_screen(
        output=output,
        sample_id=sample_id,
        surface_id=surface_id,
        source=source,
        renderer=renderer,
        checkpoint=checkpoint,
        qc_profile=qc_profile,
        qc_profile_sha256=qc_profile_sha256,
        input_sha256=sha256(input_path),
    )
    ink_used = True
    followup: dict[str, Any] | None = None
    if summary["receipts"][0]["state"] == OUTCOME_INK_SCREEN_INSUFFICIENT:
        outcome = OUTCOME_INK_SCREEN_INSUFFICIENT
    elif summary["receipts"][0]["state"] == OUTCOME_INSUFFICIENT:
        outcome = OUTCOME_INSUFFICIENT
    else:
        shadow_profile_path: Path | None = None
        shadow = qc_profile.get("ct_fiber_shadow_router")
        if isinstance(shadow, dict):
            shadow_profile_path = resolve_profile_reference(
                str(shadow["profile"]),
                str(shadow["profile_sha256"]),
                "CT/fiber shadow router profile",
            )
        followup = run_high_recall(
            output,
            summary_path,
            gate_profile_path,
            voxel_um=float(source["voxel_size_um"]),
            shadow_profile_path=shadow_profile_path,
        )
        outcome = (
            OUTCOME_RETAINED
            if int(followup["ct_retained_count"]) > 0
            else OUTCOME_NO_RETAINED
        )
    manifest = build_evidence_manifest(
        output=output,
        qc_job=qc_job,
        outcome=outcome,
        ink_used=ink_used,
        followup=followup,
        qc_profile=qc_profile,
        qc_profile_sha256=qc_profile_sha256,
        qc_profile_path=qc_profile_path,
        gate_profile_path=gate_profile_path,
    )
    evidence_root = os.environ.get("HELENA_QC_EVIDENCE_ROOT", "").strip()
    if not evidence_root:
        raise RuntimeError("HELENA_QC_EVIDENCE_ROOT is required for durable publication")
    fallback_root = os.environ.get(
        "HELENA_QC_EVIDENCE_FALLBACK_ROOT", ""
    ).strip()
    evidence_uri, publication = publish_evidence_with_failover(
        output,
        manifest,
        evidence_root,
        fallback_root,
        qc_job,
        # The attempt directory's own name. The worker mints one per attempt --
        # `{utc}-{uuid}` -- and it is the only thing in scope that distinguishes
        # this run of this job from the last one, which is exactly what the
        # evidence key was missing.
        attempt_id=output.parent.name or None,
    )
    cleanup = (
        cleanup_regenerable_payloads(output)
        if os.environ.get("HELENA_QC_CLEAN_REGENERABLE", "").strip() == "1"
        else {"enabled": False, "durable_evidence_unchanged": True}
    )
    result = {
        "schema": "campaignx.segment_qc_executor_result.v1",
        "outcome": outcome,
        "ink_used": ink_used,
        "evidence_manifest_path": str(manifest),
        "evidence_uri": evidence_uri,
        "publication": publication,
        "regenerable_cleanup": cleanup,
        "screen_summary_sha256": sha256(summary_path),
        "checkpoint_sha256": checkpoint_digest,
        "surface_qc_profile_id": qc_profile["profile_id"],
        "surface_qc_profile_sha256": qc_profile_sha256,
        "model_family": qc_profile["ink_lane"]["model_family"],
        "ink_method_id": qc_profile["ink_lane"]["method_id"],
        "ct_fiber_gate_sha256": qc_profile["ct_fiber_gate"]["profile_sha256"],
        "adapter_sha256": sha256(Path(__file__).resolve()),
        "code_commit": resolve_code_commit(),
        "retained_for_visual_review_count": (
            int(followup["ct_retained_count"]) if followup is not None else 0
        ),
        "no_automatic_acceptance": True,
    }
    write(output / "QC_EXECUTOR_RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = execute(args.input.resolve(), args.output.resolve())
    except ConfigurationError as error:
        # Distinguished from every other failure by the exit code, because that
        # is all the worker can see. It marks the job blocked instead of
        # requeuing it, so somebody is told rather than the fleet spinning.
        print(json.dumps({
            "schema": "campaignx.segment_qc_configuration_error.v1",
            "status": "BLOCKED_CONFIGURATION",
            "error": str(error),
            "no_scientific_conclusion": True,
        }, sort_keys=True), file=sys.stderr)
        return EX_CONFIG
    print(
        json.dumps(
            {
                "outcome": result["outcome"],
                "retained_for_visual_review_count": result[
                    "retained_for_visual_review_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
