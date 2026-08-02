#!/usr/bin/env python3
"""Compact one fully postprocessed Phase 4 robust window, fail closed.

The 65-slice 4 cm² TIFF stack and eight NPY probability arrays are large
derived caches.  Once CT features and the portable visual bundle exist, this
tool can:

1. verify every upstream/output hash;
2. preserve every NPY byte losslessly in a deterministic ZIP;
3. lock all source-render hashes and the exact crop recipe;
4. retain the central CT, PNGs, analyses, logs, receipts and viewer copy;
5. remove only the 64 reproducible non-central crop TIFFs and raw NPY files.

The default ``plan`` mode never mutates data.  ``compact`` is explicit and
resumable.  ``rehydrate`` restores exact NPY bytes and regenerates TIFF crops,
accepting them only when their original SHA-256 hashes match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import sys

_STAGE_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()) / "framework/stages"
for _stage_scripts in _STAGE_ROOT.glob("*/scripts"):
    _stage_scripts_text = str(_stage_scripts)
    if _stage_scripts_text not in sys.path:
        sys.path.insert(0, _stage_scripts_text)
ROOT = _STAGE_ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import Any

from PIL import Image

from framework.contracts.slice_order import ordered_tiff_files

from replica_evidence import (
    ARCHIVE_NAME,
    COMPACTION_COMPLETE_STATUS,
    COMPACTION_KIND,
    COMPACTION_RECEIPT_NAME,
    MAP_PATTERN,
    sha256_file,
)


PENDING_STATUS = "ARCHIVE_VERIFIED_PENDING_DERIVED_CACHE_REMOVAL"
FEATURE_STATUS = "FEATURES_EXTRACTED_THRESHOLDS_NOT_FROZEN"


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"unreadable JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON document is not an object: {path}")
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required file is missing: {path}")
    return {
        "path": (
            str(path.relative_to(relative_to)) if relative_to is not None else str(path)
        ),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_hash(path: Path, expected: Any, label: str) -> dict[str, Any]:
    record = file_record(path)
    if record["sha256"] != str(expected):
        raise RuntimeError(f"{label} SHA-256 mismatch: {path}")
    return record


def _unique_rows(rows: Any, *, key: str, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{label} must be a non-empty list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError(f"{label} contains a non-object row")
        value = str(row.get(key, ""))
        if not value or value in result:
            raise RuntimeError(f"{label} has an empty or duplicate {key}")
        result[value] = row
    return result


def verify_crop(
    window: Path,
    *,
    central_slice: int,
) -> tuple[
    Path,
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    tiff_dir = window / "tiffs"
    crop_path = tiff_dir / "PHYSICAL_CROP_RECEIPT.json"
    crop = read_json(crop_path)
    if (
        crop.get("kind") != "campaign_x_phase4_physical_render_crop_v1"
        or crop.get("status") != "COMPLETED"
    ):
        raise RuntimeError("physical crop receipt is not complete")
    artifact_rows = _unique_rows(
        crop.get("artifacts"),
        key="name",
        label="crop artifacts",
    )
    expected_count = int(crop.get("input", {}).get("slice_count", -1))
    if expected_count < 2 or len(artifact_rows) != expected_count:
        raise RuntimeError("crop artifact count differs from input slice count")

    central_name = f"{central_slice:02d}.tif"
    if central_name not in artifact_rows:
        raise RuntimeError(f"crop receipt lacks central TIFF {central_name}")
    crop_files: list[dict[str, Any]] = []
    removable: list[dict[str, Any]] = []
    for name, row in sorted(artifact_rows.items()):
        path = tiff_dir / name
        record = require_hash(path, row.get("sha256"), f"crop artifact {name}")
        if record["size_bytes"] != int(row.get("size_bytes", -1)):
            raise RuntimeError(f"crop artifact size mismatch: {name}")
        record["relative_to_window"] = str(path.relative_to(window))
        crop_files.append(record)
        if name != central_name:
            removable.append(record)

    input_directory = Path(str(crop["input"].get("directory", ""))).resolve()
    # Both sides of this comparison used to carry a *different* order key: the
    # render was sorted with a silent ``10**9`` reserve bucket for non-numeric
    # stems while the recipe was sorted with a bare ``int(stem)`` that raises.
    # One shared contract now orders the render, and the recipe is compared as
    # an unordered set of names so the two keys cannot drift again.
    source_files, slice_ordering = ordered_tiff_files(
        input_directory,
        require_numeric=True,
    )
    if {path.name for path in source_files} != set(artifact_rows):
        raise RuntimeError(
            "source render does not contain the exact crop-recipe slice set"
        )
    source_lock = {
        "directory": str(input_directory),
        "slice_count": len(source_files),
        "slice_ordering": slice_ordering,
        "files": [file_record(path) for path in source_files],
    }
    return crop_path, crop, crop_files, removable, source_lock


def verify_screening(
    window: Path,
    *,
    screening_name: str,
) -> tuple[
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
    list[dict[str, Any]],
]:
    screening = window / screening_name
    screening_receipt_path = screening / "INK_SCREENING_RECEIPT.json"
    screening_receipt = read_json(screening_receipt_path)
    if (
        screening_receipt.get("kind")
        != "campaign_x_phase4_timesformer_private_screening_v1"
        or screening_receipt.get("status") != "COMPLETED_DIAGNOSTIC_ONLY"
    ):
        raise RuntimeError("ink screening receipt is not complete")
    analysis_path = screening / "analysis" / "INK_STABILITY_ANALYSIS.json"
    analysis = read_json(analysis_path)
    if (
        analysis.get("kind") != "campaign_x_phase4_ink_stability_analysis_v1"
        or analysis.get("status") != "COMPLETED_DIAGNOSTIC_ONLY"
    ):
        raise RuntimeError("ink stability analysis is not complete")

    analysis_maps = _unique_rows(
        analysis.get("input", {}).get("maps"),
        key="file",
        label="analysis input maps",
    )
    runs = _unique_rows(
        screening_receipt.get("inference", {}).get("runs"),
        key="npy",
        label="screening inference runs",
    )
    if set(analysis_maps) != set(runs) or len(runs) != 6:
        raise RuntimeError("screening and analysis do not bind the same six maps")
    if any(MAP_PATTERN.fullmatch(name) is None for name in runs):
        raise RuntimeError("screening contains a noncanonical replica-map name")

    raw_rows: dict[str, dict[str, Any]] = {}
    for name, run in runs.items():
        path = screening / name
        expected = str(run.get("npy_sha256", ""))
        if str(analysis_maps[name].get("sha256", "")) != expected:
            raise RuntimeError(f"analysis/screening hash disagreement: {name}")
        raw_rows[name] = require_hash(path, expected, f"replica map {name}")

    aggregates = screening_receipt.get("aggregate_artifacts")
    if not isinstance(aggregates, dict):
        raise RuntimeError("screening receipt lacks aggregate artifacts")
    for name in ("mean_probability.npy", "stability_std.npy"):
        row = aggregates.get(name)
        if not isinstance(row, dict):
            raise RuntimeError(f"screening receipt lacks {name}")
        path = screening / name
        record = require_hash(path, row.get("sha256"), name)
        if record["size_bytes"] != int(row.get("size_bytes", -1)):
            raise RuntimeError(f"aggregate NPY size mismatch: {name}")
        raw_rows[name] = record

    removable: list[dict[str, Any]] = []
    for name, record in sorted(raw_rows.items()):
        removable.append(
            {
                **record,
                "name": name,
                "relative_to_window": str((screening / name).relative_to(window)),
            }
        )
    return (
        screening_receipt_path,
        screening_receipt,
        analysis_path,
        analysis,
        removable,
    )


def verify_review_bundle(
    *,
    window: Path,
    screening_name: str,
    central_slice: int,
    review_manifest_path: Path,
) -> dict[str, Any]:
    manifest = read_json(review_manifest_path)
    if (
        manifest.get("kind") != "campaign_x_phase4_manual_visual_review_bundle_v2"
        or manifest.get("status") != "COMPLETE"
    ):
        raise RuntimeError("manual-review bundle is not complete")
    summaries = manifest.get("window_summaries")
    if (
        not isinstance(summaries, list)
        or sum(
            isinstance(row, dict) and row.get("window") == window.name
            for row in summaries
        )
        != 1
    ):
        raise RuntimeError("manual-review bundle does not bind this window once")

    required_suffixes = {
        f"{window.name}/tiffs/{central_slice:02d}.tif",
        (f"{window.name}/{screening_name}/" "INK_SCREENING_RECEIPT.json"),
        (f"{window.name}/{screening_name}/analysis/" "INK_STABILITY_ANALYSIS.json"),
    }
    observed: set[str] = set()
    for row in manifest.get("files", []):
        if not isinstance(row, dict):
            raise RuntimeError("manual-review file inventory contains a non-object")
        source = str(row.get("source", ""))
        matched = next(
            (suffix for suffix in required_suffixes if source.endswith(suffix)),
            None,
        )
        if matched is None:
            continue
        bundle_relative = Path(str(row.get("bundle_path", "")))
        if bundle_relative.is_absolute() or ".." in bundle_relative.parts:
            raise RuntimeError("unsafe path in manual-review bundle inventory")
        bundle_path = review_manifest_path.parent / bundle_relative
        record = require_hash(
            bundle_path,
            row.get("sha256"),
            f"manual-review copy {bundle_relative}",
        )
        if record["size_bytes"] != int(row.get("size_bytes", -1)):
            raise RuntimeError("manual-review bundle size mismatch")
        observed.add(matched)
    if observed != required_suffixes:
        raise RuntimeError(
            "manual-review bundle lacks central CT, screening receipt, or analysis"
        )
    return file_record(review_manifest_path)


def verify_features(
    *,
    window: Path,
    analysis_path: Path,
    crop_files: list[dict[str, Any]],
    central_slice: int,
    feature_receipt_path: Path,
) -> dict[str, Any]:
    receipt = read_json(feature_receipt_path)
    if (
        receipt.get("kind") != "campaign_x_phase4_ct_fiber_feature_benchmark_v1"
        or receipt.get("status") != FEATURE_STATUS
    ):
        raise RuntimeError("CT feature extraction receipt is not complete")
    sources = receipt.get("sources")
    matching = [
        row
        for row in sources or []
        if isinstance(row, dict) and row.get("group_id") == window.name
    ]
    if len(matching) != 1:
        raise RuntimeError("CT feature receipt does not bind this window once")
    source = matching[0]
    if source.get("analysis_sha256") != sha256_file(analysis_path):
        raise RuntimeError("CT feature receipt analysis hash mismatch")
    by_name = {
        (
            Path(row["path"]).name
            if "path" in row
            else Path(row["relative_to_window"]).name
        ): row
        for row in crop_files
    }
    names = sorted(by_name, key=lambda name: int(Path(name).stem))
    expected = {
        "first_tiff_sha256": by_name[names[0]]["sha256"],
        "central_tiff_sha256": by_name[f"{central_slice:02d}.tif"]["sha256"],
        "last_tiff_sha256": by_name[names[-1]]["sha256"],
    }
    for field, digest in expected.items():
        if source.get(field) != digest:
            raise RuntimeError(f"CT feature receipt {field} mismatch")

    artifact = receipt.get("artifacts", {}).get("csv")
    if not isinstance(artifact, str):
        raise RuntimeError("CT feature receipt lacks its CSV artifact")
    csv_path = feature_receipt_path.parent / artifact
    require_hash(
        csv_path,
        receipt.get("artifacts", {}).get("csv_sha256"),
        "CT feature CSV",
    )
    return file_record(feature_receipt_path)


def create_deterministic_archive(
    archive_path: Path,
    members: list[dict[str, Any]],
    *,
    compresslevel: int,
) -> dict[str, Any]:
    if archive_path.exists():
        # Covers the narrow crash gap after the atomic ZIP rename and before
        # the pending receipt is written.  Exact verified bytes are reusable;
        # anything else still fails closed.
        verify_archive(archive_path, members)
        return {
            "relative_to_window": str(
                archive_path.relative_to(archive_path.parent.parent)
            ),
            "sha256": sha256_file(archive_path),
            "size_bytes": archive_path.stat().st_size,
            "members": [
                {
                    "name": row["name"],
                    "sha256": row["sha256"],
                    "size_bytes": row["size_bytes"],
                }
                for row in sorted(members, key=lambda item: item["name"])
            ],
        }
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_name(f".{archive_path.name}.partial")
    if temporary.exists():
        raise RuntimeError(f"stale partial archive requires inspection: {temporary}")
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=compresslevel,
            strict_timestamps=True,
        ) as archive:
            for row in sorted(members, key=lambda item: item["name"]):
                source = Path(row["path"])
                info = zipfile.ZipInfo(row["name"], date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info._compresslevel = compresslevel
                info.external_attr = 0o100644 << 16
                with archive.open(info, "w", force_zip64=True) as destination:
                    with source.open("rb") as input_stream:
                        shutil.copyfileobj(
                            input_stream,
                            destination,
                            length=1024 * 1024,
                        )
        verify_archive(temporary, members)
        os.replace(temporary, archive_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    verify_archive(archive_path, members)
    return {
        "relative_to_window": str(archive_path.relative_to(archive_path.parent.parent)),
        "sha256": sha256_file(archive_path),
        "size_bytes": archive_path.stat().st_size,
        "members": [
            {
                "name": row["name"],
                "sha256": row["sha256"],
                "size_bytes": row["size_bytes"],
            }
            for row in sorted(members, key=lambda item: item["name"])
        ],
    }


def verify_archive(path: Path, members: list[dict[str, Any]]) -> None:
    expected = {row["name"]: row for row in members}
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            if len(infos) != len(expected):
                raise RuntimeError("archive member count mismatch")
            if len({info.filename for info in infos}) != len(infos):
                raise RuntimeError("archive contains duplicate members")
            for info in infos:
                if Path(info.filename).name != info.filename:
                    raise RuntimeError("archive contains an unsafe member name")
                row = expected.get(info.filename)
                if row is None or info.file_size != int(row["size_bytes"]):
                    raise RuntimeError(f"archive inventory mismatch: {info.filename}")
                digest = hashlib.sha256()
                observed = 0
                with archive.open(info, "r") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        observed += len(block)
                        digest.update(block)
                if (
                    observed != int(row["size_bytes"])
                    or digest.hexdigest() != row["sha256"]
                ):
                    raise RuntimeError(
                        f"archive member verification failed: {info.filename}"
                    )
    except (OSError, zipfile.BadZipFile) as error:
        raise RuntimeError(f"invalid evidence archive: {path}") from error


def build_audit(
    *,
    window: Path,
    screening_name: str,
    central_slice: int,
    review_manifest: Path,
    feature_receipt: Path,
) -> dict[str, Any]:
    (
        crop_path,
        crop,
        crop_files,
        removable_tiffs,
        source_lock,
    ) = verify_crop(window, central_slice=central_slice)
    (
        screening_receipt_path,
        screening_receipt,
        analysis_path,
        _analysis,
        removable_npys,
    ) = verify_screening(window, screening_name=screening_name)
    review_record = verify_review_bundle(
        window=window,
        screening_name=screening_name,
        central_slice=central_slice,
        review_manifest_path=review_manifest,
    )
    feature_record = verify_features(
        window=window,
        analysis_path=analysis_path,
        crop_files=crop_files,
        central_slice=central_slice,
        feature_receipt_path=feature_receipt,
    )
    archive_path = window / screening_name / ARCHIVE_NAME
    raw_npy_bytes = sum(int(row["size_bytes"]) for row in removable_npys)
    removable_tiff_bytes = sum(int(row["size_bytes"]) for row in removable_tiffs)
    retained = [
        file_record(crop_path, relative_to=window),
        file_record(screening_receipt_path, relative_to=window),
        file_record(analysis_path, relative_to=window),
        file_record(
            window / "tiffs" / f"{central_slice:02d}.tif",
            relative_to=window,
        ),
    ]
    return {
        "kind": COMPACTION_KIND,
        "status": "VALIDATED_PLAN_NO_MUTATION",
        "generated_at_utc": utc_now(),
        "window": str(window),
        "window_id": window.name,
        "screening_name": screening_name,
        "central_slice": central_slice,
        "prerequisites": {
            "crop_receipt": file_record(crop_path),
            "screening_receipt": file_record(screening_receipt_path),
            "analysis": file_record(analysis_path),
            "manual_review_manifest": review_record,
            "ct_feature_receipt": feature_record,
        },
        "source_render_lock": source_lock,
        "crop_recipe": {
            "box_left_top_right_bottom": crop["crop"]["box_left_top_right_bottom"],
            "source_mode": crop["input"]["mode"],
            "source_shape_y_x": crop["input"]["shape_y_x"],
            "output_artifacts": crop["artifacts"],
        },
        "retained_evidence": retained,
        "derived_cache_removal": {
            "noncentral_tiffs": removable_tiffs,
            "raw_npys": removable_npys,
            "uncompressed_bytes": raw_npy_bytes + removable_tiff_bytes,
        },
        "npy_archive": {
            "relative_to_window": str(archive_path.relative_to(window)),
            "planned_members": [
                {
                    "name": row["name"],
                    "sha256": row["sha256"],
                    "size_bytes": row["size_bytes"],
                }
                for row in removable_npys
            ],
        },
        "rehydration": {
            "policy": (
                "restore exact NPY bytes from the verified ZIP; regenerate "
                "TIFF crops from the source-render lock and accept each only "
                "when it matches the original crop receipt SHA-256"
            ),
            "source_render_must_remain_available": True,
        },
        "explicit_non_claims": [
            "not automatic ink acceptance",
            "not automatic letter acceptance",
            "not deletion of primary CT or TIFXYZ sources",
        ],
    }


def compact(audit: dict[str, Any], *, compresslevel: int) -> dict[str, Any]:
    window = Path(audit["window"])
    receipt_path = window / COMPACTION_RECEIPT_NAME
    archive_path = window / audit["npy_archive"]["relative_to_window"]
    npys = audit["derived_cache_removal"]["raw_npys"]
    archive_record = create_deterministic_archive(
        archive_path,
        npys,
        compresslevel=compresslevel,
    )
    pending = {
        **audit,
        "status": PENDING_STATUS,
        "npy_archive": archive_record,
        "archive_verified_at_utc": utc_now(),
    }
    atomic_write_json(receipt_path, pending)

    for collection in (
        pending["derived_cache_removal"]["raw_npys"],
        pending["derived_cache_removal"]["noncentral_tiffs"],
    ):
        for row in collection:
            path = window / row["relative_to_window"]
            if path.is_file():
                require_hash(path, row["sha256"], "derived cache before removal")
                path.unlink()

    for row in pending["derived_cache_removal"]["raw_npys"]:
        if (window / row["relative_to_window"]).exists():
            raise RuntimeError("raw NPY cache remained after compaction")
    for row in pending["derived_cache_removal"]["noncentral_tiffs"]:
        if (window / row["relative_to_window"]).exists():
            raise RuntimeError("non-central TIFF cache remained after compaction")
    verify_archive(archive_path, npys)
    for row in pending["retained_evidence"]:
        require_hash(window / row["path"], row["sha256"], "retained evidence")

    complete = {
        **pending,
        "status": COMPACTION_COMPLETE_STATUS,
        "completed_at_utc": utc_now(),
        "storage": {
            "archive_bytes": archive_path.stat().st_size,
            "removed_raw_npy_bytes": sum(int(row["size_bytes"]) for row in npys),
            "removed_noncentral_tiff_bytes": sum(
                int(row["size_bytes"])
                for row in pending["derived_cache_removal"]["noncentral_tiffs"]
            ),
            "net_reclaimed_bytes": (
                sum(int(row["size_bytes"]) for row in npys)
                + sum(
                    int(row["size_bytes"])
                    for row in pending["derived_cache_removal"]["noncentral_tiffs"]
                )
                - archive_path.stat().st_size
            ),
        },
    }
    atomic_write_json(receipt_path, complete)
    return complete


def safe_window_member(window: Path, relative: Any) -> Path:
    candidate = Path(str(relative))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError("unsafe path in compaction receipt")
    resolved = (window / candidate).resolve()
    try:
        resolved.relative_to(window.resolve())
    except ValueError as error:
        raise RuntimeError("compaction path escapes robust window") from error
    return resolved


def receipt_window(receipt_path: Path, receipt: dict[str, Any]) -> Path:
    """Resolve a portable receipt against its current enclosing window.

    ``window`` remains historical provenance.  The operational binding is the
    receipt's current parent plus the frozen ``window_id`` basename.  This
    permits restoring a compact window under another workspace root without
    weakening its identity check.
    """

    window = receipt_path.parent.resolve()
    window_id = str(receipt.get("window_id", "")).strip()
    if not window_id or Path(window_id).name != window_id:
        raise RuntimeError("compaction receipt has an unsafe window_id")
    if window.name != window_id:
        raise RuntimeError(
            "compaction receipt window_id differs from its current directory"
        )
    return window


def validate_completed_foundation(
    receipt_path: Path,
    receipt: dict[str, Any],
) -> tuple[Path, Path, list[dict[str, Any]]]:
    """Verify immutable archive and retained evidence, allowing raw caches."""

    window = receipt_window(receipt_path, receipt)
    archive_path = safe_window_member(
        window,
        receipt.get("npy_archive", {}).get("relative_to_window"),
    )
    if sha256_file(archive_path) != receipt["npy_archive"]["sha256"]:
        raise RuntimeError("completed compaction archive hash mismatch")
    npys = receipt["derived_cache_removal"]["raw_npys"]
    verify_archive(archive_path, npys)
    for row in receipt["retained_evidence"]:
        require_hash(
            safe_window_member(window, row["path"]),
            row["sha256"],
            "retained evidence",
        )
    return window, archive_path, npys


def verify_completed_receipt(
    receipt_path: Path,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    window, _archive_path, npys = validate_completed_foundation(
        receipt_path,
        receipt,
    )
    for collection in (
        npys,
        receipt["derived_cache_removal"]["noncentral_tiffs"],
    ):
        for row in collection:
            if safe_window_member(window, row["relative_to_window"]).exists():
                raise RuntimeError(
                    "completed compaction still contains selected raw cache"
                )
    return receipt


def recompact_completed(
    receipt_path: Path,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    """Remove an exact rehydrated cache while retaining the frozen receipt."""

    window, _archive_path, npys = validate_completed_foundation(
        receipt_path,
        receipt,
    )
    present: list[tuple[Path, dict[str, Any]]] = []
    for collection in (
        npys,
        receipt["derived_cache_removal"]["noncentral_tiffs"],
    ):
        for row in collection:
            path = safe_window_member(window, row["relative_to_window"])
            if path.exists() and not path.is_file():
                raise RuntimeError(f"rehydrated cache path is not a file: {path}")
            if path.is_file():
                # Validate the complete set before removing the first byte.
                require_hash(path, row["sha256"], "rehydrated derived cache")
                present.append((path, row))
    for path, _row in present:
        path.unlink()
    for path, _row in present:
        if path.exists():
            raise RuntimeError("rehydrated cache remained after recompaction")
    verify_completed_receipt(receipt_path, receipt)
    return {
        **receipt,
        "operation": {
            "status": "RECOMPACTED_EXACT_REHYDRATED_CACHES",
            "completed_at_utc": utc_now(),
            "removed_file_count": len(present),
            "removed_bytes": sum(int(row["size_bytes"]) for _path, row in present),
            "receipt_rewritten": False,
        },
    }


def resume_compaction(receipt_path: Path) -> dict[str, Any]:
    receipt = read_json(receipt_path)
    if receipt.get("kind") != COMPACTION_KIND:
        raise RuntimeError("unexpected compaction receipt kind")
    if receipt.get("status") == COMPACTION_COMPLETE_STATUS:
        return recompact_completed(receipt_path, receipt)
    if receipt.get("status") != PENDING_STATUS:
        raise RuntimeError("existing compaction receipt is not resumable")
    window = receipt_window(receipt_path, receipt)
    archive_path = safe_window_member(
        window,
        receipt["npy_archive"]["relative_to_window"],
    )
    if sha256_file(archive_path) != receipt["npy_archive"]["sha256"]:
        raise RuntimeError("pending compaction archive hash mismatch")
    npys = receipt["derived_cache_removal"]["raw_npys"]
    verify_archive(archive_path, npys)
    for collection in (
        npys,
        receipt["derived_cache_removal"]["noncentral_tiffs"],
    ):
        for row in collection:
            path = safe_window_member(window, row["relative_to_window"])
            if path.is_file():
                require_hash(path, row["sha256"], "resumed derived cache")
                path.unlink()
    for collection in (
        npys,
        receipt["derived_cache_removal"]["noncentral_tiffs"],
    ):
        for row in collection:
            if safe_window_member(window, row["relative_to_window"]).exists():
                raise RuntimeError("resumed derived cache remained after removal")
    verify_archive(archive_path, npys)
    for row in receipt["retained_evidence"]:
        require_hash(
            safe_window_member(window, row["path"]),
            row["sha256"],
            "retained evidence",
        )
    archive_bytes = archive_path.stat().st_size
    complete = {
        **receipt,
        "status": COMPACTION_COMPLETE_STATUS,
        "completed_at_utc": utc_now(),
        "storage": {
            "archive_bytes": archive_bytes,
            "removed_raw_npy_bytes": sum(int(row["size_bytes"]) for row in npys),
            "removed_noncentral_tiff_bytes": sum(
                int(row["size_bytes"])
                for row in receipt["derived_cache_removal"]["noncentral_tiffs"]
            ),
            "net_reclaimed_bytes": (
                sum(int(row["size_bytes"]) for row in npys)
                + sum(
                    int(row["size_bytes"])
                    for row in receipt["derived_cache_removal"]["noncentral_tiffs"]
                )
                - archive_bytes
            ),
        },
    }
    atomic_write_json(receipt_path, complete)
    return complete


def resolve_source_render(
    receipt: dict[str, Any],
    *,
    source_render_dir: Path | None,
) -> dict[str, Path]:
    rows = receipt.get("source_render_lock", {}).get("files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("compaction receipt lacks source-render files")
    expected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("source-render lock contains a non-object row")
        original = Path(str(row.get("path", "")))
        name = original.name
        if not name or name in expected:
            raise RuntimeError("source-render lock has duplicate basenames")
        expected[name] = row

    if source_render_dir is None:
        frozen_directory = Path(
            str(receipt.get("source_render_lock", {}).get("directory", ""))
        )
        directory = frozen_directory.resolve()
    else:
        directory = source_render_dir.resolve()
    if not directory.is_dir():
        raise RuntimeError(f"source render directory is missing: {directory}")
    actual = {path.name: path for path in directory.glob("*.tif") if path.is_file()}
    if set(actual) != set(expected):
        raise RuntimeError(
            "relocated source render does not contain the exact frozen TIFF set"
        )
    for name, row in expected.items():
        record = require_hash(actual[name], row.get("sha256"), "source render")
        if record["size_bytes"] != int(row.get("size_bytes", -1)):
            raise RuntimeError(f"source render size mismatch: {name}")
    return actual


def rehydrate(
    receipt_path: Path,
    *,
    source_render_dir: Path | None = None,
) -> dict[str, Any]:
    receipt = read_json(receipt_path)
    if (
        receipt.get("kind") != COMPACTION_KIND
        or receipt.get("status") != COMPACTION_COMPLETE_STATUS
    ):
        raise RuntimeError("only a completed compaction can be rehydrated")
    window = receipt_window(receipt_path, receipt)
    archive_path = safe_window_member(
        window,
        receipt["npy_archive"]["relative_to_window"],
    )
    npys = receipt["derived_cache_removal"]["raw_npys"]
    if sha256_file(archive_path) != receipt["npy_archive"]["sha256"]:
        raise RuntimeError("evidence archive hash mismatch")
    verify_archive(archive_path, npys)
    source_by_name = resolve_source_render(
        receipt,
        source_render_dir=source_render_dir,
    )

    with zipfile.ZipFile(archive_path, "r") as archive:
        for row in npys:
            destination = safe_window_member(
                window,
                row["relative_to_window"],
            )
            if destination.is_file():
                require_hash(destination, row["sha256"], "rehydrated NPY")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.rehydrate")
            if temporary.exists():
                raise RuntimeError(f"stale rehydration file: {temporary}")
            with archive.open(row["name"], "r") as source:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            require_hash(temporary, row["sha256"], "restored NPY")
            os.replace(temporary, destination)

    box = tuple(
        int(value) for value in receipt["crop_recipe"]["box_left_top_right_bottom"]
    )
    artifact_rows = {
        row["name"]: row for row in receipt["crop_recipe"]["output_artifacts"]
    }
    tiff_dir = window / "tiffs"
    for name, row in sorted(artifact_rows.items()):
        destination = tiff_dir / name
        if destination.is_file():
            require_hash(destination, row["sha256"], "rehydrated TIFF")
            continue
        source = source_by_name.get(name)
        if source is None:
            raise RuntimeError(f"source render lacks {name}")
        temporary = destination.with_name(f".{destination.name}.rehydrate")
        if temporary.exists():
            raise RuntimeError(f"stale rehydration file: {temporary}")
        with Image.open(source) as image:
            image.crop(box).save(temporary, format="TIFF", compression="tiff_lzw")
        require_hash(temporary, row["sha256"], "regenerated TIFF")
        os.replace(temporary, destination)

    return {
        "status": "REHYDRATED_EXACT_CACHE_BYTES",
        "window": str(window),
        "historical_window": str(receipt.get("window", "")),
        "source_render_directory": str(next(iter(source_by_name.values())).parent),
        "source_render_override_used": source_render_dir is not None,
        "restored_npy_count": len(npys),
        "restored_tiff_count": len(artifact_rows),
        "compaction_receipt_sha256": sha256_file(receipt_path),
    }


def summary(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("status") == "VALIDATED_PLAN_NO_MUTATION":
        removal = value["derived_cache_removal"]
        return {
            "status": value["status"],
            "window": value["window"],
            "raw_npy_count": len(removal["raw_npys"]),
            "noncentral_tiff_count": len(removal["noncentral_tiffs"]),
            "candidate_reclaim_bytes_before_archive": removal["uncompressed_bytes"],
        }
    operation = value.get("operation", {})
    if operation:
        return {
            "status": operation.get("status"),
            "window": value.get("window"),
            "removed_file_count": operation.get("removed_file_count"),
            "removed_bytes": operation.get("removed_bytes"),
            "receipt_rewritten": operation.get("receipt_rewritten"),
        }
    storage = value.get("storage", {})
    return {
        "status": value.get("status"),
        "window": value.get("window"),
        "archive_bytes": storage.get("archive_bytes"),
        "net_reclaimed_bytes": storage.get("net_reclaimed_bytes"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--screening-name", required=True)
    parser.add_argument("--review-manifest", type=Path)
    parser.add_argument("--feature-receipt", type=Path)
    parser.add_argument(
        "--source-render-dir",
        type=Path,
        help=(
            "Explicit relocated source TIFF directory for rehydrate; every "
            "basename, size and SHA-256 must match the frozen source lock."
        ),
    )
    parser.add_argument("--central-slice", type=int, default=32)
    parser.add_argument(
        "--mode",
        choices=("plan", "compact", "rehydrate"),
        default="plan",
    )
    parser.add_argument("--compresslevel", type=int, default=6)
    args = parser.parse_args()

    window = args.window.resolve()
    if not window.is_dir():
        raise RuntimeError(f"robust window is missing: {window}")
    if not args.screening_name or Path(args.screening_name).name != args.screening_name:
        raise RuntimeError("screening name must be one safe path component")
    if not 0 <= args.central_slice <= 999:
        raise RuntimeError("central slice is outside the supported range")
    if not 0 <= args.compresslevel <= 9:
        raise RuntimeError("compresslevel must be between 0 and 9")
    receipt_path = window / COMPACTION_RECEIPT_NAME

    if args.mode == "rehydrate":
        result = rehydrate(
            receipt_path,
            source_render_dir=(
                args.source_render_dir.resolve()
                if args.source_render_dir is not None
                else None
            ),
        )
    elif args.mode == "compact" and receipt_path.exists():
        result = resume_compaction(receipt_path)
    else:
        if args.review_manifest is None or args.feature_receipt is None:
            raise RuntimeError(
                "plan/compact require --review-manifest and --feature-receipt"
            )
        audit = build_audit(
            window=window,
            screening_name=args.screening_name,
            central_slice=args.central_slice,
            review_manifest=args.review_manifest.resolve(),
            feature_receipt=args.feature_receipt.resolve(),
        )
        result = (
            compact(audit, compresslevel=args.compresslevel)
            if args.mode == "compact"
            else audit
        )
    print(json.dumps(summary(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
