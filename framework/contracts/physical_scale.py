"""Resolve the physical scales of an ink screening run from frozen sources.

FIX-09 removed the hardcoded 7.91 um training scale and the 9.362 um source
default, because the campaign spans 8.64 and 9.362 um acquisitions and either
default silently rescales the other cohort by 8.4%. The resolution logic first
lived in ``analyze_ink_stability``, so ``run_ink_timesformer``
imported it from there -- and inherited that module's ``scipy`` dependency.

That broke the runtime contract of the project's own pinned ink image, whose
``requirements.ink.txt`` deliberately lists only numpy, Pillow,
timesformer-pytorch, einops and safetensors. The screening runner stopped
working inside the very container built to run it.

Nothing here needs an array library: it is JSON, paths and one comparison. It
lives in ``framework/contracts`` so both consumers share one definition without
either dragging the other's dependencies into an image.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

#: Disagreement beyond this between a supplied value and a frozen one is an
#: error, never a silent rescale.
PIXEL_UM_TOLERANCE = 1e-6

_CONFIGURED_ROOT = os.environ.get("HELENA_REPO_ROOT", "").strip()
REPO_ROOT = (
    Path(_CONFIGURED_ROOT).resolve()
    if _CONFIGURED_ROOT
    else Path(__file__).resolve().parents[2]
)

DEFAULT_VOLUME_CATALOG = REPO_ROOT / "workspace" / "catalog" / "eligible_volumes.json"
DEFAULT_INK_PROFILE = (
    REPO_ROOT
    / "framework"
    / "profiles"
    / "03-ink"
    / "timesformer-gp-scroll1-screening-1.0.0.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def catalog_voxel_size_um(catalog_path: Path, sample_id: str) -> float | None:
    """Return the frozen voxel size for a sample, or None if uncatalogued."""

    catalog_path = Path(catalog_path)
    if not catalog_path.is_file():
        raise RuntimeError(f"volume catalog is missing: {catalog_path}")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    entries = catalog.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"volume catalog has no entries: {catalog_path}")
    matches = {
        float(entry["voxel_size_um"])
        for entry in entries
        if isinstance(entry, dict) and str(entry.get("sample_id")) == sample_id
    }
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(
            f"volume catalog gives {sample_id} more than one voxel size: {matches}"
        )
    return matches.pop()


def resolve_source_pixel_um(
    *,
    sample_id: str,
    catalog_path: Path,
    requested: float | None,
) -> tuple[float, dict[str, Any]]:
    """Resolve the source pixel size, failing closed on any CLI disagreement."""

    catalog_path = Path(catalog_path)
    catalogued = catalog_voxel_size_um(catalog_path, sample_id)
    if catalogued is None:
        if requested is None:
            raise RuntimeError(
                f"{sample_id} is not in {catalog_path} and no --source-pixel-um "
                "was supplied; refusing to guess the physical scale"
            )
        return float(requested), {
            "source": "CLI_UNCATALOGUED_SAMPLE",
            "catalog_path": str(catalog_path),
            "catalog_sha256": sha256_file(catalog_path),
            "catalog_voxel_size_um": None,
            "cli_source_pixel_um": float(requested),
        }
    if requested is not None and abs(float(requested) - catalogued) > PIXEL_UM_TOLERANCE:
        raise RuntimeError(
            f"--source-pixel-um {requested} disagrees with the frozen catalog "
            f"value {catalogued} for {sample_id} (tolerance {PIXEL_UM_TOLERANCE}); "
            "refusing to rescale silently"
        )
    return catalogued, {
        "source": "ELIGIBLE_VOLUMES_CATALOG",
        "catalog_path": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "catalog_voxel_size_um": catalogued,
        "cli_source_pixel_um": None if requested is None else float(requested),
    }


def resolve_training_pixel_um(
    *,
    profile_path: Path,
    requested: float | None,
) -> tuple[float, dict[str, Any]]:
    """Resolve the training pixel size from the frozen ink lane profile."""

    profile_path = Path(profile_path)
    if not profile_path.is_file():
        raise RuntimeError(f"ink lane profile is missing: {profile_path}")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    contract = profile.get("input_contract")
    if not isinstance(contract, dict) or "training_pixel_um" not in contract:
        raise RuntimeError(
            f"ink lane profile declares no training_pixel_um: {profile_path}"
        )
    declared = float(contract["training_pixel_um"])
    if requested is not None and abs(float(requested) - declared) > PIXEL_UM_TOLERANCE:
        raise RuntimeError(
            f"--training-pixel-um {requested} disagrees with the ink lane profile "
            f"value {declared} ({profile.get('profile_id')}); refusing to rescale "
            "silently"
        )
    return declared, {
        "source": "INK_LANE_PROFILE",
        "profile_path": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "profile_id": profile.get("profile_id"),
        "profile_training_pixel_um": declared,
        "cli_training_pixel_um": None if requested is None else float(requested),
    }
