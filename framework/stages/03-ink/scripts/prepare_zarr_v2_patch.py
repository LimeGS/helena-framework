#!/usr/bin/env python3
"""Extract one immutable NPY patch from an uncompressed Zarr v2 HTTP array."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, *, allow_missing: bool = False) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if allow_missing and exc.code == 404:
            return None
        raise RuntimeError(f"HTTP fetch failed ({exc.code}): {url}") from exc


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def extract_patch(
    *,
    base_url: str,
    array_path: str,
    center_xyz: tuple[int, int, int],
    shape_zyx: tuple[int, int, int],
    allow_missing_chunks: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    array_url = f"{base_url.rstrip('/')}/{array_path.strip('/')}"
    raw_metadata = fetch(f"{array_url}/.zarray")
    if raw_metadata is None:
        raise RuntimeError("missing .zarray metadata")
    metadata = json.loads(raw_metadata)
    if metadata.get("zarr_format") != 2:
        raise RuntimeError("only Zarr v2 arrays are supported")
    if metadata.get("compressor") is not None or metadata.get("filters") is not None:
        raise RuntimeError("only uncompressed, unfiltered Zarr v2 arrays are supported")
    if metadata.get("order") != "C" or metadata.get("dimension_separator") != "/":
        raise RuntimeError("array must use C order and '/' dimension separator")
    source_shape = tuple(int(value) for value in metadata["shape"])
    chunk_shape = tuple(int(value) for value in metadata["chunks"])
    if len(source_shape) != 3 or len(chunk_shape) != 3:
        raise RuntimeError("source array must be 3D Z,Y,X")
    dtype = np.dtype(metadata["dtype"])
    fill_value = metadata.get("fill_value") or 0

    center_zyx = (center_xyz[2], center_xyz[1], center_xyz[0])
    starts = tuple(center - size // 2 for center, size in zip(center_zyx, shape_zyx))
    stops = tuple(start + size for start, size in zip(starts, shape_zyx))
    if any(start < 0 or stop > limit for start, stop, limit in zip(starts, stops, source_shape)):
        raise RuntimeError("requested patch extends beyond the source array")

    output = np.full(shape_zyx, fill_value, dtype=dtype)
    fetched_chunks = 0
    missing_chunks = 0
    chunk_ranges = [range(start // chunk, (stop - 1) // chunk + 1) for start, stop, chunk in zip(starts, stops, chunk_shape)]
    for chunk_z in chunk_ranges[0]:
        for chunk_y in chunk_ranges[1]:
            for chunk_x in chunk_ranges[2]:
                chunk_indices = (chunk_z, chunk_y, chunk_x)
                chunk_starts = tuple(index * size for index, size in zip(chunk_indices, chunk_shape))
                actual_shape = tuple(min(size, limit - start) for size, limit, start in zip(chunk_shape, source_shape, chunk_starts))
                raw = fetch(
                    f"{array_url}/{chunk_z}/{chunk_y}/{chunk_x}",
                    allow_missing=allow_missing_chunks,
                )
                if raw is None:
                    missing_chunks += 1
                    continue
                expected_bytes = int(np.prod(actual_shape)) * dtype.itemsize
                if len(raw) != expected_bytes:
                    raise RuntimeError(
                        f"chunk byte count mismatch for {chunk_indices}: {len(raw)} != {expected_bytes}"
                    )
                chunk_array = np.frombuffer(raw, dtype=dtype).reshape(actual_shape, order="C")
                source_slices = []
                output_slices = []
                for axis in range(3):
                    overlap_start = max(starts[axis], chunk_starts[axis])
                    overlap_stop = min(stops[axis], chunk_starts[axis] + actual_shape[axis])
                    source_slices.append(slice(overlap_start - chunk_starts[axis], overlap_stop - chunk_starts[axis]))
                    output_slices.append(slice(overlap_start - starts[axis], overlap_stop - starts[axis]))
                output[tuple(output_slices)] = chunk_array[tuple(source_slices)]
                fetched_chunks += 1
    return output, {
        "source_zarray": metadata,
        "bounds_zyx": [[starts[i], stops[i]] for i in range(3)],
        "chunks_fetched": fetched_chunks,
        "chunks_missing_filled": missing_chunks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--array-path", default="0")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--voxel-size-um", type=float, required=True)
    parser.add_argument("--center-xyz", nargs=3, type=int, required=True)
    parser.add_argument("--shape-zyx", nargs=3, type=int, default=(256, 256, 256))
    parser.add_argument(
        "--allow-missing-chunks",
        action="store_true",
        help="Fill HTTP 404 chunks with the Zarr fill value; off by default.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite existing array: {args.output}")
    patch, details = extract_patch(
        base_url=args.base_url,
        array_path=args.array_path,
        center_xyz=tuple(args.center_xyz),
        shape_zyx=tuple(args.shape_zyx),
        allow_missing_chunks=args.allow_missing_chunks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    with temporary.open("wb") as handle:
        np.save(handle, patch, allow_pickle=False)
    os.replace(temporary, args.output)
    manifest = {
        "schema": "campaignx.ink_volumetric_patch_input.v1",
        "sample_id": args.sample_id,
        "array": {
            "path": os.path.relpath(args.output.resolve(), args.manifest.parent.resolve()),
            "sha256": sha256_file(args.output),
            "format": "npy",
            "dtype": str(patch.dtype),
            "shape_zyx": list(patch.shape),
        },
        "input_voxel_size_um": args.voxel_size_um,
        "source_volume": {
            "uri": args.base_url,
            "array_path": args.array_path,
            "voxel_size_um": args.voxel_size_um,
        },
        "extraction": {
            "center_xyz": args.center_xyz,
            "bounds_zyx": details["bounds_zyx"],
            "resampled": False,
            "chunks_fetched": details["chunks_fetched"],
            "chunks_missing_filled": details["chunks_missing_filled"],
            "missing_chunks_explicitly_allowed": args.allow_missing_chunks,
            "source_zarray": details["source_zarray"],
        },
    }
    write_new_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
