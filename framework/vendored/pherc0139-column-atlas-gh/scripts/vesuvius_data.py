"""Vesuvius Challenge data access + segment loaders (Phase 0).

Anonymous access to the AWS Open Data mirror (no credentials needed):
    s3://vesuvius-challenge-open-data  (us-east-1)

Handles both segment formats:
  - modern (AWS mirror):  TIFXYZ mesh (meta.json + x/y/z.tif) + Zarr surface-volume
  - classic (dl.ash2txt): .ppm per-pixel map + layers/NN.tif stack

ponytail: PIL reads 16-bit TIF / float TIF / PNG / JPG, so no tifffile/zarr dep
for Phase 0. Zarr *voxel* decode is deferred to Phase 1 (inference) — we only
read its .zattrs metadata here.
"""
from __future__ import annotations

import io
import json
import os

import numpy as np
from PIL import Image

BUCKET = "vesuvius-challenge-open-data"
REGION = "us-east-1"
HTTP_BASE = f"https://{BUCKET}.s3.{REGION}.amazonaws.com"


# --- S3 (anonymous) ---------------------------------------------------------

def _client():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client("s3", region_name=REGION, config=Config(signature_version=UNSIGNED))


def ls(prefix: str, delimiter: str = "/"):
    """List one level under `prefix`. Returns (subprefixes, [(key, size), ...])."""
    cli = _client()
    subprefixes, keys = [], []
    for page in cli.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=prefix, Delimiter=delimiter
    ):
        subprefixes += [p["Prefix"] for p in page.get("CommonPrefixes", [])]
        keys += [(o["Key"], o["Size"]) for o in page.get("Contents", [])]
    return subprefixes, keys


def download(key: str, dest: str) -> str:
    """Download an object to `dest`. Skips if a same-size file already exists."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    cli = _client()
    if os.path.exists(dest):
        remote = cli.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
        if os.path.getsize(dest) == remote:
            return dest  # ponytail: naive size-only cache check, good enough
    cli.download_file(BUCKET, key, dest)
    return dest


# --- image loading ----------------------------------------------------------

def read_image(path: str) -> np.ndarray:
    """Load a TIF/PNG/JPG into a numpy array. Handles 16-bit and float TIFs."""
    try:
        return np.array(Image.open(path))
    except Exception:
        import tifffile  # only needed if PIL can't decode this TIF
        return tifffile.imread(path)


# --- modern format: TIFXYZ (the .ppm replacement) ---------------------------

def read_tifxyz(dirpath: str) -> dict:
    """Load a TIFXYZ mesh dir (meta.json + x.tif + y.tif + z.tif).

    Returns dict with:
      coords : (H, W, 3) float32   volume coordinate (x, y, z) per flat pixel
      valid  : (H, W)   bool        pixels that map to a real surface point
      bbox, scale, meta            from meta.json
    """
    with open(os.path.join(dirpath, "meta.json")) as f:
        meta = json.load(f)
    x = read_image(os.path.join(dirpath, "x.tif")).astype(np.float32)
    y = read_image(os.path.join(dirpath, "y.tif")).astype(np.float32)
    z = read_image(os.path.join(dirpath, "z.tif")).astype(np.float32)
    coords = np.stack([x, y, z], axis=-1)
    # no-surface pixels are sentinel-marked (-1, or 0); real coords are all > 0
    valid = (coords > 0).all(axis=-1)
    return {"coords": coords, "valid": valid,
            "bbox": meta.get("bbox"), "scale": meta.get("scale"), "meta": meta}


def read_zarr_meta(path: str) -> dict:
    """Read an OME-Zarr .zattrs (metadata only, no voxels).

    Accepts either the zarr dir (joins .zattrs) or a direct .zattrs file path.
    """
    if os.path.isdir(path):
        path = os.path.join(path, ".zattrs")
    with open(path) as f:
        return json.load(f)


# --- classic format: .ppm per-pixel map -------------------------------------

_PPM_DTYPES = {"double": np.float64, "float": np.float32}


def parse_ppm(path: str) -> np.ndarray:
    """Parse a classic Volume Cartographer .ppm (per-pixel map).

    Text header (width/height/dim/type ...) then `<>` then row-major binary.
    Returns (H, W, dim) array; dim is usually 6 = (x,y,z, nx,ny,nz).
    """
    with open(path, "rb") as f:
        raw = f.read()
    sep = raw.find(b"<>\n")
    if sep == -1:
        raise ValueError("no '<>' header terminator — not a PPM per-pixel map")
    header = raw[:sep].decode("ascii")
    fields = {}
    for line in header.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip()] = v.strip()
    w, h, dim = int(fields["width"]), int(fields["height"]), int(fields["dim"])
    dtype = _PPM_DTYPES[fields.get("type", "double")]
    data = np.frombuffer(raw[sep + 3:], dtype=dtype, count=w * h * dim)
    return data.reshape(h, w, dim)


def write_ppm(path: str, arr: np.ndarray, dtype: str = "double") -> None:
    """Write a (H, W, dim) array as a .ppm per-pixel map (used by the self-check)."""
    h, w, dim = arr.shape
    header = (f"width: {w}\nheight: {h}\ndim: {dim}\n"
              f"ordered: true\ntype: {dtype}\nversion: 1\n<>\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.astype(_PPM_DTYPES[dtype]).tobytes())
