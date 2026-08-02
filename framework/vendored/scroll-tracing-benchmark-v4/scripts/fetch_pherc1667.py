#!/usr/bin/env python3
"""Rebuild the ignored PHerc1667 w013 band from public tifxyz files."""
import hashlib
import json
import urllib.request
from pathlib import Path

from stb.build_band import build_band

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "pherc1667.json"
OUT = ROOT / "data" / "pherc1667" / "band_w013_r1400_1600_xyz.npz"
EXPECTED_SHA256 = "45250db6c5e08e515acdd392bab660c26d8c9f3e54a80ef8d5bc27d80d6da63e"
BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    cfg = json.loads(CONFIG.read_text())
    recipe = cfg["band"]["recipe"]
    cache = ROOT / "data" / "pherc1667" / "source_tifxyz"
    cache.mkdir(parents=True, exist_ok=True)
    prefix = BUCKET + recipe["source_mesh"].rstrip("/") + "/"
    for axis in "xyz":
        destination = cache / f"{axis}.tif"
        if not destination.exists():
            print(f"downloading {prefix}{axis}.tif")
            urllib.request.urlretrieve(prefix + f"{axis}.tif", destination)
    build_band(
        cache, recipe["row_start"], recipe["row_end"], OUT,
        recipe["col_start"], recipe["col_end"], load_z=True,
    )
    got = sha256(OUT)
    if got != EXPECTED_SHA256:
        raise RuntimeError(f"PHerc1667 band hash mismatch: {got}")
    print(f"verified {OUT}: {got}")


if __name__ == "__main__":
    main()
