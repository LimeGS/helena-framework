#!/usr/bin/env python3
"""Fetch/rebuild the fixtures a fresh clone needs to run the pinned tests.

- fixtures/out_v2/ (the frozen GPU predictions) ships committed in this
  repo: nothing to fetch.
- fixtures/band_r1145_200_xyz.npz (48 MB) is NOT committed; it rebuilds
  BIT-EXACTLY from the public segment PPM (~247 MB of HTTP range requests,
  ~2 min) via the vendored scripts/build_reference_band.py, and this script
  verifies the known SHA-256 before declaring success.

Run once after cloning:  python fetch_fixtures.py
"""
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BAND = os.path.join(HERE, "fixtures", "band_r1145_200_xyz.npz")
SHA = "a93b0683df0e3a03dc656f8a259c2d83c5949215a46fec6b4655c70d660d4408"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if os.path.exists(BAND) and sha256(BAND) == SHA:
        print("band ya presente y verificada")
    else:
        print("rebuilding the band from the public PPM (~247 MB over the network)...")
        subprocess.run([sys.executable,
                        os.path.join(HERE, "scripts", "build_reference_band.py"),
                        "--out", BAND], check=False)
        # our SHA-256 check below is the authoritative verification; the
        # vendored builder may fail on post-steps tied to its home repo
        assert os.path.exists(BAND), "builder produced no band file"
        got = sha256(BAND)
        assert got == SHA, f"SHA-256 mismatch: {got}"
        print("banda reconstruida bit-exacta, SHA-256 verificada")
    out_v2 = os.path.join(HERE, "fixtures", "out_v2")
    n = len([d for d in os.listdir(out_v2) if d.startswith("seed_v2")])
    assert n == 8, f"fixtures/out_v2 incompleto: {n}/8"
    print("out_v2: 8/8 presentes. Fixtures listos: python -m pytest tests/ -q")


if __name__ == "__main__":
    main()
