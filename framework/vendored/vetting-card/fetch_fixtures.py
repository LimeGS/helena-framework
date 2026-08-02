#!/usr/bin/env python3
"""fetch_fixtures.py -- materialize the fixture windows listed in a
manifest.json into fixtures/cache/, stamping real sha256 hashes back into
the manifest in place.

    python fetch_fixtures.py --manifest fixtures/manifest.json \\
        [--cache-dir fixtures/cache] [--dry-run] \\
        [--local-source-root /path/to/parent/project]

Two kinds of entries are handled, matching the two provenance situations
in the shipped manifest (see README "Fixture manifest"):

1. provenance.type == "url" and fetchable_now == true: the source raster
   is downloaded from its public URL into <cache>/sources/, sha256-hashed,
   optionally checked against provenance.expected_size_bytes, then the
   entry's region rows/cols slice is cut and saved as <cache>/<id>.npy.
   The entry is stamped status=FETCHED_VERIFIED with the sha256 of both
   the downloaded source file and the produced fixture.

2. provenance.type == "local_render_from_public_inputs": the source
   raster was rendered locally from public inputs and has no public URL
   (the manifest documents the full recipe and pins the exact reviewed
   array bytes in content_pin). These entries are always SKIPPED for
   network fetch. If --local-source-root points at a checkout of the
   parent project that still has the source raster, the window is sliced
   from it, verified against content_pin.array_sha256, and saved to
   <cache>/<id>.npy -- the entry gets a "local_verification" block but
   its status stays NOT_PUBLICLY_FETCHABLE, because an outside claimant
   cannot reproduce this step by download. Never a substitute for a
   public URL; it exists so the pinned bytes can be re-checked and so
   vet_pipeline.py can run against the real reviewed windows on machines
   that do have the parent project.

Never fabricates a checksum: every sha256 written by this script is
computed from bytes it actually read (downloaded or sliced) in that run.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

# Fixture source rasters are legitimately gigapixel-scale (the Scroll 1
# ink TIF is 80880x15600); PIL's decompression-bomb guard must not block
# a file we chose on purpose and verify by size and hash.
Image.MAX_IMAGE_PIXELS = None


class FetchFixturesError(Exception):
    pass


def _today() -> str:
    return _dt.date.today().isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def classify(entry: dict) -> tuple[bool, str]:
    """Return (can_fetch, reason) for the NETWORK fetch path. Pure
    function, no I/O -- deliberately kept side-effect-free so this
    decision can be unit-tested without any network access.
    """
    provenance = entry.get("provenance", {})
    if not entry.get("fetchable_now"):
        reason = provenance.get("detail") or "'fetchable_now' is false in the manifest"
        return False, reason
    if provenance.get("type") != "url" or not provenance.get("url"):
        return False, "provenance.type is not 'url' (no downloadable source URL present)"
    return True, "ok"


def load_raster(path: Path) -> np.ndarray:
    """Load a source raster (.npy, or anything PIL can open: tif/png/jpg)
    as a 2D numpy array, dtype preserved where possible."""
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
    else:
        arr = np.array(Image.open(path))
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim != 2:
        raise FetchFixturesError(f"expected a 2D raster from {path.name}, got shape {arr.shape}")
    return arr


def slice_region(arr: np.ndarray, entry: dict) -> np.ndarray:
    region = entry.get("region", {})
    rows, cols = region.get("rows"), region.get("cols")
    if not rows or not cols:
        raise FetchFixturesError(f"entry {entry.get('id')} has no region rows/cols to slice")
    r0, r1 = int(rows[0]), int(rows[1])
    c0, c1 = int(cols[0]), int(cols[1])
    h, w = arr.shape
    if not (0 <= r0 < r1 <= h and 0 <= c0 < c1 <= w):
        raise FetchFixturesError(
            f"entry {entry.get('id')}: slice rows {r0}:{r1} cols {c0}:{c1} "
            f"out of bounds for source raster of shape {arr.shape}"
        )
    return arr[r0:r1, c0:c1]


def _save_fixture(window: np.ndarray, cache_dir: Path, fixture_id: str) -> tuple[Path, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{fixture_id}.npy"
    np.save(dest, window)
    info = {
        "path": str(dest),
        "shape": list(window.shape),
        "dtype": str(window.dtype),
        "array_sha256": _sha256_array(window),
        "npy_file_sha256": _sha256_file(dest),
    }
    return dest, info


def fetch_one(entry: dict, cache_dir: Path) -> dict:
    """Download entry's provenance.url, sha256 it, slice the region
    window, save both under cache_dir. Returns an updated copy of entry.
    Raises FetchFixturesError on failure -- an entry is only ever stamped
    after every verification below has passed.
    """
    url = entry["provenance"]["url"]
    sources_dir = cache_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(url).suffix or ".bin"
    source_path = sources_dir / f"{entry['id']}_source{suffix}"

    expected_size = entry["provenance"].get("expected_size_bytes")
    if not source_path.exists() or (expected_size and source_path.stat().st_size != expected_size):
        try:
            with urllib.request.urlopen(url) as resp, open(source_path, "wb") as out:
                for chunk in iter(lambda: resp.read(1 << 20), b""):
                    out.write(chunk)
        except Exception as exc:
            raise FetchFixturesError(f"failed to fetch {entry['id']} from {url}: {exc}") from exc

    size = source_path.stat().st_size
    if expected_size and size != expected_size:
        raise FetchFixturesError(
            f"{entry['id']}: downloaded size {size} != expected_size_bytes {expected_size}; not stamping"
        )
    source_sha = _sha256_file(source_path)

    arr = load_raster(source_path)
    window = slice_region(arr, entry)
    expected_shape = (
        entry["region"]["rows"][1] - entry["region"]["rows"][0],
        entry["region"]["cols"][1] - entry["region"]["cols"][0],
    )
    if window.shape != expected_shape:
        raise FetchFixturesError(
            f"{entry['id']}: sliced window shape {window.shape} != expected {expected_shape}; not stamping"
        )
    _, fixture_info = _save_fixture(window, cache_dir, entry["id"])

    updated = dict(entry)
    updated["status"] = "FETCHED_VERIFIED"
    updated["fetched_date"] = _today()
    updated["sha256"] = fixture_info["npy_file_sha256"]
    updated["fixture"] = fixture_info
    updated["source_file"] = {"path": str(source_path), "sha256": source_sha, "size_bytes": size}
    return updated


def verify_local_one(entry: dict, cache_dir: Path, local_root: Path) -> dict:
    """Slice a NOT_PUBLICLY_FETCHABLE entry's window from the parent
    project's local source raster, verify it against content_pin, and
    save it to cache. Status is deliberately NOT changed."""
    rel = entry.get("provenance", {}).get("local_source")
    if not rel:
        raise FetchFixturesError(f"{entry.get('id')}: no provenance.local_source recorded")
    src = local_root / rel
    if not src.exists():
        raise FetchFixturesError(f"{entry.get('id')}: local source not found: {src}")
    arr = load_raster(src)
    window = slice_region(arr, entry)
    pin = entry.get("content_pin") or {}
    actual = _sha256_array(window)
    if pin.get("array_sha256") and actual != pin["array_sha256"]:
        raise FetchFixturesError(
            f"{entry.get('id')}: sliced window sha256 {actual} != content_pin.array_sha256 "
            f"{pin['array_sha256']}; source raster has CHANGED since review -- not writing fixture"
        )
    _, fixture_info = _save_fixture(window, cache_dir, entry["id"])

    updated = dict(entry)
    updated["local_verification"] = {
        "date": _today(),
        "matches_content_pin": bool(pin.get("array_sha256")),
        "fixture": fixture_info,
        "note": "produced from the parent project's local source raster; NOT a public fetch",
    }
    return updated


def run(manifest_path: Path, cache_dir: Path, dry_run: bool = False,
        local_source_root: Path | None = None) -> dict:
    data = json.loads(manifest_path.read_text())
    fixtures = data.get("fixtures", [])
    summary = {"fetched": [], "verified_local": [], "skipped": [], "failed": []}
    changed = False

    for i, entry in enumerate(fixtures):
        can_fetch, reason = classify(entry)
        if can_fetch:
            if dry_run:
                summary["skipped"].append({"id": entry.get("id"), "reason": "dry-run"})
                print(f"DRY-RUN would fetch {entry.get('id')} from {entry['provenance']['url']}")
                continue
            try:
                fixtures[i] = fetch_one(entry, cache_dir)
                changed = True
                summary["fetched"].append(entry.get("id"))
                print(f"FETCHED {entry.get('id')} -> fixture sha256={fixtures[i]['sha256']}")
            except FetchFixturesError as exc:
                summary["failed"].append({"id": entry.get("id"), "error": str(exc)})
                print(f"FAILED {entry.get('id')}: {exc}", file=sys.stderr)
            continue

        is_local = entry.get("provenance", {}).get("type") == "local_render_from_public_inputs"
        if is_local and local_source_root is not None and not dry_run:
            try:
                fixtures[i] = verify_local_one(entry, cache_dir, local_source_root)
                changed = True
                summary["verified_local"].append(entry.get("id"))
                print(f"LOCAL-VERIFIED {entry.get('id')} (content_pin match; status unchanged)")
            except FetchFixturesError as exc:
                summary["failed"].append({"id": entry.get("id"), "error": str(exc)})
                print(f"FAILED {entry.get('id')}: {exc}", file=sys.stderr)
            continue

        summary["skipped"].append({"id": entry.get("id"), "reason": reason})
        print(f"SKIP  {entry.get('id')}: {reason}")

    if changed:
        data["fixtures"] = fixtures
        manifest_path.write_text(json.dumps(data, indent=2) + "\n")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch_fixtures.py",
        description="Download/slice fixture windows and sha256-stamp the manifest, in place.",
    )
    parser.add_argument("--manifest", default="fixtures/manifest.json")
    parser.add_argument("--cache-dir", default="fixtures/cache")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="list what would be fetched without touching the network or the manifest",
    )
    parser.add_argument(
        "--local-source-root", default=None,
        help="optional path to a parent-project checkout; entries with provenance.type "
             "'local_render_from_public_inputs' are sliced from it and verified against "
             "their content_pin (their status stays NOT_PUBLICLY_FETCHABLE)",
    )
    return parser


def main(argv=None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    local_root = Path(args.local_source_root) if args.local_source_root else None
    summary = run(manifest_path, Path(args.cache_dir), dry_run=args.dry_run,
                  local_source_root=local_root)
    print(
        f"done: {len(summary['fetched'])} fetched, {len(summary['verified_local'])} verified-local, "
        f"{len(summary['skipped'])} skipped, {len(summary['failed'])} failed"
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
