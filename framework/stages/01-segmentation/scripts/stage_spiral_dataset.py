#!/usr/bin/env python3
"""Put a spiral fitter's inputs where a worker can read them, once per scroll.

A P1 spiral job can be queued today and will refuse on every machine, because
nothing puts a dataset at `dataset_path`. The only way to get one is for a
person to copy files onto each host by hand, which is the opposite of having a
fleet.

Three inputs, three different natures, and treating them alike wastes two of
them.

    tracks     one file of 5.42 GB          bandwidth-bound
    lasagna    ~152,000 small objects       latency-bound
    umbilicus  a few kilobytes of JSON      not published anywhere

The first two are fetched. The third cannot be: it is annotated by a person --
ours are Aleksei Drobkov's and bruniss's, thirteen scrolls -- and lives in no
bucket. It is an artifact somebody supplies, so this refuses by name and says
how rather than inventing a URL for it.

What this produces
------------------
A directory in a cache shared per scroll, and a `SPIRAL_DATASET.json` beside it:
every file with its size, its digest and where it came from. The manifest is the
registration; the bytes stay in the cache. Thirteen scrolls at about 7 GB is 92
GB, and copying that into an artifact store to say it exists would be paying
twice for the same guarantee.

The guarantee is worth having on its own. A truncated `.dbm` is not
distinguishable from a complete one by looking at it -- the fit runs and fails
strangely much later -- so the length is checked against the source before the
download is skipped and again after it finishes.

What it deliberately does not do
--------------------------------
It does not hash 5.42 GB on every run. The digest is computed once, when the
bytes are first staged, and checked again only when asked; the per-run check is
size, which is what catches the failure this exists for at a cost a preflight
can pay.
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
from pathlib import Path
from typing import Any, Callable, Iterable

MANIFEST = "SPIRAL_DATASET.json"
SCHEMA = "campaignx.spiral_dataset.v1"

TRACKS_HOST = "https://dl.ash2txt.org/datasets/spiral_datasets"
LASAGNA_BUCKET = "vesuvius-challenge-open-data"
LASAGNA_PREFIX = "representations/predictions/lasagna"
ARRAYS = ("nx", "ny", "grad_mag")


class StagingRefused(RuntimeError):
    """The dataset cannot be assembled, said before anything is fetched."""


# ---------------------------------------------------------------- sources --

def tracks_url(scroll: str, scan: str) -> str:
    """Where one scroll's surface tracks live.

    The scan id is part of the path and is not derivable from the scroll: it is
    a timestamp of the acquisition, so it is discovered or supplied rather than
    constructed.
    """
    name = f"PHerc{scroll}_{scan}_surface_m7_L0_th0.2.dbm"
    return f"{TRACKS_HOST}/PHerc{scroll}/{scan}/tracks/{name}"


def discover_scan(scroll: str, *, opener: Callable[[str], str] | None = None) -> str:
    """The one scan id published for this scroll, or a refusal naming them all.

    Guessing is the failure mode worth avoiding: every wrong guess is a 404 that
    reads like "this scroll has no tracks" when it means "not at that path".
    """
    listing = (opener or _read_url)(f"{TRACKS_HOST}/PHerc{scroll}/")
    scans = sorted({
        part.strip("/") for part in _hrefs(listing)
        if part.strip("/").isdigit()
    })
    if not scans:
        raise StagingRefused(
            f"PHerc{scroll} publishes no scan directory under {TRACKS_HOST}; "
            "this scroll has no spiral dataset there")
    if len(scans) > 1:
        raise StagingRefused(
            f"PHerc{scroll} publishes {scans}; name one with --scan, because "
            "which acquisition the tracks came from is part of what a fit means")
    return scans[0]


def _hrefs(html: str) -> Iterable[str]:
    import re

    return re.findall(r'href="([^"]+)"', html)


def _read_url(url: str) -> str:
    completed = subprocess.run(  # noqa: S603 - argv built here
        ["curl", "-fsS", "--max-time", "120", url],
        capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise StagingRefused(f"cannot read {url}: {completed.stderr.strip()}")
    return completed.stdout


def remote_length(url: str) -> int:
    """`Content-Length` from the source, which is the only thing that says a
    local file is whole."""
    completed = subprocess.run(  # noqa: S603 - argv built here
        ["curl", "-fsSI", "--max-time", "120", url],
        capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise StagingRefused(f"cannot reach {url}: {completed.stderr.strip()}")
    for line in completed.stdout.splitlines():
        name, _, value = line.partition(":")
        if name.strip().lower() == "content-length":
            return int(value.strip())
    raise StagingRefused(f"{url} answered no Content-Length; nothing can be "
                         "verified against a source that will not say its size")


# ----------------------------------------------------------------- tracks --

def fetch_tracks(url: str, destination: Path, *,
                 length: Callable[[str], int] = remote_length,
                 run: Callable[[list[str]], int] | None = None) -> dict[str, Any]:
    """One large file, resumed if interrupted and verified against its source.

    Checked twice on purpose. Before, so a complete local copy is not fetched
    again -- 5.42 GB is fifty seconds of a good link and nine minutes of a bad
    one. After, because that is the check that catches the failure: a `.dbm`
    cut short opens, reads, and yields fewer tracks than it should.
    """
    wanted = length(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    have = destination.stat().st_size if destination.is_file() else -1
    if have == wanted:
        return {"path": destination.name, "size_bytes": wanted, "source": url,
                "fetched": False, "why": "the local copy is already whole"}

    argv = ["curl", "-fSL", "--no-progress-meter", "-C", "-",
            "-o", str(destination), url]
    code = (run or _run)(argv)
    if code != 0:
        raise StagingRefused(f"fetching {url} exited {code}")
    got = destination.stat().st_size if destination.is_file() else 0
    if got != wanted:
        raise StagingRefused(
            f"tracks are incomplete: {got} != {wanted}. A short .dbm is not "
            "distinguishable from a whole one by reading it, so this is the "
            "only place the difference is visible.")
    return {"path": destination.name, "size_bytes": got, "source": url,
            "fetched": True}


def _run(argv: list[str]) -> int:
    return subprocess.run(argv, check=False).returncode  # noqa: S603


# ---------------------------------------------------------------- lasagna --

def fetch_lasagna(scroll: str, level: str, destination: Path, *,
                  volume_name: str = "PHerc{scroll}_{array}.ome.zarr",
                  filesystem: Any | None = None,
                  workers: int = 64) -> dict[str, Any]:
    """One pyramid level of the three lasagna volumes.

    One level, because the fit reads one: fetching the whole pyramid is about
    fifty-seven times the data for nothing.

    The counting is the part that matters. In a masked volume a missing chunk is
    normal and means air; a chunk that failed to transfer also reads as air, and
    the two are identical in the array and opposite in meaning. They are counted
    apart, and a transfer failure stops the staging -- an absence does not.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if filesystem is None:
        import s3fs  # noqa: PLC0415

        filesystem = s3fs.S3FileSystem(anon=True)

    base = f"{LASAGNA_BUCKET}/PHerc{scroll}/{LASAGNA_PREFIX}"
    roots = [path for path in filesystem.ls(base) if "lasagna" in path]
    if not roots:
        raise StagingRefused(f"{base} publishes no lasagna volumes")
    root = roots[0]

    summary: dict[str, Any] = {"level": level, "arrays": {}, "absent_404": 0,
                               "transfer_failures": 0}
    for array in ARRAYS:
        name = volume_name.format(scroll=scroll, array=array)
        source = f"{root}/{name}/{level}"
        target = destination / name / level
        entries = [(key, value["size"])
                   for key, value in filesystem.find(source, detail=True).items()
                   if value.get("type") == "file"]
        if not entries:
            raise StagingRefused(
                f"{source} holds no objects; level {level} is not published for "
                f"{name}, and a level that is not there is not an empty one")
        for metadata in (".zattrs", ".zgroup"):
            try:
                filesystem.get_file(f"{root}/{name}/{metadata}",
                                    str(destination / name / metadata))
            except FileNotFoundError:
                # A pyramid without group metadata is still readable; a missing
                # optional file is not a reason to stop.
                pass

        started = time.time()
        absent, failed = 0, []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_one_object, filesystem, key, size, source, target)
                       for key, size in entries]
            for future in as_completed(futures):
                outcome = future.result()
                if outcome == "absent":
                    absent += 1
                elif outcome is not None:
                    failed.append(outcome)
        summary["arrays"][array] = {
            "objects": len(entries), "absent_404": absent,
            "transfer_failures": len(failed),
            "seconds": round(time.time() - started, 1),
        }
        summary["absent_404"] += absent
        summary["transfer_failures"] += len(failed)
        if failed:
            raise StagingRefused(
                f"{len(failed)} objects of {name} failed to transfer, which is "
                "not the same as {absent} that are not published: a chunk that "
                "did not arrive reads as air in the array exactly like a chunk "
                "that was never there, and one of those two is a hole in the "
                f"evidence. First: {failed[0]}")
    return summary


def _one_object(filesystem: Any, key: str, size: int, source: str,
                target: Path) -> str | None:
    """Fetch one object. Returns None when it landed, 'absent' when the source
    does not have it, and the key when the transfer failed.

    A 404 is not retried. Retrying an absence four times with backoff is fifteen
    seconds spent proving something the first answer already said, and on a
    masked volume it is most of them.
    """
    # An S3 key is a flat string, and nothing stops an object being stored under
    # one that carries `../` after the prefix. `relpath` happily returns those
    # segments and `target / ...` then leaves the cache entirely -- writing
    # attacker-chosen bytes anywhere the staging worker can reach, including
    # another scroll's SPIRAL_DATASET.json on the shared volume. The bucket is
    # public and explicitly untrusted, so the key is checked rather than
    # believed.
    relative = os.path.relpath(key, source)
    path = target / relative
    if os.path.isabs(relative) or not path.resolve().is_relative_to(target.resolve()):
        raise StagingRefused(
            f"the source offered an object whose key escapes the cache: "
            f"{key!r} resolves outside {target}. A key is a flat string and this "
            "one is shaped like a path traversal.")
    if path.exists() and path.stat().st_size == size:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            filesystem.get_file(key, str(path))
        except FileNotFoundError:
            return "absent"
        except Exception:  # noqa: BLE001 - a transfer failure is retried
            time.sleep(2 ** attempt * 0.25)
            continue
        if path.exists() and path.stat().st_size == size:
            return None
    return key


# -------------------------------------------------------------- umbilicus --

def require_umbilicus(dataset: Path, supplied: Path | None) -> dict[str, Any]:
    """The one input with no URL.

    Tracks and lasagna are published; the umbilicus is annotated by a person.
    Ours are Aleksei Drobkov's and bruniss's, thirteen scrolls, and they exist
    as JSON somebody produced. So this is an artifact to supply, and pretending
    otherwise would put a download in the receipt that never happened.
    """
    target = dataset / "umbilicus.json"
    if supplied is not None:
        if not Path(supplied).is_file():
            raise StagingRefused(f"the umbilicus given is not a file: {supplied}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(supplied, target)
    if not target.is_file():
        raise StagingRefused(
            "this dataset has no umbilicus.json, and there is no bucket to "
            "fetch one from: it is annotated by a person, not published. "
            "Supply it with --umbilicus <path>, or register it as an artifact "
            "for this scroll and stage from there.")

    document = json.loads(target.read_text(encoding="utf-8"))
    points = document.get("control_points")
    if not isinstance(points, list) or len(points) <= 20:
        raise StagingRefused(
            f"umbilicus.json carries {len(points) if isinstance(points, list) else 'no'} "
            "control points; the axis of a scroll is not described by a handful, "
            "and a truncated one bends the whole fit")
    zs = [point["z"] for point in points]
    return {"path": "umbilicus.json", "control_points": len(points),
            "z_range": [min(zs), max(zs)],
            "size_bytes": target.stat().st_size,
            "source": "supplied artifact; not published in any bucket"}


# --------------------------------------------------------------- manifest --

def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(dataset: Path, *, scroll: str, scan: str, level: str,
                   volume_name: str, tracks: dict[str, Any],
                   umbilicus: dict[str, Any], lasagna: dict[str, Any],
                   digests: bool = True) -> dict[str, Any]:
    """What this dataset is, so a fit can say what it read.

    Sizes for everything and digests for the small files always; the digest of
    the tracks is computed here, once, because it is the expensive one and this
    is the only moment it is worth paying for.
    """
    files: dict[str, Any] = {}
    for relative in sorted(
            path.relative_to(dataset).as_posix()
            for path in dataset.rglob("*") if path.is_file()
            and path.name != MANIFEST):
        path = dataset / relative
        entry: dict[str, Any] = {"size_bytes": path.stat().st_size}
        # Every small file is hashed. The tracks file is hashed only when asked,
        # because 5.42 GB is a minute of disk this does not need to spend twice.
        if digests and (entry["size_bytes"] < 256 * 1024 * 1024
                        or relative.startswith("tracks/")):
            entry["sha256"] = file_digest(path)
        files[relative] = entry
    return {
        "schema": SCHEMA,
        "scroll": scroll,
        "scan": scan,
        # Resolved, not templated: {scroll} is a fact the stager has and a
        # reader would have to guess. Only {array} survives, because the three
        # volumes are one setting.
        "layout": {"lasagna_volume_name": volume_name.replace(
                       "{scroll}", scroll),
                   "normal_zarr_group": level,
                   "lasagna_scale": 2 ** int(level),
                   "tracks_file": tracks["path"]},
        "tracks": tracks,
        "umbilicus": umbilicus,
        "lasagna": lasagna,
        "files": files,
        "non_claims": [
            "a complete dataset is not a good fit: it says the inputs are "
            "whole, not that the geometry they produce is right",
            "the digests are of the bytes staged here, not a claim that the "
            "source publishes the same bytes tomorrow",
        ],
    }


def verify(dataset: Path, *, hashes: bool = False) -> dict[str, Any]:
    """Check a staged dataset against its own manifest.

    Size by default and digests on request, for the reason the module docstring
    gives: the per-run check has to be cheap enough for a preflight to pay it,
    and size is what catches a truncated file.
    """
    path = dataset / MANIFEST
    if not path.is_file():
        raise StagingRefused(
            f"{dataset} carries no {MANIFEST}: it was assembled by hand and "
            "nothing can say whether it is whole")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    wrong: list[dict[str, Any]] = []
    for relative, entry in manifest["files"].items():
        here = dataset / relative
        if not here.is_file():
            wrong.append({"path": relative, "problem": "absent"})
            continue
        if here.stat().st_size != entry["size_bytes"]:
            wrong.append({"path": relative, "problem": "size",
                          "expected": entry["size_bytes"],
                          "found": here.stat().st_size})
            continue
        if hashes and entry.get("sha256") and file_digest(here) != entry["sha256"]:
            wrong.append({"path": relative, "problem": "digest"})
    return {"dataset": str(dataset), "files": len(manifest["files"]),
            "checked_digests": bool(hashes), "wrong": wrong,
            "whole": not wrong}


# -------------------------------------------------------------------- cli --

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scroll", required=True, help="e.g. 0826")
    parser.add_argument("--cache", required=True,
                        help="the shared cache root. Shared per scroll and not "
                             "per worker: three machines staging the same "
                             "scroll is three times 5.42 GB for one dataset")
    parser.add_argument("--scan", help="the acquisition id; discovered when omitted")
    parser.add_argument("--level", default="4",
                        help="which lasagna pyramid level to stage. The fit "
                             "reads one; three is 57x the data")
    parser.add_argument("--volume-name", default="PHerc{scroll}_{array}.ome.zarr",
                        help="how the lasagna volumes are named at the source")
    parser.add_argument("--umbilicus", type=Path,
                        help="the annotated umbilicus for this scroll; there is "
                             "no URL for it")
    parser.add_argument("--verify-only", action="store_true",
                        help="check a staged dataset and stage nothing")
    parser.add_argument("--verify-hashes", action="store_true",
                        help="also read every byte; minutes, not seconds")
    args = parser.parse_args(argv)

    dataset = Path(args.cache) / f"PHerc{args.scroll}"
    try:
        if args.verify_only:
            report = verify(dataset, hashes=args.verify_hashes)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["whole"] else 1

        scan = args.scan or discover_scan(args.scroll)
        umbilicus = require_umbilicus(dataset, args.umbilicus)
        url = tracks_url(args.scroll, scan)
        tracks = fetch_tracks(
            url, dataset / "tracks" / Path(url).name)
        lasagna = fetch_lasagna(args.scroll, args.level,
                                dataset / "lasagna_inputs",
                                volume_name=args.volume_name)
        manifest = build_manifest(
            dataset, scroll=args.scroll, scan=scan, level=args.level,
            volume_name=args.volume_name, tracks=tracks, umbilicus=umbilicus,
            lasagna=lasagna)
        (dataset / MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(json.dumps({"dataset": str(dataset), "scan": scan,
                          "files": len(manifest["files"]),
                          "lasagna": lasagna}, indent=2, sort_keys=True))
        return 0
    except StagingRefused as refusal:
        print(f"StagingRefused: {refusal}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
