#!/usr/bin/env python3
"""Install the ink checkpoints named by the weight manifest, and verify each one.

Every file is verified against a digest that came from upstream's own LFS
metadata rather than from a copy we made. That is the distinction that makes
this worth running: a checksum computed from the download only proves the file
did not change between two of our own reads. Comparing against what upstream
published proves the bytes are the ones the profile means.

A file that is already installed with the right digest is left alone, so this
is safe to re-run and safe to interrupt. A file whose digest does not match is
never left where a worker could load it -- it is written to a `.partial`
sibling and only moved into place once it verifies.

  install_ink_weights.py --models-root /mnt/bulk/helena/models
  install_ink_weights.py --models-root ... --only ink_9um   # one repo
  install_ink_weights.py --models-root ... --verify-only    # audit, no download

Exit status is 0 only when every selected entry is present and verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

# Under nohup stdout is a pipe, so it block-buffers: the informative lines would
# land at the end while the progress noise on stderr arrives live. A download
# that takes twenty minutes has to be readable while it runs.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

MANIFEST = Path(__file__).resolve().parents[2] / "framework/registries/ink-weights-0.1.0.json"
CHUNK = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def human(count: int) -> str:
    return f"{count / 1e9:.2f} GB" if count >= 1e9 else f"{count / 1e6:.0f} MB"


def download(url: str, target: Path, expected_size: int) -> None:
    """Stream to a .partial sibling; the caller promotes it only if it verifies."""
    partial = target.with_suffix(target.suffix + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "helena-install-ink-weights"})
    seen = 0
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            out.write(block)
            seen += len(block)
            if expected_size:
                pct = 100 * seen / expected_size
                print(f"\r    {human(seen)} / {human(expected_size)}  {pct:5.1f}%",
                      end="", file=sys.stderr, flush=True)
    print("", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-root", type=Path,
                        default=Path(os.environ.get("CX_MODELS", "/models")),
                        help="where checkpoints are installed (default: $CX_MODELS or /models)")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--only", action="append", default=[],
                        help="substring of a repo or destination; repeatable")
    parser.add_argument("--verify-only", action="store_true",
                        help="report what is installed and whether it verifies; download nothing")
    parser.add_argument("--recheck", action="store_true",
                        help="re-hash files that are already installed at the right size. "
                             "Without this, a present file of the right size is trusted, "
                             "which makes a re-run fast rather than thorough.")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    entries = manifest["entries"]
    if args.only:
        entries = [e for e in entries
                   if any(o in e["repo"] or o in e["destination"] for o in args.only)]
        if not entries:
            print(f"nothing in the manifest matches {args.only}", file=sys.stderr)
            return 2

    root = args.models_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    total = sum(e["size_bytes"] for e in entries)
    print(f"{len(entries)} checkpoints, {human(total)}, into {root}")

    installed = verified = downloaded = 0
    failures: list[str] = []

    for index, entry in enumerate(entries, 1):
        target = (root / entry["destination"]).resolve()
        if root not in target.parents:
            failures.append(f"{entry['destination']}: resolves outside {root}")
            continue
        label = f"[{index}/{len(entries)}] {entry['destination']}"

        if target.is_file():
            same_size = target.stat().st_size == entry["size_bytes"]
            if same_size and not (args.recheck or args.verify_only):
                print(f"{label}: present")
                installed += 1
                continue
            actual = sha256_file(target)
            if actual == entry["sha256"]:
                print(f"{label}: verified")
                installed += 1
                verified += 1
                continue
            failures.append(
                f"{entry['destination']}: installed file is {actual[:16]}..., the "
                f"manifest says {entry['sha256'][:16]}... -- this is not upstream's "
                "file. Move it aside and re-run rather than trusting it.")
            continue

        if args.verify_only:
            failures.append(f"{entry['destination']}: not installed")
            continue

        url = (f"https://huggingface.co/{entry['repo']}/resolve/main/"
               f"{entry['upstream_path']}?download=true")
        print(f"{label}: fetching {human(entry['size_bytes'])}")
        partial = target.with_suffix(target.suffix + ".partial")
        try:
            download(url, target, entry["size_bytes"])
        except Exception as exc:  # noqa: BLE001 -- report and carry on to the rest
            partial.unlink(missing_ok=True)
            failures.append(f"{entry['destination']}: download failed: {exc}")
            continue

        actual = sha256_file(partial)
        if actual != entry["sha256"]:
            partial.unlink(missing_ok=True)
            failures.append(
                f"{entry['destination']}: downloaded {actual[:16]}..., upstream "
                f"published {entry['sha256'][:16]}... -- discarded.")
            continue
        shutil.move(str(partial), str(target))
        print(f"{label}: verified")
        installed += 1
        verified += 1
        downloaded += 1

    print(f"\n{installed}/{len(entries)} installed, {verified} hashed this run, "
          f"{downloaded} newly downloaded")
    if failures:
        print(f"\n{len(failures)} problems:", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
