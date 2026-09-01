#!/usr/bin/env python3
"""Run every ink_9um checkpoint against the public PHerc0139 control.

Fourteen checkpoints -- two seeds by seven steps of one training run -- against
one frozen input. That is the comparison the training run exists to support,
and the model card says outright that different steps behave differently on
different segments, so which step to trust is a measurement rather than a
default somebody picks.

Why these fourteen and not every checkpoint the platform knows: the control's
surface volume is 9.362 um isotropic and these models train at 9.6 um, so a run
measures the checkpoint. The other installed models do not share that property
-- a 2 um model, a 32 um coverage model and a segmentation model handed this
volume are measuring the resample, not themselves -- and the sweep refuses them
by name rather than producing fourteen more numbers that look comparable.

Each run is independent: one failing does not stop the rest, and the summary
says which produced a live map, which produced a degenerate one, and which did
not run at all.

  sweep_ink_9um_control.py --models-root /mnt/bulk/helena/models --output /out/sweep
  sweep_ink_9um_control.py ... --dry-run     # print the commands, run nothing
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "framework/registries/ink-weights-0.1.0.json"
PROFILES = REPO / "framework/profiles/03-ink"
CONTROL = REPO / "scripts/harness/run_public_ink_control.py"
SURFACE_VOLUME = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/"
    "segments/20260112000000-w043_2026011217/surface-volumes/"
    "9.362um-1.2m-113keV-volume-20250728140407.zarr")


def profile_for(digest: str) -> dict | None:
    """The frozen profile that pins this checkpoint, if one does.

    Each of the fourteen has its own, so a receipt names the checkpoint that ran
    by profile id rather than leaving a reader to match hashes by hand.
    """
    for path in sorted(PROFILES.glob("*.json")):
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if spec.get("checkpoint_sha256") == digest:
            return spec
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-root", type=Path,
                        default=Path(os.environ.get("CX_MODELS", "/models")))
    parser.add_argument("--output", type=Path, required=True,
                        help="a directory per checkpoint is written under here")
    parser.add_argument("--surface-volume", default=SURFACE_VOLUME)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", action="append", default=[],
                        help="substring of a destination path; repeatable")
    args = parser.parse_args()

    entries = [e for e in json.loads(MANIFEST.read_text())["entries"]
               if e["repo"] == "scrollprize/ink_9um"]
    if args.only:
        entries = [e for e in entries if any(o in e["destination"] for o in args.only)]
    entries.sort(key=lambda e: e["upstream_path"])
    if not entries:
        print("no ink_9um checkpoint selected", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, entry in enumerate(entries, 1):
        checkpoint = args.models_root / entry["destination"]
        seed, step = re.match(r"hybrid_3d2d-seed(\d+)/step-(\d+)\.pth$",
                              entry["upstream_path"]).groups()
        label = f"seed{seed}-step{int(step)}"
        spec = profile_for(entry["sha256"])
        row = {"label": label, "checkpoint": str(checkpoint),
               "sha256": entry["sha256"],
               "profile_id": spec["profile_id"] if spec else None}

        if spec is None:
            # Runnable but unattributable: the receipt could not say which of the
            # fourteen produced the map, which is the only thing a sweep is for.
            row["outcome"] = "no profile pins this checkpoint"
            results.append(row)
            print(f"[{index}/{len(entries)}] {label}: skipped, {row['outcome']}")
            continue
        present = checkpoint.is_file()
        if not present and not args.dry_run:
            row["outcome"] = "not installed"
            results.append(row)
            print(f"[{index}/{len(entries)}] {label}: skipped, not installed")
            continue

        destination = args.output / label
        command = [sys.executable, str(CONTROL),
                   "--surface-volume", args.surface_volume,
                   "--checkpoint", str(checkpoint),
                   "--expected-checkpoint-sha256", entry["sha256"],
                   "--profile-id", spec["profile_id"],
                   "--output", str(destination)]
        if args.dry_run:
            # The point of a dry run is the command, so print it even for a
            # checkpoint that is not here yet -- that is exactly the case where
            # somebody is checking what a sweep would do before fetching 1.9 GB.
            print(" ".join(command) + ("" if present else "    # not installed"))
            row["outcome"] = "dry-run" if present else "dry-run, not installed"
            results.append(row)
            continue

        print(f"[{index}/{len(entries)}] {label}: running", flush=True)
        destination.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(command, check=False)
        row["exit_code"] = completed.returncode
        row["outcome"] = "ok" if completed.returncode == 0 else "failed"
        receipt = destination / "receipt.json"
        if receipt.is_file():
            try:
                loaded = json.loads(receipt.read_text(encoding="utf-8"))
                row["liveness"] = loaded.get("liveness")
                row["statistics"] = loaded.get("statistics")
            except ValueError:
                row["outcome"] = "receipt is not readable json"
        results.append(row)
        print(f"[{index}/{len(entries)}] {label}: {row['outcome']}", flush=True)

    summary = args.output / "sweep.json"
    summary.write_text(json.dumps(
        {"surface_volume": args.surface_volume,
         "models_root": str(args.models_root),
         "runs": results}, indent=2) + "\n")
    print(f"\n{summary}")
    ran = [r for r in results if r.get("outcome") == "ok"]
    print(f"{len(ran)}/{len(results)} ran clean")
    return 0 if len(ran) == len(results) or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
