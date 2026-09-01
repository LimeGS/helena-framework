#!/usr/bin/env python3
"""Assemble an ink training set, and refuse one that contains the claim.

Step five of the workflow -- run inference, label conservatively, expand the
supervision mask, retrain, test on held-out ground, repeat -- has one rule in
it that is not a matter of taste:

    Never train on region intended for prize claim

A model trained on the window you then claim has memorised the answer, and its
map over that window is evidence of nothing. Upstream's training command
cannot know which window you mean to claim. This platform does: a claim is a
bbox on a surface it already tracks, and `contamination_controls` and source
holdouts are concepts its own audits already enforce for the CT routers.

So assembling the set is where the rule lives. A tile overlapping a declared
holdout refuses the build, and a build with no declaration at all refuses too
-- an absent holdout must never read as "nothing is reserved", which is the
same failure mode as a skipped test reporting success.

What this deliberately does not do is train. Upstream ships that, and the
registry records what this platform cost itself the last time it reimplemented
an upstream runner rather than calling it: a map correlating r=0.079 instead
of r=0.885. What comes back from training is a checkpoint, and a checkpoint
enters by digest through the method registry like every other.

Non-claims
----------
* A training set is not a model, not a measurement, and not evidence of ink.
* A holdout that was declared honestly is the only thing that makes the
  resulting model's output over the claim window meaningful. This checks that
  the declaration was applied, never that it was complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

MANIFEST_NAME = "TRAINING_SET.json"


class HoldoutViolation(RuntimeError):
    """A labelled window overlaps ground reserved for a claim."""


class UndeclaredHoldout(RuntimeError):
    """No holdout was declared. Absent is not the same as empty."""


def overlaps(a: Sequence[float], b: Sequence[float]) -> bool:
    """Whether two half-open boxes share any pixel.

    Half-open on purpose: a tile that ends exactly where a holdout begins
    shares no pixel with it, and treating that as contamination would reserve
    twice the ground somebody actually reserved.
    """
    ax0, ay0, ax1, ay1 = (float(v) for v in a)
    bx0, by0, bx1, by1 = (float(v) for v in b)
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")).hexdigest()


def _check_holdout(window: dict[str, Any],
                   holdout: Sequence[dict[str, Any]]) -> None:
    surface = window.get("surface_id")
    box = window.get("bbox_xy")
    for reserved in holdout:
        if reserved.get("surface_id") != surface:
            # A holdout is a region of one surface, not a coordinate range in
            # the abstract.
            continue
        if overlaps(box, reserved["bbox_xy"]):
            raise HoldoutViolation(
                f"window {box} on {surface} overlaps reserved region "
                f"{reserved['bbox_xy']} ({reserved.get('reason') or 'no reason given'}). "
                "A model trained on the window you then claim has memorised "
                "the answer.")


def build_training_set(
    *,
    windows: Sequence[dict[str, Any]],
    holdout: Sequence[dict[str, Any]] | None,
    output: Path,
) -> dict[str, Any]:
    """Copy labelled windows into one set, refusing any that is reserved."""
    if holdout is None:
        raise UndeclaredHoldout(
            "no holdout was declared. Pass an empty list to state that nothing "
            "is reserved; absent must not read as that on its own, because "
            "forgetting to declare is exactly the mistake this refuses.")
    output = Path(output)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite: {output}")

    # Every window is checked before anything is written. A set that quietly
    # dropped a contaminated tile is a set whose contents nobody stated, so one
    # bad window refuses the whole build rather than shrinking it.
    for window in windows:
        _check_holdout(window, holdout)

    rows: list[dict[str, Any]] = []
    output.mkdir(parents=True)
    try:
        for index, window in enumerate(windows):
            source = Path(window["image"])
            image = source if source.is_file() else source / "image.tif"
            label = Path(window["label"])
            label = label if label.is_file() else label / "label.tif"
            destination = output / f"{index:04d}"
            destination.mkdir()
            shutil.copy2(image, destination / "image.tif")
            shutil.copy2(label, destination / "label.tif")
            rows.append({
                "index": index,
                "surface_id": window["surface_id"],
                "bbox_xy": [float(v) for v in window["bbox_xy"]],
                "image_sha256": _sha256_file(destination / "image.tif"),
                "label_sha256": _sha256_file(destination / "label.tif"),
            })
    except Exception:
        # A half-written training set is worse than none: it looks like a set.
        shutil.rmtree(output, ignore_errors=True)
        raise

    manifest = {
        "schema": "campaignx.ink_training_set.v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0)
        .isoformat().replace("+00:00", "Z"),
        "window_count": len(rows),
        "windows": rows,
        "holdout_declared": True,
        "holdout_count": len(holdout),
        "holdout": list(holdout),
        "holdout_sha256": _canonical_sha256(list(holdout)),
        "non_claims": [
            "this is a training set, not a model, not a measurement and not "
            "evidence of ink",
            "nothing here trains anything; a checkpoint enters by digest "
            "through the method registry like every other",
            "the holdout was applied as declared; that it was declared "
            "completely is not something this can check",
        ],
    }
    (output / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=Path, required=True,
                    help="JSON list of {surface_id, bbox_xy, image, label}")
    ap.add_argument("--holdout", type=Path, required=True,
                    help="JSON list of {surface_id, bbox_xy, reason}; pass a "
                         "file containing [] to state that nothing is reserved")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    windows = json.loads(args.windows.read_text(encoding="utf-8"))
    holdout = json.loads(args.holdout.read_text(encoding="utf-8"))
    try:
        manifest = build_training_set(
            windows=windows, holdout=holdout, output=args.output)
    except (HoldoutViolation, UndeclaredHoldout) as refused:
        print(str(refused), file=sys.stderr)
        return 3
    print(json.dumps({k: manifest[k] for k in
                      ("window_count", "holdout_count", "holdout_sha256")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
