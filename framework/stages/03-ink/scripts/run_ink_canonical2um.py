#!/usr/bin/env python3
"""Run the official canonical 2 um ink detector (ResNet-152 + 3D decoder).

This is the recipe `new_canon_autoresearch_recipe` that produced the reference
PHerc0139 prediction in which text is plainly visible. Helena Framework had it
registered as `ink-canonical-2um@1.0.0` / KNOWN_NOT_INTEGRATED and never wrote
an adapter, so the only enabled ink lane stayed the 2023 TimeSformer, which
this session showed produces no legible letters even on known ink.

Model loading is the project's own upstream code, imported unchanged, so the
architecture cannot drift from the checkpoint. Only the sliding window and the
receipt are local.

No rescaling: the reference prediction was produced at the source's native
2.399 um, so the stack is fed as-is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# The repository root, so the shared device resolver is importable from a
# script launched by path. `auto` has to mean the same thing in every runner or
# it is six defaults again, wearing one word.
_ROOT = Path(__file__).resolve()
while _ROOT != _ROOT.parent and not (_ROOT / "framework").is_dir():
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from framework.contracts.host_probe import resolve_device  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_resnet3d_3d_decoder import load_model  # noqa: E402

# The repository root, for the shared contracts. This adapter only had its own
# directory on the path, because the upstream architecture it vendors lives
# beside it -- which is also why it had no way to reach the liveness gate.
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.contracts.lane_liveness import (  # noqa: E402
    assess_liveness,
    refuse_if_not_alive,
)

Image.MAX_IMAGE_PIXELS = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ordered_layers(directory: Path) -> list[Path]:
    """Numeric stem ordering, fail-closed -- the contract this repo now requires."""
    files = list(directory.glob("*.tif"))
    if not files:
        raise RuntimeError(f"no TIFFs in {directory}")
    stems = [p.stem for p in files]
    if not all(s.isdigit() for s in stems):
        raise RuntimeError("non-numeric TIFF stem: refusing to sort lexicographically")
    if len({int(s) for s in stems}) != len(stems):
        raise RuntimeError("ambiguous numeric stems")
    return sorted(files, key=lambda p: int(p.stem))


def hann2d(n: int) -> np.ndarray:
    w = np.hanning(n + 2).astype(np.float32)[1:-1]
    k = np.outer(w, w)
    return k / float(k.max())


def grid(length: int, tile: int, stride: int) -> list[int]:
    if length < tile:
        raise RuntimeError(f"axis {length} smaller than tile {tile}")
    xs = set(range(0, length - tile + 1, stride))
    xs.add(length - tile)
    return sorted(xs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiff-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--source-pixel-um", type=float, required=True)
    ap.add_argument(
        "--expected-checkpoint-sha256",
        help="the digest the profile pins. This script computes the real one "
             "for its receipt and had nothing to compare it against -- it takes "
             "no --profile -- so a receipt could record a checkpoint nobody had "
             "checked was the intended one. Given, it is verified before the "
             "model is loaded rather than after the run.")
    ap.add_argument("--frames", type=int, default=62)
    ap.add_argument("--tile-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--depth-center", type=int, default=None)
    ap.add_argument("--min-valid-ratio", type=float, default=0.60)
    ap.add_argument("--device", default="auto",
                        help="auto (the card if this host has one), cpu, or cuda[:N] to require one")
    ap.add_argument(
        "--on-degenerate",
        choices=("fail", "warn"),
        default="fail",
        help="what to do when the output map carries no decision (default: fail closed)",
    )
    args = ap.parse_args()

    # Resolve the device before anything expensive: an explicit `cuda` on a host
    # with no card is refused here rather than several minutes into a load, and
    # `auto` becomes the word the receipt can stand behind.
    _device = resolve_device(args.device)
    args.device = _device["device"]

    # Non-empty, not merely present. The fleet worker makes every job's run
    # directory before it starts the runner -- other lanes write a file into it
    # and expect it to be there -- so refusing an existing directory refused
    # every queued run of this lane, in two and a half seconds, since the day
    # the guard was written. What it is for is not clobbering a map that is
    # already there, and an empty directory is not a map.
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output}")

    files = ordered_layers(args.tiff_dir.resolve())
    centre = args.depth_center if args.depth_center is not None else len(files) // 2
    half = args.frames // 2
    lo = centre - half
    hi = lo + args.frames
    if lo < 0 or hi > len(files):
        raise RuntimeError(
            f"{args.frames} frames centred at {centre} fall outside {len(files)} layers"
        )
    chosen = files[lo:hi]
    stack = np.stack([np.asarray(Image.open(p)) for p in chosen], axis=0)
    if stack.dtype != np.uint8:
        raise RuntimeError(f"expected uint8 layers, got {stack.dtype}")
    depth, height, width = stack.shape

    # Before the model is loaded, so a wrong checkpoint costs a hash and not a
    # run. Same reason the layer stack is validated before inference here.
    checkpoint_sha = sha256_file(args.checkpoint)
    if args.expected_checkpoint_sha256 and checkpoint_sha != args.expected_checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint {checkpoint_sha} is not the "
            f"{args.expected_checkpoint_sha256} that was expected. Nothing was run.")

    device = torch.device(args.device)
    model = load_model(str(args.checkpoint), device, num_frames=args.frames)
    model.eval()
    scale = model.get_output_scale_factor()

    # The volume stays on the CPU: 62x4096x4096 in float32 is 3.9 GiB and does
    # not fit on a shared 6 GiB card. Only the tiles go up.
    volume = torch.from_numpy(stack)
    valid_np = (stack > 0).any(axis=0)
    valid_plane = torch.from_numpy(valid_np)

    tile, stride = args.tile_size, args.stride
    weight = torch.from_numpy(hann2d(tile)).to(device)
    total = torch.zeros((height, width), dtype=torch.float32, device=device)
    norm = torch.zeros((height, width), dtype=torch.float32, device=device)

    positions = [(y, x) for y in grid(height, tile, stride) for x in grid(width, tile, stride)]
    kept = [
        (y, x)
        for (y, x) in positions
        if float(valid_plane[y : y + tile, x : x + tile].float().mean()) >= args.min_valid_ratio
    ]
    started = time.time()
    with torch.inference_mode():
        for i in range(0, len(kept), args.batch_size):
            chunk = kept[i : i + args.batch_size]
            batch = (
                torch.stack([volume[:, y : y + tile, x : x + tile] for (y, x) in chunk])
                .to(device=device, dtype=torch.float32)
                .div_(255.0)
            )
            out = model.forward(batch)
            if out.ndim == 4:
                out = out[:, 0]
            out = torch.sigmoid(out.float())
            if out.shape[-1] != tile:
                out = F.interpolate(
                    out[:, None], size=(tile, tile), mode="bilinear", align_corners=False
                )[:, 0]
            for (y, x), pred in zip(chunk, out):
                total[y : y + tile, x : x + tile] += pred * weight
                norm[y : y + tile, x : x + tile] += weight

    probability = torch.where(norm > 0, total / norm.clamp(min=1e-6), torch.zeros_like(total))
    # valid_plane lives on the CPU from the memory layout; it is masked here.
    result = np.where(valid_np, probability.detach().cpu().numpy(), 0.0)

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "probability.npy", result)
    finite = result[result > 0]
    receipt = {
        "schema": "campaignx.ink_canonical_2um_screening_receipt.v1",
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "sample_id": args.sample_id,
        "method_id": "ink-canonical-2um@1.0.0",
        "recipe": "new_canon_autoresearch_recipe",
        "checkpoint_sha256": checkpoint_sha,
        # Recorded so a reader can tell a digest that was checked against
        # something from one that was merely computed and written down.
        "checkpoint_sha256_verified": bool(args.expected_checkpoint_sha256),
        "model_loader": "ScrollPrize/villa ink-detection/optimized_inference/model_resnet3d_3d_decoder.py",
        "input": {
            "tiff_dir": str(args.tiff_dir),
            "layers_total": len(files),
            "frames": args.frames,
            "depth_center": centre,
            "layer_span": [int(chosen[0].stem), int(chosen[-1].stem)],
            "shape_depth_y_x": [depth, height, width],
            "source_pixel_um": args.source_pixel_um,
            "rescaled": False,
        },
        "sliding_window": {
            "tile_size": tile,
            "stride": stride,
            "output_scale_factor": scale,
            "candidate_tiles": len(positions),
            "kept_tiles": len(kept),
            "discarded_fraction": 1 - len(kept) / max(len(positions), 1),
            "min_valid_ratio": args.min_valid_ratio,
        },
        # The masked map over the plane the tiles actually covered -- the same
        # pair this lane already saves and reports statistics on.
        "liveness": assess_liveness(result, valid=valid_np),
        "statistics": {
            "valid_pixels": int(finite.size),
            "p50": float(np.percentile(finite[::7], 50)) if finite.size else 0.0,
            "p90": float(np.percentile(finite[::7], 90)) if finite.size else 0.0,
            "p99": float(np.percentile(finite[::7], 99)) if finite.size else 0.0,
            "max": float(finite.max()) if finite.size else 0.0,
        },
        "runtime_seconds": round(time.time() - started, 2),
        "non_claims": [
            "A probability map is not OCR and does not accept ink, text, letters or First Letters.",
            "This lane is scale-restricted: it was validated at ~2 um and must not be routed to 8.64/9.362 um targets without its own control.",
        ],
    }
    (args.output / "INK_CANONICAL_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({k: receipt[k] for k in ("statistics", "sliding_window", "runtime_seconds")}, indent=2))
    return refuse_if_not_alive(
        receipt["liveness"],
        lane="ink-canonical-2um@1.0.0",
        output=args.output,
        on_degenerate=args.on_degenerate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
