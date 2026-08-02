#!/usr/bin/env python3
"""Profile-driven ink inference: one runner, N models.

Everything model-specific comes from the ink lane profile -- ``model_type``,
``frames``, ``tile_size_y_x``, ``training_pixel_um``, ``max_clip_value``,
``model_depth``, ``n_classes``.  Routing a different detector is a different
``--profile``, not a different script.

Replaces the two near-identical ad-hoc runners (``run_canonical2um.py`` and
``run_resnet3d50.py``) that differed in four numbers and one class.

Not covered here: ``timesformer-gp-scroll1``, which keeps
``run_ink_timesformer.py``.  That adapter carries depth interpolation,
multi-centre stability aggregation and physical-scale provenance that 95
historical receipts depend on bit-for-bit; folding it in would break
comparability for no gain.  Its profile declares the same fields, so the
contract is shared even though the entry point is not.
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

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.contracts.lane_liveness import (  # noqa: E402
    assess_liveness,
    refuse_if_not_alive,
)
from framework.contracts.slice_order import ordered_tiff_files  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

PROFILE_DIR = ROOT / "framework" / "profiles" / "03-ink"

# Upstream input contract, from ScrollPrize/villa
# ink-detection/optimized_inference/inference.py: the tile is clipped to
# CFG.max_clip_value and then passed through A.ToFloat(max_value=<the same
# value>).  The divisor is the clip, never 255.
DEFAULT_MAX_CLIP_VALUE = 200


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Model construction, dispatched on the profile's model_type.
#
# Each loader imports the project's own vendored upstream module unchanged, so
# the architecture cannot drift away from the checkpoint it is paired with.
# --------------------------------------------------------------------------

def _load_resnet3d(spec: dict, checkpoint: Path, device: torch.device):
    from safetensors.torch import load_file

    from model_resnet3d import RegressionPLModel  # type: ignore

    state = load_file(str(checkpoint), device="cpu")
    model = RegressionPLModel(
        size=spec["tile_size"],
        num_frames=spec["frames"],
        model_depth=spec.get("model_depth", 50),
    )
    # Some published checkpoints carry a classifier head whose width disagrees
    # with the upstream default (resnet50_7.9um declares 1139, the code says
    # 1039).  The head is unused -- the backbone runs with forward_features --
    # but resizing it to the checkpoint keeps the load strict, so a genuine
    # architecture mismatch still fails loudly.
    head = state.get("backbone.fc.weight")
    if head is not None and head.shape[0] != model.backbone.fc.out_features:
        import torch.nn as nn

        model.backbone.fc = nn.Linear(head.shape[1], head.shape[0])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    model.to(device)
    # eval() is called for its effect, never for its return value: the upstream
    # wrappers are not nn.Module subclasses and some return None.
    model.eval()
    return model


def _load_resnet3d_3d_decoder(spec: dict, checkpoint: Path, device: torch.device):
    from model_resnet3d_3d_decoder import load_model  # type: ignore

    model = load_model(str(checkpoint), device, num_frames=spec["frames"])
    model.eval()
    return model


LOADERS = {
    "resnet3d-50": _load_resnet3d,
    "resnet3d-152": _load_resnet3d,
    "resnet3d-152-3d-decoder": _load_resnet3d_3d_decoder,
}


REGISTRY = ROOT / "framework" / "registries" / "method-capabilities-0.1.0.json"


def registry_status(method_id: str) -> dict | None:
    """What the method registry says about this method, if anything."""
    if not REGISTRY.exists():
        return None
    data = json.loads(REGISTRY.read_text())
    entries = data.get("ink") or data.get("entries") or []
    return next((e for e in entries if e.get("method_id") == method_id), None)


def check_routable(method_id: str) -> None:
    """Refuse a method the registry has already disqualified.

    The liveness gate catches a dead lane after an hour of GPU time. This
    catches the ones already known dead, before any.
    """
    entry = registry_status(method_id)
    if entry is None:
        return
    status = str(entry.get("validation_status", ""))
    if "DISQUALIFIED" in status:
        raise RuntimeError(
            f"method {method_id} is {status} in the registry and must not be routed. "
            f"{entry.get('recommended_policy', '')}"
        )


def read_profile(path: Path) -> dict:
    profile = json.loads(path.read_text())
    if profile.get("schema") != "campaignx.ink_lane_profile.v1":
        raise RuntimeError(f"not an ink lane profile: {profile.get('schema')!r}")
    contract = profile.get("input_contract", {})
    model_type = contract.get("model_type")
    if model_type not in LOADERS:
        raise RuntimeError(
            f"profile {profile['profile_id']} declares model_type={model_type!r}; "
            f"this runner dispatches {sorted(LOADERS)}. "
            "timesformer lanes use run_ink_timesformer.py."
        )
    tile = contract.get("tile_size_y_x")
    if not (isinstance(tile, list) and len(tile) == 2 and tile[0] == tile[1]):
        raise RuntimeError(f"tile_size_y_x must be a square [n, n], got {tile!r}")
    for required in ("frames", "training_pixel_um"):
        if contract.get(required) is None:
            raise RuntimeError(f"profile is missing input_contract.{required}")
    check_routable(profile["method_id"])
    return {
        "profile": profile,
        "profile_id": profile["profile_id"],
        "method_id": profile["method_id"],
        "checkpoint_sha256": profile.get("checkpoint_sha256"),
        "model_type": model_type,
        "frames": int(contract["frames"]),
        "tile_size": int(tile[0]),
        "training_pixel_um": float(contract["training_pixel_um"]),
        "training_slice_um": float(
            contract.get("training_slice_um", contract["training_pixel_um"])
        ),
        "max_clip_value": int(contract.get("max_clip_value", DEFAULT_MAX_CLIP_VALUE)),
        "model_depth": contract.get("model_depth"),
        "execution": profile.get("default_execution", {}),
    }


def resolve_profile(argument: str) -> Path:
    """Accept a path, or a bare profile id resolved inside the profile directory."""
    direct = Path(argument)
    if direct.exists():
        return direct
    stem = argument.replace("@", "-").replace(".", "-", 0)
    for candidate in sorted(PROFILE_DIR.glob("*.json")):
        try:
            if json.loads(candidate.read_text()).get("profile_id") == argument:
                return candidate
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"no profile at {argument!r} and no profile_id match ({stem})")


def depth_positions(centre: float, frames: int, step: float) -> np.ndarray:
    """Fractional source-slice indices for ``frames`` samples spaced ``step`` apart.

    The spatial resample matches apparent feature *width*; this matches apparent
    feature *depth*. Both are needed. Taking ``frames`` consecutive slices from a
    2.399 um stack for a model trained at 7.9 um hands it a slab 3.3x too thin,
    and the model answers with a near-constant map.
    """
    offsets = np.arange(frames, dtype=np.float64) - (frames - 1) / 2.0
    return centre + offsets * step


def read_interpolated_depth(files: list[Path], positions: np.ndarray) -> np.ndarray:
    """Sample the stack at fractional depths, reading each source slice once."""
    lo = int(np.floor(positions.min()))
    hi = int(np.ceil(positions.max()))
    if lo < 0 or hi > len(files) - 1:
        raise RuntimeError(
            f"depth window [{positions.min():.2f}, {positions.max():.2f}] falls "
            f"outside the {len(files)} available layers"
        )
    cache: dict[int, np.ndarray] = {}

    def layer(index: int) -> np.ndarray:
        if index not in cache:
            cache[index] = np.asarray(Image.open(files[index]), dtype=np.float32)
        return cache[index]

    planes = []
    for p in positions:
        low = int(np.floor(p))
        high = min(low + 1, len(files) - 1)
        frac = float(p - low)
        planes.append(layer(low) if frac == 0.0 else layer(low) * (1.0 - frac) + layer(high) * frac)
    return np.stack(planes, axis=0)


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
    ap.add_argument("--profile", required=True, help="path or profile_id")
    ap.add_argument("--tiff-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--source-pixel-um", type=float, required=True)
    ap.add_argument(
        "--source-slice-um",
        type=float,
        default=None,
        help="defaults to --source-pixel-um (isotropic voxels)",
    )
    ap.add_argument(
        "--upstream-dir",
        type=Path,
        required=True,
        help="directory holding the vendored villa model modules",
    )
    # Run-shape only; anything model-defining comes from the profile.
    ap.add_argument("--depth-center", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--min-valid-ratio", type=float, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--on-degenerate",
        choices=("fail", "warn"),
        default="fail",
        help="what to do when the output map carries no decision (default: fail closed)",
    )
    args = ap.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output}")

    profile_path = resolve_profile(args.profile)
    spec = read_profile(profile_path)
    execution = spec["execution"]
    stride = args.stride if args.stride is not None else execution.get("stride", spec["tile_size"] // 2)
    min_valid_ratio = (
        args.min_valid_ratio
        if args.min_valid_ratio is not None
        else execution.get("minimum_valid_ratio", 0.60)
    )

    checkpoint_sha = sha256_file(args.checkpoint)
    declared = spec["checkpoint_sha256"]
    if declared and declared != checkpoint_sha:
        raise RuntimeError(
            f"checkpoint {checkpoint_sha} does not match profile {spec['profile_id']} "
            f"which pins {declared}"
        )

    files, ordering = ordered_tiff_files(args.tiff_dir.resolve(), require_numeric=True)
    if np.asarray(Image.open(files[0])).dtype != np.uint8:
        raise RuntimeError("expected uint8 layers")
    frames = spec["frames"]
    centre = args.depth_center if args.depth_center is not None else len(files) // 2

    source_slice_um = (
        args.source_slice_um if args.source_slice_um is not None else args.source_pixel_um
    )
    depth_step = spec["training_slice_um"] / source_slice_um
    depth_index = depth_positions(float(centre), frames, depth_step)
    stack = read_interpolated_depth(files, depth_index)

    valid_source = (stack > 0).any(axis=0)      # before the clip, as upstream does
    np.clip(stack, 0, spec["max_clip_value"], out=stack)

    # Match apparent feature size to the scale the checkpoint was trained at.
    # A ratio of exactly 1.0 skips the resample so a native-scale lane stays
    # bit-identical to feeding the stack straight through.
    ratio = args.source_pixel_um / spec["training_pixel_um"]
    src_h, src_w = stack.shape[1], stack.shape[2]
    if abs(ratio - 1.0) < 1e-9:
        resized = torch.from_numpy(stack)
    else:
        resized = F.interpolate(
            torch.from_numpy(stack)[None],
            size=(round(src_h * ratio), round(src_w * ratio)),
            mode="bilinear",
            align_corners=False,
        )[0]
    depth, height, width = resized.shape

    device = torch.device(args.device)
    sys.path.insert(0, str(args.upstream_dir.resolve()))
    model = LOADERS[spec["model_type"]](spec, args.checkpoint, device)

    # The volume stays on the host: 62x4096x4096 float32 is 3.9 GiB and will not
    # fit a shared 6 GiB GTX 1660.  Only tiles are moved.
    volume = resized.div_(float(spec["max_clip_value"]))
    valid = (
        F.interpolate(
            torch.from_numpy(valid_source).float()[None, None],
            size=(height, width),
            mode="nearest",
        )[0, 0]
        > 0.5
    ).numpy()

    tile = spec["tile_size"]
    weight = torch.from_numpy(hann2d(tile)).to(device)
    total = torch.zeros((height, width), dtype=torch.float32, device=device)
    norm = torch.zeros((height, width), dtype=torch.float32, device=device)
    positions = [(y, x) for y in grid(height, tile, stride) for x in grid(width, tile, stride)]
    kept = [
        (y, x)
        for (y, x) in positions
        if valid[y : y + tile, x : x + tile].mean() >= min_valid_ratio
    ]

    started = time.time()
    with torch.inference_mode():
        for i in range(0, len(kept), args.batch_size):
            chunk = kept[i : i + args.batch_size]
            batch = torch.stack(
                [volume[:, y : y + tile, x : x + tile] for (y, x) in chunk]
            ).to(device)
            # The 3D-decoder wrapper exposes forward() but is not callable.
            out = model.forward(batch) if hasattr(model, "forward") else model(batch)
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

    prob = torch.where(norm > 0, total / norm.clamp(min=1e-6), torch.zeros_like(total))
    result = np.where(valid, prob.detach().cpu().numpy(), 0.0)

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "probability.npy", result)
    v = result[result > 0]
    s = np.sort(v[::5]) if v.size else np.array([0.0])

    def quantile(p: float) -> float:
        return float(s[min(int(len(s) * p), len(s) - 1)])

    receipt = {
        "schema": "campaignx.ink_profile_screening_receipt.v1",
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "sample_id": args.sample_id,
        "lane": {
            "profile_id": spec["profile_id"],
            "profile_path": str(profile_path),
            "profile_sha256": sha256_file(profile_path),
            "method_id": spec["method_id"],
            "model_type": spec["model_type"],
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_pinned_by_profile": bool(declared),
        },
        "input": {
            "tiff_dir": str(args.tiff_dir),
            "slice_ordering": ordering,
            "layers_total": len(files),
            "frames": frames,
            "depth_center": centre,
            "depth_positions_source_index": [float(depth_index[0]), float(depth_index[-1])],
            "depth_step_source_slices": depth_step,
            "source_slice_um": source_slice_um,
            "training_slice_um": spec["training_slice_um"],
            "depth_span_um": float((depth_index[-1] - depth_index[0]) * source_slice_um),
            "source_pixel_um": args.source_pixel_um,
            "training_pixel_um": spec["training_pixel_um"],
            "resample_ratio": ratio,
            "resampled": abs(ratio - 1.0) >= 1e-9,
            "shape_source_y_x": [src_h, src_w],
            "shape_model_y_x": [height, width],
            "max_clip_value": spec["max_clip_value"],
            "normalization": (
                f"clip(0,{spec['max_clip_value']}) then divide by "
                f"{spec['max_clip_value']} (upstream inference.py "
                "CFG.max_clip_value + A.ToFloat)"
            ),
        },
        "sliding_window": {
            "tile_size": tile,
            "stride": stride,
            "candidate_tiles": len(positions),
            "kept_tiles": len(kept),
            "min_valid_ratio": min_valid_ratio,
        },
        "statistics": {
            "valid_pixels": int(v.size),
            "p50": quantile(0.50),
            "p90": quantile(0.90),
            "p99": quantile(0.99),
            "max": float(v.max()) if v.size else 0.0,
            "fraction_above_0_5": float((v > 0.5).mean()) if v.size else 0.0,
        },
        "liveness": assess_liveness(result, valid=valid),
        "runtime_seconds": round(time.time() - started, 2),
        "non_claims": [
            "A probability map is not OCR and does not accept ink, text, letters or First Letters.",
            "A DEGENERATE liveness verdict says the map carries no decision; it is not evidence about ink either way.",
            "Scale compatibility is not a transfer control; a lane still needs its own campaign positive control before any target use.",
        ],
    }
    (args.output / "INK_PROFILE_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt["statistics"], indent=2))

    # The same refusal every lane makes. It lived here, inline, and only here:
    # three adapters had no gate at all, which is the whole reason it moved into
    # the contract beside assess_liveness.
    refusal = refuse_if_not_alive(
        receipt["liveness"],
        lane=str(spec["profile_id"]),
        output=args.output,
        on_degenerate=args.on_degenerate,
    )
    if refusal:
        return refusal
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
