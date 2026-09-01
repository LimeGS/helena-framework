#!/usr/bin/env python3
"""Run physically normalized, private TimeSformer ink screening on a CT stack.

This is a screening tool, not a letter classifier.  It preserves raw
probabilities for multiple depth centres and tiling offsets so stability can be
measured without normalizing each result by its own maximum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# The repository root, so the shared device resolver is importable from a
# script launched by path. `auto` has to mean the same thing in every runner or
# it is six defaults again, wearing one word.
_ROOT = Path(__file__).resolve()
while _ROOT != _ROOT.parent and not (_ROOT / "framework").is_dir():
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from framework.contracts.host_probe import resolve_device  # noqa: E402
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.contracts.lane_liveness import (  # noqa: E402
    assess_liveness,
    refuse_if_not_alive,
)
from framework.contracts.slice_order import ordered_tiff_files  # noqa: E402

# FIX-09: the training scale is declared by the ink lane profile, never by a
# literal here.  The contract lives in framework/contracts rather than in the
# Stage 04 analysis module, because that module needs scipy and the project's
# pinned ink image deliberately does not carry it -- importing from there
# broke this runner inside the very container built to run it.
from framework.contracts.physical_scale import (  # noqa: E402
    DEFAULT_INK_PROFILE,
    PIXEL_UM_TOLERANCE,
    resolve_training_pixel_um,
)


DEFAULT_METHOD_REGISTRY = (
    Path(__file__).resolve().parents[3]
    / "registries"
    / "method-capabilities-0.1.0.json"
)

# T4a: the frozen v1 preprocessing chain quantizes to uint8 twice -- once in
# ``interpolate_depth`` and again in ``resize_stack`` -- before ``infer_map``
# divides by 255 and hands the model float32.  ``float32`` keeps the whole
# chain in float and leaves the single normalization the network consumes.
LEGACY_PREPROCESS_PRECISION = "uint8"
PREPROCESS_PRECISIONS: dict[str, Any] = {
    "uint8": np.uint8,
    "float32": np.float32,
}
PREPROCESS_UINT8_QUANTIZATION_STEPS = {"uint8": 2, "float32": 0}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_checkpoint_identity(
    *,
    registry_path: Path,
    checkpoint_sha256: str,
    declared_model_family: str,
    allow_unregistered_checkpoint: bool,
) -> dict[str, Any]:
    if not registry_path.is_file():
        if allow_unregistered_checkpoint:
            return {
                "status": "UNREGISTERED_CHECKPOINT_EXPLICITLY_ALLOWED",
                "method_id": None,
                "registry_path": str(registry_path),
                "registry_sha256": None,
            }
        raise RuntimeError(f"method registry is required: {registry_path}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    matches = [
        entry
        for entry in registry.get("entries", [])
        if entry.get("known_checkpoint_sha256") == checkpoint_sha256
    ]
    if len(matches) > 1:
        raise RuntimeError(
            f"checkpoint appears more than once in registry: {checkpoint_sha256}"
        )
    if not matches:
        if allow_unregistered_checkpoint:
            return {
                "status": "UNREGISTERED_CHECKPOINT_EXPLICITLY_ALLOWED",
                "method_id": None,
                "registry_path": str(registry_path.resolve()),
                "registry_sha256": sha256_file(registry_path),
            }
        raise RuntimeError(
            f"checkpoint is not registered: {checkpoint_sha256}"
        )
    entry = matches[0]
    aliases = list(entry.get("receipt_model_family_aliases", []))
    if declared_model_family not in aliases:
        raise RuntimeError(
            "model family disagrees with checkpoint SHA-256: "
            f"declared={declared_model_family!r}, method={entry['method_id']!r}, "
            f"allowed={aliases!r}"
        )
    return {
        "status": "REGISTERED_CHECKPOINT_FAMILY_MATCH",
        "method_id": entry["method_id"],
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": sha256_file(registry_path),
        "accepted_model_family_aliases": aliases,
    }


def resolve_training_slice_um(
    *,
    profile_path: Path,
    requested: float | None,
) -> float:
    """Resolve the training slice pitch from the same ink lane profile.

    ``resolve_training_pixel_um`` owns the in-plane half of the FIX-09 contract.
    The depth half reuses its imported tolerance rather than declaring a second,
    driftable one.
    """

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    contract = profile.get("input_contract")
    if not isinstance(contract, dict) or "training_slice_um" not in contract:
        raise RuntimeError(
            f"ink lane profile declares no training_slice_um: {profile_path}"
        )
    declared = float(contract["training_slice_um"])
    if requested is not None and abs(float(requested) - declared) > PIXEL_UM_TOLERANCE:
        raise RuntimeError(
            f"--training-slice-um {requested} disagrees with the ink lane profile "
            f"value {declared} ({profile.get('profile_id')}); refusing to rescale "
            "silently"
        )
    return declared


def resolve_preprocess_precision(
    *,
    profile_path: Path,
    requested: str | None,
) -> str:
    """Resolve how many times preprocessing quantizes, from the lane profile.

    A profile that declares nothing keeps the frozen ``uint8`` path, so every
    existing profile reproduces bit for bit.  A profile that declares
    ``float32`` opts into the single-quantization chain, which is a different
    numeric result and therefore a different profile identity.
    """

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    contract = profile.get("input_contract")
    if not isinstance(contract, dict):
        raise RuntimeError(
            f"ink lane profile declares no input_contract: {profile_path}"
        )
    declared = contract.get("preprocess_precision", LEGACY_PREPROCESS_PRECISION)
    if declared not in PREPROCESS_PRECISIONS:
        raise RuntimeError(
            f"ink lane profile declares an unknown preprocess_precision "
            f"{declared!r} ({profile.get('profile_id')}); expected one of "
            f"{sorted(PREPROCESS_PRECISIONS)}"
        )
    if requested is not None and requested != declared:
        raise RuntimeError(
            f"--preprocess-precision {requested!r} disagrees with the ink lane "
            f"profile value {declared!r} ({profile.get('profile_id')}); refusing "
            "to change the numeric path silently"
        )
    return declared


def parse_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def depth_positions(
    center: float,
    frames: int,
    *,
    source_slice_um: float,
    training_slice_um: float,
) -> np.ndarray:
    if frames < 2 or source_slice_um <= 0 or training_slice_um <= 0:
        raise ValueError("invalid depth sampling parameters")
    offsets = np.arange(frames, dtype=np.float64) - (frames - 1) / 2.0
    return center + offsets * training_slice_um / source_slice_um


def resolve_preprocess_dtype(dtype: Any) -> np.dtype:
    resolved = np.dtype(dtype)
    if resolved not in {np.dtype(value) for value in PREPROCESS_PRECISIONS.values()}:
        raise ValueError(
            f"unsupported preprocessing dtype {resolved}; "
            f"expected one of {sorted(PREPROCESS_PRECISIONS)}"
        )
    return resolved


def interpolate_depth(
    stack: np.ndarray,
    positions: np.ndarray,
    *,
    dtype: Any = np.uint8,
) -> np.ndarray:
    """Resample the depth axis onto the training slice pitch.

    ``dtype=np.uint8`` reproduces the frozen v1 path, which rounds every
    interpolated plane back onto 256 levels.  ``dtype=np.float32`` keeps the
    interpolated values the resample actually produced.
    """

    resolved = resolve_preprocess_dtype(dtype)
    if stack.ndim != 3:
        raise ValueError("stack must be depth,y,x")
    if positions.min() < 0 or positions.max() > stack.shape[0] - 1:
        raise ValueError("depth positions extend beyond the source stack")
    output = np.empty((len(positions), *stack.shape[1:]), dtype=resolved)
    for index, position in enumerate(positions):
        lower = int(math.floor(float(position)))
        upper = min(lower + 1, stack.shape[0] - 1)
        weight = float(position - lower)
        mixed = (
            stack[lower].astype(np.float32) * (1.0 - weight)
            + stack[upper].astype(np.float32) * weight
        )
        if resolved == np.dtype(np.uint8):
            output[index] = np.rint(mixed).astype(np.uint8)
        else:
            output[index] = mixed
    return output


def grid_positions(length: int, tile: int, stride: int, offset: int) -> list[int]:
    if length < tile or tile <= 0 or stride <= 0 or offset < 0:
        raise ValueError("invalid tiling parameters")
    end = length - tile
    starts = {0, end}
    starts.update(range(offset, end + 1, stride))
    return sorted(starts)


def hann_kernel(size: int) -> np.ndarray:
    line = np.hanning(size + 2).astype(np.float32)[1:-1]
    kernel = np.outer(line, line)
    return kernel / float(kernel.max())


def load_tiff_stack(directory: Path) -> tuple[np.ndarray, list[Path], str]:
    files, ordering = ordered_tiff_files(directory)
    arrays: list[np.ndarray] = []
    for path in files:
        with Image.open(path) as image:
            array = np.asarray(image)
        if array.dtype != np.uint8:
            raise RuntimeError(
                f"TIFF slice {path.name} has dtype {array.dtype}; "
                "TimeSformer v1 accepts only native uint8 renders and will not "
                "silently cast 16-bit input"
            )
        if array.ndim != 2:
            raise RuntimeError(
                f"TIFF slice {path.name} has shape {array.shape}; "
                "TimeSformer v1 requires one grayscale plane per file"
            )
        arrays.append(array.copy())
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise RuntimeError("TIFF slices do not share one shape")
    return np.stack(arrays, axis=0), files, ordering


def resize_stack(
    stack: np.ndarray,
    *,
    target_height: int,
    target_width: int,
    dtype: Any = np.uint8,
) -> np.ndarray:
    """Resample the plane axes onto the training pixel pitch.

    ``dtype=np.uint8`` reproduces the frozen v1 path, which rounds the bilinear
    result back onto 256 levels a second time.  ``dtype=np.float32`` clamps to
    the physical 8-bit range without rounding, so the intermediate greys the
    1.18x upscale creates survive to the model input.
    """

    resolved = resolve_preprocess_dtype(dtype)

    import torch
    import torch.nn.functional as functional

    tensor = torch.from_numpy(np.ascontiguousarray(stack, dtype=np.float32))[None]
    resized = functional.interpolate(
        tensor,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )[0]
    if resolved == np.dtype(np.uint8):
        return torch.clamp(torch.round(resized), 0, 255).to(torch.uint8).numpy()
    return torch.clamp(resized, 0.0, 255.0).to(torch.float32).numpy()


def load_checkpoint_state(checkpoint: Path) -> dict[str, Any]:
    if checkpoint.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(checkpoint), device="cpu"))

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint lacks state_dict")
    return state


def load_model_architecture(config_path: Path | None) -> dict[str, int]:
    """Load the small set of architecture fields required by TimeSformer.

    The original GP Scroll1 checkpoint predates our explicit config support,
    so omitting ``config_path`` preserves its exact frozen architecture.
    """

    architecture = {
        "dim": 512,
        "depth": 8,
        "n_heads": 6,
        "num_classes": 16,
        "num_frames": 26,
        "patch_size": 16,
        "window_size": 64,
    }
    if config_path is None:
        return architecture
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    for name in architecture:
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"model config has invalid {name}: {value!r}")
        architecture[name] = value
    return architecture


def load_model(
    checkpoint: Path,
    frames: int,
    device: str,
    *,
    architecture: dict[str, int] | None = None,
):
    import torch
    import torch.nn as nn
    from timesformer_pytorch import TimeSformer

    parameters = architecture or load_model_architecture(None)
    if frames != parameters["num_frames"]:
        raise RuntimeError(
            "requested frames disagree with model config: "
            f"{frames} != {parameters['num_frames']}"
        )

    class InkTimeSformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = TimeSformer(
                dim=parameters["dim"],
                image_size=parameters["window_size"],
                patch_size=parameters["patch_size"],
                num_frames=frames,
                num_classes=parameters["num_classes"],
                channels=1,
                depth=parameters["depth"],
                heads=parameters["n_heads"],
                dim_head=64,
                attn_dropout=0.1,
                ff_dropout=0.1,
            )

        def forward(self, value):
            value = value.permute(0, 2, 1, 3, 4)
            return self.backbone(value).view(-1, 1, 4, 4)

    state = load_checkpoint_state(checkpoint)
    model = InkTimeSformer()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model.to(device).eval()


def batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def infer_map(
    stack: np.ndarray,
    model,
    *,
    device: str,
    tile_size: int,
    stride: int,
    tiling_offset: int,
    batch_size: int,
    min_valid_ratio: float,
    max_clip_value: int = 200,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    import torch
    import torch.nn.functional as functional

    height, width = stack.shape[1:]
    valid = np.any(stack != 0, axis=0)
    coordinates: list[tuple[int, int]] = []
    for y in grid_positions(height, tile_size, stride, tiling_offset):
        for x in grid_positions(width, tile_size, stride, tiling_offset):
            ratio = float(valid[y : y + tile_size, x : x + tile_size].mean())
            if ratio >= min_valid_ratio:
                coordinates.append((y, x))

    kernel = hann_kernel(tile_size)
    accumulator = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    for group in batched(coordinates, batch_size):
        patches = np.stack(
            [
                stack[:, y : y + tile_size, x : x + tile_size]
                for y, x in group
            ],
            axis=0,
        )
        tensor = torch.from_numpy(patches)[:, None].to(device).float()
        # The upstream contract is clip(0, max_clip_value) followed by
        # A.ToFloat(max_value=max_clip_value): the divisor is the clip value
        # itself, not 255. See ink-detection/optimized_inference/inference.py
        # (CFG.max_clip_value = 200, the A.ToFloat line). Dividing by 255
        # compressed the whole input to 78.4% of the contrast range the model
        # was trained on.
        tensor.clamp_(0, max_clip_value).div_(float(max_clip_value))
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")
        ):
            logits = model(tensor)
            probability = torch.sigmoid(logits.float())
            probability = functional.interpolate(
                probability,
                size=(tile_size, tile_size),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        values = probability.cpu().numpy()
        for (y, x), value in zip(group, values):
            local_valid = valid[y : y + tile_size, x : x + tile_size]
            local_weight = kernel * local_valid
            accumulator[y : y + tile_size, x : x + tile_size] += (
                value * local_weight
            )
            weights[y : y + tile_size, x : x + tile_size] += local_weight
    prediction = np.divide(
        accumulator,
        weights,
        out=np.zeros_like(accumulator),
        where=weights > 0,
    )
    all_grid_tiles = len(
        grid_positions(height, tile_size, stride, tiling_offset)
    ) * len(grid_positions(width, tile_size, stride, tiling_offset))
    candidate_tiles = len(coordinates)
    discarded_tiles = all_grid_tiles - candidate_tiles
    return prediction, weights > 0, {
        "candidate_tiles": candidate_tiles,
        "all_grid_tiles": all_grid_tiles,
        "discarded_tiles": discarded_tiles,
        "candidate_tile_fraction": (
            float(candidate_tiles / all_grid_tiles) if all_grid_tiles else 0.0
        ),
        "discarded_tile_fraction": (
            float(discarded_tiles / all_grid_tiles) if all_grid_tiles else 0.0
        ),
    }


def save_probability_png(path: Path, prediction: np.ndarray, valid: np.ndarray) -> None:
    values = prediction[valid]
    if values.size:
        lower, upper = np.percentile(values, [50, 99.5])
    else:
        lower, upper = 0.0, 1.0
    scaled = np.clip(
        (prediction - lower) / max(float(upper - lower), 1e-6),
        0,
        1,
    )
    Image.fromarray(np.rint(scaled * 255).astype(np.uint8)).save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--tiff-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        type=Path,
        help=(
            "Optional Hugging Face config.json. Omit only for the frozen "
            "GP Scroll1 small architecture."
        ),
    )
    parser.add_argument(
        "--model-family",
        default="timesformer_scroll5_july_retreat",
        help="Stable provenance label written to the receipt.",
    )
    parser.add_argument(
        "--method-registry",
        type=Path,
        default=DEFAULT_METHOD_REGISTRY,
        help="Hash-authoritative capability registry. Required by default.",
    )
    parser.add_argument(
        "--allow-unregistered-checkpoint",
        action="store_true",
        help=(
            "Explicit experimental escape hatch for an unknown checkpoint. "
            "A known hash with the wrong family is always rejected."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-centers", type=parse_ints, default=[32])
    parser.add_argument("--tiling-offsets", type=parse_ints, default=[0])
    parser.add_argument("--frames", type=int, default=26)
    parser.add_argument(
        "--ink-profile",
        type=Path,
        default=DEFAULT_INK_PROFILE,
        help=(
            "Ink lane profile that declares the training scale and the "
            "preprocessing precision. FIX-09: no scale literal lives here."
        ),
    )
    # FIX-09: no source-scale default either.  A 9.362 default silently
    # rescales the four 8.64 um rolls (PHerc0268/0800/1218/1447) by 8.4%, and
    # tests/test_ink_precision.py measures what that costs: the recovered peak
    # moves 28/36/43 um.  The caller must state the scale, and it is checked
    # against the frozen volume catalogue.
    parser.add_argument("--source-pixel-um", type=float, default=None, required=False)
    parser.add_argument("--training-pixel-um", type=float, default=None)
    parser.add_argument("--source-slice-um", type=float, default=None, required=False)
    parser.add_argument("--training-slice-um", type=float, default=None)
    parser.add_argument(
        "--preprocess-precision",
        choices=sorted(PREPROCESS_PRECISIONS),
        default=None,
        help=(
            "Optional assertion about the profile's preprocessing precision. "
            "A disagreement aborts; the profile stays authoritative."
        ),
    )
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-valid-ratio", type=float, default=0.60)
    parser.add_argument("--max-clip-value", type=int, default=200)
    parser.add_argument("--device", default="auto",
                        help="auto (the card if this host has one), cpu, or cuda[:N] to require one")
    parser.add_argument(
        "--on-degenerate",
        choices=("fail", "warn"),
        default="fail",
        help="what to do when the output map carries no decision (default: fail closed)",
    )
    args = parser.parse_args()

    # Resolve the device before anything expensive: an explicit `cuda` on a host
    # with no card is refused here rather than several minutes into a load, and
    # `auto` becomes the word the receipt can stand behind.
    _device = resolve_device(args.device)
    args.device = _device["device"]

    started = time.monotonic()
    ink_profile = args.ink_profile.resolve()
    training_pixel_um, training_scale_provenance = resolve_training_pixel_um(
        profile_path=ink_profile,
        requested=args.training_pixel_um,
    )
    training_slice_um = resolve_training_slice_um(
        profile_path=ink_profile,
        requested=args.training_slice_um,
    )
    preprocess_precision = resolve_preprocess_precision(
        profile_path=ink_profile,
        requested=args.preprocess_precision,
    )
    preprocess_dtype = PREPROCESS_PRECISIONS[preprocess_precision]
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint_identity = resolve_checkpoint_identity(
        registry_path=args.method_registry.resolve(),
        checkpoint_sha256=checkpoint_sha256,
        declared_model_family=args.model_family,
        allow_unregistered_checkpoint=args.allow_unregistered_checkpoint,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source, source_files, source_ordering = load_tiff_stack(
        args.tiff_dir.resolve()
    )
    model_config = args.model_config.resolve() if args.model_config else None
    architecture = load_model_architecture(model_config)
    if args.tile_size != architecture["window_size"]:
        raise RuntimeError(
            "tile size disagrees with model config: "
            f"{args.tile_size} != {architecture['window_size']}"
        )
    for name, value in (
        ("--source-pixel-um", args.source_pixel_um),
        ("--source-slice-um", args.source_slice_um),
    ):
        if value is None:
            raise RuntimeError(
                f"{name} is required: it has no safe default. The campaign spans "
                "8.64 and 9.362 um acquisitions, and assuming either one rescales "
                "the other by 8.4% in silence. Resolve it from "
                "workspace/catalog/eligible_volumes.json for this sample."
            )
        if value <= 0:
            raise RuntimeError(f"{name} must be positive, got {value}")

    target_height = round(source.shape[1] * args.source_pixel_um / training_pixel_um)
    target_width = round(source.shape[2] * args.source_pixel_um / training_pixel_um)
    model = load_model(
        checkpoint_path,
        args.frames,
        args.device,
        architecture=architecture,
    )
    records: list[dict[str, Any]] = []
    maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for center in args.depth_centers:
        positions = depth_positions(
            center,
            args.frames,
            source_slice_um=args.source_slice_um,
            training_slice_um=training_slice_um,
        )
        normalized = interpolate_depth(source, positions, dtype=preprocess_dtype)
        normalized = resize_stack(
            normalized,
            target_height=target_height,
            target_width=target_width,
            dtype=preprocess_dtype,
        )
        for tiling_offset in args.tiling_offsets:
            run_started = time.monotonic()
            prediction, valid, counts = infer_map(
                normalized,
                model,
                device=args.device,
                tile_size=args.tile_size,
                stride=args.stride,
                tiling_offset=tiling_offset,
                batch_size=args.batch_size,
                min_valid_ratio=args.min_valid_ratio,
                max_clip_value=args.max_clip_value,
            )
            stem = f"center-{center:03d}_offset-{tiling_offset:02d}"
            npy_path = output / f"{stem}.npy"
            png_path = output / f"{stem}.png"
            np.save(npy_path, prediction.astype(np.float16))
            save_probability_png(png_path, prediction, valid)
            values = prediction[valid]
            records.append(
                {
                    "depth_center_source_index": center,
                    "source_depth_positions": positions.tolist(),
                    "tiling_offset": tiling_offset,
                    **counts,
                    "valid_prediction_pixels": int(valid.sum()),
                    "probability_mean": float(values.mean()) if values.size else 0.0,
                    "probability_p99": (
                        float(np.percentile(values, 99)) if values.size else 0.0
                    ),
                    "duration_seconds": time.monotonic() - run_started,
                    "npy": npy_path.name,
                    "npy_sha256": sha256_file(npy_path),
                    "png": png_path.name,
                    "png_sha256": sha256_file(png_path),
                }
            )
            maps.append(prediction)
            masks.append(valid)

    stack_maps = np.stack(maps)
    common_valid = np.logical_and.reduce(masks)
    mean_map = stack_maps.mean(axis=0)
    std_map = stack_maps.std(axis=0)
    mean_npy = output / "mean_probability.npy"
    std_npy = output / "stability_std.npy"
    mean_png = output / "mean_probability.png"
    std_png = output / "stability_std.png"
    np.save(mean_npy, mean_map.astype(np.float16))
    np.save(std_npy, std_map.astype(np.float16))
    save_probability_png(mean_png, mean_map, common_valid)
    save_probability_png(std_png, std_map, common_valid)
    receipt = {
        "kind": "campaign_x_phase4_timesformer_private_screening_v1",
        "generated_at_utc": utc_now(),
        "status": "COMPLETED_DIAGNOSTIC_ONLY",
        "scope": "PRIVATE_LOCAL_FUNCTIONAL_INK_SCREENING",
        "sample_id": args.sample_id,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "model_family": args.model_family,
            "method_identity": checkpoint_identity,
            "target_overlap_known": False,
            "architecture": architecture,
            "config": (
                {
                    "path": str(model_config),
                    "sha256": sha256_file(model_config),
                }
                if model_config
                else {
                    "path": None,
                    "sha256": None,
                    "interpretation": "frozen GP Scroll1 small defaults",
                }
            ),
        },
        "input": {
            "tiff_directory": str(args.tiff_dir.resolve()),
            "slice_count": len(source_files),
            "slice_ordering": source_ordering,
            "ordered_slice_names": {
                "first": source_files[0].name,
                "middle": source_files[len(source_files) // 2].name,
                "last": source_files[-1].name,
            },
            "shape_depth_y_x": list(source.shape),
            "dtype": str(source.dtype),
            "dtype_policy": "NATIVE_UINT8_REQUIRED_FAIL_CLOSED",
            "source_slice_hashes": {
                source_files[0].name: sha256_file(source_files[0]),
                source_files[len(source_files) // 2].name: sha256_file(
                    source_files[len(source_files) // 2]
                ),
                source_files[-1].name: sha256_file(source_files[-1]),
            },
        },
        "physical_normalization": {
            "source_pixel_um": args.source_pixel_um,
            "training_pixel_um": training_pixel_um,
            "source_slice_um": args.source_slice_um,
            "training_slice_um": training_slice_um,
            "target_shape_y_x": [target_height, target_width],
            "frames": args.frames,
            "training_scale_provenance": training_scale_provenance,
            "preprocess_precision": preprocess_precision,
            "preprocess_uint8_quantization_steps": (
                PREPROCESS_UINT8_QUANTIZATION_STEPS[preprocess_precision]
            ),
            "model_input_normalization": (
                f"clamp(0,{args.max_clip_value})/{args.max_clip_value} to float32 in infer_map"
            ),
        },
        "inference": {
            "depth_centers": args.depth_centers,
            "tiling_offsets": args.tiling_offsets,
            "tile_size": args.tile_size,
            "stride": args.stride,
            "batch_size": args.batch_size,
            "min_valid_ratio": args.min_valid_ratio,
            "device": args.device,
            "device_selection": _device,
            "runs": records,
            "tile_filter_summary": {
                "all_grid_tiles": sum(
                    int(record["all_grid_tiles"]) for record in records
                ),
                "candidate_tiles": sum(
                    int(record["candidate_tiles"]) for record in records
                ),
                "discarded_tiles": sum(
                    int(record["discarded_tiles"]) for record in records
                ),
                "discarded_tile_fraction": (
                    sum(int(record["discarded_tiles"]) for record in records)
                    / sum(int(record["all_grid_tiles"]) for record in records)
                    if sum(int(record["all_grid_tiles"]) for record in records)
                    else 0.0
                ),
            },
        },
        "aggregate_artifacts": {
            path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in (mean_npy, std_npy, mean_png, std_png)
        },
        "common_valid_pixels": int(common_valid.sum()),
        # The ensemble mean over the pixels every model in it could see. The
        # per-model maps are not assessed separately: the mean is what the
        # receipt publishes and what a screen would read, so it is the map that
        # has to carry a decision.
        "liveness": assess_liveness(mean_map, valid=common_valid),
        "duration_seconds": time.monotonic() - started,
        "explicit_non_claims": [
            "not automatic letter acceptance",
            "not a First Letters submission claim",
            "not independent external validation",
            "model output requires raw-CT and stability review",
        ],
    }
    write_json(output / "INK_SCREENING_RECEIPT.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    refusal = refuse_if_not_alive(
        receipt["liveness"],
        lane=str(args.model_family or "timesformer"),
        output=output,
        on_degenerate=args.on_degenerate,
    )
    if refusal:
        return refusal
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
