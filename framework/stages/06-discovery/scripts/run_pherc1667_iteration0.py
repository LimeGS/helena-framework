#!/usr/bin/env python3
"""Run the frozen PHerc1667 iteration-0 ink model on one surface stack.

This runner follows the executable tiling recipe published with
``scrollprize/PHerc.1667-iteration-0``:

* exactly 62 uint8 surface layers;
* 256 x 256 spatial tiles;
* overlapping quarter-resolution predictions upsampled and averaged;
* raw values clipped to [0, 200];
* an explicit, receipt-recorded intensity transform.  The model's training
  pipeline used Albumentations ``Normalize(mean=0, std=1)``, which divides
  uint8 inputs by the default ``max_pixel_value=255``.  The published tiling
  snippet omits that division, so the runner requires the caller to name the
  interpretation rather than silently choosing one.

The output is a screening map.  It is not an ink, letter, or First Letters
classification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from framework.contracts.slice_order import (  # noqa: E402
    ordered_tiff_files as _ordered_tiff_files,
)


EXPECTED_DEPTH = 62
EXPECTED_TILE = 256
INPUT_TRANSFORMS = (
    "clip-divide-255",
    "clip-float-no-scaling",
)


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


def ordered_tiff_files(directory: Path) -> tuple[list[Path], str]:
    """Apply the shared slice-order contract with this runner's extra rules."""

    return _ordered_tiff_files(
        directory,
        suffixes=(".tif", ".tiff"),
        require_numeric=True,
        require_contiguous=True,
    )


def load_stack(directory: Path, *, reverse_layers: bool) -> tuple[
    np.ndarray, list[Path], str
]:
    files, ordering = ordered_tiff_files(directory)
    if len(files) != EXPECTED_DEPTH:
        raise RuntimeError(
            f"model requires exactly {EXPECTED_DEPTH} layers; found {len(files)}"
        )
    arrays: list[np.ndarray] = []
    for path in files:
        with Image.open(path) as opened:
            value = np.asarray(opened.convert("L"), dtype=np.uint8)
        arrays.append(value)
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise RuntimeError("TIFF layers do not share one shape")
    stack = np.stack(arrays, axis=0)
    if reverse_layers:
        stack = stack[::-1].copy()
        files = list(reversed(files))
        ordering += "_THEN_REVERSED"
    return stack, files, ordering


def grid_positions(length: int, tile: int, stride: int) -> list[int]:
    if length < tile or tile <= 0 or stride <= 0:
        raise ValueError("invalid tiling dimensions")
    end = length - tile
    starts = set(range(0, end + 1, stride))
    starts.add(end)
    return sorted(starts)


def batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for index in range(0, len(values), size):
        yield values[index : index + size]


def derive_fragment_mask(stack: np.ndarray) -> np.ndarray:
    if stack.ndim != 3:
        raise ValueError("stack must be depth,y,x")
    return np.any(stack != 0, axis=0)


def preprocess_stack(stack: np.ndarray, transform: str) -> np.ndarray:
    clipped = np.clip(stack, 0, 200).astype(np.float32)
    if transform == "clip-divide-255":
        return clipped / 255.0
    if transform == "clip-float-no-scaling":
        return clipped
    raise ValueError(f"unsupported input transform: {transform}")


def load_fragment_mask(
    mask_path: Path | None,
    *,
    expected_shape: tuple[int, int],
    stack: np.ndarray,
) -> tuple[np.ndarray, str]:
    if mask_path is None:
        return derive_fragment_mask(stack), "DERIVED_ANY_NONZERO_ACROSS_DEPTH"
    with Image.open(mask_path) as opened:
        mask = np.asarray(opened.convert("L")) > 0
    if mask.shape != expected_shape:
        raise RuntimeError(
            f"fragment mask shape {mask.shape} != stack shape {expected_shape}"
        )
    return mask, "EXPLICIT_NONZERO_MASK"


def eligible_coordinates(
    mask: np.ndarray,
    *,
    tile: int,
    stride: int,
    min_valid_ratio: float,
) -> tuple[list[tuple[int, int]], int]:
    if not 0.0 < min_valid_ratio <= 1.0:
        raise ValueError("min-valid-ratio must be in (0,1]")
    height, width = mask.shape
    kept: list[tuple[int, int]] = []
    considered = 0
    for y in grid_positions(height, tile, stride):
        for x in grid_positions(width, tile, stride):
            considered += 1
            if float(mask[y : y + tile, x : x + tile].mean()) >= min_valid_ratio:
                kept.append((y, x))
    return kept, considered


def infer(
    stack: np.ndarray,
    mask: np.ndarray,
    model: Any,
    *,
    device: str,
    stride: int,
    batch_size: int,
    min_valid_ratio: float,
    inference_dtype: str,
    input_transform: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    import torch
    import torch.nn.functional as functional

    coordinates, considered = eligible_coordinates(
        mask,
        tile=EXPECTED_TILE,
        stride=stride,
        min_valid_ratio=min_valid_ratio,
    )
    if not coordinates:
        raise RuntimeError("no eligible tiles after applying the fragment mask")

    height, width = mask.shape
    prediction_sum = np.zeros((height, width), dtype=np.float32)
    prediction_count = np.zeros((height, width), dtype=np.float32)
    transformed = preprocess_stack(stack, input_transform)
    use_amp = inference_dtype == "float16"
    device_type = torch.device(device).type
    if use_amp and device_type != "cuda":
        raise RuntimeError("float16 inference is supported only on CUDA")

    with torch.inference_mode():
        for group in batched(coordinates, batch_size):
            tiles = np.stack(
                [
                    transformed[
                        :,
                        y : y + EXPECTED_TILE,
                        x : x + EXPECTED_TILE,
                    ]
                    for y, x in group
                ],
                axis=0,
            )
            tensor = torch.from_numpy(tiles).unsqueeze(1).float().to(device)
            with torch.autocast(
                device_type=device_type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(tensor).logits
                probability = torch.sigmoid(logits)
                probability = functional.interpolate(
                    probability,
                    size=(EXPECTED_TILE, EXPECTED_TILE),
                    mode="bilinear",
                    align_corners=False,
                )
            values = probability[:, 0].float().cpu().numpy()
            for (y, x), value in zip(group, values):
                prediction_sum[
                    y : y + EXPECTED_TILE,
                    x : x + EXPECTED_TILE,
                ] += value
                prediction_count[
                    y : y + EXPECTED_TILE,
                    x : x + EXPECTED_TILE,
                ] += 1.0

    prediction = np.divide(
        prediction_sum,
        prediction_count,
        out=np.zeros_like(prediction_sum),
        where=prediction_count > 0,
    )
    return prediction, prediction_count, {
        "tiles_considered": considered,
        "tiles_inferred": len(coordinates),
        "tiles_skipped": considered - len(coordinates),
    }


def save_fixed_scale_png(path: Path, value: np.ndarray) -> None:
    image = np.clip(np.rint(value * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(image, mode="L").save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--tiff-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fragment-mask", type=Path)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--min-valid-ratio", type=float, default=1.0)
    parser.add_argument("--device", default="auto",
                        help="auto (the card if this host has one), cpu, or cuda[:N] to require one")
    parser.add_argument(
        "--inference-dtype",
        choices=("float32", "float16"),
        default="float16",
    )
    parser.add_argument(
        "--input-transform",
        choices=INPUT_TRANSFORMS,
        required=True,
        help=(
            "Frozen intensity interpretation. Use clip-divide-255 to reproduce "
            "Albumentations Normalize(mean=0,std=1,max_pixel_value=255)."
        ),
    )
    parser.add_argument("--reverse-layers", action="store_true")
    parser.add_argument("--source-pixel-um", type=float, required=True)
    parser.add_argument("--source-slice-um", type=float, required=True)
    parser.add_argument("--model-revision", required=True)
    args = parser.parse_args()

    # Resolve the device before anything expensive: an explicit `cuda` on a host
    # with no card is refused here rather than several minutes into a load, and
    # `auto` becomes the word the receipt can stand behind.
    _device = resolve_device(args.device)
    args.device = _device["device"]

    if args.stride <= 0 or args.stride > EXPECTED_TILE:
        raise ValueError("stride must be in [1,256]")
    if args.source_pixel_um <= 0 or args.source_slice_um <= 0:
        raise ValueError("physical resolutions must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("output directory must be absent or empty")
    args.output.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoModel

    torch.manual_seed(130697)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(130697)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    started = time.monotonic()
    stack, source_files, ordering = load_stack(
        args.tiff_dir.resolve(),
        reverse_layers=args.reverse_layers,
    )
    mask, mask_policy = load_fragment_mask(
        args.fragment_mask.resolve() if args.fragment_mask else None,
        expected_shape=stack.shape[1:],
        stack=stack,
    )
    model = AutoModel.from_pretrained(
        str(args.model_dir.resolve()),
        trust_remote_code=True,
        local_files_only=True,
    ).eval().to(args.device)
    prediction, count, tiling = infer(
        stack,
        mask,
        model,
        device=args.device,
        stride=args.stride,
        batch_size=args.batch_size,
        min_valid_ratio=args.min_valid_ratio,
        inference_dtype=args.inference_dtype,
        input_transform=args.input_transform,
    )

    probability_path = args.output / "probability.npy"
    count_path = args.output / "prediction_count.npy"
    preview_path = args.output / "probability-fixed-scale.png"
    coverage_path = args.output / "inference-coverage.png"
    np.save(probability_path, prediction)
    np.save(count_path, count)
    save_fixed_scale_png(preview_path, prediction)
    Image.fromarray((count > 0).astype(np.uint8) * 255, mode="L").save(
        coverage_path
    )

    covered = count > 0
    receipt = {
        "kind": "campaign_x_phase4_pherc1667_iteration0_inference_v1",
        "generated_at_utc": utc_now(),
        "status": "COMPLETED_SCREENING_ONLY",
        "sample_id": args.sample_id,
        "model": {
            "directory": str(args.model_dir.resolve()),
            "revision": args.model_revision,
            "input_depth": EXPECTED_DEPTH,
            "input_tile_y_x": [EXPECTED_TILE, EXPECTED_TILE],
            "transform": args.input_transform,
        },
        "input": {
            "directory": str(args.tiff_dir.resolve()),
            "shape_depth_y_x": list(stack.shape),
            "ordering": ordering,
            "reverse_layers": args.reverse_layers,
            "source_pixel_um": args.source_pixel_um,
            "source_slice_um": args.source_slice_um,
            "layers": [
                {
                    "path": str(path.resolve()),
                    "numeric_stem": int(path.stem),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in source_files
            ],
        },
        "fragment_mask": {
            "path": (
                str(args.fragment_mask.resolve()) if args.fragment_mask else None
            ),
            "policy": mask_policy,
            "valid_fraction": float(mask.mean()),
            "minimum_valid_ratio_per_tile": args.min_valid_ratio,
        },
        "inference": {
            "device": args.device,
            "device_selection": _device,
            "dtype": args.inference_dtype,
            "stride": args.stride,
            "batch_size": args.batch_size,
            "seed": 130697,
            "deterministic_algorithms": True,
            **tiling,
            "runtime_seconds": time.monotonic() - started,
        },
        "screening_summary": {
            "covered_pixel_fraction": float(covered.mean()),
            "mean_probability_covered": float(prediction[covered].mean()),
            "maximum_probability": float(prediction.max()),
            "fraction_covered_ge_0_5": float(
                (prediction[covered] >= 0.5).mean()
            ),
            "fraction_covered_ge_0_7": float(
                (prediction[covered] >= 0.7).mean()
            ),
        },
        "outputs": {
            "probability_npy": {
                "path": probability_path.name,
                "sha256": sha256_file(probability_path),
            },
            "prediction_count_npy": {
                "path": count_path.name,
                "sha256": sha256_file(count_path),
            },
            "probability_fixed_scale_png": {
                "path": preview_path.name,
                "sha256": sha256_file(preview_path),
            },
            "inference_coverage_png": {
                "path": coverage_path.name,
                "sha256": sha256_file(coverage_path),
            },
        },
        "explicit_non_claims": [
            "not an ink classification",
            "not a letter classification",
            "not a First Letters claim",
        ],
    }
    receipt_path = args.output / "INFERENCE_RECEIPT.json"
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
