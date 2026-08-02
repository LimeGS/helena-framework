#!/usr/bin/env python3
"""Load or run the frozen 3D DINO-guided ink checkpoint with strict provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METHOD_ID = "ink-3d-dino-guided@1.0.0"
PROFILE_RELATIVE_PATH = Path(
    "framework/profiles/03-ink/ink-3d-dino-guided-diagnostic-1.0.0.json"
)

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.contracts.lane_liveness import (  # noqa: E402
    assess_liveness,
    refuse_if_not_alive,
)



def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def resolve_array(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    array_path = Path(manifest["array"]["path"])
    if not array_path.is_absolute():
        array_path = manifest_path.parent / array_path
    array_path = array_path.resolve()
    if not array_path.is_file():
        raise RuntimeError(f"input array does not exist: {array_path}")
    require_hash(array_path, manifest["array"]["sha256"], "input array")
    return array_path


def validate_manifest(manifest_path: Path, *, model_voxel_um: float) -> tuple[dict[str, Any], Path]:
    manifest = read_object(manifest_path)
    if manifest.get("schema") != "campaignx.ink_volumetric_patch_input.v1":
        raise RuntimeError("unsupported input manifest schema")
    if manifest["array"].get("format") != "npy":
        raise RuntimeError("only immutable NPY patch inputs are supported")
    if tuple(manifest["array"]["shape_zyx"]) != (256, 256, 256):
        raise RuntimeError("DINO input must be exactly 256x256x256 voxels")
    voxel_um = float(manifest["input_voxel_size_um"])
    if abs(voxel_um - model_voxel_um) > 0.01:
        raise RuntimeError(
            f"input voxel size {voxel_um} um is incompatible with frozen model scale {model_voxel_um} um"
        )
    if manifest["extraction"].get("resampled") and "resampling_receipt" not in manifest["extraction"]:
        raise RuntimeError("resampled inputs require a resampling_receipt")
    return manifest, resolve_array(manifest_path, manifest)


def load_execution_profile(
    profile_path: Path, *, repo_root: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    repo_root = repo_root.resolve()
    expected_profile_path = (repo_root / PROFILE_RELATIVE_PATH).resolve()
    if profile_path.resolve() != expected_profile_path:
        raise RuntimeError(
            f"profile must be the repository-pinned profile: {expected_profile_path}"
        )
    profile = read_object(expected_profile_path)
    if profile.get("schema") != "campaignx.ink_lane_profile.v1":
        raise RuntimeError("unsupported ink lane profile schema")
    if profile.get("method_id") != METHOD_ID:
        raise RuntimeError(f"profile method mismatch: {profile.get('method_id')}")
    for field in (
        "checkpoint_sha256",
        "config_sha256",
        "pinned_source_revision",
        "adapter",
    ):
        if not profile.get(field):
            raise RuntimeError(f"profile is missing required identity field: {field}")
    native_voxel_um = profile.get("input_contract", {}).get("native_voxel_um")
    if native_voxel_um is None:
        raise RuntimeError("profile has no native_voxel_um")
    adapter_path = (repo_root / profile["adapter"]).resolve()
    if adapter_path != Path(__file__).resolve():
        raise RuntimeError(
            f"profile adapter does not resolve to this executable: {adapter_path}"
        )
    return profile, {
        "path": PROFILE_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(expected_profile_path),
        "adapter": profile["adapter"],
        "adapter_sha256": sha256_file(adapter_path),
    }


def install_torch_compiler_compatibility(torch: Any) -> bool:
    """Allow current Villa decorators on PyTorch versions lacking reason=."""
    try:
        import inspect

        if "reason" in inspect.signature(torch.compiler.disable).parameters:
            return False
    except (TypeError, ValueError):
        pass
    original = torch.compiler.disable

    def compatible_disable(fn=None, recursive=True, reason=None):  # noqa: ARG001
        return original(fn, recursive=recursive)

    torch.compiler.disable = compatible_disable
    return True


def normalized_model_config(embedded: dict[str, Any]) -> dict[str, Any]:
    model_config = dict(embedded.get("model_config") or {})
    model_config.setdefault("patch_size", tuple(embedded["patch_size"]))
    model_config.setdefault("train_patch_size", tuple(embedded["patch_size"]))
    model_config.setdefault("batch_size", embedded.get("batch_size", 1))
    model_config.setdefault("train_batch_size", embedded.get("batch_size", 1))
    model_config.setdefault("in_channels", embedded.get("in_channels", 1))
    model_config.setdefault("targets", embedded["targets"])
    model_config.setdefault("model_name", embedded.get("wandb_run_name", "ink_3d_dino_guided"))
    model_config.setdefault("enable_deep_supervision", False)
    return model_config


def load_model(*, checkpoint_path: Path, config_path: Path, villa_python_root: Path, device: str):
    sys.path.insert(0, str(villa_python_root))
    import torch

    compatibility_patch = install_torch_compiler_compatibility(torch)
    from vesuvius.models.build.build_network_from_config import NetworkFromConfig

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "ema_model" not in checkpoint or "config" not in checkpoint:
        raise RuntimeError("checkpoint must contain ema_model and embedded config")
    external_config = read_object(config_path)
    if checkpoint["config"] != external_config:
        raise RuntimeError("embedded checkpoint config differs from frozen config.json")
    model_config = normalized_model_config(checkpoint["config"])

    class MinimalConfigManager:
        def __init__(self, config: dict[str, Any]):
            self.model_config = config
            self.targets = config["targets"]
            self.train_patch_size = tuple(config["train_patch_size"])
            self.train_batch_size = int(config["train_batch_size"])
            self.in_channels = int(config["in_channels"])
            self.autoconfigure = bool(config.get("autoconfigure", False))
            self.enable_deep_supervision = bool(config.get("enable_deep_supervision", False))
            self.model_name = config["model_name"]
            self.spacing = [1] * len(self.train_patch_size)

    model = NetworkFromConfig(MinimalConfigManager(model_config))
    state = checkpoint["ema_model"]
    prefixes = ("module.", "_orig_mod.")

    def clean_key(key: str) -> str:
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        return key

    model.load_state_dict({clean_key(key): value for key, value in state.items()}, strict=True)
    model.eval().to(device)
    return torch, model, compatibility_patch, model_config


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "infer-patch"), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--villa-python-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--on-degenerate",
        choices=("fail", "warn"),
        default="fail",
        help="what to do when the output map carries no decision (default: fail closed)",
    )
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--probability-output", type=Path)
    parser.add_argument("--projection-output", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    profile, profile_identity = load_execution_profile(
        args.profile, repo_root=args.repo_root
    )
    checkpoint_sha256 = profile["checkpoint_sha256"]
    config_sha256 = profile["config_sha256"]
    villa_expected_revision = profile["pinned_source_revision"]
    model_voxel_size_um = float(profile["input_contract"]["native_voxel_um"])
    require_hash(args.checkpoint, checkpoint_sha256, "checkpoint")
    require_hash(args.config, config_sha256, "config")
    if not args.villa_python_root.is_dir():
        raise RuntimeError(f"Villa Python root does not exist: {args.villa_python_root}")
    villa_repo = args.villa_python_root
    while villa_repo != villa_repo.parent and not (villa_repo / ".git").exists():
        villa_repo = villa_repo.parent
    revision = git_revision(villa_repo)
    if revision != villa_expected_revision:
        raise RuntimeError(f"Villa revision mismatch: {revision} != {villa_expected_revision}")

    torch, model, compatibility_patch, model_config = load_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        villa_python_root=args.villa_python_root,
        device=args.device,
    )
    receipt: dict[str, Any] = {
        "schema": "campaignx.ink_3d_dino_execution.v2",
        "generated_at_utc": utc_now(),
        "mode": args.mode,
        "status": "MODEL_LOAD_VERIFIED" if args.mode == "preflight" else "COMPLETED_DIAGNOSTIC_ONLY",
        "model": {
            "method_id": METHOD_ID,
            "checkpoint_sha256": checkpoint_sha256,
            "config_sha256": config_sha256,
            "weights_key": "ema_model",
            "normalization": "percentile_minmax_p1_p99",
            "native_voxel_size_um": model_voxel_size_um,
            "patch_size_zyx": list(model_config["train_patch_size"]),
        },
        "profile": profile_identity,
        "runtime": {
            "villa_revision": revision,
            "torch_version": torch.__version__,
            "device": args.device,
            "torch_compiler_compatibility_patch": compatibility_patch,
        },
        "non_claims": [
            "model loading or response does not validate transfer",
            "a silent response does not prove absence of ink",
            "this execution does not identify letters",
        ],
    }
    if args.mode == "infer-patch":
        if not args.input_manifest or not args.probability_output or not args.projection_output:
            raise RuntimeError("infer-patch requires input manifest and both outputs")
        if args.probability_output.exists() or args.projection_output.exists():
            raise RuntimeError("refusing to overwrite an existing model output")
        manifest, array_path = validate_manifest(
            args.input_manifest, model_voxel_um=model_voxel_size_um
        )
        import numpy as np
        from PIL import Image

        volume = np.load(array_path, allow_pickle=False).astype(np.float32, copy=False)
        if volume.shape != (256, 256, 256):
            raise RuntimeError(f"array shape mismatch: {volume.shape}")
        lower, upper = np.percentile(volume, (1.0, 99.0))
        if float(upper - lower) <= 1e-8:
            normalized = np.zeros_like(volume, dtype=np.float32)
        else:
            normalized = (
                (np.clip(volume, lower, upper) - lower) / float(upper - lower)
            ).astype(np.float32, copy=False)
        tensor = torch.from_numpy(normalized[None, None]).to(args.device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda" if args.device.startswith("cuda") else "cpu",
            dtype=torch.bfloat16,
            enabled=args.device.startswith("cuda"),
        ):
            output = model(tensor)
            logits = output["ink"] if isinstance(output, dict) else output
            probability = torch.sigmoid(logits).float().cpu().numpy()[0, 0]
        args.probability_output.parent.mkdir(parents=True, exist_ok=True)
        temporary_probability = args.probability_output.with_name(args.probability_output.name + ".partial")
        with temporary_probability.open("wb") as handle:
            np.savez_compressed(handle, probability=probability.astype(np.float16))
        os.replace(temporary_probability, args.probability_output)
        projection = probability.max(axis=0)
        projection_u8 = np.clip(projection * 255.0, 0, 255).astype(np.uint8)
        temporary_projection = args.projection_output.with_name(args.projection_output.name + ".partial")
        Image.fromarray(projection_u8, mode="L").save(temporary_projection, format="PNG")
        os.replace(temporary_projection, args.projection_output)
        receipt["input"] = {
            "manifest_sha256": sha256_file(args.input_manifest),
            "sample_id": manifest["sample_id"],
            "array_sha256": manifest["array"]["sha256"],
            "input_voxel_size_um": manifest["input_voxel_size_um"],
        }
        # Every voxel of the cube is a prediction: the model is handed the whole
        # normalised volume and there is no masked-out region, so the mask is
        # genuinely all-true rather than the contract's default of "> 0". That
        # default would silently drop voxels whose sigmoid underflowed to zero,
        # which are exactly the confident negatives.
        receipt["liveness"] = assess_liveness(
            probability, valid=np.ones(probability.shape, dtype=bool))
        receipt["output"] = {
            "probability_sha256": sha256_file(args.probability_output),
            "projection_sha256": sha256_file(args.projection_output),
            "minimum": float(probability.min()),
            "maximum": float(probability.max()),
            "mean": float(probability.mean()),
            "fraction_ge_0_5": float((probability >= 0.5).mean()),
        }
    write_new_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if "liveness" in receipt:
        return refuse_if_not_alive(
            receipt["liveness"],
            lane="ink-3d-dino-guided@1.0.0",
            output=args.receipt.parent,
            on_degenerate=args.on_degenerate,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
