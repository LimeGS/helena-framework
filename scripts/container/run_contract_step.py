#!/usr/bin/env python3
"""Validate one immutable Helena Framework container step and write its receipt.

This is deliberately an execution *contract*, not a second orchestrator.  It
does not choose scientific parameters and it never deletes cache artifacts.
An actual stage launcher may run only after this contract has validated the
locked manifest and emitted the receipt it will extend with output hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


KIND = "campaign_x_container_stage_manifest_v1"
RECEIPT_KIND = "campaign_x_container_execution_receipt_v1"
STEPS = {"vc3d-grow", "ink-inference", "qc", "high-recall"}
TERMINAL_HIGH_RECALL = {"COMPLETED", "FAILED_CLOSED", "NOT_REQUIRED"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("manifest must be a JSON object")
    return value


def bounded_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if root.resolve() not in candidate.parents and candidate != root.resolve():
        raise ValueError(f"input path escapes its snapshot: {relative}")
    return candidate


def as_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def validate_manifest(
    manifest: dict[str, Any], *, manifest_path: Path, input_root: Path, model_root: Path, step: str, allow_unpinned: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if manifest.get("kind") != KIND:
        raise ValueError(f"manifest.kind must be {KIND}")
    if manifest.get("step") != step:
        raise ValueError(f"manifest step {manifest.get('step')!r} does not match --step {step!r}")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("manifest.run_id must be a non-empty string")
    image = manifest.get("image")
    if not isinstance(image, dict) or not isinstance(image.get("reference"), str):
        raise ValueError("manifest.image.reference is required")
    if not allow_unpinned and "@sha256:" not in image["reference"]:
        raise ValueError("image reference must be pinned by immutable digest")
    runtime_image = os.environ.get("HELENA_IMAGE_REF")
    if runtime_image and runtime_image != image["reference"]:
        raise ValueError("manifest image does not match the immutable image selected by the runtime")
    source = manifest.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("git_commit"), str):
        raise ValueError("manifest.source.git_commit is required")

    checked_inputs: list[dict[str, Any]] = []
    for item in as_list(manifest.get("inputs", []), "manifest.inputs"):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            raise ValueError("each input requires path and sha256")
        source_path = bounded_child(input_root, item["path"])
        if not source_path.is_file():
            raise ValueError(f"declared input is missing: {item['path']}")
        actual = sha256(source_path)
        if actual != item["sha256"]:
            raise ValueError(f"input hash mismatch: {item['path']}")
        checked_inputs.append({"path": item["path"], "sha256": actual, "size_bytes": source_path.stat().st_size})

    models: list[dict[str, Any]] = []
    for item in as_list(manifest.get("models", []), "manifest.models"):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            raise ValueError("each model requires id, path and sha256")
        model_path = bounded_child(model_root, item["path"])
        if not model_path.is_file():
            raise ValueError(f"declared model is missing: {item['path']}")
        actual = sha256(model_path)
        if actual != item["sha256"]:
            raise ValueError(f"model hash mismatch: {item['path']}")
        models.append({"id": item["id"], "path": item["path"], "sha256": actual, "size_bytes": model_path.stat().st_size})

    cleanup = manifest.get("cleanup", {})
    if cleanup is not None and not isinstance(cleanup, dict):
        raise ValueError("manifest.cleanup must be an object")
    retained = cleanup.get("retained_artifacts", []) if isinstance(cleanup, dict) else []
    high_recall_status = cleanup.get("high_recall_status") if isinstance(cleanup, dict) else None
    if cleanup.get("requested", False) and retained and high_recall_status not in TERMINAL_HIGH_RECALL:
        raise ValueError("cleanup blocked: retained artifacts are declared downstream inputs")
    return checked_inputs, models


def receipt_base(
    *, status: str, step: str, manifest: dict[str, Any], manifest_path: Path, inputs: list[dict[str, Any]], models: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "kind": RECEIPT_KIND,
        "schema_version": 1,
        "status": status,
        "generated_at_utc": utc_now(),
        "run_id": manifest.get("run_id"),
        "step": step,
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "source": manifest.get("source"),
        "image": manifest.get("image"),
        "inputs": inputs,
        "models": models,
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "command": list(sys.argv),
        "runtime": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            "container_image": os.environ.get("HELENA_IMAGE_REF"),
        },
        "policy": [
            "inputs are read-only snapshot paths verified by sha256",
            "receipt path must not overwrite a prior receipt",
            "this validation step performs no scientific inference or cleanup",
        ],
    }


def write_receipt(output: Path, payload: dict[str, Any]) -> Path:
    receipt = output / "execution-receipt.json"
    if receipt.exists():
        raise ValueError(f"receipt already exists; refusing overwrite: {receipt}")
    temporary = output / ".execution-receipt.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=sorted(STEPS), required=True)
    parser.add_argument("--manifest", type=Path, default=Path("/inputs/stage-manifest.json"))
    parser.add_argument("--input-root", type=Path, default=Path("/inputs"))
    parser.add_argument("--model-root", type=Path, default=Path("/models"))
    parser.add_argument("--output", type=Path, default=Path("/outputs"))
    parser.add_argument("--allow-unpinned-development", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Required for the initial vertical slice.")
    args = parser.parse_args()
    try:
        if not args.dry_run:
            raise ValueError("only --dry-run is implemented; attach an explicit stage launcher after equivalence testing")
        if not args.manifest.is_file():
            raise ValueError(f"manifest is missing: {args.manifest}")
        if not args.input_root.is_dir() or not args.output.is_dir():
            raise ValueError("input root and output directory must already exist")
        if any(args.output.iterdir()):
            raise ValueError("output directory must be new and empty")
        manifest = load_object(args.manifest)
        inputs, models = validate_manifest(
            manifest, manifest_path=args.manifest, input_root=args.input_root, model_root=args.model_root, step=args.step,
            allow_unpinned=args.allow_unpinned_development,
        )
        receipt = receipt_base(
            status="DRY_RUN_VALIDATED", step=args.step, manifest=manifest, manifest_path=args.manifest,
            inputs=inputs, models=models,
        )
        write_receipt(args.output, receipt)
        print(json.dumps({"status": receipt["status"], "receipt": str(args.output / "execution-receipt.json")}, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        # A failure is evidence too.  Preserve it only when the contract still
        # owns a new output directory; an existing directory is never touched.
        if args.output.is_dir() and not any(args.output.iterdir()):
            failed = {
                "kind": RECEIPT_KIND,
                "schema_version": 1,
                "status": "CONTRACT_FAILED",
                "generated_at_utc": utc_now(),
                "step": args.step,
                "manifest": {
                    "path": str(args.manifest),
                    "sha256": sha256(args.manifest) if args.manifest.is_file() else None,
                },
                "command": list(sys.argv),
                "error": str(error),
                "policy": ["failure receipt written only into a new empty output directory", "no prior output was overwritten"],
            }
            try:
                write_receipt(args.output, failed)
            except (OSError, ValueError):
                pass
        print(f"CONTRACT_FAILED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
