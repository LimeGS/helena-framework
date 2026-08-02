#!/usr/bin/env python3
"""Assemble verified compact ScrollFiesta shard receipts into one weld input."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


BACKENDS = Path(__file__).resolve().parents[1] / "backends"
sys.path.insert(0, str(BACKENDS))

from scrollfiesta.cube_stage import CubeStageError, plan_cubes  # noqa: E402
from scrollfiesta.receipt import file_artifact, sha256_file, write_new_json  # noqa: E402


SCHEMA = "campaignx.scrollfiesta_cube_shard_assembly_request.v1"
REQUIRED = {
    "schema",
    "canonical_roi_level0_zyx",
    "level",
    "cube_edge_voxels",
    "shard_receipts",
    "output_dir",
}


def file_uri_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise CubeStageError(f"only local file artifacts can be assembled: {uri}")
    return Path(unquote(parsed.path))


def hardlink_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=os.link)


def assemble(request: dict, request_path: Path) -> Path:
    if not isinstance(request, dict) or set(request) != REQUIRED:
        keys = set(request) if isinstance(request, dict) else set()
        raise CubeStageError(
            f"request keys mismatch: missing={sorted(REQUIRED-keys)}, "
            f"unknown={sorted(keys-REQUIRED)}"
        )
    if request["schema"] != SCHEMA:
        raise CubeStageError(f"expected schema {SCHEMA}")
    output = Path(request["output_dir"])
    if not output.is_absolute() or output.exists():
        raise CubeStageError(f"immutable absolute output_dir required: {output}")
    roi = tuple(int(item) for item in request["canonical_roi_level0_zyx"])
    expected_plans = plan_cubes(
        roi,
        level=int(request["level"]),
        cube_edge_voxels=int(request["cube_edge_voxels"]),
    )
    expected = {plan.cube_id for plan in expected_plans}
    receipt_paths = [Path(item).resolve(strict=True) for item in request["shard_receipts"]]
    if not receipt_paths:
        raise CubeStageError("at least one shard receipt is required")

    source_identity = None
    scientific_parameters = None
    runtime_identity = None
    rows = []
    observed: set[str] = set()
    for receipt_path in receipt_paths:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema") != "campaignx.scrollfiesta_compact_cube_stage.v1":
            raise CubeStageError(f"not a compact cube-stage receipt: {receipt_path}")
        if receipt.get("status") != "SUCCEEDED" or receipt.get("complete_fraction") != 1.0:
            raise CubeStageError(f"shard did not finish successfully: {receipt_path}")
        identity = receipt.get("source")
        parameters = receipt.get("parameters", {})
        runtime = receipt.get("runtime")
        science = {
            key: parameters.get(key)
            for key in ("halo", "threshold", "resolved_upstream_threshold", "cube_edge_voxels")
        }
        if source_identity is None:
            source_identity = identity
            scientific_parameters = science
            runtime_identity = runtime
        elif identity != source_identity or science != scientific_parameters:
            raise CubeStageError("shard source or scientific parameters disagree")
        elif runtime != runtime_identity:
            raise CubeStageError("shard runtime binaries disagree")
        for result in receipt.get("results", []):
            cube_id = str(result.get("cube_id"))
            if cube_id not in expected or cube_id in observed:
                raise CubeStageError(f"unexpected or duplicate cube: {cube_id}")
            if result.get("returncode") != 0 or result.get("produced_obj") is not True:
                raise CubeStageError(f"non-successful cube result: {cube_id}")
            obj_artifact = result.get("obj", {})
            obj = file_uri_path(str(obj_artifact.get("uri"))).resolve(strict=True)
            if sha256_file(obj) != obj_artifact.get("sha256") or obj.stat().st_size != obj_artifact.get("bytes"):
                raise CubeStageError(f"cube artifact hash/size mismatch: {cube_id}")
            expected_name = f"{cube_id}_step12_final_all.obj"
            if obj.name != expected_name or obj.parent.name != f"{cube_id}_step12_final":
                raise CubeStageError(f"unexpected final OBJ layout: {obj}")
            observed.add(cube_id)
            rows.append(
                {
                    "cube_id": cube_id,
                    "source_receipt": file_artifact(receipt_path),
                    "source_obj": obj_artifact,
                    "source_final_dir": obj.parent,
                }
            )
    if observed != expected:
        raise CubeStageError(f"incomplete shard union: missing={sorted(expected-observed)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    dump = output / "dump"
    dump.mkdir()
    assembled_results = []
    try:
        for row in sorted(rows, key=lambda item: item["cube_id"]):
            cube_id = row["cube_id"]
            final_name = f"{cube_id}_step12_final"
            destination = dump / cube_id / final_name
            destination.parent.mkdir()
            hardlink_tree(row["source_final_dir"], destination)
            assembled_obj = destination / f"{final_name}_all.obj"
            if sha256_file(assembled_obj) != row["source_obj"]["sha256"]:
                raise CubeStageError(f"assembled OBJ changed: {cube_id}")
            assembled_results.append(
                {
                    "cube_id": cube_id,
                    "source_receipt": row["source_receipt"],
                    "source_obj": row["source_obj"],
                    "assembled_obj": file_artifact(assembled_obj),
                }
            )
        receipt_path = output / "CUBE_SHARD_ASSEMBLY_RECEIPT.json"
        write_new_json(
            receipt_path,
            {
                "schema": "campaignx.scrollfiesta_cube_shard_assembly.v1",
                "status": "SUCCEEDED",
                "request": file_artifact(request_path),
                "source": source_identity,
                "scientific_parameters": scientific_parameters,
                "runtime": runtime_identity,
                "canonical_roi_level0_zyx": list(roi),
                "planned_cube_count": len(expected),
                "assembled_cube_count": len(assembled_results),
                "results": assembled_results,
                "assembly_mode": "HARDLINK_FINAL_STEP12",
                "complete_fraction": 1.0,
                "partial_weld_allowed": False,
                "physical_mesh_fusion_performed": False,
                "ink_used": False,
            },
        )
        return receipt_path
    except Exception:
        # The output remains an immutable failed attempt. It is never a weld input
        # because no success receipt exists.
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-spec", type=Path, required=True)
    args = parser.parse_args(argv)
    request_path = args.run_spec.resolve(strict=True)
    try:
        receipt = assemble(json.loads(request_path.read_text(encoding="utf-8")), request_path)
    except (CubeStageError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
