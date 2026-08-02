#!/usr/bin/env python3
"""Package and gate upstream Volume Cartographer's TIFXYZ merge.

This wrapper contains no stitching algorithm.  It resolves certified platform
artifacts, verifies their immutable bytes, invokes the pinned upstream
``vc_merge_tifxyz`` binary, and refuses to publish unless seam and geometry
evidence pass the frozen profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "framework/contracts/pipeline_phases.json").is_file())
STAGE1 = ROOT / "framework/stages/01-segmentation"
FLEET = ROOT / "framework/stages/03-ink/fleet"
for path in (ROOT, STAGE1, FLEET):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class MergeRefused(RuntimeError):
    """The proposed merge lacks compatible inputs or passing evidence."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def validate_parent_compatibility(parents: list[dict[str, Any]]) -> dict[str, Any]:
    if not 2 <= len(parents) <= 64:
        raise MergeRefused("parent count must be between 2 and 64")
    required = ("surface_id", "sample_id", "source_snapshot_id", "artifact_uri",
                "artifact_sha256", "ct_uri", "coordinate_frame", "voxel_size_um")
    for parent in parents:
        for field in required:
            if parent.get(field) in (None, ""):
                raise MergeRefused(f"parent {parent.get('surface_id')} has no {field}")
        if parent.get("geometry_qc_state") != "GEOMETRY_CERTIFIED":
            raise MergeRefused(
                f"geometry_qc_state for {parent['surface_id']} is "
                f"{parent.get('geometry_qc_state')}, not GEOMETRY_CERTIFIED")

    reference = parents[0]
    for field in ("sample_id", "source_snapshot_id", "ct_uri", "coordinate_frame",
                  "voxel_size_um"):
        expected = canonical(reference[field])
        if any(canonical(parent[field]) != expected for parent in parents[1:]):
            raise MergeRefused(f"parents disagree on {field}")
    known_ct_hashes = {str(parent["ct_sha256"]) for parent in parents
                       if parent.get("ct_sha256")}
    if len(known_ct_hashes) > 1:
        raise MergeRefused("parents disagree on ct_sha256")
    return {
        "sample_id": reference["sample_id"],
        "source_snapshot_id": reference["source_snapshot_id"],
        "ct_uri": reference["ct_uri"],
        "ct_sha256": next(iter(known_ct_hashes), None),
        "coordinate_frame": reference["coordinate_frame"],
        "voxel_size_um": reference["voxel_size_um"],
    }


def declared_edges(rows: list[list[str | None]]) -> set[frozenset[str]]:
    edges: set[frozenset[str]] = set()
    for row_index, row in enumerate(rows):
        for column, name in enumerate(row):
            if name is None:
                continue
            neighbours = []
            if column + 1 < len(row):
                neighbours.append(row[column + 1])
            if row_index + 1 < len(rows) and column < len(rows[row_index + 1]):
                neighbours.append(rows[row_index + 1][column])
            for neighbour in neighbours:
                if neighbour is not None and neighbour != name:
                    edges.add(frozenset((name, neighbour)))
    return edges


def validate_layout(rows: Any, artifact_ids: list[str]
                    ) -> list[list[str | None]]:
    """Recheck the fan-in graph inside the worker trust boundary."""
    if not isinstance(rows, list) or not rows:
        raise MergeRefused("rows must be a non-empty array")
    if any(not isinstance(row, list) or not row for row in rows):
        raise MergeRefused("every layout row must be a non-empty array")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise MergeRefused("rows must be rectangular")
    normalized: list[list[str | None]] = []
    flattened: list[str] = []
    for row in rows:
        normalized_row = []
        for value in row:
            if value is None:
                normalized_row.append(None)
            elif isinstance(value, str) and value.strip():
                normalized_row.append(value)
                flattened.append(value)
            else:
                raise MergeRefused("layout cells must be artifact ids or null")
        normalized.append(normalized_row)
    if len(flattened) != len(set(flattened)):
        raise MergeRefused("each parent may appear in rows only once")
    if set(flattened) != set(artifact_ids):
        raise MergeRefused("rows must contain every parent exactly once")
    edges = declared_edges(normalized)
    adjacency = {value: set() for value in artifact_ids}
    for edge in edges:
        left, right = tuple(edge)
        adjacency[left].add(right)
        adjacency[right].add(left)
    visited = {artifact_ids[0]}
    pending = [artifact_ids[0]]
    while pending:
        value = pending.pop()
        for neighbour in adjacency[value] - visited:
            visited.add(neighbour)
            pending.append(neighbour)
    if visited != set(artifact_ids):
        raise MergeRefused("rows describes a disconnected surface layout")
    return normalized


def evaluate_seam_qc(summary: dict[str, Any], expected_edges: set[frozenset[str]],
                     policy: dict[str, Any]) -> dict[str, Any]:
    observed: dict[frozenset[str], dict[str, Any]] = {}
    for edge in summary.get("edges") or []:
        key = frozenset((str(edge.get("a")), str(edge.get("b"))))
        observed[key] = edge
    if policy.get("require_all_declared_edges", True):
        missing = expected_edges - observed.keys()
        extra = observed.keys() - expected_edges
        if missing or extra:
            raise MergeRefused(
                f"upstream edge set differs: missing={len(missing)} extra={len(extra)}")

    checked = []
    for key in sorted(expected_edges, key=lambda edge: sorted(edge)):
        edge = observed[key]
        label = "<->".join(sorted(key))
        anchors = int(edge.get("anchor_count", edge.get("ransac_total", 0)))
        inliers = int(edge.get("ransac_inliers", 0))
        total = int(edge.get("ransac_total", 0))
        fraction = inliers / total if total else 0.0
        sigma = float(edge.get("ransac_sigma_in", float("inf")))
        overlap_a = int(edge.get("real_overlap_A", 0))
        overlap_b = int(edge.get("real_overlap_B", 0))
        if anchors < int(policy["minimum_anchor_count_per_edge"]):
            raise MergeRefused(f"{label} has only {anchors} anchors")
        if inliers < int(policy["minimum_ransac_inliers_per_edge"]):
            raise MergeRefused(f"{label} has only {inliers} RANSAC inliers")
        if fraction < float(policy["minimum_inlier_fraction_per_edge"]):
            raise MergeRefused(f"{label} inlier fraction {fraction:.3f} is too low")
        if sigma > float(policy["maximum_ransac_sigma_in_voxels"]):
            raise MergeRefused(f"{label} RANSAC sigma {sigma:.3f} is too high")
        if policy.get("require_real_overlap_on_both_parents") and (
                overlap_a <= 0 or overlap_b <= 0):
            raise MergeRefused(f"{label} has no real overlap on both parents")
        checked.append({"edge": sorted(key), "anchor_count": anchors,
                        "ransac_inliers": inliers, "ransac_total": total,
                        "inlier_fraction": fraction, "ransac_sigma_in": sigma,
                        "real_overlap": [overlap_a, overlap_b]})
    return {"schema": "campaignx.vc3d_merge_seam_qc.v1", "status": "PASS",
            "edges": checked, "policy": policy, "generated_at_utc": utc_now()}


def upstream_command(*, binary: Path, obj2tifxyz: Path, merge_json: Path,
                     paths_dir: Path, reference: str, ransac_seed: int,
                     anchor_cap: int, strip_cols: int) -> list[str]:
    if ransac_seed <= 0:
        raise MergeRefused("ransac_seed must be a deterministic nonzero integer")
    return [str(binary), "--merge", str(merge_json),
            "--paths-dir", str(paths_dir),
            "--obj2tifxyz", str(obj2tifxyz),
            "--ref", reference, "--ransac-seed", str(ransac_seed),
            "--anchor-cap", str(anchor_cap), "--strip-cols", str(strip_cols)]


def verify_runtime(profile: dict[str, Any]) -> dict[str, Any]:
    runtime = profile["runtime"]
    expected_image = str(runtime["villa_image"])
    observed_image = os.environ.get("HELENA_VILLA_IMAGE_DIGEST", "")
    if observed_image != expected_image:
        raise MergeRefused(
            "HELENA_VILLA_IMAGE_DIGEST does not equal the digest-pinned profile")
    checked = {}
    for name, declaration in runtime["binaries"].items():
        path = Path(declaration["path"])
        if not path.is_file():
            raise MergeRefused(f"runtime binary is missing: {name} at {path}")
        observed = file_sha256(path)
        if observed != declaration["sha256"]:
            raise MergeRefused(f"runtime binary digest differs: {name}")
        checked[name] = {"path": str(path), "sha256": observed}
    receipt_path = Path(runtime["toolchain_receipt"])
    if not receipt_path.is_file():
        raise MergeRefused(f"toolchain receipt is missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if profile["source"]["commit"] not in canonical(receipt):
        raise MergeRefused("toolchain receipt does not bind the pinned Villa commit")
    return {"image": observed_image, "binaries": checked,
            "toolchain_receipt": str(receipt_path),
            "toolchain_receipt_sha256": file_sha256(receipt_path)}


def materialize_parents(parents: list[dict[str, Any]], aliases: dict[str, str],
                        paths_dir: Path) -> list[dict[str, Any]]:
    from fleet.certifier import load_qc_adapter

    adapter = load_qc_adapter()
    materialized = []
    for parent in parents:
        alias = aliases[str(parent["surface_id"])]
        destination = paths_dir / alias
        manifest = adapter.materialize_surface(
            str(parent["artifact_uri"]), str(parent["artifact_sha256"]), destination)
        materialized.append({
            "surface_id": parent["surface_id"], "alias": alias,
            "artifact_uri": parent["artifact_uri"],
            "artifact_sha256": parent["artifact_sha256"],
            "manifest_sha256": hashlib.sha256(
                canonical(manifest).encode("utf-8")).hexdigest(),
        })
    return materialized


def locate_upstream_output(paths_dir: Path, aliases: Iterable[str]) -> tuple[Path, Path]:
    alias_set = set(aliases)
    summaries = [path for path in paths_dir.rglob("*_summary.json")
                 if path.parent.name not in alias_set]
    if len(summaries) != 1:
        raise MergeRefused(f"expected one upstream summary, found {len(summaries)}")
    summary_path = summaries[0]
    target = summary_path.parent
    for name in ("x.tif", "y.tif", "z.tif", "meta.json"):
        if not (target / name).is_file():
            raise MergeRefused(f"upstream output is missing {name}")
    return target, summary_path


def publish_set(source: Path, names: list[str], *, schema: str, store_spec: str,
                sample_id: str, key: str, attempt: str) -> dict[str, Any]:
    from fleet.artifact_store import open_artifact_store
    from fleet.common import content_sha256

    files = {name: {"size_bytes": (source / name).stat().st_size,
                    "sha256": file_sha256(source / name)} for name in names}
    manifest = {"schema": schema, "files": files,
                "artifact_sha256": content_sha256(files)}
    store = open_artifact_store(store_spec)
    staged = store.stage(source, attempt, manifest)
    promoted = store.promote(staged, sample_id, key, manifest)
    return {**promoted, "artifact_sha256": manifest["artifact_sha256"],
            "manifest": manifest}


def run(args: argparse.Namespace) -> dict[str, Any]:
    from fleet.common import stable_id
    from fleet.finalizer import certify_surface_geometry, inspect_tifxyz
    from job_store import InkJobStore

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise MergeRefused(f"refusing non-empty output directory: {output}")
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if profile.get("profile_id") != "vc3d-tifxyz-merge@1.0.0":
        raise MergeRefused("unexpected merge profile")
    if args.ransac_seed != int(profile["parameters"]["ransac_seed"]):
        raise MergeRefused("ransac_seed differs from frozen profile")
    for name in ("anchor_cap", "strip_cols"):
        if getattr(args, name) != int(profile["parameters"][name]):
            raise MergeRefused(f"{name} differs from frozen profile")

    environment_name = args.db.removeprefix("postgres-env://")
    if not args.db.startswith("postgres-env://") or not os.environ.get(environment_name):
        raise MergeRefused("--db must name a populated postgres-env:// variable")
    store = InkJobStore(os.environ[environment_name])
    artifact_ids = json.loads(args.artifact_ids_json)
    rows = validate_layout(json.loads(args.rows_json), artifact_ids)
    parents = store.merge_surfaces(artifact_ids)
    source = validate_parent_compatibility(parents)
    if args.reference_artifact_id not in artifact_ids:
        raise MergeRefused("reference_artifact_id is not a parent")

    runtime = verify_runtime(profile)
    aliases = {surface_id: f"s{index:03d}"
               for index, surface_id in enumerate(artifact_ids)}
    alias_rows = [[aliases[value] if value is not None else None for value in row]
                  for row in rows]
    work = output / "work"
    paths_dir = work / "paths"
    paths_dir.mkdir(parents=True)
    materialized = materialize_parents(parents, aliases, paths_dir)
    # JSON arrays preserve explicit empty cells. Whitespace-delimited strings
    # cannot represent a hole and would silently change the declared graph.
    write_json(output / "merge.json", {"rows": alias_rows})
    profile_sha = file_sha256(args.profile)
    assembly = {
        "schema": "campaignx.vc3d_tifxyz_assembly_manifest.v1",
        "profile_id": profile["profile_id"], "profile_sha256": profile_sha,
        "source": source, "parents": materialized, "rows": rows,
        "alias_rows": alias_rows,
        "reference_artifact_id": args.reference_artifact_id,
        "reference_alias": aliases[args.reference_artifact_id],
        "parameters": {"ransac_seed": args.ransac_seed,
                       "anchor_cap": args.anchor_cap, "strip_cols": args.strip_cols},
    }
    write_json(output / "assembly_manifest.json", assembly)

    command = upstream_command(
        binary=Path(runtime["binaries"]["vc_merge_tifxyz"]["path"]),
        obj2tifxyz=Path(runtime["binaries"]["vc_obj2tifxyz_legacy"]["path"]),
        merge_json=output / "merge.json", paths_dir=paths_dir,
        reference=aliases[args.reference_artifact_id],
        ransac_seed=args.ransac_seed, anchor_cap=args.anchor_cap,
        strip_cols=args.strip_cols)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = time.monotonic() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    (output / "stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (output / "stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode:
        raise MergeRefused(f"vc_merge_tifxyz exited {completed.returncode}")

    upstream_dir, upstream_summary = locate_upstream_output(paths_dir, aliases.values())
    for name in ("x.tif", "y.tif", "z.tif", "meta.json"):
        shutil.copy2(upstream_dir / name, output / name)
    summary = json.loads(upstream_summary.read_text(encoding="utf-8"))
    shutil.copy2(upstream_summary, output / "vc_merge_tifxyz_summary.json")
    obj_path = Path(str(summary.get("obj_out") or ""))
    if not obj_path.is_file():
        candidates = list(upstream_dir.glob("*.obj"))
        if len(candidates) != 1:
            raise MergeRefused("upstream produced no unique combined OBJ")
        obj_path = candidates[0]
    shutil.copy2(obj_path, output / "combined.obj")

    seam_qc = evaluate_seam_qc(
        summary, declared_edges(alias_rows), profile["seam_gate"])
    write_json(output / "SEAM_QC.json", seam_qc)
    geometry = certify_surface_geometry(output)
    write_json(output / "GEOMETRY_CERTIFICATION.json", geometry)
    if geometry.get("geometry_qc_state") != "GEOMETRY_CERTIFIED":
        raise MergeRefused(
            f"merged geometry is {geometry.get('geometry_qc_state')}, not certified")
    inspection = inspect_tifxyz(output, float(source["voxel_size_um"]))

    license_path = ROOT / profile["source"]["license_path"]
    notice_path = ROOT / profile["source"]["notice_path"]
    shutil.copy2(license_path, output / "VOLUME_CARTOGRAPHER_LICENSE.txt")
    shutil.copy2(notice_path, output / "VOLUME_CARTOGRAPHER_NOTICE.txt")
    write_json(output / "CORRESPONDING_SOURCE.json", {
        **profile["source"], "profile_sha256": profile_sha,
        "runtime": runtime, "generated_at_utc": utc_now(),
    })

    scientific_names = ["x.tif", "y.tif", "z.tif", "meta.json"]
    scientific_files = {name: file_sha256(output / name) for name in scientific_names}
    surface_id = stable_id("vc3d-tifxyz-merge", {
        "source_snapshot_id": source["source_snapshot_id"],
        "parents": [{"surface_id": row["surface_id"],
                     "artifact_sha256": row["artifact_sha256"]}
                    for row in materialized],
        "files": scientific_files, "profile_sha256": profile_sha,
    })
    publication = publish_set(
        output, scientific_names,
        schema="campaignx.merged_tifxyz_artifact_set.v1",
        store_spec=args.artifact_store, sample_id=source["sample_id"],
        key=surface_id, attempt=f"{surface_id}-scientific")
    job_id = os.environ.get("HELENA_JOB_ID", "")
    if not job_id:
        raise MergeRefused("HELENA_JOB_ID is required for lineage registration")
    receipt = {
        "schema": "campaignx.vc3d_tifxyz_merge_receipt.v1", "status": "PASS",
        "job_id": job_id, "surface_id": surface_id,
        "artifact_uri": publication["artifact_uri"],
        "artifact_sha256": publication["artifact_sha256"],
        "profile_id": profile["profile_id"], "profile_sha256": profile_sha,
        "source": profile["source"], "runtime": runtime,
        "parents": materialized, "assembly_manifest_sha256": file_sha256(
            output / "assembly_manifest.json"),
        "seam_qc_sha256": file_sha256(output / "SEAM_QC.json"),
        "geometry_certification_sha256": file_sha256(
            output / "GEOMETRY_CERTIFICATION.json"),
        "parameters": {"rows": rows,
                       "reference_artifact_id": args.reference_artifact_id,
                       "ransac_seed": args.ransac_seed,
                       "anchor_cap": args.anchor_cap,
                       "strip_cols": args.strip_cols},
        "outputs": {
            path.name: {"sha256": file_sha256(path),
                        "size_bytes": path.stat().st_size}
            for path in sorted(output.iterdir()) if path.is_file()
        },
        "execution": {"argv": command, "exit_code": completed.returncode,
                      "elapsed_seconds": round(elapsed, 3),
                      "user_cpu_seconds": round(after.ru_utime - before.ru_utime, 3),
                      "system_cpu_seconds": round(after.ru_stime - before.ru_stime, 3),
                      "max_rss": after.ru_maxrss,
                      "input_blocks": after.ru_inblock - before.ru_inblock,
                      "output_blocks": after.ru_oublock - before.ru_oublock,
                      "major_page_faults": after.ru_majflt - before.ru_majflt,
                      "minor_page_faults": after.ru_minflt - before.ru_minflt,
                      "voluntary_context_switches": after.ru_nvcsw - before.ru_nvcsw,
                      "involuntary_context_switches": after.ru_nivcsw - before.ru_nivcsw},
        "generated_at_utc": utc_now(), "non_claims": profile["non_claims"],
    }
    write_json(output / "MERGE_RECEIPT.json", receipt)
    evidence_names = sorted(path.name for path in output.iterdir() if path.is_file())
    evidence = publish_set(
        output, evidence_names, schema="campaignx.vc3d_merge_evidence_set.v1",
        store_spec=args.artifact_store, sample_id=source["sample_id"],
        key=f"evidence-{surface_id}", attempt=f"{surface_id}-evidence")
    surface = {
        "surface_id": surface_id, "source_snapshot_id": source["source_snapshot_id"],
        "sample_id": source["sample_id"],
        "artifact_sha256": publication["artifact_sha256"],
        "artifact_uri": publication["artifact_uri"],
        "bbox_xyz": inspection["bbox_xyz"],
        "sample_points": inspection["sample_points"],
        "area_cm2": inspection["area_cm2"], "state": "MERGED",
        "physical_qc_state": "UNVALIDATED",
        "geometry_qc_state": geometry["geometry_qc_state"],
        "parent_surface_ids": artifact_ids,
        "evidence_uri": evidence["artifact_uri"],
        "evidence_sha256": evidence["artifact_sha256"],
        "profile_id": profile["profile_id"], "profile_sha256": profile_sha,
    }
    registration = store.register_merged_surface(
        surface, parents, job_id=job_id,
        qc_profile_id=profile["physical_qc"]["profile_id"])
    write_json(output / "EVIDENCE_PUBLICATION.json", {
        "schema": "campaignx.vc3d_merge_evidence_publication.v1",
        "evidence_uri": evidence["artifact_uri"],
        "evidence_sha256": evidence["artifact_sha256"],
        "registration": registration,
    })
    shutil.rmtree(work, ignore_errors=True)
    return {**receipt, "evidence_uri": evidence["artifact_uri"],
            "evidence_sha256": evidence["artifact_sha256"]}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--artifact-ids-json", required=True)
    ap.add_argument("--rows-json", required=True)
    ap.add_argument("--reference-artifact-id", required=True)
    ap.add_argument("--ransac-seed", type=int, required=True)
    ap.add_argument("--anchor-cap", type=int, required=True)
    ap.add_argument("--strip-cols", type=int, required=True)
    ap.add_argument("--artifact-store", required=True)
    ap.add_argument("--output", type=Path, required=True)
    return ap


def main() -> int:
    try:
        run(parser().parse_args())
    except Exception as error:  # noqa: BLE001 - runner records fail-closed reason
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
