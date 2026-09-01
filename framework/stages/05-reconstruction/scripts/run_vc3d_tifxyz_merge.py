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


def _refuse_non_finite(literal: str) -> Any:
    raise ValueError(f"{literal} is not JSON and cannot be hashed")


def canonical(value: Any) -> str:
    """The one JSON representation this lane hashes or compares documents in.

    It calls ``fleet.common.canonical_bytes`` rather than restating its flags.
    Restating them is what broke this: the lane declared its own ``json.dumps``,
    left ``ensure_ascii`` at the default True, and so produced a different digest
    for the same manifest than the one the control plane stored -- two content
    ids for one document, which is the same as having none.  A copy of a
    canonicalization becomes a second canonicalization the day either is edited,
    so there is no copy here.

    The one thing added on top is a refusal ``canonical_bytes`` does not make.
    ``NaN`` and ``Infinity`` are not JSON; ``json.dumps`` writes them as bare
    literals, and a digest over bytes no conforming parser will accept cannot be
    reproduced by whatever later checks it.  ``parse_constant`` fires on exactly
    those three literals and never on a string that merely contains them.
    """
    from fleet.common import canonical_bytes

    text = canonical_bytes(value).decode("utf-8")
    json.loads(text, parse_constant=_refuse_non_finite)
    return text


def manifest_sha256(manifest: dict[str, Any]) -> str:
    """The digest of a whole artifact-set manifest document."""
    return hashlib.sha256(canonical(manifest).encode("utf-8")).hexdigest()


def artifact_sha256(files: dict[str, Any]) -> str:
    """The digest of an artifact set, which is the digest of its inventory."""
    return hashlib.sha256(canonical(files).encode("utf-8")).hexdigest()


def require_canonical_manifest(
    manifest: Any, *, expected_artifact_sha256: str | None = None,
    schema: str | None = None,
) -> dict[str, Any]:
    """Refuse any manifest whose stated digest is not its inventory's digest.

    ``artifact_sha256`` is what names a surface in the catalogue, and it is
    supposed to be ``content_sha256`` of the ``files`` block sitting beside it.
    Nothing on this path ever recomputed it: materialization compared the field
    to the catalogue and materialization is where a merge decides which bytes it
    is about to stitch.  A field compared to another copy of itself is a hash
    that verifies nothing, so this derives the digest from the inventory the
    manifest actually carries and refuses the manifest when the two disagree.
    """
    if not isinstance(manifest, dict):
        raise MergeRefused("artifact manifest is not a mapping")
    stated_schema = manifest.get("schema")
    if not isinstance(stated_schema, str) or not stated_schema:
        raise MergeRefused("artifact manifest names no schema")
    if schema is not None and stated_schema != schema:
        raise MergeRefused(
            f"artifact manifest schema is {stated_schema}, not {schema}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise MergeRefused("artifact manifest has no file inventory")
    for name, entry in files.items():
        if (not isinstance(entry, dict)
                or not isinstance(entry.get("sha256"), str)
                or not _is_sha256(entry["sha256"])
                or not isinstance(entry.get("size_bytes"), int)
                or isinstance(entry.get("size_bytes"), bool)
                or entry["size_bytes"] < 0):
            raise MergeRefused(f"artifact manifest entry is unusable: {name}")
    stated = manifest.get("artifact_sha256")
    if not isinstance(stated, str) or not _is_sha256(stated):
        raise MergeRefused("artifact manifest has no lowercase artifact_sha256")
    derived = artifact_sha256(files)
    if stated != derived:
        raise MergeRefused(
            "artifact manifest artifact_sha256 is not the digest of its own "
            f"inventory: it says {stated} and the inventory is {derived}")
    if expected_artifact_sha256 is not None and stated != expected_artifact_sha256:
        raise MergeRefused(
            "artifact manifest is not the catalogue artifact: it is "
            f"{stated} and the catalogue names {expected_artifact_sha256}")
    return manifest


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef"
                                    for character in value)


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


def evaluate_collision_qc(parents: list[dict[str, Any]], paths_dir: Path,
                          expected_edges: set[frozenset[str]],
                          policy: dict[str, Any]) -> dict[str, Any]:
    """Every pair, not only the ones the layout declared.

    Seam QC certifies each edge it was told about, and the loop above iterates
    `expected_edges`. Pairs nobody declared neighbours are never looked at --
    and a scroll is one sheet, so two surfaces five wraps apart in the layout
    can still end up in the same place.

    They did. In the PHerc826 reconstruction w045 and w046 pass within 34 um of
    each other across 62% of their extent, and w041/w042 within 74 um, against a
    control pair of true neighbours at 165 um. Papyrus is 100-200 um thick.
    Every one of those surfaces is GEOMETRY_CERTIFIED and every declared seam
    passed: the contradiction is between a pair the layout never named, so
    nothing in this file could reach it.

    Undeclared neighbours in contact are reported rather than refused. Two
    surfaces may legitimately be adjacent without the layout saying so -- the
    layout is a stitching order, not a claim about geometry. What cannot be
    legitimate is two sheets closer than one sheet is thick, and that is what
    this returns.
    """
    from framework.contracts import winding

    located: dict[str, Any] = {}
    for parent in parents:
        artifact = str(parent["artifact_id"])
        points = _read_tifxyz_points(paths_dir, artifact)
        if points is None:
            continue
        located[artifact] = points

    if len(located) < 2:
        return {"schema": "campaignx.vc3d_merge_collision_qc.v1",
                "status": "NOT_EVALUATED",
                "why": "fewer than two parents could be read as geometry",
                "generated_at_utc": utc_now()}

    centre, axis = _scroll_frame(list(located.values()))
    at = {name: winding.locate(pts, centre, axis) for name, pts in located.items()}

    findings = []
    names = sorted(at)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if at[a] is None or at[b] is None:
                continue
            declared = frozenset((a, b)) in expected_edges
            verdict, evidence = winding.compare(at[a], at[b])
            if verdict == winding.CONTRADICTED:
                findings.append({"pair": [a, b], "declared_neighbours": declared,
                                 "verdict": verdict, **evidence})

    undeclared = [f for f in findings if not f["declared_neighbours"]]
    status = "PASS" if not undeclared else "CONTRADICTED"
    if undeclared and policy.get("refuse_on_undeclared_collision", False):
        first = undeclared[0]
        raise MergeRefused(
            f"{first['pair'][0]} and {first['pair'][1]} are not declared "
            f"neighbours and pass within {first['separation_um']:.0f} um, "
            "which is thinner than a sheet of papyrus")
    return {"schema": "campaignx.vc3d_merge_collision_qc.v1", "status": status,
            "pairs_examined": len(names) * (len(names) - 1) // 2,
            "contradictions": findings,
            "undeclared_contradictions": len(undeclared),
            "policy": {"refuse_on_undeclared_collision":
                       bool(policy.get("refuse_on_undeclared_collision", False))},
            "generated_at_utc": utc_now()}


def _read_tifxyz_points(paths_dir: Path, artifact: str):
    """The parent's geometry, subsampled. None when it cannot be read.

    Subsampled because this runs on every merge and the verdict turns on where
    a surface sits, not on its detail: locate() takes medians.
    """
    directory = paths_dir / artifact
    try:
        import numpy as np
        import tifffile
    except ImportError:
        return None
    try:
        x, y, z = (tifffile.imread(str(directory / f"{c}.tif")).astype(float)
                   for c in "xyz")
    except Exception:
        return None
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    pts = pts[np.isfinite(pts).all(axis=1) & (pts > 0).all(axis=1)]
    if len(pts) < 200:
        return None
    return pts[:: max(1, len(pts) // 4000)]


def _scroll_frame(clouds):
    """Centre and axis from the parents themselves, so this needs no P0 record."""
    import numpy as np
    cloud = np.vstack(clouds)
    centre = cloud.mean(axis=0)
    step = max(1, len(cloud) // 20000)
    _, _, vh = np.linalg.svd(cloud[::step] - centre, full_matrices=False)
    return centre, vh[0]


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
        # The adapter verifies the files against the manifest and the manifest
        # against the catalogue.  Neither check derives artifact_sha256 from the
        # inventory, so this does, inside the boundary that is about to stitch
        # these bytes into a canonical surface.
        require_canonical_manifest(
            manifest,
            expected_artifact_sha256=str(parent["artifact_sha256"]))
        materialized.append({
            "surface_id": parent["surface_id"], "alias": alias,
            "artifact_uri": parent["artifact_uri"],
            "artifact_sha256": parent["artifact_sha256"],
            "manifest_sha256": manifest_sha256(manifest),
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

    files = {name: {"size_bytes": (source / name).stat().st_size,
                    "sha256": file_sha256(source / name)} for name in names}
    manifest = {"schema": schema, "files": files,
                "artifact_sha256": artifact_sha256(files)}
    # Publication and revalidation have to be the same statement about the same
    # bytes, so publication is held to the check materialization applies.  It
    # costs one hash and it is the only thing that makes the digest downstream
    # recomputes mean what this side intended by it.
    require_canonical_manifest(manifest, schema=schema)
    store = open_artifact_store(store_spec)
    staged = store.stage(source, attempt, manifest)
    promoted = store.promote(staged, sample_id, key, manifest)
    return {**promoted, "artifact_sha256": manifest["artifact_sha256"],
            "manifest": manifest}


def run(args: argparse.Namespace) -> dict[str, Any]:
    from fleet import surface_routing
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
    # The store resolves under its transaction, and the wrapper independently
    # rechecks the retained lineage before it downloads a byte or starts the
    # upstream binary.  This protects forged/replayed job rows that bypassed
    # queue admission.
    from fleet.canonical_lineage import require_canonical_lineage
    for parent in parents:
        payload = dict(parent.get("payload") or {})
        require_canonical_lineage(
            boundary="P8_PARENT_MATERIALIZATION",
            controlled_mission=(
                parent.get("controlled_first_letters") is True
                or payload.get("controlled_first_letters") is True
            ),
            authoritative_lineage=(
                parent.get("authoritative_lineage")
                or payload.get("authoritative_lineage")
            ),
            allow_unvalidated=(
                parent.get("allow_unvalidated")
                if "allow_unvalidated" in parent
                else payload.get("allow_unvalidated")
            ),
        )
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

    edges = declared_edges(alias_rows)
    seam_qc = evaluate_seam_qc(summary, edges, profile["seam_gate"])
    write_json(output / "SEAM_QC.json", seam_qc)

    # Seam QC has now certified every edge the layout declared. Nothing has
    # looked at the pairs it did not, and that is where w045/w046 hid.
    collision_qc = evaluate_collision_qc(
        parents, paths_dir, edges, profile.get("collision_gate", {}))
    write_json(output / "COLLISION_QC.json", collision_qc)
    geometry = certify_surface_geometry(
        output, voxel_um=float(source["voxel_size_um"]))
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

    # The merged sheet is described once, here, and every consumer below reads
    # this one dict: the routing receipt, the merge receipt, and the catalogue
    # registration. Two descriptions of one surface is two surfaces the day
    # either is edited.
    #
    # Its extent is `inspection`'s -- measured by inspect_tifxyz over the merged
    # x/y/z grids upstream just produced and this lane just published, under the
    # same frozen triangulation and the same invalid-coordinate policy every
    # other surface in the catalogue is measured with. No parent's area is read
    # and none is inherited. A merge is not an exemption from the floor and it
    # is not a promotion across it: stitching changes how much papyrus is in one
    # artifact, which is exactly the quantity the floor is about.
    merged_surface = {
        "surface_id": surface_id,
        "source_snapshot_id": source["source_snapshot_id"],
        "sample_id": source["sample_id"],
        "artifact_sha256": publication["artifact_sha256"],
        "artifact_uri": publication["artifact_uri"],
        "bbox_xyz": inspection["bbox_xyz"],
        "sample_points": inspection["sample_points"],
        "area_cm2": inspection["area_cm2"], "state": "MERGED",
        "physical_qc_state": "UNVALIDATED",
        "geometry_qc_state": geometry["geometry_qc_state"],
        "parent_surface_ids": artifact_ids,
        "profile_id": profile["profile_id"], "profile_sha256": profile_sha,
    }
    # Built by the router's own assembly rather than by this lane, for the
    # reason receipt_for_surface exists: a second place that names the receipt's
    # fields is a second receipt format, and the one that drifts is the one
    # nobody ran the test against. These are the bytes both stores will hold.
    routing_receipt = surface_routing.receipt_for_surface(merged_surface)
    write_json(output / "ROUTING_RECEIPT.json", routing_receipt)

    receipt = {
        "schema": "campaignx.vc3d_tifxyz_merge_receipt.v1", "status": "PASS",
        "job_id": job_id, "surface_id": surface_id,
        # Which scroll this is. Every other receipt says so and this one did
        # not, so the panel -- which reads the scroll off the receipt and
        # refuses to parse it out of a directory name -- listed the run under a
        # scroll called "?", with a run count beside it in P0's own inventory.
        "sample_id": source["sample_id"],
        "artifact_uri": publication["artifact_uri"],
        "artifact_sha256": publication["artifact_sha256"],
        # The measurement, and the decision it produced, travel with the
        # receipt. Without them a successful P8 result says nothing about
        # whether the sheet it names is large enough to ask a question of.
        "area_cm2": inspection["area_cm2"],
        "area_measurement": {
            "schema": "campaignx.merged_surface_measurement.v1",
            "measured_by": "fleet.finalizer.inspect_tifxyz",
            # The canonical digest require_canonical_manifest derived from the
            # inventory of exactly the files measured here, so the area is bound
            # to those bytes and not to a second name for them.
            "artifact_sha256": publication["artifact_sha256"],
            "voxel_size_um": source["voxel_size_um"],
            "shape": inspection["shape"],
            "finite_coordinate_count": inspection["finite_coordinate_count"],
            "valid_triangle_count": inspection["valid_triangle_count"],
            "invalid_coordinate_policy": inspection["invalid_coordinate_policy"],
            "bbox_xyz": inspection["bbox_xyz"],
        },
        "routing_receipt": routing_receipt,
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
    # The evidence set is published after the merge receipt, because the receipt
    # is a member of it. Nothing here is in the routing digest, so the receipt
    # decided above is the receipt this surface registers with.
    surface = {
        **merged_surface,
        "evidence_uri": evidence["artifact_uri"],
        "evidence_sha256": evidence["artifact_sha256"],
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
