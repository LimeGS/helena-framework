"""Convert the neural-tracing-audit reference band (band_r1145_200_xyz.npz)
into a strip-v0 file: registry entry UC-01.

Input layout (documented in release/neural-tracing-audit/docs/NT_AUDIT.md
and README.md of that repo; verified against the real file):
    xyz   (200, 25706, 3) float32 -- x, y, z in 7.91 um voxel coordinates
    valid (200, 25706)    bool
    row0  ()              int64   -- first band row in the source PPM (1145)

PROVENANCE: the winding-class assignment below (umbilicus constants,
row-100-seeded 2D phase unwrap with cross-row 2*pi branch matching, class =
rint((U - U_center) / 2*pi), per-class point collection with a >= 100
point floor) is ported from
release/neural-tracing-audit/gate_3090/score_native.py (lines: the
CX/CY constants through the `trees` construction loop), which itself
documents its derivation from winding_audit_v4.py. That logic is UNCHANGED
except for:
  - dropping the KD-tree construction (the strip only needs the class
    labels; trees are rebuilt by the scorer),
  - keeping per-point U values so coverage-boundary edge flags can be
    computed (strip_format.edge_flags_from_phase), which the source script
    expressed instead as its band row-0/row-199 exclusion rule at scoring
    time,
  - remapping the signed classes (-2..+1) to sequential wrap ids (0..3),
    recorded in meta["source_winding_class_to_wrap"].

The umbilicus (CX, CY) below was fitted in the source audit from 25,376
PPM normal lines (NT_AUDIT.md, Stage 2); it is used ONLY as the unwrap
phase reference, never in any distance computation -- same as the source.

Usage:
    python strips/UC-01/convert_ntaudit_band.py /path/to/band_r1145_200_xyz.npz \
        --out strips/UC-01/UC-01.npz
    # to also carry per-cell normals (recovered from the PPM's nx,ny,nz
    # channels; see this strip's README) so the CT cross-check can run:
    python strips/UC-01/convert_ntaudit_band.py /path/to/band_r1145_200_xyz.npz \
        --out strips/UC-01/UC-01.npz \
        --normals-npy /path/to/band_normals_r1145_200.npy
Then qualify:
    python qualify_strip.py strips/UC-01/UC-01.npz            # checks a-c
    python qualify_strip.py strips/UC-01/UC-01.npz --ct-check  # all four
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

# make the repo root importable when run as strips/UC-01/convert_...py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from make_strip import EDGE_MARGIN_REVOLUTIONS, assign_tier, compute_pitch
from strip_format import SCHEMA_VERSION, edge_flags_from_phase, save_strip

# ---- constants ported from gate_3090/score_native.py (unchanged) ----
CX, CY = 1809.26609333, 1732.01937758  # fitted umbilicus, unwrap reference only
C0, C1 = 12750, 12950                  # seed window columns (recorded in meta)
CLASSES = list(range(-3, 4))
VOX_UM = 7.91
MIN_CLASS_POINTS = 100

CT_VOLUME_URL = (
    "https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/"
    "volumes_zarr_standardized/53keV_7.91um_Scroll3.zarr"
)


def unwrap_band(bxyz: np.ndarray, bvalid: np.ndarray) -> np.ndarray:
    """PROVENANCE: verbatim port of the 2D winding-consistent unwrap from
    gate_3090/score_native.py (itself from winding_audit_v4.py, where its
    self-test measured p90 0.1 vox). Logic unchanged."""
    theta = np.where(
        bvalid, np.arctan2(bxyz[..., 1] - CY, bxyz[..., 0] - CX), np.nan
    )
    U = np.full_like(theta, np.nan)
    REF = 100
    cols_ref = np.where(bvalid[REF])[0]
    U[REF] = np.interp(
        np.arange(bxyz.shape[1]), cols_ref, np.unwrap(theta[REF, cols_ref])
    )
    for direction in (+1, -1):
        prev = U[REF].copy()
        r = REF + direction
        while 0 <= r <= bxyz.shape[0] - 1:
            u_r = theta[r] + 2 * np.pi * np.round((prev - theta[r]) / (2 * np.pi))
            carry = ~np.isfinite(u_r)
            u_r[carry] = prev[carry]
            U[r] = u_r
            prev = u_r
            r += direction
    return np.where(bvalid, U, np.nan)


def convert(npz_path: Path, out_path: Path,
            normals_path: Path = None) -> None:
    d = np.load(npz_path)
    bxyz = d["xyz"].astype(np.float64)
    bvalid = d["valid"]
    row0 = int(d["row0"])
    print(f"loaded band: xyz {bxyz.shape}, valid {bvalid.shape}, row0={row0}")

    # Optional per-cell surface normals, aligned to the SAME (row, col) band
    # grid as `xyz`. The source xyz npz is an xyz-only snapshot; the PPM the
    # band was cut from actually carries six channels (x, y, z, nx, ny, nz),
    # so the normals are recoverable by range-downloading the same rows of
    # the PPM and keeping channels 3:6 (see this strip's README rebuild
    # recipe). They are needed ONLY by qualify_strip.py --ct-check, which
    # samples CT intensity along them; without them the strip still qualifies
    # on checks (a)-(c) but the CT cross-check cannot run.
    bnrm = None
    if normals_path is not None:
        bnrm = np.load(normals_path)
        if bnrm.shape != bxyz.shape:
            raise ValueError(
                f"normals {bnrm.shape} do not match the band xyz grid "
                f"{bxyz.shape}; they must be the same (rows, cols, 3) layout"
            )
        print(f"loaded normals: {bnrm.shape} (|n| median "
              f"{float(np.median(np.linalg.norm(bnrm[bvalid], axis=-1))):.3f})")

    U_valid = unwrap_band(bxyz, bvalid)

    # PROVENANCE: class assignment as in score_native.py (unchanged):
    # classes are relative to the seed-center angle.
    U_center = U_valid[100, (C0 + C1) // 2]
    w = (U_valid - U_center) / (2 * np.pi)
    cls = np.where(np.isfinite(w), np.rint(w), 99).astype(np.int64)
    populations = {int(n): int((cls == n).sum()) for n in CLASSES}
    print("class populations:", populations)

    rr_all, cc_all = np.where(bvalid)
    U_flat = U_valid[rr_all, cc_all]
    cls_flat = cls[rr_all, cc_all]
    xyz_flat = bxyz[rr_all, cc_all]
    nrm_flat = bnrm[rr_all, cc_all] if bnrm is not None else None

    kept_classes = [
        n for n in CLASSES if int((cls_flat == n).sum()) >= MIN_CLASS_POINTS
    ]
    print("classes kept (>= 100 points):", kept_classes)

    # strip-v0 adaptation: signed classes -> sequential wrap ids, ascending
    class_to_wrap = {n: i for i, n in enumerate(sorted(kept_classes))}
    margin_rad = EDGE_MARGIN_REVOLUTIONS * 2 * np.pi

    wraps, edges = {}, {}
    normals_out = {} if nrm_flat is not None else None
    for n, wid in class_to_wrap.items():
        sel = cls_flat == n
        wraps[wid] = xyz_flat[sel].astype(np.float32)
        edges[wid] = edge_flags_from_phase(U_flat[sel], margin_rad)
        if normals_out is not None:
            normals_out[wid] = nrm_flat[sel].astype(np.float32)
        print(f"  class {n:+d} -> wrap {wid}: {int(sel.sum())} points, "
              f"{int(edges[wid].sum())} edge-flagged")

    pitch_native = compute_pitch(wraps)
    pitch_um = {k: v * VOX_UM for k, v in pitch_native.items()}
    tier = assign_tier(pitch_um["median"])
    print(f"pitch: median {pitch_native['median']:.2f} vox = "
          f"{pitch_um['median']:.1f} um (p10 {pitch_um['p10']:.1f}, "
          f"p90 {pitch_um['p90']:.1f}) -> tier {tier}")

    source_checksum = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    print(f"source sha256: {source_checksum}")
    normals_checksum = (
        hashlib.sha256(Path(normals_path).read_bytes()).hexdigest()
        if normals_path is not None else None
    )

    if normals_out is not None:
        normals_meta = (
            "present: per-cell surface normals recovered from PPM channels "
            "3:6 (nx, ny, nz), aligned to the same band grid as xyz; enables "
            "qualify_strip.py --ct-check. See README rebuild recipe. "
            f"normals source sha256: {normals_checksum}"
        )
    else:
        normals_meta = (
            "not present in the source npz (xyz-only snapshot); the "
            "optional CT check needs normals and cannot run on this "
            "strip as converted"
        )

    meta = {
        "strip_id": "UC-01",
        "scroll": "PHerc0332 / Scroll 3",
        "segment_id": "20240618142020",
        "window": {
            "source": "PPM rows 1145-1344 inclusive (200 rows), full width "
                      "25706 columns",
            "row0": row0,
            "seed_window_cols": [C0, C1],
            "revolutions": 3.03,
        },
        "voxel_size_um": VOX_UM,
        "tier": tier,
        "schema_version": SCHEMA_VERSION,
        "source_checksum": source_checksum,
        "source_file": "band_r1145_200_xyz.npz (release/neural-tracing-audit)",
        "source_winding_class_to_wrap": {
            str(n): wid for n, wid in class_to_wrap.items()
        },
        "source_class_populations": {str(k): v for k, v in populations.items()},
        "umbilicus_xy": [CX, CY],
        "ct_volume_url": CT_VOLUME_URL,
        "normals": normals_meta,
        "normals_source_checksum": normals_checksum,
        "built_by": "strips/UC-01/convert_ntaudit_band.py",
        "provenance": "release/neural-tracing-audit/docs/NT_AUDIT.md",
    }

    save_strip(out_path, wraps, normals=normals_out, pitch_um=pitch_um,
               meta=meta, edges=edges)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("band_npz", help="path to band_r1145_200_xyz.npz")
    parser.add_argument("--out", default=str(Path(__file__).parent / "UC-01.npz"))
    parser.add_argument(
        "--normals-npy", default=None,
        help="OPTIONAL (rows, cols, 3) .npy of per-cell surface normals "
             "aligned to the band grid (PPM channels 3:6). When given, the "
             "strip carries per-wrap normals so qualify_strip.py --ct-check "
             "can run; see this strip's README for how to produce it.",
    )
    args = parser.parse_args()
    convert(Path(args.band_npz), Path(args.out),
            normals_path=Path(args.normals_npy) if args.normals_npy else None)


if __name__ == "__main__":
    main()
