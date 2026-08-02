"""Rebuild the bundled local multi-wrap reference from the public PPM.

The PPM stores little-endian float64 records ``x,y,z,nx,ny,nz`` in row-major
order. This command range-fetches the exact 200-row benchmark band, writes the
float32 XYZ reference NPZ, and preserves the 200x200 seed-normal crop needed by
the historical null baseline and independent CT-spacing check.
"""

import argparse
from pathlib import Path
import urllib.request as urlrequest

import numpy as np


DEFAULT_PPM = (
    "https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/"
    "paths/20240618142020/20240618142020.ppm"
)
POINT_BYTES = 6 * 8


def parse_ppm_header(header):
    marker = header.find(b"<>\n")
    if marker < 0:
        raise ValueError("PPM header terminator '<>\\n' was not found")
    fields = {}
    for line in header[:marker].decode("ascii").strip().splitlines():
        key, value = line.split(": ", 1)
        fields[key] = value
    return int(fields["width"]), int(fields["height"]), marker + 3


def row_byte_range(data_offset, width, row_start, row_count):
    start = data_offset + row_start * width * POINT_BYTES
    end = start + row_count * width * POINT_BYTES - 1
    return start, end


def fetch_range(url, start, end, timeout=300):
    request = urlrequest.Request(url, headers={"Range": f"bytes={start}-{end}"})
    payload = urlrequest.urlopen(request, timeout=timeout).read()
    expected = end - start + 1
    if len(payload) != expected:
        raise ValueError(
            f"range {start}-{end} returned {len(payload)} bytes; expected {expected}"
        )
    return payload


def parse_rows(payload, row_count, width):
    expected_values = row_count * width * 6
    values = np.frombuffer(payload, dtype="<f8")
    if values.size != expected_values:
        raise ValueError(f"row payload has {values.size} values; expected {expected_values}")
    return values.reshape(row_count, width, 6)


def load_seed_grid(seed_dir):
    import tifffile

    seed_dir = Path(seed_dir)
    arrays = [tifffile.imread(seed_dir / f"{axis}.tif") for axis in "xyz"]
    return np.stack(arrays, axis=-1).astype(np.float32)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppm-url", default=DEFAULT_PPM)
    parser.add_argument("--row-start", type=int, default=1145)
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--seed-col-start", type=int, default=12750)
    parser.add_argument("--seed-cols", type=int, default=200)
    parser.add_argument("--chunk-rows", type=int, default=8)
    parser.add_argument("--out", default="band_r1145_200_xyz.npz")
    parser.add_argument("--normals-out", default="seed_normals.npy")
    parser.add_argument("--seed-dir", default="seed_segment")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.rows <= 0 or args.seed_cols <= 0 or args.chunk_rows <= 0:
        raise ValueError("row, seed-column, and chunk counts must be positive")

    header = fetch_range(args.ppm_url, 0, 4095, timeout=60)
    width, height, data_offset = parse_ppm_header(header)
    if args.row_start < 0 or args.row_start + args.rows > height:
        raise ValueError("requested rows are outside the PPM")
    seed_stop = args.seed_col_start + args.seed_cols
    if args.seed_col_start < 0 or seed_stop > width:
        raise ValueError("requested seed columns are outside the PPM")

    xyz = np.empty((args.rows, width, 3), dtype=np.float32)
    valid = np.empty((args.rows, width), dtype=bool)
    normals = np.empty((args.rows, args.seed_cols, 3), dtype=np.float32)
    for local_start in range(0, args.rows, args.chunk_rows):
        count = min(args.chunk_rows, args.rows - local_start)
        source_row = args.row_start + local_start
        start, end = row_byte_range(data_offset, width, source_row, count)
        rows = parse_rows(fetch_range(args.ppm_url, start, end), count, width)
        chunk_xyz = rows[..., :3]
        xyz[local_start:local_start + count] = chunk_xyz.astype(np.float32)
        valid[local_start:local_start + count] = (chunk_xyz > 0).all(axis=-1)
        normals[local_start:local_start + count] = rows[
            :, args.seed_col_start:seed_stop, 3:6
        ].astype(np.float32)
        print(f"fetched rows {source_row}-{source_row + count - 1}", flush=True)

    np.savez_compressed(args.out, xyz=xyz, valid=valid, row0=args.row_start)
    np.save(args.normals_out, normals)

    seed = load_seed_grid(args.seed_dir)
    rebuilt_seed = xyz[:, args.seed_col_start:seed_stop]
    if seed.shape != rebuilt_seed.shape or not np.array_equal(seed, rebuilt_seed):
        raise ValueError("rebuilt PPM crop does not exactly match bundled seed tifxyz")
    norms = np.linalg.norm(normals, axis=-1)
    if not np.isfinite(normals).all() or np.any(norms < 1e-6):
        raise ValueError("seed normals contain invalid vectors")
    print(
        f"wrote {args.out} and {args.normals_out}; valid={int(valid.sum())}; "
        f"seed match exact; normal norm median={float(np.median(norms)):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
