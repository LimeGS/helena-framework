# UC-01 — PHerc 0332 (Scroll 3) compressed multi-wrap band

The first registered reference strip: the 200-row, 3.03-revolution band of
segment `20240618142020` that the
[neural-tracing-audit](../../../neural-tracing-audit/) benchmark was built
on. Registered **by recipe**, not by bundling — the ~48 MB source band and
the ~127 MB converted strip (with normals; ~66 MB without) are not committed
to this repo; you rebuild them locally with the steps below and verify the
checksum.

"UC" = ultra-compressed, after the audit's seed window. See the honest
tier note below: the *band-wide* measured pitch lands in the medium tier;
the ultra-compressed figure refers to the seed-window core.

## Provenance

| Field | Value |
|---|---|
| Scroll / volume | PHerc 0332 / Scroll 3, `PHerc332.volpkg`, 53keV 7.91 µm standardized zarr |
| Segment | `20240618142020` (public segmentation, unstated authorship) |
| Band | PPM rows 1145–1344 inclusive (200 rows), full width 25,706 columns |
| Source file | `band_r1145_200_xyz.npz` from the neural-tracing-audit repo |
| Source sha256 | `a93b0683df0e3a03dc656f8a259c2d83c5949215a46fec6b4655c70d660d4408` (computed from the local file; matches that repo's `docs/CHECKSUMS.txt`) |
| Technical account | `release/neural-tracing-audit/docs/NT_AUDIT.md` and `docs/BENCHMARK.md` |

### Qualification numbers measured in the SOURCE AUDIT (cited, see provenance)

These were measured by the original audit on its 200×200 seed window with
its grid-based methodology — they are provenance, not this repo's output:

- Self-test: seed points vs their own wrap through the full pipeline =
  **0.000 vox** (median and max).
- Wrong-side discrimination control: front predictions scored against the
  wrong-side wrap fail **99.1%**.
- Constant-offset null baseline: **100% wrong** in both directions, at
  ~2× the distance of the real network predictions.
- Independent CT intensity cross-check: median inter-sheet spacing
  **13.0 vox = 103 µm** (p10–p90 8–22 vox) along seed normals — confirms
  the compressed spacing from raw image intensity, independent of the
  segmentation.

### Qualification numbers measured HERE (strip-v0 conversion, full band)

Measured by `qualify_strip.py` on the converted `UC-01.npz`
(2026-07-10, config recorded inside the report): **QUALIFIED**.

- (a) Self-test: max distance 0.000000 vox, 100% nearest-own-wrap.
- (b) Wrong-side separation: min fail 99.62% over all six adjacent pairs
  (bar ≥ 95%).
- (c) Null baseline (2× local-gap overshoot): min wrong-hop 99.16% over
  the three front pairs (bar ≥ 99%). Coverage-hole seed exclusions per
  pair: 4.3k / 12.0k / 103.7k of 250k tested — the last pair (wrap 2→3)
  excludes 41% of seeds because the outermost wrap's coverage is patchy
  relative to wrap 2. This is consistent with the source audit itself,
  which excluded ~53–54% of its front-direction cells on the same band
  for reference-coverage reasons.
- (d) CT intensity cross-check: **run and passed** (2026-07-10). Streamed
  raw CT from the public `53keV_7.91um_Scroll3.zarr` (level 0) along the
  strip's recovered per-wrap normals: median inter-sheet spacing
  **13.0 vox = 102.8 µm** (p10–p90 8–29 vox, n=83 spacings over 9 sampled
  cells) vs the strip's 141.8 µm pitch → **ratio 0.73**, inside the
  accepted band [0.5, 2.0]. This reproduces the source audit's 13.0 vox /
  103 µm seed-normal figure — now measured band-wide *through the shipped
  `check_ct`*, not merely cited. ~35 MB streamed. Earlier the converted
  strip carried no normals (the source `.npz` is an xyz-only snapshot) and
  the CT check could not run; the strip is now rebuilt with normals
  recovered from the PPM's `nx,ny,nz` channels (recipe below), which is
  what lets check (d) run — see `meta["normals"]` inside the strip.

### Measured strip properties (full band, this conversion)

- 4 wraps (source winding classes −2, −1, 0, +1 → wrap ids 0–3; mapping
  recorded in `meta["source_winding_class_to_wrap"]`). Class populations
  match the source audit's published table exactly
  (114,509 / 1,654,338 / 1,675,193 / 1,631,237).
- Band-wide pitch: median 17.93 vox = **141.8 µm** (p10 72.6, p90 528.1)
  → tier **medium** under the PROVISIONAL thresholds.
- Honest tier note: the audit's seed window (columns 12,750–12,950) is
  the ultra-compressed core, with local gaps of 8.8–11.2 vox
  (70–88 µm ≈ ultra) and CT-confirmed 103 µm spacing. The band-wide
  median (141.8 µm) pools all 3 revolutions including wider outer
  regions, hence medium. The strip is registered with the measured
  band-wide number, not the headline core number.

## Rebuild recipe

1. Obtain the source band (either):
   - copy `band_r1145_200_xyz.npz` from the neural-tracing-audit repo
     (bundled there, 47 MB), or
   - rebuild it from the public PPM via HTTP range requests: rows
     1145–1344 of
     `https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths/20240618142020/20240618142020.ppm`
     (header 74 bytes, then row-major float64 x,y,z,nx,ny,nz, width
     25,706 columns) — exact recipe at the bottom of that repo's
     `docs/NT_AUDIT.md`.
2. Verify: `shasum -a 256 band_r1145_200_xyz.npz` must print the sha256
   in the table above. Do not proceed on a mismatch.
3. (Optional, only for the CT cross-check) recover per-cell normals from
   the SAME PPM rows and save them aligned to the band grid. Channels 3:6
   of the PPM are `nx,ny,nz`; the xyz `.npz` is a 3-channel snapshot that
   dropped them, but they are recoverable from the same range request:

   ```python
   import urllib.request, numpy as np
   ppm = "https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths/20240618142020/20240618142020.ppm"
   W, DIM, HDR, r0, nrows = 25706, 6, 74, 1145, 200     # header ends at byte 74
   rb = W * DIM * 8
   req = urllib.request.Request(ppm, headers={"Range": f"bytes={HDR+r0*rb}-{HDR+(r0+nrows)*rb-1}"})
   band = np.frombuffer(urllib.request.urlopen(req).read(), "<f8").reshape(nrows, W, DIM)
   np.save("band_normals_r1145_200.npy", band[:, :, 3:6].astype(np.float32))  # ~235 MB fetched
   ```

   The recovered `band[:,:,0:3]` matches `band_r1145_200_xyz.npz`'s `xyz`
   bit-for-bit at every valid cell, which guarantees the normals are
   aligned to the same points.
4. Convert and qualify (from this repo's root):

```bash
# geometry-only strip (checks a-c); omit --normals-npy to skip the CT check
python strips/UC-01/convert_ntaudit_band.py /path/to/band_r1145_200_xyz.npz \
    --out strips/UC-01/UC-01.npz \
    --normals-npy /path/to/band_normals_r1145_200.npy
# all four checks, including the network CT cross-check:
python qualify_strip.py strips/UC-01/UC-01.npz --ct-check   # needs zarr fsspec aiohttp
```

The converter prints the class populations — check them against the table
above — and records the source file's sha256 (and the normals file's
sha256, when given) into the strip's meta.

## Converter status

`convert_ntaudit_band.py` was written against the documented npz layout
and **tested against the real band file** (present locally at conversion
time): class populations reproduced the source audit's published numbers
exactly, and the resulting strip passed qualification as reported above.
Its optional `--normals-npy` path was exercised end-to-end on 2026-07-10:
normals recovered from the PPM were attached, and `qualify_strip.py
--ct-check` then streamed the live CT volume and passed check (d) with the
numbers above.
