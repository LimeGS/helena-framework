# Streaming tools for the 2.4 µm Herculaneum rescans

## Introduction

Starting in 2026, the Vesuvius Challenge began releasing new BM18 CT rescans
of the Herculaneum scrolls at 2.4 µm resolution. Each volume is 8–123 TB.

This is a small, standalone toolkit for four specific jobs we needed while
working on Scrolls 2, 3, 4, and 5 this campaign: bridging a legacy segment
into a new rescan, carving a renderable region out of unsegmented territory,
moving existing ink labels onto a fresh render, and running three specific
third-party ink models with their exact preprocessing. Every script streams
only the chunks a job needs (no local copy of a multi-terabyte volume),
prints its byte budget up front, and resumes after a crash. Built and used
for: both Scroll 3 primes at 62 native layers, the first ink inference ever
run on Scroll 2's April-2026 rescan, six "virgin wedge" renders in
unsegmented territory, and a dual-energy pair on Scroll 1. Total streamed
across all of it: **~600 GB, out of volumes summing to more than 200 TB.**

**This is not a replacement for the official `vesuvius` package**
(`ScrollPrize/villa`), which already has broader, more capable machinery
for most of this — see the next section before reaching for anything here.

## Relationship to the official `vesuvius` package

**The gap this repo fills: nothing official renders a `.ppm` at all any more,
and nothing ever bridged one onto a different volume.**

Checked against `ScrollPrize/villa` at HEAD, two ways:

- The pip-installable `vesuvius` package has no PPM support. Its modules are
  built around **tifxyz** — `vesuvius.tifxyz` (reader, writer, hierarchical
  tiling, upsampling), `neural_tracing`, `ink_detection`. A code search scoped
  to `vesuvius/` returns zero references to `.ppm`.
- `volume-cartographer` has 49 apps and **none of them names PPM**. The
  `vc_render --output-ppm` / `vc_layers_from_ppm` pair this section used to
  point at is gone: a filename search returns nothing and the whole
  subdirectory carries two incidental mentions of the format. What is there is
  `vc_render_tifxyz`, beside `vc_flatten`, `vc_obj2tifxyz`, `vc_tifxyz2obj`
  and `flatboi`.

So the official path moved to tifxyz, and the earlier claim here — that PPM
support exists upstream and merely lacks cross-volume bridging — is no longer
true. It was accurate when written.

PPM is still the *only* format many 2023/2024-era segments are published in
(our own `vesuvius_data.py` in the parent project documents that split:
"classic" `.ppm` versus "modern" tifxyz), so for those segments there is now no
official route at all — not to a newer rescan, and not within their own volume
either.

**The open question this raises, which this repo does not answer:** whether
converting those PPMs to tifxyz and using the official path is preferable to
maintaining a renderer here. That depends on whether the conversion preserves
what the segment needs, which has not been measured.

| Job | Official `vesuvius`/`villa` | This repo |
|---|---|---|
| Render a segment in its own volume | `vc_render_tifxyz` (`volume-cartographer`), for a surface in **tifxyz**. Nothing official renders a `.ppm` any more: the `vc_render --output-ppm` / `vc_layers_from_ppm` pair named here before is gone, and none of volume-cartographer's 49 apps names the format. | not needed here |
| Bridge a segment/labels across a `transform.json`-related *different* volume pair | `vesuvius.data.affine` (`read_transform_json`, `resample_label_to_image_grid`) has the cross-frame math; `mesh_to_surface` can render an `.obj` against an arbitrary target volume. **None of the tools with cross-volume bridging take a `.ppm` as input** — `vc_layers_from_ppm`'s PPM support only renders against its own source volume | `render_scroll2.py` / `render_scroll3.py` take a `.ppm` directly, apply the same bridging math (reimplemented independently — see Validation), and stream a stack out against a *different* volume. The one-command path for PPM-only segments that need a new rescan |
| Trace surface into unsegmented territory | `neural_tracing` — a full neural-net-based tracer with its own training/inference pipeline, actively integrated into VC3D; almost certainly more capable than what's here | `winding_tracer.py` + `wedge_extract.py` — a v0 radial ray-caster, far simpler, useful if you want something you can read start-to-end in an afternoon, and its output is a `.ppm` too (so it composes with the renderers above) |
| Move labels across volumes | `vesuvius.data.affine.resample_label_to_image_grid` resamples label *arrays* (zarr-to-zarr) — the docstring's own phrasing ("the label slab fetched from disk") assumes the label is already an indexable array. A raw label PNG needs converting first | `label_transport.py` starts from a `.ppm` + a plain label PNG directly, no conversion step, HTTP-Range-fetches only the rows touched |
| Unified inference CLI | `vesuvius.predict` — a real documented entrypoint, for models trained within the `vesuvius` framework (nnU-Net-v2-compatible) | `adapters/` wrap three *specific* pre-existing models that shipped as independent scripts from their own authors/papers, outside that framework — narrower in scope, not a general replacement |
| Lazy, no-full-download remote access | Already the case — `vesuvius`'s data layer reads zarr via `zarr`/`fsspec`, which is chunk-on-demand by design | Same idea, implemented directly against S3 with an explicit printed byte budget and per-stripe resume file |

Summary: the underlying math (transform bridging, cross-frame resampling)
is not novel — it's in the official package, better-maintained, and for
surface tracing specifically the official tool is more capable than ours.
The real, narrow reason this repo's renderer and label-transport tools are
worth having alongside the official package is the input format: if what
you have is a `.ppm` (still common for older segments) and a plain label
PNG, there is currently no official one-command path from that to a
rendered stack or a transported label on a new volume — it would need an
undocumented PPM→obj conversion and a separate label→zarr conversion
first.

## Layout

```
render_scroll.py        chunk-gather re-renderer, any masked OME-Zarr rescan
render_scroll3.py       the PHerc 0332 2.399 um variant it replaced (kept)
render_scroll2.py       the PHercParis3 2.400 um variant it replaced (kept)
winding_tracer.py       radial winding map r(z, theta, w) over surface predictions
wedge_extract.py        (winding, z-band) region -> synthetic PPM ("virgin wedge")
label_transport.py      move human ink labels onto a re-render via HTTP Range
adapters/                uniform CLI for three specific third-party ink models
  infer_resnet3d_1667.py    PHerc.1667 ResNet3D family
  infer_dino3d.py           volumetric DINO-guided nnU-Net, spot-check/eval harness
  sweep_dino3d.py           same model, full-PPM batched sweep
  infer_timesformer.py      GP-era TimeSformer (patched upstream copy)
requirements.txt        core deps; adapter extras documented inline
tests/test_tools.py     11 offline unit tests, no network/GPU/scan data
```

Requires Python >= 3.9 and network access to the public S3 bucket /
`dl.ash2txt.org` (both anonymous). `pip install -r requirements.txt` covers
the core tools; the adapters' torch stack is pinned in a comment there,
with the install-order warning that costs hours if ignored (see the
Adapters cheatsheet below).

## Cheatsheets

### Scenario A — I have a legacy segment PPM and want a fresh stack from a new rescan

```bash
# 62 native-resolution layers (canonical-reader style), Scroll 3:
python render_scroll3.py --ppm 20240716140050.ppm --out seg/20240716140050 \
       --layers 62 --spacing 1.0 --conc 64

# Scroll 2, legacy-spaced 26-layer stack (era-model style): omit --spacing,
# the per-layer step is derived from the transform matrix and printed at
# startup instead of assumed from nominal voxel sizes:
python render_scroll2.py --ppm 20230709155141.ppm --out seg/20230709155141

# Kill it anywhere (Ctrl-C, OOM, network drop) and rerun the exact same
# command — done.json resumes per-stripe. render_meta.json refuses to
# resume if --stripe/--layers/the PPM changed (loud assert, never a
# silently mixed stack).
```

Useful flags: `--stripe` (rows per output stripe, default 640), `--conc`
(concurrent chunk fetches, default 64), `--max-gb` (abort if the printed
budget exceeds this, default 14.0), `--pilot-stripe N` (render just stripe
N first, to sanity-check before committing to the full job).

Adapting to a volume that doesn't exist yet is an argument now:

```
python render_scroll.py --ppm seg.ppm --out out/seg \
  --volume PHercXXXX/volumes/<the masked OME-Zarr>.zarr
```

It used to be a five-line constants change and a copied file -- `VOLKEY`,
`VOLSHAPE` and the nominal spacing ratio -- with this section telling you to
diff the two renderers, which is an instruction to fork. Two of those three
were never ours to write down: the shape and the chunk edge are in the
volume's own `.zarray`, so they are read, and a mistyped shape now fails at
the source instead of rendering the wrong region. The third, the voxel size,
is parsed from the key. Checked against both files' hardcoded values:
identical for both scrolls.

`render_scroll2.py` and `render_scroll3.py` are still here and unchanged.

### Scenario B — I want to look at scroll territory that has no segment yet

For anything beyond a quick look, try the official `neural_tracing` module
in `ScrollPrize/villa` first — it's a proper neural surface tracer, actively
maintained, and almost certainly handles harder cases than the simple radial
ray-caster below. This is the "read it in an afternoon" version:

```bash
# 1. Build the winding map once (Scroll 3 surface predictions):
python winding_tracer.py --ntheta 1440 --procs 54
# --pilot first, if you want the ground-truth-mesh sanity check to print
# before committing to the full-scroll trace:
python winding_tracer.py --pilot

# 2. Turn any (winding, z-band) region into a synthetic PPM:
python wedge_extract.py --z0 5200 --z1 5600 --winding 40 --out w40.ppm

# 3. Feed that PPM into the Scenario A renderer like any other segment:
python render_scroll3.py --ppm w40.ppm --out seg/w40 --layers 62 --spacing 1.0
```

Caveats stated plainly in the code: wedge normals are radial
approximations — fine for detection experiments, not for a final reading
render. The tracer is a v0 that inherits any sheet-switching already
present in the community surface predictions; a closed 360° loop
incrementing the winding count by exactly one is the built-in audit that
catches when it hasn't.

### Scenario C — I have human ink labels and want them on a new-volume render

```bash
python label_transport.py --seg 20231005123336 \
       --label labels/20231005123336_inklabels.png \
       --size 1280 --max-windows 4
```

This picks the label-densest, non-overlapping windows first (so a
`--max-windows` budget buys the most-informative crops), HTTP-Range
fetches only the PPM rows those windows need — not the full (up to 15 GB)
segment PPM — and writes, per window, an `.npz` (`img`, `label`, `mask`)
plus a QC overlay JPEG (render mid-layer with the label boundary drawn on
top). **Look at the QC overlay before trusting a window** — it's how you
catch a transform-alignment problem instead of training on it. The
packager itself is beta; the alignment principle (real letterforms tracing
correctly on a 2.4 µm re-render) is demonstrated, not yet hardened.

`--render-cmd` lets you point this at either renderer (defaults to
`render_scroll3.py`); `--out` / `--tmp` control where packaged windows and
scratch bands land.

### Scenario D — I have a rendered stack and want an ink-probability map

If your model was trained within the official `vesuvius` framework
(nnU-Net-v2-compatible), use `vesuvius.predict` — it's the maintained,
general entrypoint. The three adapters below are for models that, as far as
we could tell, aren't wrapped by it: independent scripts released by their
own authors alongside specific papers/community posts, each with its own
easy-to-miss preprocessing convention. All three need a checkpoint you
source yourself — see "Data & checkpoint sources" below before running any
of these.

```bash
# PHerc.1667 ResNet3D family (HF checkpoints, iteration-0..5):
python adapters/infer_resnet3d_1667.py --stack stack.u8 --shape 62,H,W \
       --model models/iteration-5 --out pred.png

# DINO-guided volumetric nnU-Net, full-PPM batched sweep (prefetch
# overlapped with GPU forward — this is the one you want for coverage;
# infer_dino3d.py itself is a smaller spot-check/eval harness with a
# built-in air control, not a general sweep tool):
python adapters/sweep_dino3d.py --ppm seg/20240716140050.ppm --out out_sweep \
       --batch 6 --workers 24

# GP-era TimeSformer (patched upstream copy — model and tiling math
# unchanged, six bugfixes documented inline, see the table below):
python adapters/infer_timesformer.py --segment_id 20240716140050 \
       --segment_path eval_scrolls --model_path outputs/.../epoch=7.ckpt \
       --out_path out_ts --reverse 0
```

The detail that matters per adapter — get any of these wrong and the model
still runs, it just quietly outputs something worse:

| Adapter | Model family | The detail that matters |
|---|---|---|
| `infer_resnet3d_1667.py` | PHerc.1667 iteration-0..5 (HF) | `uint8 clip[0,200] / 255` — the paper's normalization, not plain `/255` (shifts output means by over a point on our renders) |
| `infer_dino3d.py`, `sweep_dino3d.py` | ink_3d_dino_guided, volumetric 256³ | builds the nnU-Net from the checkpoint's own embedded config; percentile 1/99 min-max on valid voxels; the sweep overlaps S3 prefetch with GPU forward instead of a naive per-cell loop that starves on fetch |
| `infer_timesformer.py` | GP-era TimeSformer | upstream reference with six documented fixes: a `--reverse` flag that upstream parsed but never applied, and a max-normalisation of the saved map that made every peak exactly 1.0 |

**Install order matters and costs hours if you get it wrong:** pin
`torch==2.6.0+cu124 torchvision==0.21.0+cu124` installed together and
*last*. Letting `timm` or `transformers` resolve torch afterwards pulls a
cu13 wheel that fails on 5xx drivers with a misleading "driver too old"
error — we lost hours to this twice, hence the warning both here and at
the top of every adapter file.

### Scenario E — the volume isn't chunked at all (raw TIFF slices)

Not every volume ships as zarr. Several energy volumes exist only as
uncompressed single-strip TIFF slices (177 MB each, tens of thousands of
slices). Because they're uncompressed, one HTTP Range request per slice
fetches exactly the row band a surface crop needs — the same
header-parse-then-band-fetch pattern `label_transport.py` uses for PPM
rows, applied to a raw TIFF stack instead. This repo doesn't ship a
ready-made script for that path (the dual-energy companion repo's
`render_332_pair.py` does, sampling a surface against a 348 GB volume for
~5 GB); the pattern is documented here because it generalizes to any raw
TIFF stack on the server, not just that one pair.

## Data & checkpoint sources

Nothing in this repo trains a model or hosts a checkpoint. Every adapter
runs someone else's public model; **we did not train any of the three
below**, and none of the checkpoints these scripts default to are ours to
redistribute. Where a script's default path looks like a local training
path (e.g. `outputs/vesuvius/pretraining_all/...`), that's inherited
verbatim from the upstream/official script's own example — you need to
substitute your own copy of the real checkpoint.

| Adapter | Model | Confirmed source | What we could confirm |
|---|---|---|---|
| `infer_resnet3d_1667.py` | PHerc.1667 ResNet3D, iteration-0..5 | Hugging Face (`trust_remote_code=True`), published alongside the Scroll 4 reading paper | Model family and hosting platform confirmed; we did not pin down the exact HF repo id while writing this README — search the paper's model/data availability section or the Vesuvius Challenge HF org |
| `infer_dino3d.py`, `sweep_dino3d.py` | `ink_3d_dino_guided` — volumetric nnU-Net (142M), 256³ raw patches, ships with a reference embedding (`avg_ref_embedding.npy`) | Official community model, identified via the Vesuvius Challenge Discord/model catalog | The `models1667/dino3d/ckpt_78k_fullsup.pth` default is only the local filename we gave our downloaded copy — "78k" is the upstream authors' own training-step count, not ours. Exact download URL not re-confirmed for this README; check the Discord `#ink-detection` model links |
| `infer_timesformer.py` | GP-era (2023) TimeSformer, from `ScrollPrize/villa` | Official upstream inference script, patched here (see the fixes table above) | The default `model_path` is upstream's own placeholder filename (`valid_20230827161847_0_fr_i3depoch=7.ckpt`), not a real path on your system — get a real checkpoint from the `villa` repo/Discord and point `--model_path` at it |

The winding tracer's input is a similar case: `winding_tracer.py` reads a
local `preds/surface.zarr` (the "community surface-prediction zarr for
Scroll 3" in its docstring) — a third-party auto-segmentation output, not
something this repo produces or redistributes. Point it at your own copy;
we did not pin its exact bucket location in this README.

By contrast, the renderers' own inputs *are* fully self-contained: segment
PPMs come from the standard public segment layout on
`vesuvius-challenge-open-data` / `dl.ash2txt.org` (the same place the
official `vesuvius` package reads them from), and `transform.json` is
fetched live from the public bucket at render time — no separate download
step for either.

## How each piece was validated

Every component was checked against an independent reference before we
trusted its output; the checks are reproducible and most run on every
invocation:

- **Transform bridge (both renderers).** Re-derived on every run and
  checked against the published landmark pairs before any chunk is
  fetched — mean |delta| per axis over the landmark set is printed (4.0
  vox for PHercParis3, 19.6 vox for PHerc 0332 — small against the
  128-voxel chunk size), with a hard assert at 60 vox. The dual-energy
  companion repo measured 1.1 vox for its PHercParis4 pair with a
  stricter per-point metric. The inversion + matrix-derived-scale math
  also has an offline unit test against a synthetic transform. This math
  was written independently of `vesuvius.data.affine` (see "Relationship
  to the official package" above) rather than imported from it; the two
  implementations agreeing on landmark residuals is itself a
  cross-check, not just an assertion that we trust our own code.
- **Rendered stacks.** A re-render of a Scroll 1 window, using the same
  chunk-gather + surface-normal-stepping sampling core (geometry supplied
  by the segment mesh rather than PPM + transform), read with the
  production ink model, correlates **r = 0.89 (ds8)** with the officially
  published ink map of the same region — two independent render chains
  agreeing on the result. On Scroll 3, where no official maps exist (PPM +
  transform path), the 27-patch full-prime mosaic showed 95.9% coverage
  with independently rendered patches agreeing in their overlap regions.
  Note the renderers sample the nearest voxel, no interpolation — correct
  for ink-detection stacks, but switch to trilinear sampling if you need
  sub-voxel fidelity for a display render.
- **Winding tracer.** Pilot mode validates against a ground-truth segment
  mesh: 3.42 vox median radial error vs 20–40 vox wrap spacing (`--pilot`
  reproduces this number; the same crossing geometry is unit-tested
  against a synthetic annulus mask). That pilot number predates this
  release's fix to a half-voxel inward bias in run-start indexing — the
  fix can only tighten it, not worsen it.
- **Wedge extractor.** Six wedges extracted at 99–100% trace coverage; the
  synthetic-PPM header contract is unit-tested against the renderers' own
  parser, and the winding-audit property described above (closed 360°
  loop increments w by exactly one) is the built-in cross-check.
- **Label transport.** QC overlays trace real letterforms on a 2.4 µm
  re-render — pixel-level alignment demonstrated visually on every window
  produced. The window-selection logic (density + non-overlap) is
  unit-tested. The packager itself remains beta.
- **Adapters.** `infer_resnet3d_1667`'s clip-at-200 normalization is
  unit-tested directly; `infer_dino3d` asserts 0 missing / 0 unexpected
  keys on checkpoint load and ships an air-patch (empty-space) control
  group so a systematic false-positive shows up immediately; the
  TimeSformer adapter's five patches are documented line-by-line against
  the unmodified upstream file, and its own tests are deliberately left to
  upstream rather than duplicated here.

## Testing

Offline unit tests — no network, no GPU, no scan data (transforms
mocked, masks synthetic):

```bash
python -m pytest tests/          # or, dependency-free:
python tests/test_tools.py
```

11 tests, each exercising one piece of the validation story above rather
than a synthetic coverage target:

| Test | What it locks down |
|---|---|
| `test_parse_ppm_header_roundtrip` | PPM header parse → write is lossless |
| `test_load_transform_inverts_and_derives_scale` | transform inversion + matrix-derived layer spacing, against a synthetic transform |
| `test_fetch_chunk_404_means_empty` | a missing chunk (outside the scanned volume) is treated as empty, not an error |
| `test_tracer_finds_synthetic_ring` | ray-crossing geometry finds the exact center of a synthetic annulus (post half-voxel-bias fix) |
| `test_pick_windows_density_and_separation` | label-transport window selection prefers dense, non-overlapping regions |
| `test_pick_windows_empty_label` | the same selector degrades safely on a label with no ink at all |
| `test_fetch_rows_batches_and_reassembles` | batched HTTP-Range fetches reassemble to the same bytes as one big request (mocked server) |
| `test_wedge_header_contract` | a wedge-extractor PPM header parses cleanly with the renderers' own parser |
| `test_preprocess_tile_clips_at_200` | the ResNet3D-1667 adapter's normalization clips at 200 before dividing by 255 |
| `test_gaussian_kernel_shape_and_peak` | the adapters' blending kernel is centered and correctly shaped |
| `test_norm_pminmax` | percentile 1/99 min-max normalization behaves correctly on edge-case inputs |

`adapters/infer_timesformer.py` is a patched copy of the upstream
`ScrollPrize/villa` inference script and is deliberately left to
upstream's own test suite; its five patches are marked `# PATCH` inline
for line-by-line review instead.

## License

MIT. Scan data referenced is CC BY-NC 4.0 from the Vesuvius Challenge.
