---
license: cc-by-nc-4.0
task_categories:
- image-classification
tags:
- vesuvius-challenge
- herculaneum
- papyrology
- image-classification
- weak-supervision
- human-in-the-loop
pretty_name: Herculaneum legibility proxy — training labels
---

# Herculaneum legibility proxy — training labels (proxy_v4 / proxy_s2s3_v1)

This is the **label set and training recipe** behind the `proxy_v4` /
`proxy_s2s3_v1` model card:
https://huggingface.co/LimeGS/herculaneum-legibility-proxy.
Read that first — it documents the model, the preprocessing, the
thresholds, and the validation results. This dataset only covers how the
model was *trained*: the human labeling protocol, the exact windows it
produced, and a script to turn them back into pixels and a checkpoint.

`proxy_v4` is a ResNet-18 binary classifier that scores ~1 cm windows of
Vesuvius Challenge ink-detection maps for legible-text likelihood (connected
Greek letterforms vs. fiber noise/damage). `proxy_s2s3_v1` is the same model
fine-tuned with extra negatives from self-rendered Scroll 2/3 maps.

## Files

| File | Checksum |
|---|---|
| `train_labels.jsonl` | MD5 `7af281941a56` |
| `fiber_negatives_50.jsonl` | MD5 `8b9bd89c4a85` |
| `full_index_complete.json` | SHA256 `4d393d70ce886ed62b7e73e365f1d01cbe7f6efa37168fb3f27ade2b89d6e7a8` |

## What's in this release

- `full_index_complete.json` — the full scoring index this release's model
  produced over all 78 official Scroll 1 ink maps: 125,298 rows of
  `{panel, y, x, v3}`. The score key is named `v3` for legacy reasons
  (the field predates the final checkpoint); every score was produced by
  `proxy_v4.pt`. Gold = `v3 >= 0.9`.
- `summarize_index.py` — recomputes, with asserts, every published bucket
  and denominator from that index (125,298 / 15,386 / union 5,678 / 4,424 =
  77.9% / classics 2,460 / redundant 7,248).
- `eval_checkpoint.py` — frozen evaluator: deterministically reconstructs
  the held-out validation split (81 positives / 123 negatives) and scores
  the published checkpoint. Expected: AUROC 0.9853457794.
- `fiber_negatives_50.jsonl` — the 50 GPU-verified-fiber negative windows
  (weight 1.0) the original `proxy_v4` run trained on: the top-50, by
  score, of the project's full-grid candidate ranking. Same row schema as
  the main file; 12 of the 50 fall in the validation panels and `train.py`
  routes them there, as in the original run.
- `train_labels.jsonl` — 686 rows, one JSON object per line. Every row is a
  `{panel/segment, y, x, window size}` coordinate into an already-public
  CC BY-NC 4.0 map, plus the human (or human-implied) verdict, a training
  weight, and full provenance back to the original labeling-session file.
- `generate_crops.py` — turns those coordinates back into the exact 160×160
  training crops, using the identical preprocessing documented in the model
  card (percentile [2, 99.5] stretch over active pixels → resize → pad for
  undersized fragments). Its S1 downloader is source-locked to
  `2.4um-0.22m-78keV-volume-20260411134726` and refuses missing or ambiguous
  matches; some panel prefixes also publish a different 1.129 µm ds8 map.
- `train.py` — the consolidated training loop (ResNet-18, BCE-with-logits,
  AdamW, per-example weighting, spatial holdout) that turns crops back into
  a checkpoint with the same architecture as the published one.
- `requirements.txt` — the pinned dependency versions this release was
  built and smoke-tested with.
- `test_package.py` — self-checks (row counts, schema, id uniqueness, the
  12-in-validation fiber invariant, index SHA256, and exact S1 download
  source selection); runs from any directory with no network.
- `VERIFICATION.md` — this release's own QC pass: every row count checked
  against the source files, and checked against the model card's prose
  claims (with one confirmed discrepancy — see below).

## Why no images are bundled

The label rows are coordinates, not pixels. Every map they point into is
either:

1. an **official** Vesuvius Challenge ink-detection map (Scroll 1 /
   PHercParis4, the June-2026 `w010`–`w100` panel series), CC BY-NC 4.0,
   downloadable from the public anonymous S3 mirror; or
2. a **self-rendered** Scroll 2/3 mosaic produced by this project's own
   rendering pipeline (used only because no official ink map existed yet
   for those regions) — see "Known limitation" below for what that means
   for reproducibility.

Publishing coordinates instead of crops keeps this release at a few hundred
KB instead of redistributing gigabytes of imagery, while still making the
exact training set fully specified and (for the S1 portion) fully
regenerable by anyone with access to the public maps.

For Scroll 1, crop and score reproduction must use the 2.4 µm render family
named above. Scores do not transfer across render families, even when both
products are published under the same panel prefix.

## Labeling protocol

**Scroll 1 (PHercParis4), three rounds, all against the 16 official ds8
`w010`–`w100` panels, 512×512 px windows (≈1 cm physical):**

1. **`browse_all`** (`human_marks_v1.json`, 398 rows) — a reviewer paged
   through model-agnostic candidate windows and tagged each `interesting`
   or `unsure`. 393 interesting → positive; 5 unsure → excluded.
2. **`model_proposed`** (`round2_top120.json` / `round2_confirmed.json`,
   120 rows) — the top-120 windows by an early classifier's score were
   shown to the reviewer for confirmation. All 120 were confirmed (0
   rejected) → positive. (`round2_top120.json` is the pre-confirmation
   candidate list; `round2_confirmed.json` — the file the training scripts
   actually read — has the identical 120 coordinates plus a `human:
   "confirmed"` field, so both are represented by one row each, sourced
   from `round2_confirmed.json`.)
3. **`uncertainty_sampled`** (`round3_marks.json` / `round3_uncertain.json`)
   — 132 windows in the model's ambiguous score band (0.35–0.70-ish) were
   pulled as `round3_uncertain.json`; the reviewer explicitly tagged 45 of
   them in `round3_marks.json` (28 `interesting` → positive, 17 `unsure` →
   excluded). The other 132 − 45 = **87** were shown to the reviewer during
   the same pass but never flagged — treated as **implied negatives at
   weight 0.7** (round `uncertainty_sampled_implied`, source
   `round3_uncertain.json`), on the reasoning that a genuinely interesting
   window in a browsed set would have been flagged.

Additionally, **`verification_only`** (`verify6.json`, 6 rows) — six
already-gold (score 1.0) windows that were independently cross-checked
against a *native-resolution* CT render on a different pipeline
(`tools/session-20260706/verify6_driver.sh`) as a sanity check, not a
labeling round. This file has no interesting/negative verdict field and is
never read by either training script — included here for transparency of
everything that touched these maps, marked `split: excluded`,
`label: unsure`. Two of the six coincide exactly with an existing
round-1/2/3 positive coordinate (flagged per-row in the `note` field).

**Scroll 2/3, one round (`context_reviewed`, `s2s3_context_review_final.json`,
30 rows):** a first pass (`s2s3_seed_round1_marks.json`) tagged 30
candidate windows from the self-rendered S2/S3 mosaics as 7 `interesting` /
23 `unsure`. Because these were sparse, low-signal fragments, every one was
re-reviewed with 3×3 spatial context added around it; most downgraded to
`noise`. Final: 23 `noise` (→ negative, weight 1.0), 3 `interesting` + 4
`unsure` (→ excluded, never trained on — these are the only S2/S3 windows
that stayed live candidates after context review, and none were confirmed
as actual text).

## Row counts (verified against the actual files — see VERIFICATION.md for the full derivation)

| Scroll | Round | Rows | Label | Split | Weight |
|---|---|---:|---|---|---|
| S1 | browse_all | 393 | positive | train | 1.0 |
| S1 | browse_all | 5 | unsure | excluded | — |
| S1 | model_proposed | 120 | positive | train | 1.0 |
| S1 | uncertainty_sampled | 28 | positive | train | 1.0 |
| S1 | uncertainty_sampled | 17 | unsure | excluded | — |
| S1 | uncertainty_sampled_implied | 87 | negative | train | 0.7 |
| S1 | verification_only | 6 | unsure | excluded | — |
| **S1 total** | | **656** | 541 positive / 87 negative / 28 excluded | | |
| S2/S3 | context_reviewed | 23 | negative | train | 1.0 |
| S2/S3 | context_reviewed | 7 | unsure | excluded | — |
| **S2/S3 total** | | **30** | 0 positive / 23 negative / 7 excluded | | |
| **Grand total** | | **686** | | | |

## Known limitation — this is not the complete negative set the published checkpoints trained on

The label rows above cover every window that a human explicitly looked at.
The actual training runs (`train_proxy_v4.py`, `train_proxy_s2s3_v1.py` in
the project's internal archive) also mixed in two more
negative sources that are **not human labels** and so are deliberately not
in `train_labels.jsonl`:

1. **50 GPU-verified-fiber windows** (weight 1.0) — the top-50 of an
   automatically-scored full-grid candidate ranking over all S1 windows
   (a model/heuristic output, not a human judgment — which is why they ship
   separately from the human labels). **Included in this release as
   `fiber_negatives_50.jsonl`** (2026-07-11 peer-review fix: an earlier
   version omitted them, which made the published AUROC non-reproducible
   since 12 of the 50 sit in the validation panels). Provenance caveat: the
   original script read a session copy of the ranking; the shipped file
   applies the same selection rule (first 50 of the score-sorted ranking)
   to the archived ranking file.
2. **Procedural background negatives** (~30 per panel that contains a
   positive mark, weight 1.0, fixed seed) — a content-thresholded random
   sample with no human judgment involved at all, fully regenerable from
   the raw map pixels alone. `train.py` regenerates this automatically from
   `--maps-dir` (same directory used for crop generation), reproducing the
   original scripts' algorithm exactly.

Separately, and more importantly for `proxy_s2s3_v1`: the model card states
"192 additional negatives (23 human-reviewed noise at weight 1.0, **169
implicit at 0.7**)". This release accounts for the 23. **The 169 implicit
negatives' coordinates could not be recovered** — `train_proxy_s2s3_v1.py`
reads them from a ~199-item candidate gallery (`gallery_items`, loaded via
an `S2S3_ITEMS` environment variable pointing at a session-specific
scratchpad path) that no longer exists anywhere in the project repo or
filesystem. Only the 30 windows that a human actually reviewed
(`s2s3_context_review_final.json`) survived. This means:

- `train_labels.jsonl` is a complete, verified record of every S2/S3 window
  a human looked at.
- It is **not** sufficient to exactly reproduce `proxy_s2s3_v1.pt`'s
  training run — `train.py --mode s2s3` will fine-tune on 23 S2/S3
  negatives instead of 192, and should be expected to land close to but not
  identical to the published checkpoint's behavior on self-rendered maps.
  See VERIFICATION.md for the full accounting.

## How to reproduce the model end to end

```bash
# 0. Verify the PUBLISHED checkpoint first (deterministic; expected
#    AUROC 0.9853457794 on 81 positives / 123 negatives):
python generate_crops.py --labels train_labels.jsonl \
    --maps-dir /path/to/s1_official_panels --out ./crops
python generate_crops.py --labels fiber_negatives_50.jsonl \
    --maps-dir /path/to/s1_official_panels --out ./crops
python eval_checkpoint.py --checkpoint /path/to/proxy_v4.pt \
    --maps-dir /path/to/s1_official_panels --crops-dir ./crops

# 1. (optional, for the s2s3 variant) S2/S3 crops need the self-rendered
#    harvest (project-internal, not redistributed — see "Known limitation"):
python generate_crops.py --labels train_labels.jsonl \
    --maps-dir /path/to/s1_official_panels \
    --s2s3-dir /path/to/s2s3_harvest --out ./crops

# 2. Retrain a proxy_v4 equivalent (ImageNet init; labels + fiber negatives
#    + procedural background negatives, original spatial holdout):
python train.py --mode s1 --labels train_labels.jsonl --crops-dir ./crops \
    --maps-dir /path/to/s1_official_panels \
    --extra-negatives fiber_negatives_50.jsonl --out proxy_v4_reproduced.pt

# 3. Fine-tune the proxy_s2s3_v1 equivalent (see "Known limitation" above —
#    won't be bit-identical to the published checkpoint):
python train.py --mode s2s3 --labels train_labels.jsonl --crops-dir ./crops \
    --maps-dir /path/to/s1_official_panels \
    --init-checkpoint proxy_v4_reproduced.pt --out proxy_s2s3_v1_reproduced.pt

# 4. Compare: outputs are plain resnet18()+Linear(512,1) state_dicts,
#    122 tensors, same keys/shapes as the published checkpoints. Score your
#    retrained model with eval_checkpoint.py --checkpoint <yours>; expect
#    close to, not exactly, the published 0.9853457794.
```

Trains in minutes on a laptop (Apple MPS, CUDA, or CPU) — this was always meant to
be a small, cheaply-reproducible model, not a reason to withhold the labels.

## Provenance & license

Labels are the project's own human review, over coordinates into publicly
released Vesuvius Challenge data (CC BY-NC 4.0). Released under the same
CC BY-NC 4.0 to match the model card. No official-team code, checkpoints,
or imagery are redistributed here — only coordinates and a from-scratch
training script.
