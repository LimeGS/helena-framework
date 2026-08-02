# Verification note

QC pass run while consolidating `train_labels.jsonl` from
`data/letters/s1_atlas/triage/*.json`. Every count below was produced by
reading the actual files and, for the S1 numbers, by reproducing
`train_proxy_v4.py`'s and `train_proxy_s2s3_v1.py`'s own positive/negative/
exclusion logic line-for-line against the real files (not by re-typing their
docstring summaries). The build script used is not part of this release;
its logic is reproduced in prose below and its output (`train_labels.jsonl`)
is checked in.

## Per-source-file row counts

| File | Rows | Breakdown |
|---|---:|---|
| `human_marks_v1.json` | 398 | `tag`: 393 `interesting`, 5 `unsure` |
| `round2_top120.json` | 120 | raw model-proposed candidates, no verdict field |
| `round2_confirmed.json` | 120 | same 120 coordinates as `round2_top120.json` + `human: "confirmed"` on all 120 (verified: `set(top120 coords) == set(confirmed coords)`, i.e. 120/120 confirmation, 0 rejected) |
| `round3_marks.json` | 45 | `tag`: 28 `interesting`, 17 `unsure` |
| `round3_uncertain.json` | 132 | candidate coordinates only, no verdict field; verified all 45 `round3_marks.json` coordinates are an exact subset of these 132 |
| `verify6.json` | 6 | coordinates + `v3` score only, no verdict field; not read by any training script |
| `s2s3_seed_round1_marks.json` | 30 | `tag`: 7 `interesting`, 23 `unsure` |
| `s2s3_context_review_final.json` | 30 | same 30 ids as the seed file; `tagFinal`: 23 `noise`, 4 `unsure`, 3 `interesting` |

## S1 derived counts (reproducing `train_proxy_v4.py`'s own logic)

```
pos_r1 (human_marks_v1.json, tag=="interesting")            = 393
pos_r2 (round2_confirmed.json, all rows)                    = 120
pos_r3 (round3_marks.json, tag=="interesting")               =  28
                                                       TOTAL POSITIVES = 541

unsure_r1 (human_marks_v1.json, tag=="unsure")                =  5
unsure_r3 (round3_marks.json, tag=="unsure")                  = 17
                                                       TOTAL EXCLUDED  =  22

neg_r3 implied (round3_uncertain.json rows NOT in round3_marks.json) = 87
  (132 candidates − 45 explicitly tagged = 87; verified no coordinate
  overlap between this set and the 541 positives, and no internal
  duplicates)
```

All of the above were computed directly against the real files (see
`train_labels.jsonl` for the row-level result) — not copied from the model
card or the training scripts' comments.

## S2/S3 derived counts (reproducing `train_proxy_s2s3_v1.py`'s own logic)

```
noise (tagFinal=="noise", weight 1.0 negative)         = 23
excluded (tagFinal in {"interesting","unsure"})        =  7  (3 interesting + 4 unsure)
                                                  TOTAL REVIEWED = 30
```

`train_proxy_s2s3_v1.py` does **not** read `s2s3_context_review_final.json`
directly for its full negative set — it reads a separate `gallery_items`
list (loaded from a path given by the `S2S3_ITEMS` environment variable,
which pointed at `.../s2s3_gallery/items.json` under an ephemeral,
session-specific scratch directory outside the repository).
`excluded_ids`/`noise_ids` are derived
from `s2s3_context_review_final.json` as above, but `neg_s2s3` (the actual
192-item training negative set the model card describes) is built by
iterating the full `gallery_items` list and labeling anything **not** in
`excluded_ids` as an implicit negative (weight 0.7 unless it's in
`noise_ids`, weight 1.0).

**That `gallery_items`/`items.json` file was searched for and does not
exist** — not in the project repo (`git log --all` shows no trace under
`triage/`), not in `/tmp`, not in `/private/tmp`. It lived in a prior
session's ephemeral scratchpad directory and is gone. This means:

- The 23 `noise` (weight 1.0) and 7 excluded rows are fully verified and
  included in `train_labels.jsonl`.
- The 169 "implicit" negatives (weight 0.7) the model card describes cannot
  be reconstructed from anything currently on disk — their exact
  coordinates, and even the true size of the 199-item candidate gallery
  they were drawn from, are not independently verifiable in this session.
  **Not fabricated or approximated** — simply absent from
  `train_labels.jsonl`, and called out here and in `DATASET_CARD.md`.

## Match / mismatch against the model card's "Training summary"

| Model card claim | Verified count | Status |
|---|---|---|
| "546 positive windows ... (398 browse-all + 120 model-proposed/human-confirmed + 28 uncertainty-sampled)" | **541** (393 + 120 + 28) | **WAS A MISMATCH, now corrected.** The "398 browse-all" figure was the *total* row count of `human_marks_v1.json` (398), not the count of rows tagged `interesting` (393) — 5 rows are `tag: "unsure"` and are correctly excluded elsewhere, but the positive-count arithmetic in the model card's original prose never subtracted them. `train_proxy_v4.py`'s own docstring has the same error ("Pos (546): 398 r1 + 120 r2 confirmadas + 28 r3" — but its *code* correctly filters to `tag=="interesting"`, giving 393, not 398). The actual number of positive training windows, reproduced from the real files and the real code path, is **541** — the model card was updated 2026-07-08 to state 541 with an inline correction note; this file's finding was the source of that fix. |
| "87 human-implied negatives at weight 0.7" | **87** | **MATCH.** |
| "22 human-'unsure' windows excluded" | **22** (5 + 17) | **MATCH.** |
| "192 additional negatives (23 human-reviewed noise at weight 1.0, 169 implicit at 0.7)" | 23 confirmed; 169 **unverifiable** | **PARTIAL — flagged above.** The 23 is exactly right. The 169 cannot be confirmed or denied; its source file does not exist in this environment. |
| "7 human-flagged candidate windows excluded from training entirely" | **7** (3 interesting + 4 unsure) | **MATCH.** |

### Bottom line

Four of five aggregate claims check out exactly. One did not at the time
this file was first written: **positive windows are 541, not 546** — a
small but real and precisely-explained discrepancy (a stale
unsure-inclusive round-1 total propagated into the model card's sum),
traced to its exact origin in both the model card prose and the original
training script's own docstring. `train_labels.jsonl` and
`DATASET_CARD.md` both report 541. This file was originally written
read-only against the model card, per that task's instructions, flagging
the mismatch for reconciliation rather than fixing it directly — that
reconciliation has since happened: the model card was
corrected to 541 on 2026-07-08 (see its inline "Corrected 2026-07-08"
note), so this is now a closed discrepancy, not an open one.

## Checkpoint architecture check

Loaded the real checkpoints directly:

```
data/letters/s1_atlas/triage/proxy_v4.pt        -> 122 tensors, md5[:12]=c869ce189f2a
data/letters/s1_atlas/triage/proxy_s2s3_v1.pt   -> 122 tensors, md5[:12]=95b6839c58c8
```

Both md5 prefixes match the model card's stated values
exactly. Compared both state_dicts' keys and shapes against a fresh
`torchvision.models.resnet18()` with `fc = nn.Linear(512, 1)`: identical key
set, zero shape mismatches, in both directions (v4 vs fresh, s2s3_v1 vs
fresh, v4 vs s2s3_v1). `train.py` in this release builds the model the same
way (`build_model()`), and a real 1-epoch smoke run (see below) confirms
this in practice, not just by inspection.

## Functional smoke test (not a static-only review)

Both `generate_crops.py` and `train.py` were run against the real local
project data (16 official S1 panel jpgs already present in
`data/letters/s1_atlas/`, and the real S2/S3 harvest under
`data/letters/s1_atlas/harvest_20260706/`):

- `generate_crops.py` produced valid 160×160 grayscale JPEGs for both S1
  rows (from panel jpgs) and S2/S3 rows (from `.npy` mosaics/harvest
  arrays, including the small-fragment padding path — verified a
  `cls_px=57` row actually renders as a padded canvas with black corners).
- `train.py --mode s1`, run for 1 epoch on a 2-panel subset (140 label
  rows, one of the two panels held out as val), completed without error and
  produced a checkpoint (AUROC 0.873 on the tiny val slice — not meant to
  be representative, this is a smoke test, not a real training run).
- `train.py --mode s2s3`, initialized from the above checkpoint and given a
  small S2/S3 negative slice, also completed for 1 epoch without error.
- The `--mode s1` output checkpoint's `state_dict` was compared directly
  against the real `proxy_v4.pt`: identical key set, 122/122 tensors,
  zero shape mismatches.

No full training run (which would take longer, and would only approximate
`proxy_v4.pt`/`proxy_s2s3_v1.pt` anyway given the 169-negative gap above)
was executed — out of scope per the task's instructions.

## Judgment calls made

1. **Row ids.** The schema asks for a "stable string" id but doesn't
   specify a format. Used `s1_<round>_<panel>_y<y>_x<x>` and
   `s2s3_<original gallery id>`. Discovered during a uniqueness check that
   2 of the 6 `verify6.json` windows share exact coordinates with an
   existing round-1 positive (they were drawn *from* the positive pool for
   independent native-render verification) — an id scheme keyed only on
   coordinates would have collided; the round is folded into the id to
   guarantee uniqueness, and the coincidence is flagged in a `note` field
   on the affected rows instead of silently dropping one.
2. **Scope of `train_labels.jsonl`.** Limited strictly to the 8 listed
   human-labeling files. The two automated negative-mining components used
   by the real training scripts (50 GPU-verified-fiber windows from
   `v1_all_candidates.json`, and procedural background sampling) are
   **not** included as static rows — they're not human labels, and the
   background sampler in particular needs raw pixel content to run, not
   just coordinates. Both are instead reproduced as code in `train.py`
   (`--extra-negatives` hook, `sample_background_negatives()`), documented
   in `DATASET_CARD.md`'s "Known limitation" section. This was a deliberate
   choice to keep "training labels" (what the task asked to verify and
   consolidate) distinct from "training recipe" (what `train.py` is for),
   rather than silently blending an unverifiable file into the label set.
3. **`round2_top120.json` vs `round2_confirmed.json`.** Both list the same
   120 windows; only `round2_confirmed.json` is read by the training
   script. Emitted one row per window (sourced from `round2_confirmed.json`)
   rather than two near-duplicate rows, with the 120/120-confirmed fact
   noted per-row and in `DATASET_CARD.md`.
4. **`verify6.json`'s missing label field.** This file has no
   interesting/negative/unsure verdict — it's a coordinate list for an
   out-of-band native-render QC pass, not a labeling-round output. Emitted
   with `label: "unsure"`, `split: "excluded"` rather than inferring
   `positive` from its `v3: 1.0` score field (that's a *model* score, not a
   human verdict, and conflating the two would misrepresent what this file
   actually is).
5. **S2/S3 `win`/`cls_px` per row.** These vary per segment (unlike S1's
   fixed 512). Computed directly from `tools/session-20260706/
  s2_small_ids.txt` (segment height/width/area) using the exact formula in
  `train_proxy_s2s3_v1.py`, and stored per-row as extra `win`/`cls_px`
  fields so `generate_crops.py` doesn't need to depend on that metadata
  file at all — it just reads the two numbers straight off each row.
6. **The 169-implicit-negative gap.** Did not attempt to approximate,
   re-derive, or silently drop this discrepancy — see the dedicated section
   above. This is the single largest gap between this release and full
   bit-for-bit training-data completeness, and is surfaced prominently in
   both this file and `DATASET_CARD.md` rather than buried.


## Pre-upload pass (2026-07-11)

Re-review of the whole release before its HuggingFace upload. Changes made
and re-verified:

1. **Crops are now written as lossless PNG** (were JPEG q95). The original
   training scripts cropped in memory and never wrote intermediate files;
   PNG preserves the preprocessing pipeline's exact uint8 output, so the
   "exact training crops" claim now holds bit-for-bit. `generate_crops.py`
   and `train.py` updated together.
2. **`--download` is now a verified code path** (was best-effort and
   unexercised, via a project-internal helper). It fetches the official ds8
   ink map from the public anonymous bucket using the key layout
   `PHercParis4/segments/<panel>/ink-detection/downsampled/<...>-ds8.jpg`,
   stdlib only. Live test 2026-07-11: the fetched file for panel
   `20260623141924-w010-027` is **SHA256-identical** to the local panel the
   index and training used (`eef262bc44bf2d78…`) — the reproduction inputs
   are provably the official product.
3. **`train.py` now also selects CUDA** when available (was MPS/CPU only).
4. Re-ran the functional smoke with the updated scripts: 140 PNG crops from
   two real panels (0 skipped), `--mode s1` 1 epoch, val AUROC 0.885 on the
   tiny slice (8-jul JPEG smoke: 0.873 — same ballpark, as expected), output
   state_dict 122/122 tensors with key set and shapes identical to the
   published `proxy_v4.pt`.
5. Coordinate audit: all 656 S1 label windows verified in-bounds against
   the real panel dimensions (0 out-of-range). `train_labels.jsonl`: 686
   rows, 686 unique ids, per-round counts re-verified identical to the
   tables above.
6. One internal absolute path in this file's own prose was generalized, and
   cross-references now point at the HF model/dataset repos instead of the
   project's internal folder layout.


## Independent-review fix (2026-07-29)

An independent reproduction audit found a real bug in the optional S1
`generate_crops.py --download` path. At audit time, 19 of the 28 canonical
wrap-series panels exposed both a 1.129 µm and a 2.4 µm `-ds8.jpg` under the
same S3 prefix. The old code selected the first lexicographic key, so it
could download the 1.129 µm render instead of the 2.4 µm render used by the
training run and full index.

This does **not** change the checkpoint, labels, local-map scoring run, or any
published index count. It affected regeneration from a missing local map via
`--download`.

The downloader now requires exactly one key containing
`2.4um-0.22m-78keV-volume-20260411134726`. It fails closed when that source
is missing or ambiguous. `test_package.py` covers dual-family ordering,
missing-source, and ambiguous-source cases.

`--download` only fetches missing files. Anyone who populated `--maps-dir`
with the older downloader should remove and re-fetch those panel JPGs; this
script cannot infer the source family from the generic local filename alone.


## Peer-review fixes (2026-07-11, second pass)

An external peer review of the whole package confirmed four blockers; all
verified against the real files and fixed:

1. **Dedup accounting**: the June `w104-106` segment has no published ink
   map; the July panel covering that range (2,648 windows, 131 gold) is
   unique territory. The
   classic bucket's 23 panels reduce to 18 physical surfaces by name
   lineage (an offset-0 re-render, a `_v14`, a `_copy`, a `_v2_flatboi`, a
   `_v8`). Canonical numbers now in the model card and the project's
   `CLASSIC_EQUIVALENCE.md`: wrap-series union 5,678 gold; classics 2,460
   (separate bucket); redundant 7,248; invariant total 15,386 unchanged.
2. **Reproducibility of the full proxy_v4 recipe**: the 50 GPU-verified
   fiber negatives now SHIP as `fiber_negatives_50.jsonl` (top-50 of the
   score-sorted ranking; verified sorted-descending; 12 of 50 in the
   validation panels). `train.py --extra-negatives` now routes rows through
   the same VAL_PANELS spatial split as every other row (previously it sent
   them all to train, which could not reproduce the original validation
   composition). Smoke re-run: fiber crops 50/50 generated, val grew by
   exactly the 12 val-panel fiber rows, 1-epoch AUROC 0.972 on the smoke
   slice.
3. Added `requirements.txt` (pinned versions used for this release) and
   `test_package.py` (self-checks on row counts, schema, id uniqueness and
   the 12-in-val invariant; passes).
4. Framing: PHerc 0139 human review is now stated as 54/63 clear (85.7%) +
   9 possible + 0 rejected, single reviewer — not "100% precision"; the
   reproducibility claim in the model card is scoped (architecture +
   behavior, not bit-for-bit; index generator out of scope).


## Third pass (2026-07-11, pre-upload blockers)

1. **The full index is now part of this release**: `full_index_complete.json`
   (125,298 rows, SHA256
   `4d393d70ce886ed62b7e73e365f1d01cbe7f6efa37168fb3f27ade2b89d6e7a8`),
   with `summarize_index.py` recomputing every published bucket under
   asserts (rows/panels/gold, union 5,678, beyond-w037 4,424 = 77.9%,
   classics 2,460, redundant 7,248 = 224+739+6,285). Wording fixed
   throughout: the June `w104-106` SEGMENT exists in the bucket but has no
   published ink map (it is not a missing wrap range).
2. **Frozen checkpoint evaluator**: `eval_checkpoint.py` deterministically
   rebuilds the held-out validation split (81 positives / 123 negatives:
   val-panel labeled rows + 12 val-panel fiber negatives + seed-0
   background negatives) and scores the published `proxy_v4.pt`:
   **AUROC 0.9853457794**, matching the independent reviewer's
   reconstruction to 10 decimal places. This also validates the shipped
   PNG-crop pipeline end to end against the published checkpoint.
3. `test_package.py` now runs from any CWD (paths relative to the file) and
   checks the index SHA256 when the file is present.
4. Claim language: the six-zone CT replication is now described as a
   reported internal check (scripts preserved, environment not packaged);
   the w037 frontier is a provisional proxy with an explicit physical
   sizing argument (83 cm of papyrus through w037 vs 65-90 cm for 11
   columns) and is no longer called conservative; "100% precision" phrasing
   removed project-wide in favor of "120/120 reviewer-confirmed on
   model-selected candidates".
