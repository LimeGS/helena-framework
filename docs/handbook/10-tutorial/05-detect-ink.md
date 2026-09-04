---
title: Step 4 — Detect ink
summary: Run a pretrained detector over the stack, and get a probability map.
---

The walkthrough uses the pretrained cross-scroll 9 µm models. Helena has five
detector adapters; the 9 µm one is the closest equivalent and the one to start
with.

## Choose a lane

| Profile | Adapter | Reads | Trained at |
| --- | --- | --- | --- |
| `ink-9um-hybrid-3d2d-screening@1.0.0` | `run_ink_9um.py` | a surface volume | 9.6 µm |
| `ink-canonical-2um-screening@1.1.0` | `run_ink_canonical2um.py` | a layer stack | 2.399 µm |
| `timesformer-gp-scroll1-screening@1.1.0` | `run_ink_timesformer.py` | a layer stack | 7.91 µm, resampled |
| `resnet50-7.9um-scroll1-frags-screening@1.0.0` | `run_ink.py` | a layer stack | 7.9 µm, resampled |

> **Trap** Write the `@version`. A profile id is matched exactly, so
> `ink-canonical-2um-screening` without one is refused as unknown. And the
> version is not cosmetic here: `@1.0.0` routes to the generic runner, and the
> record left by the profile that supersedes it says that runner's map
> correlated r = 0.079 with the community map for that recipe, on ground where
> `@1.1.0`'s own render matched at r = 0.98. They are not comparable.

A fifth adapter, `run_ink_3d_dino.py`, is declared `unroutable` — it takes a
patch manifest rather than a layer stack — and the queue refuses it by name.

The lane is chosen by naming a **profile**, not by picking a script. The profile
pins the checkpoint and its digest, the training scale, and the model family, so
two runs of the same profile are comparable.

> **Trap** A lane is scale-restricted. The 2 µm canonical lane must not be
> pointed at a 9 µm cohort without its own control at that scale; the profile
> says so in its own `comparability` note, and the number that matters is
> `source_pixel_um` — stated, never defaulted. An 8.4% error moves the recovered
> peak by tens of microns.

## Queue it

Open the P5 launcher, and name **exactly one** input — `layer_stack` (a P4 job
id), `tiff_dir` (a directory this worker can see), or `surface_volume` (an
OME-Zarr already at the model's scale). Naming two is refused.

- the **layer stack** — the P4 job, by id, not a directory;
- the **checkpoint**, whose digest is verified before inference;
- `source_pixel_um` — the scale the stack was rendered at. Required for
  `tiff_dir` and `layer_stack`; a `surface_volume` is read as it is and does
  not take it;
- `depth_center`, `stride`, `batch_size`, `tile_size` if you are tuning.

The 9 µm lane pools the stack into the volume its model wants, in the same
process, which is why it is one job and not two. It needs `source_pixel_um` for
exactly that reason: pooling has to know what it is pooling from, and assuming
2.4 would turn a native 9 µm render into a 38 µm one.

## What comes out

A probability map: one float per pixel, brighter meaning more likely ink. A
direction-both run publishes `probability.npy` and `probability_reverse.npy` —
a map and its reverse. Both, with the run's `p50`, `p99` and the spread between
them, show up on P5's Maps tab once the job finishes.

It is a **screening output, not a reading**. The map says where a model responded,
which is a different claim from where the ink is.

Next: [read the result](#/docs/tutorial/read-the-result).
