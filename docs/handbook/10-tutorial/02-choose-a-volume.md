---
title: Step 1 — Choose a volume
summary: Pin the scan and the number that makes every later micron comparable.
---

The walkthrough opens a scroll in VC3D and streams it from the open-data
collection. In Helena the equivalent is P0: choose the scan, and freeze the
scale.

## Do this

1. Open **Scrolls**. The catalogue is the frozen list of what this deployment
   can reach; a scroll not in it is not a scroll you can start from.
2. Find your scroll and read the declared **µm per voxel**. Check it against the
   scan you think you chose — a scroll often has several, at different energies
   and resolutions, and they are not interchangeable.
3. On **Mission**, add the scroll to your mission.

That is the whole of P0. Nothing computes; it records a choice.

## Why the scale matters more than it looks

Every later phase resamples in microns. A detector trained at 7.91 µm handed a
stack rendered at 9.362 µm sees the wrong physical thickness — and nothing
downstream notices. There is no error. It simply produces a worse map, and you
spend a day wondering about the model.

> **Trap** The single most expensive mistake in this pipeline is a scale that is
> wrong by 8%. It never announces itself. It moves the recovered depth by tens
> of microns and degrades every map made from it.

## What you should see

The scroll appears on your mission with its scale and a source snapshot. From
here every artifact binds to that snapshot, which is what lets a result made in
six months still say what it came from.

Next: [recover a surface](#/docs/tutorial/recover-a-surface).
