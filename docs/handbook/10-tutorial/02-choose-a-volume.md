---
title: Step 1 — Choose a volume
summary: Pin the scan and the number that makes every later micron comparable.
---

The walkthrough opens a scroll in VC3D and streams it from the open-data
collection. In Helena the equivalent is P0: choose the scan, and freeze the
scale.

## Do this

1. Open **P0** under Mission; its view is called **Scrolls**. The table lists
   every scroll the open-data bucket exposes. A frozen catalogue — seeded from a
   committed file and refreshed from the bucket on startup and daily — supplies
   the declared scale for the scrolls it describes, and only a name it (or
   another source this deployment has registered) can resolve to a real volume
   can actually be added to a mission.
2. Find your scroll and read the declared **µm per voxel**. Check it against the
   scan you think you chose — a scroll often has several, at different energies
   and resolutions, and they are not interchangeable. A blank cell means the
   catalogue does not describe this scan; the scale is unpinned, not merely
   unknown.
3. Tick the scroll and click **Apply** — say why only if the mission has already
   produced work, since a first selection applies straight away. Applying both
   adds the scroll to the mission and freezes what P0 decided for it in the same
   step. A name with no volume this deployment can read is refused here rather
   than at P1. If a selection was made before that automatic freeze existed and
   nothing shows under "What P0 produced", a separate **Record what P0 decided**
   button there catches it up.

That is the whole of P0. Nothing computes; it records a choice.

## Why the scale matters more than it looks

Every later phase resamples in microns. A detector trained at 7.91 µm handed a
stack rendered at 9.362 µm sees the wrong physical thickness — and nothing
downstream notices. There is no error. It simply produces a worse map, and you
spend a day wondering about the model.

> **Trap** The single most expensive mistake in this pipeline is a scale that is
> wrong by 8.4%. It never announces itself. It moves the recovered depth by
> tens of microns and degrades every map made from it.

## What you should see

The scroll appears on your mission with its declared scale, frozen from the
catalogue. The content-locked source snapshot — a URI, a digest and a scale
bound together — is not part of this; it is registered once source intake runs
at P1. From then on every artifact binds to that snapshot, which is what lets a
result made in six months still say what it came from.

Next: [recover a surface](#/docs/tutorial/recover-a-surface).
