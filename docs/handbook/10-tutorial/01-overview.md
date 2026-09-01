---
title: From a CT scan to letters
summary: The Scroll Prize walkthrough, done in Helena — what you will run, in what order, and what each step costs.
---

The Scroll Prize published a walkthrough that goes from a CT scan to visible
letters: open the volume, recover a surface, flatten and render it, run ink
inference, look at the result and iterate. This tutorial does the same route
through Helena, and says where Helena's version differs and why.

The difference in one sentence: the walkthrough runs tools on a workstation, and
Helena runs the same tools as queued jobs that record what they consumed. The
commands are largely the same underneath, and so is what they will accept — an
exploration run has the same reach as running the scripts yourself, including
the escape hatches. What you gain is the receipt.

Certification is the mode that costs something. It removes the escape hatches:
`allow_unvalidated` stops being honoured, and an input has to resolve to a
lineage this control plane recorded rather than one the request asserted. That
is a deliberate trade you make per mission, not a tax on using the platform.

## What you will do

| Step | Blog | Helena | Roughly |
| --- | --- | --- | --- |
| 1 | Open the scroll in VC3D | [Choose a volume](#/docs/tutorial/choose-a-volume) — P0 | minutes |
| 2 | Recover the surface (spiral fitter) | [Recover a surface](#/docs/tutorial/recover-a-surface) — P1, then P2 | hours on a GPU |
| 3 | Flatten and render | [Flatten and render](#/docs/tutorial/flatten-and-render) — P3, P4 | tens of minutes |
| 4 | Ink inference | [Detect ink](#/docs/tutorial/detect-ink) — P5 | minutes |
| 5 | Inspect, label, iterate | [Read the result](#/docs/tutorial/read-the-result) — P6, P7, P9 | as long as you like |

## Before you start

- **A mission.** Create one on the Mission page and add the scroll to it. For a
  first run, make it an exploration mission — no `campaign_kind`. You will
  spend the first attempt learning what the inputs look like, and a
  certification run refuses the shortcuts that make learning fast.
- **A worker with a GPU**, visible on the Fleet page and polling. The spiral fit
  and the detectors need one; nothing in this tutorial runs on the panel host.
- **The inputs the spiral fitter reads**: an umbilicus, tracks, and the lasagna
  normal volumes. These are not all downloadable — see
  [Recover a surface](#/docs/tutorial/recover-a-surface), which is the step
  where people actually get stuck.

> **Cost** A whole-scroll spiral fit is hours of GPU time. Fit a z-range first —
> `z_begin` 10000 to `z_end` 11000 is about a thousand slices and is enough to
> tell whether the inputs are right. Discovering that the umbilicus was wrong
> after six hours is the expensive way to learn it.

## What this tutorial does not do

It does not train a detector. Step 5 of the walkthrough — label strokes, expand
the mask, fine-tune, test on held-out regions — is a research loop, and Helena's
role in it is to keep the runs comparable rather than to run the training. The
[screening](#/docs/phases/p7) page covers what the platform will and will not
say about a map.

It also does not cover [certification](#/docs/start/what-helena-is). Once the
route works, running it inside a campaign mission is a matter of creating the
mission differently; the phases are the same.
