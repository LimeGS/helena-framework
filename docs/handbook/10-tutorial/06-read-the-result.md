---
title: Step 5 — Read the result, and iterate
summary: Whether the map decided anything, what a vetting card is for, and where the research loop starts.
---

The walkthrough's last step is a loop: run inference, label the clear strokes,
expand the mask, fine-tune, test on held-out regions. Helena does not run that
loop. What it does is make each pass through it comparable to the last, and
refuse the two ways a map lies.

## First: did the map decide anything?

A detector handed an unusable input still produces a map. A flat field at 0.51
is a perfectly valid array of floats.

P6 is the liveness check, and a P5 job with no liveness verdict is **failed**,
not passed. Storing null read exactly like a lane that had been checked and
found alive, which is how three adapters went a long time without the gate and
nobody saw it.

> **Trap** `DEGENERATE` is a real outcome, not a crash. The lane produced a map
> that carries no decision. That is information about the input, and it is
> recorded as a refusal rather than a failure.

## Then: screening and the vetting card

Queue **P7** with a bounding box and `px_um`, and name the map exactly one of
two ways — the P5 job id plus both of its digests, or a path this worker can
see. It produces `verdict.json` and `VETTING_CARD.md`: the card is a Markdown
document, not an image, written to be read in a review and quoted in one.

The walkthrough's own criterion is a good one and Helena does not replace it:
coherent rows aligned with the papyrus fibres, and at least ten visible legible
letters in about 4 cm². What the platform adds is that the card records which
map, from which render, from which surface, from which scan — so a claim can be
retraced rather than re-argued.

## Reading it honestly

Three cautions the walkthrough also gives, and this platform is built around:

- **Do not train on discovery regions.** A model fine-tuned on the area you
  intend to claim will find letters there. Helena marks discovery-namespace
  surfaces as non-canonical for this reason.
- **Keep the receptive field small.** A large one hallucinates plausible
  letterforms out of texture.
- **Verify against false positives.** Held-out regions, and the reverse map.

> **Note** The platform will never tell you a map contains text. It will tell
> you a map is live, what it came from, and whether two runs of it agree. The
> reading is yours.

## Where to go from here

- [The phases](#/docs/phases/p0) — the same steps, exhaustively, with every
  field and every trap.
- [Certification and exploration](#/docs/start/what-helena-is) — when you want
  this route to produce a finding rather than a picture.
- [Seed agreement](#/docs/phases/p2) — running a thing twice and asking whether
  it agreed with itself.
