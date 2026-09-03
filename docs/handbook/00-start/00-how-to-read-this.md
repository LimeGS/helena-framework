---
title: How to read this handbook
summary: Where things are, and the four kinds of aside that mean something specific.
---

Documentation has three tabs. This is the Handbook: the walkthrough, the phases,
and every panel page with what its controls do, written by hand. Developer
reference and API reference sit beside it, generated from the code they
describe — contracts, profiles and how to put your own tool into a phase on
one, the HTTP routes themselves on the other. Reach for those when you need the
exact shape of something rather than an explanation of it.

Sections down the left, one page at a time, and the current page's own headings
under it. The filter searches the full text of every page, not just the titles —
if you half-remember a phrase, type the phrase.

![The handbook's navigation](handbook-navigation.png "Sections, pages, and the current page's headings in place. Every heading is linkable: hover it for the anchor.")

Every heading has an anchor, and the address bar carries the page you are on, so
a link to a paragraph is a link somebody else can follow.

## The four asides

They are not decoration and the vocabulary is closed — the build refuses one it
does not recognise.

> **Note** Something worth knowing that is not a hazard. Usually a distinction
> the interface does not make for you.

> **Trap** A way to get a wrong answer while everything reports success. These
> are the ones to read even when you are skimming; most of them are failures
> this platform has actually had.

> **Cost** Something expensive in GPU time or storage, where the cheap version
> of the same experiment exists and is worth doing first.

> **Certification** A rule that applies only inside a
> [certification run](#/docs/start/what-helena-is), where the platform refuses
> what an exploration run allows.

## Where to start

- Never run this before: [What Helena is](#/docs/start/what-helena-is), then the
  [tutorial](#/docs/tutorial/overview).
- Running it already and need a specific control: go to
  [the phase](#/docs/phases/p0), or filter.
- Putting your own tool into it:
  [adding a lane](#/docs/reference/extending).
- Something is broken: [when something is wrong](#/docs/operations/troubleshooting).
