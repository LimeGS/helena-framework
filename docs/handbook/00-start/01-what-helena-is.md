---
title: What Helena is
summary: A pipeline that turns a CT scan into evidence about ink, and keeps the receipt for every step.
---

Helena takes a CT scan of a carbonised scroll and produces, at the far end, a
picture of a papyrus surface with a claim about where the ink is. Between those
two things are ten phases, and the reason there are ten rather than one is that
each of them can fail in a way that still looks like it worked.

The platform exists to make that impossible to miss. Every phase records what it
consumed, what it produced, and the digest of both. A result that cannot say
what it came from is refused rather than filed.

Helena does not reimplement the algorithms it runs. VC3D, m7, ScrollFiesta, the
Volume Cartographer flatteners and the ink models stay exactly what they are; a
phase calls them through a stable adapter and records what each was given and
what it returned. A phase implemented as a separate component rather than
inline in `framework/stages/` is registered in `framework/registries/` and
vendored under `framework/vendored/`, each component with a `VENDOR.json`
recording the source commit and a hash per file. Only the technique is
imported; findings stay at the source.

## The shape of a run

Work enters as a **job**, queued against a **phase**, inside a **mission**.

- A **phase** is a step: recover a surface, flatten it, render it, detect ink.
  Phases are numbered P0 to P9 and each has its own page in this handbook.
- A **lane** is one way of doing a phase. P4 has two, P8 has three, P5 has five
  detectors. Choosing a lane is choosing a method, and the receipt records
  which one ran.
- A **mission** is the project the work belongs to. Nothing exists outside one:
  coverage, backlogs and job counts are all computed against a mission's
  scrolls, so a job queued outside one is work nobody can find afterwards.

A worker claims the job, runs it, and publishes what it made to the artifact
store. Which worker may claim it depends on the capability the job declares —
most phases run on a CPU-only fleet, and only ink detection and surface QC need
a GPU. The panel never runs the work itself — it writes a row and waits, which
is why a panel outage does not stop a render that is already going.

## Two kinds of run, and the difference matters

This is the distinction that shapes everything else, and it is a property of the
**mission**, not of the job.

An **exploration run** is a mission with no `campaign_kind`. Lineage is recorded
but not enforced: you can point a phase at a directory you made by hand, run a
detector on it, and look at what comes out. Nothing about it is second-class —
most real work starts here — but its outputs are not admissible as a finding.

A **certification run** is a mission bound to a campaign. You declare one thing
— `campaign_kind`, whose only value is `FIRST_LETTERS_DISCOVERY` — and the panel
pins the rest: the policy id, the policy digest, the deployed revision and your
name, with a binding digest stamped over the set so changing any of them by hand
makes the mission unreadable.

> **Trap** Declaring it is not the whole story. The ink queue asks whether a
> **campaign budget admission** has been registered for this mission, and the
> surface boundaries ask whether the surface row itself is marked. A mission
> with `campaign_kind` and no registered admission is gated as generic.

From that moment every phase boundary asks a harder question:

- the resolved lineage must be **complete**: right schema, canonical, a
  recognised surface state, no ambiguity or hash conflict, and an external
  surface needs a recorded admission digest;
- `allow_unvalidated` must be literally `False`. Anything else — including a
  null, including absent — is refused, so a P4, P5, P7 or P8 job in a controlled
  mission that names no surface is refused outright;
- the source snapshot, the policy digest and the deployed revision are pinned
  into every receipt, so a result produced in six months can still say what
  produced it.

What is **not** on that list is most of what this platform refuses, and it is a
long list. In **both** modes: a surface may never state its own lineage; a
discovery-namespace lineage is refused at every boundary; a job needs a mission;
P3 needs a surface that is both `GEOMETRY_CERTIFIED` and CT-supported; P4 checks
the stack it wrote; a P5 job whose receipt carries no liveness verdict is failed.

Those are properties of the phases and an exploration run gets all of them.
Certification adds provenance you cannot assert for yourself; it does not switch
on the engineering.

> **Certification** A certification run is slower and refuses more. That is the
> whole point: it is the mode in which a result is allowed to be a claim about a
> scroll rather than a picture you generated.

You choose between them once, when you create the mission. You cannot promote an
exploration run into a certified one afterwards — the evidence it would need was
never collected, and manufacturing it later is precisely what the mode exists to
prevent.

## What a phase actually guarantees

Each phase publishes a receipt with the digests of its inputs and outputs, the
lane that ran, and the parameters it was given. Two properties follow, and both
are worth knowing before you trust a green result:

- **A receipt proves what ran, not that the result is good.** P4 can render 33
  slices of nothing at all. The phase pages say, for each one, how to tell a
  usable result from an expensive one.
- **An exit code proves even less.** Several upstream tools report success after
  failing to open their input. Where Helena knows about it, the phase verifies
  its own output before calling the job a success — P4 counts the slices it
  wrote and checks the middle one is not a constant, and refuses if it is.

A third discipline sits beside those two: geometry certification, CT support,
the detector's own response and human review are four separate judgements, and
none of them stands in for another. A surface can be CT-supported and
geometrically rejected at once — they answer different questions, and a phase
page shows each as its own field rather than folding them into one verdict.

## Where to go next

- [The pipeline, end to end](#/docs/start/pipeline) — the ten phases and what
  passes between them.
- [Tutorial](#/docs/tutorial/overview) — a scan to a picture of letters,
  following the route the Scroll Prize walkthrough takes.
- [The phases](#/docs/phases/p0) — one page each, exhaustive, with the traps.
