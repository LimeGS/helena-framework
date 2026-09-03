---
title: Adding a lane or a phase
summary: What a new way of doing a step has to declare, and what it gets for free.
---

Adding a module that does a phase differently is one entry in a table plus its
parameters. Nothing in the worker, the panel or the command builder needs to
know it exists: the form, the guide and the routing all read the same
declarations.

## What a lane declares

| Key | What it is |
| --- | --- |
| `runner` | the program, relative to the repository root |
| `required` | parameters without which the job is refused |
| `profiles` | an **allowlist of versioned profile ids**; anything else, or none, is refused |
| `sample_ids` | pins the lane to named scrolls and refuses every other sample |
| `flags` | parameter → command-line flag |
| `fixed` | leading arguments, e.g. a subcommand |
| `output_flag` | a flag that gets the job's own output directory |
| `json_flags` | values serialised as one compact JSON token |
| `defaults` | a callable of `output_dir`, for values derived from the run |
| `image` | the runtime image this lane needs |
| `gpu_required` | admission. **Defaults to `True`** — a CPU lane must say `False` or only GPU workers claim it |
| `build` | `"legacy"` opts **out** of the declarative builder |

> **Trap** `build` reads backwards. Declaring nothing gets you the declarative
> builder; `build: "legacy"` is the escape hatch that routes to the phase's
> hand-written branch. Nine of the eleven registered lanes are legacy — the
> declarative path is the minority, not the majority.

> **Note** `minimum_vram_gb` is read on the claim and **no lane declares one**,
> so it is always zero. It is real on the segmentation task contract, and as a
> floor a worker raises for itself after a recorded GPU OOM.

A lane with its own `image` makes its worker a **specialist**: it was built to
carry that lane's frozen environment and nothing else. A worker running
something else refuses the job by name rather than taking it and failing at the
first import.

## What a P5 adapter declares

Detectors are a table of their own, because the shape differs:

| Key | What it is |
| --- | --- |
| `receipt` | the file the runner writes about itself |
| `profile_flag`, `profile_as` | how the profile reaches it — by id, or as a path |
| `needs` | parameters without which it cannot run |
| `flags` | the flat mapping |
| `upstream` | that it imports a vendored architecture, and how it reaches it |
| `unroutable` | that it cannot be queued, and why |
| `image` | the runtime image. **For P5 this — not the lane — is what makes a worker a specialist** |

> **Note** An adapter the queue has not been taught is **refused by name**.
> Falling back to the default runner is how a TimeSformer lane once came to be
> handed `--upstream-dir`, which it has no argument for.

## Rules a new lane inherits

You do not have to implement these, and you cannot opt out of them:

- **Parameters are a contract.** A field not in the phase's table is refused,
  not passed through. Numbers get a floor, and a ceiling **where one is
  declared** — several have none. Add one for any size a worker would otherwise
  have to lease before discovering it.
- **Paths are checked for shape, not for place.** Validation requires absolute
  and free of `..`; the run directory is where the builder *defaults* them, not
  a boundary the validator enforces. The containment check is on the worker.
- **Server-owned fields stay server-owned.** Where a job publishes and its
  control binding cannot arrive in a request.
- **The command is built one way.** `command_for` is the only path from a queued
  row to a process, so widening what a caller can run is an edit there rather
  than a consequence of a request.
- **Inputs are checked on the worker.** It says which path and whether the
  problem is the path or the permission — after the claim, so a bad one costs an
  attempt. Only the runtime image filters before a claim.
- **A new phase needs two table entries for its lineage boundary.** The boundary
  name must be in the enum and have a matching incomplete-reason, or the gate
  raises "unknown canonical-lineage boundary".

## What you should add yourself

- **A profile**, frozen, with the digest of anything it pins.
- **An output check.** The phase should look at what it produced before calling
  the job a success. P3 parses its own TIFXYZ; P4 counts slices and checks the
  middle one is not constant; every P5 adapter calls `assess_liveness` from
  `framework/contracts/lane_liveness.py` on its own map and gets back `ALIVE`,
  `DEGENERATE` or `EMPTY` — a checkpoint can load cleanly, hashes and all, and
  still have an untrained decoder that answers every input with the same
  narrow band of numbers, and only the shape of the output distribution
  catches that. `refuse_if_not_alive` turns anything but `ALIVE` into a
  `LANE_NOT_USABLE` marker and a non-zero exit, unless the job set
  `on_degenerate` to `warn`; a new ink adapter that skips the call fails the
  suite, which checks every `run_ink*.py` script on disk by name. The general
  rule: an exit code is not evidence of work done.
- **A page here.** A lane nobody documented is a lane somebody works out by
  clicking.
