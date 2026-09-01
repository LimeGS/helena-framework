---
title: Models, Modules, Hosts and Users
summary: The weights, the extension points, the machines and the accounts.
---

## Models

The weights the platform needs, and getting them onto the machine.

A **profile** is authoritative about a checkpoint's identity: it names a SHA-256
and treats the path as runtime input. So this page is not a catalogue of models
somebody thinks are good — it is the list of digests this checkout declares, and
whether each one is present here.

Two documents declare, and the difference between them matters:

- a **profile** names the one checkpoint a lane froze;
- the **weight manifest**, `framework/registries/ink-weights-0.1.0.json`, names
  what upstream published — every file, with the digest upstream published for
  it.

They are not the same list, and deriving the page from profiles alone hid the
gap. `scrollprize/ink_9um` publishes a training run: two seeds by seven steps,
fourteen checkpoints. A profile names exactly one, which is correct for a
profile and wrong for the page — the other thirteen were not missing and not
installed, they were absent from the question. All fourteen are listed now, and
each has its own profile so a job can name one.

> **Note** Every digest in the manifest was read from upstream's own LFS
> metadata, not computed from a copy we downloaded. That is the difference
> between proving the bytes are upstream's and proving that two of our own
> copies agree with each other.

> **Trap** A profile that pins **no** digest used to run with no verification at
> all, in silence, and still wrote a receipt carrying the checkpoint's hash —
> indistinguishable from a verified one. It is now refused: weights silently
> different from what the receipt names are an irreproducible result that reads
> as reproducible.

### Installing

`scripts/models/install_ink_weights.py` fetches what the manifest names,
verifies each file against it, and only moves a download into place once it
verifies. It is safe to interrupt and safe to re-run — a file already installed
with the right digest is left alone.

```
install_ink_weights.py --models-root /mnt/bulk/helena/models
install_ink_weights.py --models-root ... --only ink_9um   # one repository
install_ink_weights.py --models-root ... --verify-only    # audit, download nothing
```

> **Cost** The full set is 17.4 GB across 32 files. Put the models root on a
> volume with room: the `ink_3d_dino_guided` checkpoints are 1.7 GB each.

> **Trap** A job that names weights nobody installed is refused when it is
> queued, not an hour later in a worker's log — but only when this panel can see
> the models root. On a split deployment where panel and workers do not share
> one, absence here proves nothing about the worker, so the check declines to
> guess rather than becoming something an operator has to switch off.

The page's real content is the resolution state per model, and there are eight:

| State | Means |
| --- | --- |
| `exact` | published, and the bytes match the digest |
| `mismatch` | published, and they do not |
| `pickle_only` | published, and you cannot install it — see below |
| `no_safetensors` | the repo has none |
| `gated` | it needs an acceptance you have not given |
| `not_published` | the profile names a model nobody published |
| `unreachable` | the hub did not answer |
| `no_family` | the profile names no model family to look up |

> **Trap** **Safetensors only.** A `pickle_only` model is a published model you
> cannot install here, and that is deliberate: safetensors cannot carry code.

## Modules

What every phase can be done with, and what is switched on.

The platform has three extension mechanisms and each is right for what it does:

- a **lane** is a program,
- a **profile** is a model with its scale,
- a **seeder** chooses a point.

This page does not add a fourth. What is registered is read from the same
declarations the queue routes with, so a lane that appears here exists and one
that does not cannot be queued.

> **Note** It is not read-only. Modules can be enabled and disabled from here,
> and a new P5 profile can be **registered** from a Hugging Face repo id,
> adapter, training scale and frame count. That is a write.

## Hosts

The machines, their roles, and what they reported about themselves.

The assignable roles are `segment`, `render`, `ink`, `mesh` and `build`, and the
server refuses anything else. A role describing where infrastructure lives is a
fact about the deployment rather than something to be inferred.

> **Trap** **Roles do not route work.** They decide which images a host is
> expected to hold and nothing else — no claim path filters on them. A host with
> no `ink` role still claims ink work if a worker is running on it. The control
> looks like it routes, and does not; the API says so in its own response.

> **Trap** Neither does enabling or disabling a host. That flag greys the host
> in the launcher's picker and drops it from the hardware tile, and nothing in
> the claim reads it. A "disabled" host keeps taking work.

Registering one takes an `ssh` target, and that field is validated hard: the
format is `[user@]host` and nothing else. A leading dash once made
`-oProxyCommand=` code execution in the panel container.

Host state is measured, not assumed — by the **worker's** probe, not by this
page. The claim filters on the `has_gpu` and VRAM the worker reported, probed
before its first claim, because a host with no card must not take a job that
needs one: it would fail it, burn an attempt, and leave the queue looking broken
rather than misrouted.

> **Note** GPUs are merged **by uuid and kept for an hour**, because a worker
> that sees no card would otherwise write an empty list over a real one and
> leave this page reporting no hardware on a machine that has just finished a
> render. The cost is that a genuinely removed card lingers here for an hour.

> **Trap** A host reporting healthily is not the same as its workers claiming.
> The host report is written on its own timer, in a separate branch of the same
> loop, so it keeps reporting while a claim beside it is blocked. For that
> question use [the Fleet page](#/docs/panel/fleet).

## Users

Accounts. **No roles**: everyone who can sign in can do everything, including
adding and removing accounts.

That is stated on the page rather than left to be discovered, because a
permission model people assume exists is worse than one they know is absent.

Three carve-outs:

- the **last** account cannot be deleted;
- the **first** can only be created from loopback;
- deleting an account kills its live sessions immediately, while changing a
  password does not — those sessions stay open.

**Machine tokens are administered here too.** Minted and revoked from this page,
shown once and never again, and refused if the name collides with a person's.
They are not "do everything": they reach the artifact endpoints and nothing
else, so a leaked one cannot queue GPU work or read somebody's missions.
`last_used_utc` is written once a day, so a token nobody uses is visible as one
to revoke.
