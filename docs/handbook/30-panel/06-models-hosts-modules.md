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
install_ink_weights.py --models-root /path/to/models
install_ink_weights.py --models-root ... --only ink_9um   # one repository
install_ink_weights.py --models-root ... --verify-only    # audit, download nothing
```

That writes into the models volume directly, from whatever machine runs it.
`scripts/harness/install_declared_weights.py` does the equivalent job through
the panel instead: it reads `GET /api/models?resolve=1` for what a frozen
profile still needs, then calls `POST /api/models/download` once per row
against that profile's own digest — the same request the page's Download
button makes, including for a `pickle_only` checkpoint, since the request
carries the hash. It takes a panel URL and a user's credentials, `--only
<substring>` to filter by upstream repository or destination path, and
`--dry-run` to plan without fetching anything.

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
| `gated` | it needs an acceptance you have not given, or the repository does not exist — the hub answers 401 for both, so as not to leak which |
| `not_published` | the profile names a model nobody published |
| `unreachable` | the hub did not answer |
| `no_family` | the profile names no model family to look up |

> **Trap** **Safetensors freely; a pickle only against a hash.** A `.bin`,
> `.pt` or `.pth` checkpoint runs code when it is loaded, on a GPU worker, so
> `POST /api/models/download` fetches one only when the request states the
> `expect_sha256` it must have, and deletes rather than installs a file whose
> bytes disagree. The page's own Download button sends the profile's digest;
> the free-form field does not, so a `pickle_only` model is one you fetch by
> request with the hash, not by clicking.

## Modules

What every phase can be done with, and what is switched on.

A module has one of five kinds, and each is a different contract rather than a
synonym for the others:

| Kind | Is |
| --- | --- |
| `lane` | a program the queue starts |
| `profile` | a model with its weights and physical scale, routed to an adapter |
| `backend` | what grows a surface, for segmentation |
| `seeder` | what chooses the point to grow from |
| `source` | where the frozen scroll catalog is read from |

This page does not add a sixth. What is registered is read from the same
declarations the queue routes with, so a lane that appears here exists and one
that does not cannot be queued. P0 reports only `source`, naming the
`CX_SCROLL_SOURCE` in effect; P1 reports backends and seeders instead of
lanes; P5 reports profiles instead of lanes; every other phase reports lanes.

> **Note** It is not read-only. Modules can be enabled and disabled from here,
> and a new P5 profile can be **registered** from a Hugging Face repo id, an
> adapter, training scale, frame count, a revision (`main` if left blank) and
> the checkpoint file (`model.safetensors` unless changed). Left with nothing
> said about what it does not claim, the profile says so itself: nothing has
> been validated against a known positive on this deployment, which is true
> and is what a reader needs. That is a write.

## Hosts

The machines, their roles, and what they reported about themselves.

The assignable roles are `segment`, `render`, `ink`, `mesh` and `build`, and the
server refuses anything else:

| Role | Means |
| --- | --- |
| `segment` | grows surfaces with VC3D — CPU only, so any host can take it |
| `render` | turns a surface into a layer stack — no GPU |
| `ink` | the only stage that needs a GPU worth having |
| `mesh` | comparative backend, research only — its surfaces are not catalogued |
| `build` | compiles the images, which is why they are built where they run |

A host can carry a role this page does not offer, such as `postgres` on the one
running the control-plane database. That is shown, greyed, rather than dropped:
a role describing where infrastructure lives is a fact about the deployment
rather than something to be inferred, and a request from this page that cannot
express it must not be able to remove it by saving an unrelated change.

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

Registering with `provision` set — the default — also runs
`containers/provision-host.sh` against that target in the background: Docker
if it is missing, the worker and tunnel images streamed over SSH, the compose
files and an env file under `/etc/helena` on the target, then the tunnel and
the worker as compose projects. `GET /api/hosts/{id}/provision` reads the log
back. The panel image does not carry `containers/`, so a deployment running
only the image answers **503**: the host is registered, and provisioning is
the part that did not happen — bring it up with the compose files and it
reports itself.

That last sentence is literal, not a figure of speech: a host that nobody
registered still registers itself, the first time its worker or its
`host_report.py` reports in. That first report inserts the row with no `ssh`
target and a note pointing back here — "registered by its own report; add an
ssh target under Configuration -> Hosts to provision it from the panel" — so
the fleet counts its hardware from the start instead of the worker running
every job on a machine this page had never heard of. Add the `ssh` target by
hand afterwards to make it provisionable; nothing else about the row changes.

Host state is measured, not assumed — by the **worker's** probe, not by this
page. The claim filters on the `has_gpu` and VRAM the worker reported, probed
before its first claim, because a host with no card must not take a job that
needs one: it would fail it, burn an attempt, and leave the queue looking broken
rather than misrouted.

The table itself reports what was measured: GPUs by name and utilisation, one
line each; cores, with the host's own total alongside when this worker is
confined to fewer; RAM free — `MemAvailable`, what a new process could
actually get, reclaimable cache included — over the total; and disk free on
the volume runs land on, not on `/`. "Last seen" reads **live** for two
different reasons that share one word: the host the panel itself runs on,
which it can simply measure rather than wait for a report, and any host whose
only report is a segmentation worker's heartbeat, which carries admission
capabilities — a GPU, if there is one — and nothing else, so cores, RAM and
disk stay dashes there even while it says live. Every other host shows the
timestamp of its last report, or **never**.

`GET /api/hosts/{id}/images` is a second check, reachable but not on this
page yet: it asks the host over SSH which `helena-*` image digests it is
actually running, compares them against what its roles require, and reports
the drift — a tag identifies nothing, which is how two hosts came to run
different bytes under the same name. It answers "not reachable" rather than
guessing for a host with no `ssh` target or one that does not answer.

> **Note** GPUs are merged **by uuid and kept for an hour**, because a worker
> that sees no card would otherwise write an empty list over a real one and
> leave this page reporting no hardware on a machine that has just finished a
> render. The cost is that a genuinely removed card lingers here for an hour.

> **Trap** A host reporting healthily is not the same as its workers claiming.
> The host report is written on its own timer, in a separate branch of the same
> loop, so it keeps reporting while a claim beside it is blocked. For that
> question use [the queue page](#/docs/panel/fleet).

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
