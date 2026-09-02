# Reproducing the public ink control

Everything below runs from inputs anybody can download, on a machine that starts
with nothing. No account of ours, no credential, no access to our deployment.

It drives the ink chain **through Helena's API** -- the job is queued, a worker
claims it, and the control fetches back what that worker published. Nothing
reads a path on the worker's disk. An earlier version of this page described a
local subprocess instead, which proves the tooling works and says nothing about
the queue, the worker, the routing or the publication; that is the half a
reviewer is being asked to trust, so it is the half this now exercises.

Written from a run on a rented RTX 5090 that had nothing installed on it. The
seven steps are the seven things that run needed, in order.

## What it proves, and what it does not

It proves this platform can drive the recommended ink-detection tooling end to
end on public data, through its own queue, and that each boundary either passed
or said why not.

It is not a reading, not an ink claim, and not the nine-boundary First Letters
campaign control -- that is a different receipt with a different schema, and the
evaluator refuses the substitution. The receipt carries these as `non_claims`
rather than leaving them to a reader.

## Inputs, and where they come from

| Input | Source | Credential |
|---|---|---|
| Surface volume, 9.362 um isotropic | `https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/segments/20260112000000-w043_2026011217/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr` | none |
| Checkpoint `hybrid_3d2d-seed42/step-075000.pth` | `https://huggingface.co/scrollprize/ink_9um` | none, not gated |

The https form of the volume, not the `s3://` one: the control reads the
volume's own `.zattrs` over plain HTTP, deliberately anonymous, because a
control that proves "public" while holding credentials has proved nothing about
what a stranger can obtain. The lane reads the same form. Same object, one
address, both halves agree.

The checkpoint's SHA-256 is
`e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab` and its size
is 138360039 bytes. The control verifies both before running and records the
digest it saw.

---

## 1. Two prerequisites that are not Helena's

**The NVIDIA driver has to match the card.** Blackwell -- a 5090, for instance --
requires the *open* kernel modules, and a host carrying the proprietary ones
reports no device at all while `lspci` plainly shows the GPU:

    NVRM: The NVIDIA GPU 0000:00:07.0 (PCI ID: 10de:2b85)
    NVRM: installed in this system requires use of the NVIDIA open kernel modules.

    apt-get purge -y '^nvidia-.*' '^libnvidia-.*'
    apt-get install -y nvidia-driver-575-open
    reboot

**Docker has to be able to reach the card.** That is the NVIDIA Container
Toolkit, and without it the workers build, start, claim work and find no device.
The installer warns when the daemon has no `nvidia` runtime; it does not install
it for you, because that is a system package and your machine's business.

    # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
    nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
    docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L    # must list the card

## 2. Install

    curl -fsSL https://raw.githubusercontent.com/LimeGS/helena-framework/main/install.sh | sh -s -- --gpu

One command. It clones, builds the panel, compiles volume-cartographer -- expect
an hour or two -- and starts the workers. It wants `git`, `curl` and Docker
with Compose v2, and about 6 GB free under Docker's root, and it checks all of
that before it downloads anything. The checkout lands in `./helena` (set
`HELENA_DIR` to put it elsewhere); that directory is `<checkout>` in step 7. On
the machine this page was written from it brought up nine containers and left
four workers polling.

The tenth was the surface-QC runtime, which wanted a checkpoint nothing
downloaded, and sat restarting. The deploy fetches that checkpoint from Hugging
Face now, against the digest the weights registry pins, before it starts the
QC stack; only a fetch that fails leaves it looping. It is P2's either way, and
this control does not touch it.

## 3. Claim the first account

    curl -sk -c cookies https://127.0.0.1:8800/api/session/bootstrap \
      -H 'Content-Type: application/json' \
      -d '{"username":"you","password":"at-least-ten-characters"}'

Loopback only, and it closes once an account exists. The answer signs you in
and sets the session cookie, so `-c cookies` keeps it in a jar, and every
`curl` below sends it back with `-b cookies`. To sign in again later, `POST
/api/session` with the same body and `-c cookies` again.

## 4. Ask the panel for the checkpoint

    curl -sk -b cookies -X POST https://127.0.0.1:8800/api/models/download \
      -H 'Content-Type: application/json' \
      -d '{"repo":"scrollprize/ink_9um",
           "file":"hybrid_3d2d-seed42/step-075000.pth",
           "name":"ink_9um",
           "expect_sha256":"e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab"}'

`expect_sha256` is required here and not optional paperwork: upstream publishes
this checkpoint as `.pth`, which is a pickle, and the panel fetches one only
against the hash it must have -- deleting it rather than installing it if the
bytes disagree. `GET /api/models` then reports it as installed and says which
profiles declare it.

Earlier versions of this page wrote the file into the volume with a `docker run`
instead, because the API refused both the format and the subdirectory. A control
that reaches into the machine it is testing is not testing the interface anybody
else would use, so both refusals were changed rather than documented.

## 5. Make a mission

Nothing may be queued outside one, and that rule lives in the store rather than
the panel.

    curl -sk -b cookies -X POST https://127.0.0.1:8800/api/missions \
      -H 'Content-Type: application/json' \
      -d '{"mission_id":"public-control","name":"Public ink control",
           "scrolls":["PHerc0139"]}'

PHerc0139 is the scroll on purpose. It is absent from the frozen eligible
catalogue: the First Letters control policy pins it as the development control,
and a control scroll may never be an evaluation scroll. The panel accepts it
here because that policy names its volumes. Any other name has to be in the
catalogue, or the request is refused with the names the deployment knows.

## 6. Freeze and select the P0 artifact

The segmentation control makes these two requests itself. Here they are yours
to make: PHerc0139 is the control scroll, and the panel will not queue an ink
job on the control scroll until the mission has selected a P0 artifact -- a
control run has to name which scan it used, and the panel refuses to guess:

    409: control scope requires an explicit selected P0 artifact

Two calls. The first registers what P0 produced and returns `artifacts`, one
per scroll, each with an `artifact_id` of the form `p0:PHerc0139:<12 hex>`;
the second chooses it. The selection is the whole map, never a patch.

    curl -sk -b cookies -X POST \
      https://127.0.0.1:8800/api/missions/public-control/artifacts/freeze-p0

    curl -sk -b cookies -X POST https://127.0.0.1:8800/api/missions/public-control/selection \
      -H 'Content-Type: application/json' \
      -d '{"choices":{"P0/PHerc0139":"<artifact_id from the call above>"},
           "reason":"public ink control"}'

## 7. Run it

    docker run --rm --network host \
      -v <checkout>:/repo:ro -v <output dir>:/out \
      -e HELENA_PANEL_PASSWORD='<the password from step 3>' \
      -e HELENA_PANEL_TLS_INSECURE=1 \
      helena-ink-9um:local \
      python /repo/scripts/harness/run_public_ink_control.py \
        --panel https://127.0.0.1:8800 --user you --mission public-control \
        --sample-id PHerc0139 \
        --checkpoint-path /models/ink_9um/hybrid_3d2d-seed42/step-075000.pth \
        --expected-checkpoint-sha256 e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab \
        --source-pixel-um 9.362 \
        --surface-volume <the volume URL above> \
        --output /out

`helena-ink-9um:local` is the 9 um lane image the GPU deploy built in step 2;
`docker images | grep helena-ink-9um` shows it. The control runs in it because
it reads the map back with numpy, which the panel image does not carry.
`<checkout>` is the directory the installer cloned into and `<output dir>` any
empty directory of yours. `--user` is the account from step 3; the password
comes from `HELENA_PANEL_PASSWORD` or `--password`.

`HELENA_PANEL_TLS_INSECURE=1` because the panel's certificate is self-signed on
first boot. Point `--panel` at a name that certificate covers and you can drop
it; on the machine that just installed itself, you cannot.

Exit status is 0 on `CONTROL_PASS` and 3 otherwise. Inference took twenty-four
to twenty-eight minutes on one 5090 across the three runs below.

## What is not an API call, and why

Steps 4 to 7 are requests. Nothing reads or writes the deployment's disk: the
checkpoint is placed by the models endpoint, the mission and its P0 selection
are requests, the job is queued, and the map comes back by the `artifact_uri`
the worker published. The control mounts no volume of the deployment's -- only
somewhere to write its own output.

Steps 1 to 3 are not, and cannot be. They are what happens before a platform
exists: a kernel driver, a container runtime, and the installer that creates the
thing an API could be served from. There is no interface to call on a machine
that has none.

Step 7 runs a client. That client is a program you run, like `curl` above, and
it can run anywhere that can reach `--panel`; it does not have to be on the
deployment's host.

## What it writes

    PUBLIC_INK_CONTROL.json   the stage-survival receipt, content-addressed
    ARTIFACT_SET.json         the digests of what the worker published
    probability.npy           the map, and its reverse

## Reading the receipt

Each row of `stages` carries a `boundary`, a `terminal_state` -- `PASS`,
`INCOMPLETE`, `FAILED` or `NOT_RUN_PREREQUISITE` -- a `reason_code` saying why,
and `elapsed_seconds`. `control_state` is `CONTROL_PASS` only when every
boundary passed. The first non-passing boundary owns the outcome and is named
in `first_nonpassing_boundary`; every row after it is normalised to
`NOT_RUN_PREREQUISITE` with its counts and hashes emptied, so a receipt cannot
show a later boundary passing over an earlier failure. `content_sha256` is the
SHA-256 of the receipt itself, serialised with sorted keys and no whitespace
before that field was added; a receipt that has been edited no longer matches
its own digest.

Four fields are worth reading directly:

* `stages[INK].resource_identity.through` is `helena-queue`. A run through the
  queue and a run beside it are different claims and must not be readable as the
  same one -- the other value is `local-subprocess`.
* `stages[INK].resource_identity.job_id` and `artifact_uri` name the job the
  worker claimed and where it published, so the receipt can be checked against
  the deployment rather than believed.
* `stages[CHECKPOINT].resource_identity.established_by` is `helena-api`. The
  other value is `local-file`, which means the control hashed a file it could
  see rather than asking the deployment -- a claim about the machine the control
  ran on, not about the one that ran the job.
* `stages[PUBLIC_SOURCE].resource_identity.credentials_used` is `false`.
* `stages[LIVENESS].counts.p50` should sit near 0.25. That is this recipe's
  documented no-ink floor -- it trains with BCE label smoothing 0.5 -- so a map
  whose p50 is 0.000 is empty, not no-ink. A misconfigured batch size produced
  exactly that, exit 0 and all, and the liveness gate is what caught it.

## Runs

    2026-09-02  CONTROL_PASS  helena-queue  the same fresh machine as the
                                              segmentation control      run-2026-09-02-clean

    2026-09-01  CONTROL_PASS  helena-queue  rented 5090          run-2026-09-01-vast
    2026-09-01  CONTROL_PASS  helena-queue  the same, reinstalled from scratch
                                            with the command in step 2
                                                                 run-2026-09-01-clean

Comparing digests only works against a run on the same GPU. Ours and the rented
card produce different bytes and the same statistics to four decimals -- p50
0.2784, p99 0.7765, std 0.1674 -- which is floating-point accumulation, not a
different result. Compare those.
