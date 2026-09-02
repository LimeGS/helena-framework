# Running the fleet as a non-root user

Every image here ran as uid 0. A bug in a runner — a tar, a zarr reader, a
model loader handed an attacker-shaped file — had the whole container and every
volume mounted into it, and `no-new-privileges` was not set either.

`runtime.compose.yaml` already did this. The rest now do too, in two groups.

**Running as 1000:1000**: the ink slots, the panel and its init, surface-qc.
Verified on a fleet host before the files changed — a worker imports
psycopg/numpy/tifffile, writes runs and artifacts, reaches the database, runs
the migrations and claims; the panel imports and writes its state directory.

**0:0 by compose default, 1000:1000 by configuration**: `segment`,
`fleet-runner`, `preflight` and `spiral`. They run `helena-worker-cpp`, whose
interpreter uv installed under `/root` at mode 0700 — as any other user Python
could not read its own stdlib and died with `No module named 'encodings'`
before reaching `main()`. Its entrypoint also ran `mkdir -p $ARTIFACT_ROOT` on
an `s3://` URL, which root created as a junk directory and a normal user simply
cannot.

Both are fixed in `Containerfile.worker-cpp` (`/root` is made traversable and
the interpreter world-readable) and `worker-entrypoint.sh` (only directory
roots are created). `segment.compose.yaml` and `spiral.compose.yaml` still
default `HELENA_RUN_AS` to `0:0`, but the template `segment.env.example`, which
the deploy seeds into `config/segment.env`, and `provision-host.sh` both set it
to `1000:1000`, so a host configured from the templates runs these four as
1000 on an image that carries the fix. Check an image before pointing an older
env file at it:

```bash
docker run --rm --user 1000:1000 --entrypoint sh helena-worker-cpp:<tag> \
  -c 'python3 -c "print(1)"'
```

Flipping it on an image without the fix is how those four containers spent a
few minutes crash-looping.

**Left alone entirely**: `postgres`, because the official image already drops
to its own user over a data directory that user owns; and `backup`, which
declares `user: root` for its own reasons and did so before this change.

## What a host has to do once

The volumes have to be owned by that uid, or the containers start and fail on
their first write. On a host that has not been converted, the failure is a
permission error at startup rather than silent damage.

The named volumes need nothing by hand: the platform stack's `prepare-volumes`
service chowns `helena-panel-state`, `helena-runs`, `helena-artifacts` and
`helena-models` to `HELENA_RUN_AS` on every `up`, and only while they are still
root's. Host paths bound in their place are yours to own, and the deploy does
it for the runs directories it knows about:

```bash
sudo chown -R 1000:1000 \
  /path/to/artifacts \
  /path/to/panel-state \
  /path/to/runs
```

Adjust the paths to whatever `HELENA_INK_ARTIFACTS`, `HELENA_PANEL_STATE` and
`HELENA_FLEET_RUNS` resolve to on that host. Postgres's own data directory is
not in the list and must not be chowned.

The chown is safe to run while the fleet is up as root: root writes regardless,
so it is a no-op until the containers are recreated.

## Verifying before you commit to it

Both halves were checked this way before the compose files changed,
and it is the cheapest way to check a new host:

```bash
docker run --rm --user 1000:1000 --network host \
  -v /path/to/runs:/path/to/runs \
  -v /path/to/artifacts:/artifacts \
  --env-file config/ink.env \
  helena-worker-gpu:<tag> sh -c 'id; touch /artifacts/.probe && rm /artifacts/.probe'
```

`config/` is the checkout's configuration directory, which the deploy seeds
from the templates; a host configured before that keeps its files under
`/etc/helena`.

## Going back

`HELENA_RUN_AS=0:0` in the host's env file restores the old behaviour without
editing a compose file, which is what to reach for if a lane turns out to need
something only root can do. That would be worth reporting rather than leaving
set.
