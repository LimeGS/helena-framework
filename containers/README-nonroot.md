# Running the fleet as a non-root user

Every image here ran as uid 0. A bug in a runner — a tar, a zarr reader, a
model loader handed an attacker-shaped file — had the whole container and every
volume mounted into it, and `no-new-privileges` was not set either.

`runtime.compose.yaml` already did this. The rest now do too, in two groups.

**Running as 1000:1000**: the ink slots, the panel and its init, surface-qc.
Verified on gpu-1 before the files changed — a worker imports
psycopg/numpy/tifffile, writes runs and artifacts, reaches the database, runs
the migrations and claims; the panel imports and writes its state directory.

**Still 0:0, on purpose**: `segment`, `fleet-runner`, `preflight` and `spiral`.
They run `helena-worker-cpp`, whose interpreter uv installed under `/root` at mode
0700 — as any other user Python cannot read its own stdlib and dies with
`No module named 'encodings'` before reaching `main()`. Its entrypoint also ran
`mkdir -p $ARTIFACT_ROOT` on an `s3://` URL, which root created as a junk
directory and a normal user simply cannot.

Both are fixed in `Containerfile.worker-cpp` and `worker-entrypoint.sh`, and
neither takes effect until those images are rebuilt. Flip the default in
`segment.compose.yaml` and `spiral.compose.yaml` once a rebuilt image passes:

```bash
docker run --rm --user 1000:1000 --entrypoint sh helena-worker-cpp:<tag> \
  -c 'python3 -c "print(1)"'
```

Flipping it before that is how those four containers spent a few minutes
crash-looping.

**Left alone entirely**: `postgres`, because the official image already drops
to its own user over a data directory that user owns; and `backup`, which
declares `user: root` for its own reasons and did so before this change.

## What a host has to do once

The volumes have to be owned by that uid, or the containers start and fail on
their first write. On a host that has not been converted, the failure is a
permission error at startup rather than silent damage.

```bash
sudo chown -R 1000:1000 \
  /ssd/vc3d/artifacts \
  /ssd/vc3d/panel-state \
  /ssd/docker/volumes/helena-models/_data
```

`/mnt/bulk/helena/runs` is already `1000:1000` on gpu-1; check yours. Adjust the
paths to whatever `HELENA_INK_ARTIFACTS`, `HELENA_PANEL_STATE` and the models
volume resolve to on that host. Postgres's own data directory is not in the
list and must not be chowned.

The chown is safe to run while the fleet is up as root: root writes regardless,
so it is a no-op until the containers are recreated.

## Verifying before you commit to it

Both halves were checked on gpu-1 this way before the compose files changed,
and it is the cheapest way to check a new host:

```bash
docker run --rm --user 1000:1000 --network host \
  -v /mnt/bulk/helena/runs:/mnt/bulk/helena/runs \
  -v /ssd/vc3d/artifacts:/artifacts \
  --env-file /etc/helena/ink.env \
  helena-worker-gpu:<tag> sh -c 'id; touch /artifacts/.probe && rm /artifacts/.probe'
```

## Going back

`HELENA_RUN_AS=0:0` in the host's env file restores the old behaviour without
editing a compose file, which is what to reach for if a lane turns out to need
something only root can do. That would be worth reporting rather than leaving
set.
