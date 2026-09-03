---
title: Backups and object storage
summary: Two different buckets: where phases publish, and where the control plane's copy goes. Both optional, both yours.
---

Helena needs no bucket to be complete. A deployment stores artifacts on a
volume of the panel host and keeps its database in the postgres container. Two
things change that, and they are separate decisions with separate credentials.

## Where phases publish

Each phase writes to a store the panel names. All four default to a path
inside the artifact volume the platform mounts -- `CX_RENDER_STORE`, for
instance, is `/artifacts/layer-stacks-v1` unless it is overridden. Point them
at your own bucket only when you have one, in `config/panel.env`:

```
CX_FLATTEN_STORE=s3://your-bucket/flattened-v1
CX_RENDER_STORE=s3://your-bucket/layer-stacks-v1
CX_INK_STORE=s3://your-bucket/ink-maps-v1
CX_RECONSTRUCTION_STORE=s3://your-bucket/reconstruction-v1
```

Cleared to empty rather than left at that default, a store blocks its phase at
the queue: P3, P5 and P8's merge lane refuse to enqueue with a 409 until it is
set again, and P4 refuses too unless the request opts into
`allow_local_layers` for a deliberate single-machine run.

The credentials for those stores live in the control plane, not on the workers:
**Configuration → Settings → Credentials**. The names it accepts are
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`,
`AWS_DEFAULT_REGION`, `AWS_REGION` and `AWS_ENDPOINT_URL`. They are write-only:
the page reports which are set and never returns a value. A worker adopts them
when it starts, so a machine joining the fleet needs a database URL and nothing
else.

An environment variable on a worker wins over the control plane, deliberately.
A key left in a worker's env file makes the panel's copy inert for that host,
and the worker logs that it is shadowing the panel. Keep credentials in one
place.

Give the key the least it needs: put and get on the prefixes above, list on the
bucket. It never deletes.

## Where the control plane's copy goes

Everything a phase produces is published by digest and outlives the machine.
The record of *which* of them is certified, which cell was attempted and what
every verdict was is one PostgreSQL on one host. `helena-backup` is the copy of
that.

Every interval, and once at start, it uploads:

| what | how | to |
|---|---|---|
| the database | `pg_dump --format=custom --no-owner --no-acl`, then `pg_restore --list` to prove it parses | `<prefix>/postgres/<utc>.dump` |
| the panel's state directory | `tar.gz` | `<prefix>/panel-state/<utc>.tgz` |
| the runs directory | `tar.gz` | `<prefix>/runs/<utc>.tgz` |
| a receipt naming each file with its SHA-256 and size | JSON | `<prefix>/receipts/<utc>.json` |

### Turning it on

1. A bucket, and a key that can put and list on one prefix of it. Retention is
   a lifecycle rule on the bucket: the script uploads and forgets, because a
   deleter with credentials is a different risk from a writer with credentials.
2. `config/aws.env`, mode 0600, on the panel host:

   ```
   AWS_ACCESS_KEY_ID=...
   AWS_SECRET_ACCESS_KEY=...
   AWS_DEFAULT_REGION=eu-central-1
   # anything that is not AWS, such as MinIO:
   # AWS_ENDPOINT_URL=http://127.0.0.1:9000
   ```

   This file is read by the backup container, and also by the panel itself
   for the read-only S3 surface previews a TIFXYZ needs -- one credential to
   rotate rather than a third copy of it. It is not the panel's credential
   store above, and it is not a worker's env file.
3. In `config/platform.env`:

   ```
   HELENA_BACKUP_S3=s3://your-bucket/helena
   HELENA_BACKUP_INTERVAL_HOURS=24
   ```

   `HELENA_AWS_ENV` names the credentials file if it is somewhere other than
   `config/aws.env`. An interval of `0` runs one round and exits, which is how
   to test it.
4. Run the deploy again (`containers/deploy-platform.sh nogpu` or `gpu`). It
   adds the `backup` profile when `HELENA_BACKUP_S3` is set and says so;
   without it, it says the service is not started.

Then look:

```
docker logs helena-backup
aws s3 ls --recursive s3://your-bucket/helena/
```

The first round on a large `runs` directory is a long `tar.gz`; that cost
repeats every interval, so on a fleet with terabytes of runs point
`HELENA_PANEL_RUNS` at what you want copied or lengthen the interval.

### Restoring

A dump that parses is not a proven recovery. Once, on a clean PostgreSQL:

```
aws s3 cp s3://your-bucket/helena/postgres/<utc>.dump control-plane.dump
sha256sum control-plane.dump         # against the receipt
pg_restore --no-owner --clean --if-exists -d "$CX_DB" control-plane.dump
```

The panel state archive unpacks into the panel state volume; runs into the runs
directory. Bring the panel up afterwards, not before.

### What this is not

* Not a backup of published artifacts: those are already in the store above,
  by digest, and a bucket is not backed up by copying it into itself.
* Not versioned: every round is a full copy, and the bucket's lifecycle rule
  decides what stays.
* Not a test of the current database: the per-round check is that the dump
  lists its objects, which is the strongest thing available without a second
  database standing by.

## Where the files are

Everything above lives in `config/` in the checkout, git-ignored, next to the
code that reads it. A host configured before that directory existed keeps its
files under `/etc/helena`, and the deploy says so each time it runs. Nothing
here is entered through the panel except the credentials in the first section.
