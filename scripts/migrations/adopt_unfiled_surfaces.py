#!/usr/bin/env python3
"""Give surfaces that belong to no mission one, so phases can be queued on them.

216 of the 220 PHerc826 surfaces -- 332 cm2 of certified geometry, the bulk of
what this platform has recovered -- match no mission by any of the three routes
`MISSION_SURFACE_PREDICATE` accepts. They read as `unfiled`, which is a
read-only view of what predates missions rather than a scope. Nothing can be
queued against them: P3 answers `considered: 0` and exits zero.

There is no adoption path in the panel. `POST /api/missions/{id}/artifacts/
backfill` registers surfaces in a mission's artifact register on disk, which
serves lineage and does not touch any branch of the predicate. So this writes
the third branch directly: `payload->>'mission_id'`.

What it deliberately does not touch: a surface already linked to a real mission
through segment_tasks. Setting a second mission on one of those would make it
belong to two at once, and every count in the panel is mission-scoped.

  adopt_unfiled_surfaces.py --mission <id> --sample PHerc826            # dry run
  adopt_unfiled_surfaces.py --mission <id> --sample PHerc826 --apply
  adopt_unfiled_surfaces.py --rollback backup-<stamp>.json

Every applied run writes a backup naming each surface it changed. The rollback
removes the key only from the surfaces that backup lists, so a surface adopted
by some other means afterwards is left alone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SELECT_CANDIDATES = """
SELECT s.surface_id
  FROM segment_surfaces s
 WHERE s.sample_id = %s
   AND s.payload->>'mission_id' IS NULL
   AND NOT EXISTS (
     SELECT 1 FROM segment_artifact_sets art
       JOIN segment_attempts a ON a.attempt_id = art.attempt_id
       JOIN segment_tasks t ON t.task_id = a.task_id
      WHERE art.manifest->>'artifact_sha256' = s.artifact_sha256
        AND t.mission_id IS NOT NULL
        AND t.mission_id <> 'unfiled')
 ORDER BY s.surface_id
"""


def connect(dsn: str):
    import psycopg  # noqa: PLC0415
    return psycopg.connect(dsn, connect_timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dsn", default=os.environ.get("CX_DB"),
                        help="defaults to $CX_DB")
    parser.add_argument("--mission")
    parser.add_argument("--sample", default="PHerc826")
    parser.add_argument("--apply", action="store_true",
                        help="without this the script only reports")
    parser.add_argument("--rollback", type=Path,
                        help="undo a previous run from its backup file")
    parser.add_argument("--backup-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    if not args.dsn:
        raise SystemExit("no DSN: pass --dsn or set CX_DB")

    if args.rollback:
        record = json.loads(args.rollback.read_text())
        ids = record["surface_ids"]
        print(f"removing mission_id from {len(ids)} surfaces "
              f"(set to {record['mission_id']} at {record['applied_at_utc']})")
        with connect(args.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """UPDATE segment_surfaces
                      SET payload = payload - 'mission_id'
                    WHERE surface_id = ANY(%s)
                      AND payload->>'mission_id' = %s""",
                (ids, record["mission_id"]))
            print(f"  {cur.rowcount} rows reverted")
            conn.commit()
        return 0

    if not args.mission:
        raise SystemExit("--mission is required unless --rollback is given")

    with connect(args.dsn) as conn, conn.cursor() as cur:
        cur.execute(SELECT_CANDIDATES, (args.sample,))
        ids = [row[0] for row in cur.fetchall()]
        print(f"{len(ids)} surfaces of {args.sample} belong to no mission")
        if not ids:
            return 0
        for surface in ids[:5]:
            print(f"  {surface}")
        if len(ids) > 5:
            print(f"  ... and {len(ids) - 5} more")

        if not args.apply:
            print("\\ndry run; nothing written. Re-run with --apply")
            return 0

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = args.backup_dir / f"adopt-surfaces-backup-{stamp}.json"
        backup.write_text(json.dumps({
            "applied_at_utc": stamp, "mission_id": args.mission,
            "sample_id": args.sample, "surface_ids": ids,
            "note": "these surfaces had no mission_id in payload before this run",
        }, indent=1))
        print(f"\\nbackup written to {backup}")

        cur.execute(
            """UPDATE segment_surfaces
                  SET payload = coalesce(payload, '{}'::jsonb)
                                || jsonb_build_object('mission_id', %s::text)
                WHERE surface_id = ANY(%s)
                  AND payload->>'mission_id' IS NULL""",
            (args.mission, ids))
        changed = cur.rowcount
        conn.commit()
        print(f"{changed} surfaces adopted into {args.mission}")
        if changed != len(ids):
            print(f"warning: expected {len(ids)}; something changed underneath",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
