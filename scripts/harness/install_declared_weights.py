#!/usr/bin/env python3
"""Install every checkpoint the deployment's profiles declare, through the API.

    install_declared_weights.py --panel https://localhost:8800 --user <user> \\
        --password-file <file> [--only <substring>] [--dry-run]

A review of the running deployment found the scaffolding complete -- 33
profiles pinning checkpoints by digest, a Models page reporting `writable`, a
refusal at queue time when the weight is missing -- and the content absent:
"4 of 33 installed by hash". That is what blocks a sweep.

scripts/models/install_ink_weights.py already installs in bulk, by writing
into the models volume from outside. The download endpoint's own docstring
says why that is the wrong door: the panel is the one process allowed to write
a checkpoint, because a worker that could overwrite one could change what a
frozen profile means. So this asks the panel what it lacks (`GET
/api/models?resolve=1`, which resolves each missing digest on Hugging Face)
and asks it to fetch each one (`POST /api/models/download`) against the digest
the profile pins -- the same request the Models page makes, once per row.

A pickle (`.ckpt`, `.pth`) is fetched only against a hash, and the hash given
here is always the profile's own, so this never asks for bytes nobody named.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_client import Panel, PanelError  # noqa: E402

FETCHABLE = ("exact", "pickle_only")


def read_password(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def plan(rows: list[dict], only: list[str]) -> tuple[list[dict], list[dict]]:
    """What to fetch, and what cannot be fetched and why."""
    fetch, blocked = [], []
    for row in rows:
        if row.get("installed"):
            continue
        label = f"{row.get('upstream') or ''} {row.get('expected_path') or ''}"
        if only and not any(o in label for o in only):
            continue
        found = row.get("hugging_face") or {}
        if found.get("state") in FETCHABLE and found.get("file"):
            fetch.append(row)
        else:
            blocked.append(row)
    return fetch, blocked


def request_for(row: dict) -> dict:
    found = row["hugging_face"]
    body = {"repo": found["repo"], "file": found["file"],
            "revision": str(found.get("revision") or "main"),
            "expect_sha256": row["checkpoint_sha256"]}
    # Where it lands under the models root: the directory the registry expects,
    # so a profile that names the path finds it there. Installed-ness is by
    # hash regardless.
    expected = str(row.get("expected_path") or "")
    if "/" in expected:
        body["name"] = expected.split("/", 1)[0]
    return body


def main(argv: list[str] | None = None, panel: Panel | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel",
                        default=os.environ.get("HELENA_PANEL", "https://localhost:8800"))
    parser.add_argument("--user", required=panel is None)
    parser.add_argument("--password-file", type=Path, required=panel is None,
                        help="read at run time; never echoed or stored")
    parser.add_argument("--only", action="append", default=[],
                        help="substring of the upstream repo or expected path; repeatable")
    parser.add_argument("--dry-run", action="store_true",
                        help="say what would be fetched and fetch nothing")
    parser.add_argument("--insecure", action="store_true",
                        help="the panel serves its own certificate")
    args = parser.parse_args(argv)

    if panel is None:
        panel = Panel(args.panel, insecure=args.insecure)
        panel.call("POST", "/api/session",
                   {"username": args.user, "password": read_password(args.password_file)})

    before = panel.call("GET", "/api/models?resolve=1")
    rows = list(before.get("checkpoints") or [])
    installed_before = sum(1 for r in rows if r.get("installed"))
    print(f"{installed_before} of {len(rows)} installed by hash")
    if not before.get("writable", True):
        print("the models root is not writable by the panel; nothing can be "
              "installed through it", file=sys.stderr)
        return 2

    fetch, blocked = plan(rows, args.only)
    for row in blocked:
        found = row.get("hugging_face") or {}
        print(f"  cannot fetch {row.get('upstream')}: {found.get('state') or 'unresolved'}"
              + (f" -- {found['why']}" if found.get("why") else ""))
    if not fetch:
        print("nothing to fetch")
        return 0 if not blocked else 1

    failures = 0
    for row in fetch:
        body = request_for(row)
        size = (row.get("hugging_face") or {}).get("bytes") or row.get("size_bytes")
        print(f"  {body['repo']}/{body['file']}"
              + (f" ({size / 1e6:.0f} MB)" if size else "")
              + (" -- dry run" if args.dry_run else ""))
        if args.dry_run:
            continue
        try:
            panel.call("POST", "/api/models/download", body, timeout=3600)
        except PanelError as refused:
            failures += 1
            print(f"    refused: HTTP {refused.status} {refused.body[:200]}",
                  file=sys.stderr)

    if args.dry_run:
        return 0
    after = panel.call("GET", "/api/models")
    installed_after = sum(1 for r in (after.get("checkpoints") or []) if r.get("installed"))
    print(f"{installed_after} of {len(rows)} installed by hash")
    return 1 if failures or blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
