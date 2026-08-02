#!/usr/bin/env bash
# Local, network-free provenance backup.  Addresses FIX-01 of
# the 2026-07-24 pipeline audit for the case where no Git remote is
# configured and none may be created.
#
# It writes two independent artifacts plus a receipt:
#
#   1. campaign-x-<utc>.bundle       every ref, verifiable with `git bundle verify`
#   2. campaign-x-<utc>-durable.tar  untracked files the README declares durable
#                                    (a bundle carries committed history only)
#   3. PROVENANCE_BACKUP_RECEIPT.json  sizes, SHA-256, HEAD, ref list
#
# Contacts no network and reads no credentials.
#
# Usage:
#   bash scripts/harness/backup_provenance_local.sh DESTINATION_DIR
#
# DESTINATION_DIR must be outside the repository so the backup is not an input
# to itself.  Prefer a different physical volume: a bundle beside the working
# tree protects against history rewrite and .git corruption, not disk loss.
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 DESTINATION_DIR" >&2
  exit 64
fi

repo_root="$(git rev-parse --show-toplevel)"
destination="$1"

case "$(cd "$destination" 2>/dev/null && pwd -P || echo "$destination")" in
  "$repo_root"|"$repo_root"/*)
    echo "DESTINATION_DIR must be outside the repository: $repo_root" >&2
    exit 65
    ;;
esac

mkdir -p "$destination"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
bundle="$destination/campaign-x-$stamp.bundle"
durable="$destination/campaign-x-$stamp-durable.tar"
receipt="$destination/PROVENANCE_BACKUP_RECEIPT_$stamp.json"

cd "$repo_root"

# Committed history, every ref.
git bundle create "$bundle" --all
git bundle verify "$bundle" >/dev/null

# Untracked files the README classifies as "keep until downstream closeout".
# A bundle cannot carry them: they are not committed.  The raw six-replica .npy
# maps are the declared input of the Module 04 high-recall route.
untracked_list="$(mktemp)"
trap 'rm -f "$untracked_list"' EXIT
git ls-files --others --exclude-standard -z -- workspace \
  | tr '\0' '\n' \
  | grep -E '(center-[0-9]+_offset-[0-9]+\.npy|GROWTH_RECEIPT\.json|meta\.json|/(x|y|z)\.tif)$' \
  > "$untracked_list" || true

untracked_count="$(wc -l < "$untracked_list" | tr -d ' ')"
if [[ "$untracked_count" -gt 0 ]]; then
  tar -cf "$durable" -T "$untracked_list"
else
  tar -cf "$durable" --files-from /dev/null
fi

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
bytes() { wc -c < "$1" | tr -d ' '; }

python3 - "$bundle" "$durable" "$receipt" "$untracked_count" <<'PY'
import hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

bundle, durable, receipt, untracked_count = sys.argv[1:5]

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()

payload = {
    "schema": "campaignx.provenance_backup_receipt.v1",
    "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "head_commit": git("rev-parse", "HEAD"),
    "refs": git("for-each-ref", "--format=%(refname) %(objectname)").splitlines(),
    "remotes": git("remote", "-v").splitlines(),
    "bundle": {
        "path": bundle,
        "size_bytes": Path(bundle).stat().st_size,
        "sha256": digest(bundle),
        "verified": True,
    },
    "durable_untracked": {
        "path": durable,
        "size_bytes": Path(durable).stat().st_size,
        "sha256": digest(durable),
        "file_count": int(untracked_count),
    },
    "non_claims": [
        "A bundle beside the working tree protects against history rewrite and .git corruption, not disk loss.",
        "Off-volume or off-site replication remains a separate manual step.",
        "This receipt does not assert that the untracked selection is exhaustive.",
    ],
}
Path(receipt).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({
    "head": payload["head_commit"],
    "refs": len(payload["refs"]),
    "remotes": len(payload["remotes"]),
    "bundle_bytes": payload["bundle"]["size_bytes"],
    "durable_files": payload["durable_untracked"]["file_count"],
    "receipt": receipt,
}, indent=2))
PY
