#!/usr/bin/env bash
# Stream an entire Vast /workspace tree to S3 without placing AWS credentials
# or a local archive on the Vast host.  The AWS CLI runs only on the Mac.
#
# Usage:
#   scripts/harness/backup_vast_workspace_to_s3.sh \
#     root@HOST PORT ~/.ssh/KEY BUCKET S3_PREFIX LOCAL_RECEIPT_DIR EXPECTED_BYTES
#
# The archive preserves the remote /workspace directory exactly as a tar stream
# (including symlinks, xattrs, ACLs, numeric owners and sparse files).  It is
# zstd-compressed on Vast, SHA-256-hashed while it crosses the Mac, then written
# as one multipart S3 object.  The manifest and its hash are uploaded first.
set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "usage: $0 REMOTE PORT SSH_KEY BUCKET PREFIX RECEIPT_DIR EXPECTED_BYTES" >&2
  exit 64
fi

remote="$1"
port="$2"
ssh_key="$3"
bucket="$4"
prefix="${5%/}"
receipt_dir="$6"
expected_bytes="$7"

mkdir -p "$receipt_dir"

manifest="$receipt_dir/workspace-manifest.tsv"
manifest_sha="$receipt_dir/workspace-manifest.tsv.sha256"
archive_sha="$receipt_dir/workspace.tar.zst.sha256"
archive_sha_fifo="$receipt_dir/workspace.tar.zst.sha256.fifo"
archive_key="$prefix/workspace.tar.zst"
manifest_key="$prefix/workspace-manifest.tsv"
manifest_sha_key="$prefix/workspace-manifest.tsv.sha256"
archive_sha_key="$prefix/workspace.tar.zst.sha256"

ssh_base=(ssh -i "$ssh_key" -p "$port" -o BatchMode=yes "$remote")

"${ssh_base[@]}" 'cd / && find workspace -xdev -type f -printf "%s\t%T@\t%p\n" | LC_ALL=C sort' > "$manifest"
shasum -a 256 "$manifest" > "$manifest_sha"

aws s3 cp "$manifest" "s3://$bucket/$manifest_key" --no-progress
aws s3 cp "$manifest_sha" "s3://$bucket/$manifest_sha_key" --no-progress

# No archive file is created locally or remotely: tee has one hashing consumer
# and one S3 multipart-upload consumer.  pipefail makes a remote tar/zstd or
# upload failure terminal instead of emitting a misleading receipt.
rm -f "$archive_sha_fifo"
mkfifo "$archive_sha_fifo"
trap 'rm -f "$archive_sha_fifo"' EXIT
shasum -a 256 < "$archive_sha_fifo" > "$archive_sha" &
sha_pid=$!
"${ssh_base[@]}" 'cd / && tar --xattrs --acls --sparse --numeric-owner --ignore-failed-read -cf - workspace | zstd -T0 -3' \
  | tee "$archive_sha_fifo" \
  | aws s3 cp - "s3://$bucket/$archive_key" --expected-size "$expected_bytes" --no-progress
wait "$sha_pid"

aws s3 cp "$archive_sha" "s3://$bucket/$archive_sha_key" --no-progress
aws s3api head-object --bucket "$bucket" --key "$archive_key" \
  --query '{ContentLength:ContentLength,ETag:ETag,VersionId:VersionId,ServerSideEncryption:ServerSideEncryption}' \
  --output json > "$receipt_dir/HEAD_OBJECT.json"

printf 'BACKUP_COMPLETE\narchive=s3://%s/%s\nmanifest=s3://%s/%s\n' \
  "$bucket" "$archive_key" "$bucket" "$manifest_key"
