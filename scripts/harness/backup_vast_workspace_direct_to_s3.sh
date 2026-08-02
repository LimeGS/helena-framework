#!/usr/bin/env bash
# Run on a Vast host with short-lived AWS credentials supplied only through its
# process environment.  It writes no credentials to disk.  The whole remote
# /workspace tree is archived as a compressed stream and uploaded directly to
# S3; no data crosses the operator's Mac.
#
# Usage:
#   bash backup_vast_workspace_direct_to_s3.sh BUCKET PREFIX RECEIPT_DIR EXPECTED_BYTES
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 BUCKET PREFIX RECEIPT_DIR EXPECTED_BYTES" >&2
  exit 64
fi

bucket="$1"
prefix="${2%/}"
receipt_dir="$3"
expected_bytes="$4"

case "$receipt_dir" in
  /workspace|/workspace/*)
    echo "RECEIPT_DIR must be outside /workspace so the archive input stays immutable" >&2
    exit 65
    ;;
esac

mkdir -p "$receipt_dir"

manifest="$receipt_dir/workspace-manifest.tsv"
manifest_sha="$receipt_dir/workspace-manifest.tsv.sha256"
archive_sha="$receipt_dir/workspace.tar.zst.sha256"
archive_sha_fifo="$receipt_dir/workspace.tar.zst.sha256.fifo"
secret_exclusion_receipt="$receipt_dir/EXCLUDED_SENSITIVE_FILES.json"
archive_key="$prefix/workspace.tar.zst"
manifest_key="$prefix/workspace-manifest.tsv"
manifest_sha_key="$prefix/workspace-manifest.tsv.sha256"
archive_sha_key="$prefix/workspace.tar.zst.sha256"
secret_exclusion_receipt_key="$prefix/EXCLUDED_SENSITIVE_FILES.json"

# These are credentials/one-shot tokens rather than scientific artifacts.  The
# paths are explicit on purpose: broad '*token*' exclusions would silently omit
# source files and model metadata with harmless names.
sensitive_path_1="/workspace/campaign-x-vc3d-mcp-runtime/token"
sensitive_path_2="/workspace/local-holdout-v2-private-cafcb39b/LOCAL_HOLDOUT_V2_ATTEMPT_TOKEN.bin"

mapfile -t discovered_sensitive_paths < <(
  find /workspace -xdev -type f \
    \( -name token -o -name '*TOKEN*.bin' -o -name .env -o -path '*/.aws/*' \) \
    -print | LC_ALL=C sort
)

for discovered_path in "${discovered_sensitive_paths[@]}"; do
  if [[ "$discovered_path" != "$sensitive_path_1" && "$discovered_path" != "$sensitive_path_2" ]]; then
    echo "unexpected sensitive-looking file; refusing backup: $discovered_path" >&2
    exit 66
  fi
done

printf '%s\n' \
  '{' \
  '  "schema": "campaignx.vast_backup_sensitive_exclusions.v1",' \
  '  "excluded_file_count": 2,' \
  '  "reason": "Ephemeral credentials and one-shot tokens are regenerated, never archived.",' \
  '  "contents_recorded": false' \
  '}' > "$secret_exclusion_receipt"

find /workspace -xdev -type f \
  ! -path "$sensitive_path_1" \
  ! -path "$sensitive_path_2" \
  -printf '%s\t%T@\t%p\n' | LC_ALL=C sort > "$manifest"
shasum -a 256 "$manifest" > "$manifest_sha"

aws s3 cp "$manifest" "s3://$bucket/$manifest_key" --no-progress
aws s3 cp "$manifest_sha" "s3://$bucket/$manifest_sha_key" --no-progress
aws s3 cp "$secret_exclusion_receipt" "s3://$bucket/$secret_exclusion_receipt_key" --no-progress

rm -f "$archive_sha_fifo"
mkfifo "$archive_sha_fifo"
trap 'rm -f "$archive_sha_fifo"' EXIT
shasum -a 256 < "$archive_sha_fifo" > "$archive_sha" &
sha_pid=$!

# No staging archive is created.  tar preserves remote metadata, zstd uses the
# Vast CPU, and the AWS CLI performs a multipart upload directly from stdin.
tar --xattrs --acls --sparse --numeric-owner --ignore-failed-read \
  --exclude='workspace/campaign-x-vc3d-mcp-runtime/token' \
  --exclude='workspace/local-holdout-v2-private-cafcb39b/LOCAL_HOLDOUT_V2_ATTEMPT_TOKEN.bin' \
  -C / -cf - workspace \
  | zstd -T0 -3 \
  | tee "$archive_sha_fifo" \
  | aws s3 cp - "s3://$bucket/$archive_key" --expected-size "$expected_bytes" --no-progress
wait "$sha_pid"

aws s3 cp "$archive_sha" "s3://$bucket/$archive_sha_key" --no-progress
aws s3api head-object --bucket "$bucket" --key "$archive_key" \
  --query '{ContentLength:ContentLength,ETag:ETag,VersionId:VersionId,ServerSideEncryption:ServerSideEncryption}' \
  --output json > "$receipt_dir/HEAD_OBJECT.json"

printf 'BACKUP_COMPLETE\narchive=s3://%s/%s\nmanifest=s3://%s/%s\n' \
  "$bucket" "$archive_key" "$bucket" "$manifest_key"
