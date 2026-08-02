#!/usr/bin/env bash
# Build a same-platform, immutable ScrollFiesta runtime bundle for internal
# Helena Framework research. This script never fetches source. `uv` may download
# only artifacts admitted by the hash-bearing frozen lock unless UV_OFFLINE=1.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: build_native_bundle.sh --scrollfiesta-source DIR --villa-runtime DIR --output DIR [--jobs N]

Prerequisites:
  * frozen ScrollFiesta checkout already materialized locally;
  * uv, CMake, C/C++ compiler, OpenMP and system libtiff;
  * a separately verified Villa runtime directory containing
    VILLA_RUNTIME_MANIFEST.json, flatboi and vc_obj2tifxyz_legacy.

The output path must not exist. No CT, m7, model, credential or scientific
result belongs in either input or output.
EOF
  exit 2
}

sf_source=""
villa_runtime=""
output=""
jobs="${HELENA_BUILD_JOBS:-2}"
while (($#)); do
  case "$1" in
    --scrollfiesta-source) sf_source="$2"; shift 2 ;;
    --villa-runtime) villa_runtime="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --jobs) jobs="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n "$sf_source" && -n "$villa_runtime" && -n "$output" ]] || usage
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs must be a positive integer" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "output already exists: $output" >&2; exit 2; }
output_parent="$(dirname "$output")"
[[ -d "$output_parent" ]] || { echo "output parent does not exist: $output_parent" >&2; exit 2; }
output="$(cd "$output_parent" && pwd)/$(basename "$output")"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="$(cd "$script_dir/.." && pwd)"
lock="$runtime_dir/locks/source-lock.json"
license_inventory="$runtime_dir/licenses/license-inventory.json"
python_bin="${PYTHON:-python3}"

for command in git cmake ctest patch uv tar "$python_bin"; do
  command -v "$command" >/dev/null || { echo "missing build command: $command" >&2; exit 2; }
done

tmp="$(mktemp -d "${TMPDIR:-/tmp}/helena-scrollfiesta-build.XXXXXX")"
stage="$(mktemp -d "${output}.staging.XXXXXX")"
trap 'rm -rf "$tmp" "$stage"' EXIT
mkdir -p "$tmp/receipts"
"$python_bin" "$script_dir/verify_source_lock.py" \
  --source "$sf_source" --lock "$lock" --output "$tmp/receipts/SOURCE_VERIFICATION.json"
"$python_bin" "$script_dir/verify_villa_runtime.py" \
  --root "$villa_runtime" --output "$tmp/receipts/VILLA_VERIFICATION.json"

# Use git archive so untracked files, credentials and previous build outputs
# can never leak from the source checkout into the bundle.
source_copy="$tmp/source"
mkdir "$source_copy"
git -C "$sf_source" archive --format=tar HEAD | tar -xf - -C "$source_copy"
for patch_file in "$runtime_dir"/patches/*.patch; do
  patch --batch --forward -d "$source_copy" -p1 < "$patch_file"
done
"$python_bin" - "$runtime_dir/patches" "$stage/PATCH_MANIFEST.json" <<'PY'
import hashlib, json, pathlib, sys
patch_dir, output = map(pathlib.Path, sys.argv[1:])
patches = []
for path in sorted(patch_dir.glob("*.patch")):
    patches.append({
        "name": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
output.write_text(json.dumps({
    "schema": "campaignx.scrollfiesta_patch_manifest.v1",
    "patches": patches,
}, indent=2, sort_keys=True) + "\n")
PY

cmake -S "$source_copy" -B "$tmp/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DSCROLLFIESTA_BUILD_TOOLS=ON \
  -DSCROLLFIESTA_BUILD_TESTS=ON \
  -DSCROLLFIESTA_WITH_TIFF=ON \
  -DSCROLLFIESTA_OPENMP=ON \
  -DSCROLLFIESTA_INSTALL=ON \
  -DCMAKE_INSTALL_PREFIX="$stage"
cmake --build "$tmp/build" --parallel "$jobs"
ctest --test-dir "$tmp/build" --output-on-failure
cmake --install "$tmp/build"

# The upstream Python lock is authoritative. A venv is a same-platform native
# build/test environment. Runtime packages are installed into a relocatable
# target directory for the exact Python ABI recorded in the receipt.
uv sync --project "$source_copy/python" --locked
uv run --project "$source_copy/python" pytest \
  "$source_copy/python/tests" \
  -m "not network" \
  --ignore "$source_copy/python/tests/test_network.py"
uv export --project "$source_copy/python" --locked --no-dev --no-emit-project \
  --output-file "$tmp/runtime-requirements.txt"

mkdir -p "$stage/bin" "$stage/python-packages" "$stage/share/licenses/scrollfiesta"
uv pip install --python "$python_bin" --target "$stage/python-packages" \
  --require-hashes --requirements "$tmp/runtime-requirements.txt"
cp -a "$source_copy/python/src/scrollunwrap" "$stage/python-packages/scrollunwrap"
cp "$tmp/runtime-requirements.txt" "$stage/PYTHON_REQUIREMENTS.lock"
for tool in flatboi vc_obj2tifxyz_legacy; do
  relative="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["artifacts"][sys.argv[2]]["path"])' "$villa_runtime/VILLA_RUNTIME_MANIFEST.json" "$tool")"
  cp "$villa_runtime/$relative" "$stage/bin/$tool"
  chmod 0555 "$stage/bin/$tool"
done
cat > "$stage/bin/scrollunwrap" <<'EOF'
#!/bin/sh
set -eu
root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
export SCROLLFIESTA_CUBE_MESH="$root/bin/cube_mesh"
export SCROLLFIESTA_GRID_WELD="$root/bin/grid_weld"
export PYTHONPATH="$root/python-packages${PYTHONPATH:+:$PYTHONPATH}"
exec "${SCROLLFIESTA_PYTHON:-python3}" -m scrollunwrap.cli "$@"
EOF
chmod 0555 "$stage/bin/scrollunwrap"
rm -f "$stage/bin/grid_pipeline"
cp "$source_copy/LICENSE" "$stage/share/licenses/scrollfiesta/LICENSE"
cp "$source_copy/THIRD_PARTY_LICENSES.md" "$stage/share/licenses/scrollfiesta/THIRD_PARTY_LICENSES.md"
cp "$lock" "$stage/SOURCE_LOCK.json"
cp "$license_inventory" "$stage/LICENSE_INVENTORY.json"
cp "$tmp/receipts/SOURCE_VERIFICATION.json" "$stage/SOURCE_VERIFICATION.json"
cp "$tmp/receipts/VILLA_VERIFICATION.json" "$stage/VILLA_VERIFICATION.json"
printf '%s\n' 'campaignx.scrollfiesta_runtime_bundle.v1' > "$stage/BUNDLE_SCHEMA"

"$python_bin" - "$stage" "$tmp/build" "$lock" <<'PY'
import hashlib, json, os, pathlib, platform, subprocess, sys
root, build, lock_path = map(pathlib.Path, sys.argv[1:])
def output(*cmd):
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout.strip()
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
cache = (build / "CMakeCache.txt").read_text()
c_compiler = next(line.split("=", 1)[1] for line in cache.splitlines() if line.startswith("CMAKE_C_COMPILER:FILEPATH="))
receipt = {
    "schema": "campaignx.scrollfiesta_runtime_build_receipt.v1",
    "status": "BUILT_AND_TESTED",
    "scientific_data_used": False,
    "distribution": "INTERNAL_RESEARCH_ONLY",
    "platform": platform.platform(),
    "machine": platform.machine(),
    "cmake_version": output("cmake", "--version").splitlines()[0],
    "compiler": output(c_compiler, "--version").splitlines()[0],
    "python_version": platform.python_version(),
    "python_abi": sys.implementation.cache_tag,
    "source_lock_sha256": sha(lock_path),
    "patch_manifest_sha256": sha(root / "PATCH_MANIFEST.json"),
    "source_verification_sha256": sha(root / "SOURCE_VERIFICATION.json"),
    "villa_verification_sha256": sha(root / "VILLA_VERIFICATION.json"),
    "tests": ["ctest:PASS", "python-non-network:PASS"],
    "required_commands": ["cube_mesh", "grid_weld", "flatboi", "vc_obj2tifxyz_legacy"],
    "forbidden_inputs": ["CT", "m7", "models", "credentials", "scientific_results"],
}
(root / "BUILD_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY
"$python_bin" "$script_dir/generate_sbom.py" \
  --root "$stage" --source-lock "$lock" --output "$stage/SBOM.spdx.json"

"$python_bin" "$script_dir/generate_sha256s.py" \
  --root "$stage" --output "$stage/SHA256SUMS"
"$python_bin" "$script_dir/verify_runtime_bundle.py" --root "$stage"
chmod -R a-w "$stage"
mv "$stage" "$output"
trap - EXIT
rm -rf "$tmp"
printf 'SCROLLFIESTA_NATIVE_BUNDLE_READY %s\n' "$output"
