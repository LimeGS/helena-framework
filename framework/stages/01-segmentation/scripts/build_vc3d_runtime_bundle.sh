#!/bin/sh
# Build a relocatable Linux VC3D/MCP runtime from already-verified binaries.
# The destination host supplies only glibc, the NVIDIA driver and CUDA driver.
set -eu

if [ "$#" -ne 4 ]; then
  echo "usage: build_vc3d_runtime_bundle.sh OUTPUT_TGZ VC_GROW_BINARY VC_MCP_SERVER VC_RENDER_TIFXYZ" >&2
  exit 2
fi

output="$1"
grow="$(readlink -f "$2")"
mcp="$(readlink -f "$3")"
renderer="$(readlink -f "$4")"
[ -x "$grow" ] || { echo "grow binary is not executable: $grow" >&2; exit 2; }
[ -x "$mcp" ] || { echo "MCP binary is not executable: $mcp" >&2; exit 2; }
[ -x "$renderer" ] || { echo "renderer binary is not executable: $renderer" >&2; exit 2; }

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT HUP INT TERM
mkdir -p "$stage/bin" "$stage/libexec"

collect_dependencies() {
  binary="$1"
  library_path="$2"
  if [ -n "$library_path" ]; then
    env LD_LIBRARY_PATH="$library_path" ldd "$binary"
  else
    ldd "$binary"
  fi | awk '
    /=> \/[^ ]+/ { print $3 }
    /^[[:space:]]*\// { print $1 }
  '
}

add_component() {
  component="$1"
  name="$2"
  binary="$3"
  library_path="$4"
  component_root="$stage/libexec/$component"
  component_lib="$stage/lib/$component"
  mkdir -p "$component_root" "$component_lib"
  cp -L "$binary" "$component_root/$name"
  if [ -n "$library_path" ]; then
    unresolved="$(env LD_LIBRARY_PATH="$library_path" ldd "$binary" | grep 'not found' || true)"
  else
    unresolved="$(ldd "$binary" | grep 'not found' || true)"
  fi
  if [ -n "$unresolved" ]; then
    printf '%s\n' "$unresolved" >&2
    echo "unresolved dependency for $component" >&2
    exit 3
  fi
  collect_dependencies "$binary" "$library_path" | sort -u > "$stage/dependencies-$component.txt"
  while IFS= read -r dependency; do
    [ -f "$dependency" ] || continue
    base="$(basename "$dependency")"
    case "$base" in
      ld-linux-*|libc.so.*|libm.so.*|libdl.so.*|libpthread.so.*|librt.so.*)
        continue
        ;;
    esac
    destination="$component_lib/$base"
    if [ -e "$destination" ]; then
      old="$(sha256sum "$destination" | awk '{print $1}')"
      new="$(sha256sum "$dependency" | awk '{print $1}')"
      [ "$old" = "$new" ] || {
        echo "dependency basename collision inside $component: $base" >&2
        exit 3
      }
      continue
    fi
    cp -L "$dependency" "$destination"
  done < "$stage/dependencies-$component.txt"

  printf '%s\n' \
    '#!/bin/sh' \
    'root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"' \
    "exec env LD_LIBRARY_PATH=\"\$root/lib/$component\" \"\$root/libexec/$component/$name\" \"\$@\"" \
    > "$stage/bin/$name"
  chmod 755 "$stage/bin/$name" "$component_root/$name"
}

# A bundle can combine tools compiled on different compatible Linux builders.
# Their libraries must never share one flat directory: libvc_core.so and other
# ABI-sensitive names may differ while all three tools remain independently
# valid. Callers can provide the library roots needed to resolve each input.
add_component grow vc_grow_seg_from_seed "$grow" "${VC3D_BUNDLE_GROW_LIBRARY_PATH:-}"
add_component mcp vc_mcp_server "$mcp" "${VC3D_BUNDLE_MCP_LIBRARY_PATH:-}"
add_component render vc_render_tifxyz "$renderer" "${VC3D_BUNDLE_RENDER_LIBRARY_PATH:-}"

printf '%s\n' \
  '#!/bin/sh' \
  'VC3D_RUNTIME_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"' \
  'export VC3D_RUNTIME_ROOT' \
  'export VC3D_GROW_BINARY="$VC3D_RUNTIME_ROOT/bin/vc_grow_seg_from_seed"' \
  'export VC_MCP_SERVER_BINARY="$VC3D_RUNTIME_ROOT/bin/vc_mcp_server"' \
  'export VC3D_RENDER_BINARY="$VC3D_RUNTIME_ROOT/bin/vc_render_tifxyz"' \
  > "$stage/activate.sh"
chmod 755 "$stage/activate.sh" "$stage/bin/"*
(
  cd "$stage"
  find bin lib libexec -type f -print | LC_ALL=C sort | xargs sha256sum > SHA256SUMS
)
printf '%s\n' "campaignx.vc3d_runtime_bundle.v3" > "$stage/BUNDLE_SCHEMA"
mkdir -p "$(dirname "$output")"
tar -C "$stage" -czf "$output" .
sha256sum "$output"
