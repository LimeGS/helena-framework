#!/bin/sh
# Build a relocatable Linux VC3D/MCP runtime from already-verified binaries.
# The destination host supplies only glibc, the NVIDIA driver and CUDA driver --
# and a python3, if the MCP component is the source one (see below).
#
# The MCP argument takes either form:
#
#   an executable   the original native vc_mcp_server, packaged with its ldd
#                   closure like the other two. Its source is gone: it cannot be
#                   rebuilt, only copied from a bundle that already has it.
#   a directory     the Python server that replaced it -- stdlib only, and
#                   already what every worker in this fleet actually runs. Point
#                   this at framework/stages/01-segmentation/mcp.
#
# The second form is what makes this buildable from a checkout. grow and render
# come out of the villa image, which is compiled from a pinned commit; the MCP
# was the one input that could not be produced, and it had already been
# reimplemented -- the image build just kept asking for the binary.
set -eu

if [ "$#" -ne 4 ]; then
  echo "usage: build_vc3d_runtime_bundle.sh OUTPUT_TGZ VC_GROW_BINARY MCP VC_RENDER_TIFXYZ" >&2
  echo "  MCP is the native vc_mcp_server, or the directory holding server.py" >&2
  exit 2
fi

output="$1"
grow="$(readlink -f "$2")"
mcp="$(readlink -f "$3")"
renderer="$(readlink -f "$4")"
[ -x "$grow" ] || { echo "grow binary is not executable: $grow" >&2; exit 2; }
[ -x "$renderer" ] || { echo "renderer binary is not executable: $renderer" >&2; exit 2; }
if [ -d "$mcp" ]; then
  mcp_kind=source
  for required in server.py seed_candidates.py; do
    [ -f "$mcp/$required" ] || {
      echo "$mcp is a directory but has no $required; it is not the MCP source" >&2
      exit 2
    }
  done
elif [ -x "$mcp" ]; then
  mcp_kind=native
else
  echo "MCP is neither an executable nor a directory holding server.py: $mcp" >&2
  exit 2
fi

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
# The source MCP needs no library closure: it is stdlib-only Python, and
# server.py puts its own directory on sys.path, so the two files travel together
# and resolve each other wherever the bundle is unpacked. The launcher has the
# same shape as the native ones -- callers exec bin/vc_mcp_server either way and
# cannot tell, which is the point.
add_source_component() {
    component="$1"
    name="$2"
    source_dir="$3"
    component_root="$stage/libexec/$component"
    mkdir -p "$component_root"
    # Only the sources. __pycache__ carries paths from the build host and would
    # make two bundles from identical inputs differ.
    for file in "$source_dir"/*.py; do
      [ -f "$file" ] || continue
      cp -L "$file" "$component_root/"
    done
    chmod 644 "$component_root"/*.py
    # What it needs that the bundle cannot carry. numpy is a compiled extension:
    # packaging it relocatably is a different and much worse problem than saying
    # it is required, so the bundle declares it and whoever unpacks installs it.
    # Without this the image builds, --help answers, and the first seed search
    # dies on ModuleNotFoundError.
    if [ -f "$source_dir/requirements.txt" ]; then
      cp -L "$source_dir/requirements.txt" "$stage/PYTHON_REQUIREMENTS"
      chmod 644 "$stage/PYTHON_REQUIREMENTS"
    fi
    printf '%s\n' \
      '#!/bin/sh' \
      'root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"' \
      "exec \"\${VC3D_PYTHON:-python3}\" \"\$root/libexec/$component/server.py\" \"\$@\"" \
      > "$stage/bin/$name"
    chmod 755 "$stage/bin/$name"
}

add_component grow vc_grow_seg_from_seed "$grow" "${VC3D_BUNDLE_GROW_LIBRARY_PATH:-}"
if [ "$mcp_kind" = native ]; then
  add_component mcp vc_mcp_server "$mcp" "${VC3D_BUNDLE_MCP_LIBRARY_PATH:-}"
else
  add_source_component mcp vc_mcp_server "$mcp"
fi
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
# v4 is not a better v3, it is a different requirement: the host now needs a
# python3 as well as glibc. A consumer that cannot promise one has to be able to
# tell the two apart before it unpacks, and the schema line is where it looks.
if [ "$mcp_kind" = native ]; then
  printf '%s\n' "campaignx.vc3d_runtime_bundle.v3" > "$stage/BUNDLE_SCHEMA"
else
  printf '%s\n' "campaignx.vc3d_runtime_bundle.v4" > "$stage/BUNDLE_SCHEMA"
fi
mkdir -p "$(dirname "$output")"
tar -C "$stage" -czf "$output" .
sha256sum "$output"
