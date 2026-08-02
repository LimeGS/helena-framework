#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <libigl-root> <source.cpp> <output-binary>" >&2
  exit 2
fi

libigl_root=$(cd "$1" && pwd)
source_file=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
output_file=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")

g++ -std=c++17 -O2 -DNDEBUG \
  -I"$libigl_root/include" \
  -I/usr/include/eigen3 \
  "$source_file" \
  -lgmp -lmpfr \
  -o "$output_file"

usage_output=$({ "$output_file" 2>&1 || test $? -eq 2; })
grep -Fq 'usage: helena_mesh_self_intersection' <<<"$usage_output"
sha256sum "$output_file"
