# ScrollFiesta runtime package

This directory implements **I1 only** of the approved ScrollFiesta + VC3D
hybrid plan. It packages ScrollFiesta as an optional geometry backend while
VC3D remains the active backend. A successful build is a technical result; it
is not scientific evidence and does not authorize replacing VC3D.

## Frozen inputs

- ScrollFiesta repository commit
  `f344c17931b9e264a17c8d760a4c478390133bd4` (`0.9.0` native,
  `scrollunwrap==0.1.0`).
- Python `uv.lock` SHA-256
  `62ef615c57d76670228a77e98ceea407e244ccd778ccd97802193ac20e6d4a63`.
- Volume Cartographer commit
  `05dcf0349356bc833670d61e5eca00be58376e35` for `flatboi` and
  `vc_obj2tifxyz_legacy`.
- Every remaining tree and file identity is in `locks/source-lock.json`.

The existing sparse Helena Framework Villa checkout cannot build the two external
tools. Therefore I1 accepts a separately built Villa runtime only when its
`VILLA_RUNTIME_MANIFEST.json` names the frozen commit and carries the exact
SHA-256 of both executables. Paths are always passed explicitly; upstream
sibling-directory defaults are forbidden.

To package those already-compiled Linux tools, first produce a build receipt
with schema `campaignx.villa_toolchain_receipt.v1`, the frozen source
commit/tree, absolute `build_root`, and non-empty `c_compiler`, `cxx_compiler`,
`cmake_version`, `build_type` and `build_command` fields. Then run:

```bash
python3 containers/images/scrollfiesta/scripts/package_villa_runtime.py \
  --villa-source /absolute/villa-05dcf034 \
  --build-root /absolute/villa-build \
  --flatboi /absolute/villa-build/bin/flatboi \
  --obj2tifxyz /absolute/villa-build/bin/vc_obj2tifxyz_legacy \
  --toolchain-receipt /absolute/VILLA_TOOLCHAIN_RECEIPT.json \
  --source-lock containers/images/scrollfiesta/locks/source-lock.json \
  --output /absolute/new/villa-runtime
```

The packager runs `/usr/bin/ldd` on both executables, rejects missing
libraries and any dependency resolved inside the source or build trees, copies
only the two tools, emits hashes plus the toolchain/linkage receipt, and
atomically publishes a read-only output. It never overwrites an existing path.

## License gate

This runtime is **internal research only**. ScrollFiesta is MIT, but Triangle
restricts commercial distribution and the vendored `andres/graph` copy does
not carry a license text. Consequently:

- local research builds are allowed with all notices preserved;
- the image and native bundle must not be pushed, published, sold or included
  in a commercial system;
- an OCI recipe does not resolve binary redistribution rights;
- `licenses/license-inventory.json` and `SBOM.spdx.json` are mandatory.

## Build a native bundle

The builder never fetches source: materialize the two frozen source/runtime
inputs first, then run. `uv` may fetch only Python artifacts admitted by the
hash-bearing lock; set `UV_OFFLINE=1` when a prefilled cache is required:

```bash
containers/images/scrollfiesta/scripts/build_native_bundle.sh \
  --scrollfiesta-source /absolute/frozen/scrollfiesta \
  --villa-runtime /absolute/verified/villa-runtime \
  --output /absolute/new/scrollfiesta-runtime-0.1.0 \
  --jobs 8
```

The output path must not exist. The builder:

1. verifies the Git commit/tree, dependency trees and source hashes;
2. copies only `git archive HEAD`, then applies the one-line `<stdlib.h>`
   compatibility patch;
3. builds with TIFF, OpenMP, the shared C API and upstream tests enabled;
4. runs CTest and all non-network Python tests;
5. verifies and incorporates the two frozen Villa executables;
6. writes locked dependencies and `scrollunwrap` into a relocatable target
   directory for the exact Python ABI recorded in the build receipt;
7. removes `grid_pipeline`, which is deliberately not an I1 dependency;
8. generates receipts, notices, SPDX SBOM and full `SHA256SUMS`;
9. verifies and makes the completed bundle read-only before atomic publish.

The native bundle is a same-OS/architecture fallback for hosts that block
container namespaces. It is not a universal binary archive.

## Build the internal OCI image

Create a sterile named context:

```bash
python3 containers/images/scrollfiesta/scripts/make_oci_context.py \
  --bundle /absolute/scrollfiesta-runtime-0.1.0 \
  --output /absolute/new/scrollfiesta-oci-context
```

Then build with a base reference pinned by digest:

```bash
docker build \
  --file containers/images/Containerfile.scrollfiesta \
  --build-context scrollfiesta_runtime=/absolute/scrollfiesta-oci-context \
  --build-arg SCROLLFIESTA_BASE_IMAGE='registry/repo@sha256:<64 hex>' \
  --build-arg BASE_IMAGE='registry/repo@sha256:<64 hex>' \
  --tag helena-scrollfiesta:0.1.0 \
  containers/images
```

Use `make build-scrollfiesta` from `containers/images` in normal operation.
The image context contains only the verified tarball. Repository code, CT, m7,
models, credentials and outputs are runtime mounts and never image layers.

## Runtime contract

The scientific adapter must pass all executable paths explicitly:

```text
SCROLLFIESTA_CUBE_MESH=/opt/campaignx/scrollfiesta/bin/cube_mesh
SCROLLFIESTA_GRID_WELD=/opt/campaignx/scrollfiesta/bin/grid_weld
SCROLLFIESTA_FLATBOI=/opt/campaignx/scrollfiesta/bin/flatboi
SCROLLFIESTA_OBJ2TIFXYZ=/opt/campaignx/scrollfiesta/bin/vc_obj2tifxyz_legacy
```

`scrollunwrap` streams public Zarr inputs, so its service may receive network
access at runtime. No cloud credential is embedded; public anonymous access or
a separately mounted short-lived credential is an operations decision. Every
scientific run still requires a new immutable output prefix and the fail-closed
I2 adapter.

## What I1 does not prove

- It does not validate ZYX → XYZ conversion or TIFXYZ coordinates; that is the
  mandatory I2/I3 asymmetric fixture.
- It does not prove topology, correct sheet identity, usable flattening or
  text.
- It does not allow ScrollFiesta output to be fused automatically with VC3D.
- It does not change the active VC3D backend.
