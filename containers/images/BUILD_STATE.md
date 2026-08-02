# What builds, and what does not

Recorded here rather than in the registries: those carry a committed schema and
a field invented to hold this would break it. The registries say what a method
*is*; this says what happened the last time somebody tried to build it.

Verified 2026-07-26.

## helena-villa — WORKING

`make build-villa`. 2.32 GB, 47 apps, volume-cartographer at the commit the
ScrollFiesta source lock pins.

All four binaries the framework names start, and `ldd` reports zero unresolved
sonames:

    vc_grow_seg_from_seed   vc_render_tifxyz
    vc_obj2tifxyz_legacy    flatboi

No `LD_LIBRARY_PATH` wrapper. Upstream sets an install RPATH of `$ORIGIN/../lib`
and the image preserves that relationship, which is what makes the difference
from the hand-assembled runtime on the host: that one was copied without its
libraries and cannot start without a wrapper script.

This image is also where `build-scrollfiesta-src` gets `flatboi` and
`vc_obj2tifxyz_legacy`, which the ScrollFiesta native bundle takes as inputs and
cannot produce.

## helena-scrollfiesta — BLOCKED ON ITS OWN TESTS

Everything up to the test suite works. The source lock verifies against the
pinned commit, the Villa runtime packages cleanly, and the native code compiles.
Then ctest fails 2 of 3, and the failures are not about packaging:

    sf_api_dlopen:
      FAIL: audit no non-manifold edges
      FAIL: audit one boundary loop
      FAIL: audit is a disk
      FAIL: cleanup keeps verts
      FAIL: cleanup keeps faces
      FAIL: cleanup provenance is identity on a clean sheet

Six mesh-topology correctness checks, on a clean test sheet, returning wrong
answers. The build is left failing. Skipping the tests would produce an image
that runs, for a backend already carrying `FAILED_REFERENCE_CONTROL`, whose own
geometric checks disagree with it -- which is the exact shape of thing this
framework exists to refuse.

The likely cause is the toolchain, not the code: this builds on the Villa image,
which is Ubuntu 26.04 with gcc 15 because that is what upstream
volume-cartographer's own Dockerfile pins, while `scrollfiesta/README.md`
describes an Ubuntu 24.04 base. Resolving it means either building Villa on
24.04 as well, or establishing that gcc 15 is sound for this code. Both are
decisions about what to trust, so neither is made here.

## helena-thaumato — RECIPE WRITTEN, BUILD NOT YET COMPLETED

Read `Containerfile.thaumato` before building it. Upstream moved
ThaumatoAnakalyptor to `villa/deprecated/`, and the registry's
`UPSTREAM_ONLY_NOT_LOCALLY_VALIDATED` is right. The image exists to make it
comparable, not to adopt it.

Their `DockerfileThaumato` cannot be repeated -- Miniconda `latest`, three
build-time clones, eight CUDA architectures. Pinned and checksummed here, with
the arch list defaulting to the one card this fleet has.
