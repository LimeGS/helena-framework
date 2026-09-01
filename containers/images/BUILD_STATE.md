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

## helena-villa-python — WORKING

Verified 2026-08-23 on gpu-1. 6.28 GB.

    make build-villa-python \
      VILLA_PYTHON_BASE_IMAGE=<minimal base, sha256-pinned> \
      VILLA_SRC=<checkout at the volume_cartographer lock commit> \
      UV_CONTEXT=<directory containing a uv binary>

The spiral fitter (P1) and lasagna (P3): the villa Python that runs on a GPU,
from the same commit `build-villa` compiles the binaries out of. They share one
image because lasagna's only third-party imports are torch and numpy, which the
spiral project's own declared dependencies already bring; two would duplicate a
multi-gigabyte torch layer to no end.

Not in `helena-villa`, for the reason that image gives about its own toolchain:
it is a runtime, and a compiled binary that never imports torch should not
carry it.

The base is minimal, not CUDA. uv installs the interpreter -- the spiral project
declares `requires-python >= 3.14` -- and PyPI's linux torch wheels carry their
own CUDA runtime as `nvidia-*` packages. `build-essential` is installed in the
build stage only: `posix-ipc` ships no wheel and compiles from source, and the
first attempt failed on a base with no `cc`.

Checked at build time rather than at run time:

    source lock verified for 3 files
    spiral: default_config present; shared deps installed
    lasagna: fit.py compiles

The `default_config` check is the one that matters. The spiral adapter parses
that dict out of upstream's source to validate override keys, so if upstream
ever stops declaring it the adapter refuses every run -- and this fails the
build instead of discovering it in production.

## the spiral lane, inside helena-worker-cpp — WORKING

The same shape as the 9 um lane above and gone for the same reason: it was a
second full copy of the worker with one directory added. `Containerfile.worker-cpp`
carries it at `/opt/lanes/spiral` under its `with_lane` target, and
`containers/build-worker.sh` picks the target by whether the lane image is here.

    docker build --target with_lane \
      --build-arg BASE_IMAGE=<the built helena-villa> \
      --build-arg LANE_IMAGE=<the built helena-villa-python> \
      -f containers/images/Containerfile.worker-cpp -t helena-worker-cpp:local .

## the 9 um lane, inside helena-worker-gpu — WORKING

There is no separate worker image for this lane any more. It was 16.5 GB
against the 7.83 GB worker it was built from, differing by one directory, and
the two lanes could not coexist because both landed at `/opt/villa`.

`Containerfile.worker-gpu` has two targets now: `runtime`, and `with_lane`
which copies the lane to `/opt/lanes/ink-9um`. `deploy-platform.sh` chooses by
whether the lane image is on the host, so a host without it never needs it.

    docker build --target with_lane \
      --build-arg BASE_IMAGE=<the built helena-gpu-runtime> \
      --build-arg LANE_IMAGE=<the built helena-ink-9um> \
      -f containers/images/Containerfile.worker-gpu -t helena-worker-gpu:local .

Two environments in one image. `helena-ink-9um` cannot claim a job -- it has no
psycopg and no boto3 -- and adding them means installing on top of a `uv sync
--frozen` lock, which spends the exact property that lock exists for. So the
lane image's `/opt/villa` is copied onto the worker image and the adapter is
pointed at it with `HELENA_INK_9UM_PYTHON`. The same shape `helena-gpu-runtime`
already is.

Both halves are checked at build time, because the failure mode is a worker
that starts, claims a P5 job, and only then discovers it cannot run it.

`HELENA_RUNTIME_IMAGE` for this worker is `helena-ink-9um`, not the composed
name: the lane declares which image it needs, and this worker carries it.

Verified through the queue on 2026-08-26, which is the check that matters:
`p5-b7cd63a5e68f4c`, enqueued through the panel's API with the general ink
worker polling the same queue, claimed by this image and run to `succeeded` in
1,557 s -- 21 layers pooled to a ~9.6 um isotropic OME-Zarr, 10,885 blocks of
inference, `ALIVE` at p50 0.2666666805744171 over 49,126,400 valid pixels, and
the map published to `/artifacts/ink-maps-v1/...` with its manifest.

Those figures are identical to the digit to the same lane run directly, outside
Helena, on the same input. That is what makes this a check on the integration
rather than on the model: same bytes in, same bytes out, by a different route.

Building it is what found the flaw below, and running it found three more --
a worker that burned a job it could not run, a pooling step in the wrong
interpreter, and an `--output` that meant a file where every other P5 lane
means a directory. None of the three were reachable from a unit test: they all
live in the joint between the worker, the queue and the image.

### Why the 9 um venv is interpreter-pinned and self-contained

`uv sync` left `/opt/villa/ink-venv/bin/python` a symlink to
`/usr/local/bin/python3.11` -- the base image's interpreter, which uv preferred
because it already satisfied `requires-python`. Correct in that image and a
dangling symlink the moment `/opt/villa` is copied onto any other base, which is
what composition does. `UV_PYTHON_PREFERENCE=only-managed` puts the interpreter
under `UV_PYTHON_INSTALL_DIR`, so it travels with the tree that references it.

`UV_PYTHON=3.11` pins the minor version, because only-managed otherwise takes
the newest interpreter `requires-python` allows and this lock was resolved
against 3.11: on 3.14 several pinned wheels have no matching tag, uv falls back
to building them, and the build fails for want of a compiler -- a failure whose
obvious "fix" is a toolchain in a runtime image. The lock decides the
interpreter.

A build-time check now refuses a venv whose interpreter resolves outside
`/opt/villa`, so this cannot regress into a composed image that fails at run
time instead.

## helena-ink-9um — WORKING

Verified 2026-08-23 on gpu-1. 8.7 GB.

    make build-ink-9um \
      VILLA_PYTHON_BASE_IMAGE=<minimal base, sha256-pinned> \
      VILLA_INK_SRC=<checkout at the villa_ink_detection lock commit> \
      UV_CONTEXT=<directory containing a uv binary>

`koine_machines.inference.infer`, the runner the 9 um ink lane profile names.
A *second* villa revision: it does not exist at the volume_cartographer commit
and lives on the merge-ink-pipelines branch, so the source lock pins that commit
separately and `VILLA_INK_SRC` is a separate checkout. Pointing it at `VILLA_SRC`
builds an image whose runner is not the one the profile names.

Why this is its own image rather than a lane inside `helena-ink`, which is where
an ink model would normally go: models are runtime mounts here, and a new model
does not earn an image. This is not a new model, it is a new dependency set.
Measured 2026-08-23:

    helena-ink runtime      torch 2.5.1+cu124, Python 3.11.10
    ink-detection's lock    torch 2.10.0

Five major versions apart, and the lock is installed `--frozen` on purpose --
loosening it to make one environment hold both is precisely what the method
registry records going wrong before, when a recipe run by the wrong runner
produced a map correlating r=0.079 against r=0.885.

The build context is the checkout root, not `ink-detection`: that tree declares
`vesuvius = { path = "../vesuvius", editable = true }`, and a context holding
only `ink-detection` cannot satisfy its own lock. `--frozen` refusing there was
the lock doing its job.

Checked at build time:

    source lock verified for 3 files
    9 um runner starts

The checkpoint is not in the image. It is 138 MB, one of fourteen upstream
publishes, and which one ran belongs in a receipt rather than in an image tag --
so it arrives on the `/models` mount like every other checkpoint.
