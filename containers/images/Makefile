SHELL := /bin/sh
.DEFAULT_GOAL := help

BUILD_COMMIT ?= $(shell git rev-parse HEAD 2>/dev/null || printf unknown)
SOURCE_DATE_EPOCH ?= $(shell git show -s --format=%ct HEAD 2>/dev/null || printf 0)

.PHONY: help check-base-images check-scrollfiesta-base-image check-surface-qc-images check-all-images build-scrollfiesta build-ink build-qc build-orchestrator build-surface-qc build-villa build-villa-python build-ink-9um build-scrollfiesta-src build-thaumato build-worker

help:
	@printf '%s\n' 'Checks:  check-base-images check-scrollfiesta-base-image check-surface-qc-images check-all-images'
	@printf '%s\n' 'Assemble: build-scrollfiesta build-ink build-qc build-orchestrator build-surface-qc build-worker'
	@printf '%s\n' 'Compile:  build-villa build-scrollfiesta-src build-thaumato'
	@printf '%s\n' 'Villa Python: build-villa-python build-ink-9um'
	@printf '%s\n' ''
	@printf '%s\n' 'The villa-python targets need a source checkout, a uv binary and a'
	@printf '%s\n' 'sha256-pinned minimal base, and take them by name:'
	@printf '%s\n' '  build-villa-python  VILLA_SRC=      spiral fitter (P1) + lasagna (P3)'
	@printf '%s\n' '  build-ink-9um       VILLA_INK_SRC=  the 9 um ink runner'
	@printf '%s\n' 'Both also need VILLA_PYTHON_BASE_IMAGE= and UV_CONTEXT=. The two source'
	@printf '%s\n' 'variables are different checkouts at different commits: the 9 um runner'
	@printf '%s\n' 'does not exist at the volume_cartographer one. See BUILD_STATE.md.'
	@printf '%s\n' 'build-surface-qc requires INK_IMAGE and VILLA_IMAGE, both sha256-pinned.'
	@printf '%s\n' ''
	@printf '%s\n' 'The assemble targets unpack a runtime tarball somebody built elsewhere.'
	@printf '%s\n' 'The compile targets build that runtime from pinned source, so no binary'
	@printf '%s\n' 'enters the stack without a recipe. build-villa produces the VC3D apps and'
	@printf '%s\n' 'the flatboi/vc_obj2tifxyz_legacy that build-scrollfiesta-src needs.'

# One digest check for every image reference. $(1) is the variable name, so the
# error message can name the offending variable instead of only its value.
define require_digest
@printf '%s\n' "$($(1))" | grep -Eq '^[^[:space:]]+@sha256:[0-9a-f]{64}$$' || { printf '%s\n' "$(1) must be a non-empty, lowercase sha256-pinned reference (got: '$($(1))')" >&2; exit 2; }
endef

check-base-images:
	$(call require_digest,INK_BASE_IMAGE)
	$(call require_digest,QC_BASE_IMAGE)
	$(call require_digest,ORCHESTRATOR_BASE_IMAGE)

check-scrollfiesta-base-image:
	$(call require_digest,SCROLLFIESTA_BASE_IMAGE)

# Containerfile.gpu-runtime takes a base and two parent images: INK_IMAGE (the
# frozen six-replica TimeSformer environment, copied in as /opt/conda) and
# VILLA_IMAGE (the compiled toolchain, from which the three-tool bundle is
# assembled in a stage). Neither was validated anywhere before FIX-16, so this
# could be built on floating tags and the result was not reproducible.
#
# VC3D_IMAGE is gone with helena-vc3d: it existed only to be this image's other
# parent, and the tools are compiled from a pinned commit now. VILLA_COMMIT and
# VILLA_TREE are labelled onto the result, so what is auditable is the source
# rather than a digest of an image nobody published.
check-surface-qc-images:
	$(call require_digest,INK_IMAGE)
	$(call require_digest,VILLA_IMAGE)

check-all-images: check-base-images check-scrollfiesta-base-image check-surface-qc-images

define build_image
docker build --file Containerfile.$(1) --build-arg BASE_IMAGE="$($(2)_BASE_IMAGE)" --build-arg BUILD_COMMIT="$(BUILD_COMMIT)" --build-arg SOURCE_DATE_EPOCH="$(SOURCE_DATE_EPOCH)" --tag helena-$(1):local .
endef

define build_scrollfiesta_image
docker build --file Containerfile.scrollfiesta --build-context scrollfiesta_runtime="$(SCROLLFIESTA_RUNTIME_CONTEXT)" --build-arg BASE_IMAGE="$(SCROLLFIESTA_BASE_IMAGE)" --build-arg BUILD_COMMIT="$(BUILD_COMMIT)" --build-arg SOURCE_DATE_EPOCH="$(SOURCE_DATE_EPOCH)" --tag helena-scrollfiesta:local .
endef

# The repository is the context: the bundle stage needs the MCP source. It used
# to be this directory, when the tools arrived as an already-built image.
define build_surface_qc_image
docker build --file Containerfile.gpu-runtime --build-arg BASE_IMAGE="$(SURFACE_QC_BASE_IMAGE)" --build-arg CA_CERTIFICATES_VERSION="$(CA_CERTIFICATES_VERSION)" --build-arg INK_IMAGE="$(INK_IMAGE)" --build-arg VILLA_IMAGE="$(VILLA_IMAGE)" --build-arg VILLA_COMMIT="$(VILLA_COMMIT)" --build-arg VILLA_TREE="$(VILLA_TREE)" --build-arg BUILD_COMMIT="$(BUILD_COMMIT)" --build-arg SOURCE_DATE_EPOCH="$(SOURCE_DATE_EPOCH)" --tag helena-gpu-runtime:local ../..
endef

build-scrollfiesta: check-scrollfiesta-base-image
	@test -n "$(SCROLLFIESTA_RUNTIME_CONTEXT)" || { printf '%s\n' 'SCROLLFIESTA_RUNTIME_CONTEXT is required' >&2; exit 2; }
	@test -f "$(SCROLLFIESTA_RUNTIME_CONTEXT)/scrollfiesta-runtime.tgz" || { printf '%s\n' 'SCROLLFIESTA_RUNTIME_CONTEXT must contain scrollfiesta-runtime.tgz' >&2; exit 2; }
	$(build_scrollfiesta_image)

build-ink: check-base-images
	$(call build_image,ink,INK)

build-qc: check-base-images
	$(call build_image,qc,QC)

build-orchestrator: check-base-images
	$(call build_image,orchestrator,ORCHESTRATOR)

# `campaignx-surface-qc:0.1.1` ran QC in production without this target
# existing: it was built by hand, and Containerfile.gpu-runtime plus
# requirements.surface-qc.txt were excluded by .dockerignore, so the image was
# not reproducible from the repository at all.
build-surface-qc: check-surface-qc-images
	@test -f Containerfile.gpu-runtime || { printf '%s\n' 'Containerfile.gpu-runtime is missing' >&2; exit 2; }
	@test -f requirements.surface-qc.txt || { printf '%s\n' 'requirements.surface-qc.txt is missing' >&2; exit 2; }
	$(build_surface_qc_image)

# The same omission as build-surface-qc above, made again: helena-worker-cpp:local
# was running the fleet on two hosts with no target that builds it. It was put
# together by hand, so what was in it could only be discovered by looking inside
# it -- and a code change reached the hosts only if somebody remembered the same
# command. VILLA_IMAGE is sha256-pinned because this image inherits VC3D's
# binaries from it.
build-worker:
	$(call require_digest,VILLA_IMAGE)
	@test -f "$(UV_CONTEXT)/uv" || { printf '%s\n' 'UV_CONTEXT must contain a uv binary' >&2; exit 2; }
	docker build --file Containerfile.worker-cpp \
	  --build-context repo="$(CURDIR)/../.." \
	  --build-context uv_context="$(UV_CONTEXT)" \
	  --build-arg BASE_IMAGE="$(VILLA_IMAGE)" \
	  --build-arg VILLA_IMAGE_DIGEST="$(VILLA_IMAGE)" \
	  --build-arg BUILD_COMMIT="$(BUILD_COMMIT)" \
	  --build-arg SOURCE_DATE_EPOCH="$(SOURCE_DATE_EPOCH)" \
	  --tag helena-worker-cpp:local .

# ---------------------------------------------------------------------------
# Compile-from-source targets.
#
# VILLA_SRC is a checkout of ScrollPrize/villa at the commit recorded in
# scrollfiesta/locks/source-lock.json. Nothing is fetched during the build: a
# build that clones is a build whose output depends on the day it ran.
# ---------------------------------------------------------------------------

VILLA_COMMIT ?= $(shell python3 -c "import json;print(json.load(open('scrollfiesta/locks/source-lock.json'))['volume_cartographer']['commit'])" 2>/dev/null)
SCROLLFIESTA_COMMIT ?= $(shell python3 -c "import json;print(json.load(open('scrollfiesta/locks/source-lock.json'))['scrollfiesta']['commit'])" 2>/dev/null)
# The 9 um ink runner is a *second* villa revision: koine_machines does not
# exist at VILLA_COMMIT. Read separately so a build cannot quietly use one
# checkout for both.
VILLA_INK_COMMIT ?= $(shell python3 -c "import json;print(json.load(open('scrollfiesta/locks/source-lock.json'))['villa_ink_detection']['commit'])" 2>/dev/null)

define require_source_tree
@test -n "$($(1))" || { printf '%s\n' "$(1) is required: a checkout of the pinned source" >&2; exit 2; }
@test -d "$($(1))/.git" -o -d "$($(1))" || { printf '%s\n' "$(1) is not a directory: $($(1))" >&2; exit 2; }
endef

build-villa: check-base-images
	$(call require_source_tree,VILLA_SRC)
	@test -n "$(VILLA_COMMIT)" || { printf '%s\n' 'VILLA_COMMIT could not be read from the source lock' >&2; exit 2; }
	@git -C "$(VILLA_SRC)" rev-parse HEAD 2>/dev/null | grep -Fxq "$(VILLA_COMMIT)" \
	  || { printf '%s\n' "VILLA_SRC is not at the locked commit $(VILLA_COMMIT)" >&2; exit 2; }
	docker build --file Containerfile.villa \
	  --build-context villa_src="$(VILLA_SRC)" \
	  --build-arg BASE_IMAGE="$(VILLA_BASE_IMAGE)" \
	  --build-arg VILLA_COMMIT="$(VILLA_COMMIT)" \
	  --build-arg VILLA_TREE="$$(git -C "$(VILLA_SRC)" rev-parse 'HEAD^{tree}')" \
	  --build-arg BUILD_COMMIT="$(BUILD_COMMIT)" \
	  --build-arg SOURCE_DATE_EPOCH="$(SOURCE_DATE_EPOCH)" \
	  --tag campaignx-villa:local .

# The villa Python that runs on a GPU: spiral (P1) and lasagna (P3), from the
# same checkout build-villa compiles the binaries out of. One image because
# lasagna's third-party imports are torch and numpy, which the spiral project's
# own dependencies already bring; two would duplicate a multi-gigabyte layer.
build-villa-python:
	$(call require_digest,VILLA_PYTHON_BASE_IMAGE)
	$(call require_source_tree,VILLA_SRC)
	@test -n "$(VILLA_COMMIT)" || { printf '%s\n' 'VILLA_COMMIT could not be read from the source lock' >&2; exit 2; }
	@test -f "$(UV_CONTEXT)/uv" || { printf '%s\n' 'UV_CONTEXT must contain a uv binary' >&2; exit 2; }
	@git -C "$(VILLA_SRC)" rev-parse HEAD 2>/dev/null | grep -Fxq "$(VILLA_COMMIT)" \
	  || { printf '%s\n' "VILLA_SRC is not at the locked commit $(VILLA_COMMIT)" >&2; exit 2; }
	docker build --file Containerfile.villa-python \
	  --build-context villa_src="$(VILLA_SRC)" \
	  --build-context uv_context="$(UV_CONTEXT)" \
	  --build-context repo="$(CURDIR)/../.." \
	  --build-arg BASE_IMAGE="$(VILLA_PYTHON_BASE_IMAGE)" \
	  --build-arg VILLA_COMMIT="$(VILLA_COMMIT)" \
	  --build-arg VILLA_TREE="$$(git -C "$(VILLA_SRC)" rev-parse 'HEAD^{tree}')" \
	  --build-arg BUILD_COMMIT="$(BUILD_COMMIT)" \
	  --build-arg SOURCE_DATE_EPOCH="$(SOURCE_DATE_EPOCH)" \
	  --tag helena-villa-python:local .

# VILLA_INK_SRC is a *separate* checkout: the 9 um runner lives on a branch the
# volume_cartographer commit predates, so pointing this at VILLA_SRC would build
# an image whose runner is not the one the lane profile names.
build-ink-9um:
	$(call require_digest,VILLA_PYTHON_BASE_IMAGE)
	$(call require_source_tree,VILLA_INK_SRC)
	@test -n "$(VILLA_INK_COMMIT)" || { printf '%s\n' 'VILLA_INK_COMMIT could not be read from the source lock' >&2; exit 2; }
	@test -f "$(UV_CONTEXT)/uv" || { printf '%s\n' 'UV_CONTEXT must contain a uv binary' >&2; exit 2; }
	@git -C "$(VILLA_INK_SRC)" rev-parse HEAD 2>/dev/null | grep -Fxq "$(VILLA_INK_COMMIT)" \
	  || { printf '%s\n' "VILLA_INK_SRC is not at the locked commit $(VILLA_INK_COMMIT)" >&2; exit 2; }
	docker build --file Containerfile.ink-9um \
	  --build-context villa_ink_src="$(VILLA_INK_SRC)" \
	  --build-context uv_context="$(UV_CONTEXT)" \
	  --build-context repo="$(CURDIR)/../.." \
	  --build-arg BASE_IMAGE="$(VILLA_PYTHON_BASE_IMAGE)" \
	  --build-arg VILLA_INK_COMMIT="$(VILLA_INK_COMMIT)" \
	  --build-arg BUILD_COMMIT="$(BUILD_COMMIT)" \
	  --build-arg SOURCE_DATE_EPOCH="$(SOURCE_DATE_EPOCH)" \
	  --tag helena-ink-9um:local .

# A worker that can both claim a job and run the 9 um lane. Two environments in
# one image, because claiming needs psycopg and the lane's lock is --frozen and
# must not gain it. INK_9UM_IMAGE and BASE_IMAGE are both already-built images
# of this platform, so neither is pinned by digest here: they are named by the
# tags this Makefile produced.
# A worker that can both claim a P1 job and run the spiral fitter, composed the
# same way and for the same reason: claiming and filing a fitted winding needs
# psycopg, numpy and tifffile, and the fitter is resolved against a Python no
# distribution ships. Both build-args name images this Makefile produced, so
# neither is pinned by digest here.
# VILLA_IMAGE must be sha256-pinned: this image inherits Villa's binaries, so a
# floating tag would make its provenance a guess.
build-scrollfiesta-src: check-scrollfiesta-base-image
	$(call require_digest,VILLA_IMAGE)
	$(call require_source_tree,SCROLLFIESTA_SRC)
	$(call require_source_tree,VILLA_SRC)
	@test -f "$(UV_CONTEXT)/uv" || { printf '%s\n' 'UV_CONTEXT must contain a uv binary' >&2; exit 2; }
	docker build --file Containerfile.scrollfiesta-src \
	  --build-context sf_src="$(SCROLLFIESTA_SRC)" \
	  --build-context villa_src="$(VILLA_SRC)" \
	  --build-context repo="$(CURDIR)/../.." \
	  --build-context uv_context="$(UV_CONTEXT)" \
	  --build-arg BASE_IMAGE="$(SCROLLFIESTA_BASE_IMAGE)" \
	  --build-arg VILLA_IMAGE="$(VILLA_IMAGE)" \
	  --build-arg SCROLLFIESTA_COMMIT="$(SCROLLFIESTA_COMMIT)" \
	  --build-arg BUILD_JOBS="$(or $(BUILD_JOBS),2)" \
	  --build-arg BUILD_COMMIT="$(BUILD_COMMIT)" \
	  --build-arg SOURCE_DATE_EPOCH="$(SOURCE_DATE_EPOCH)" \
	  --tag helena-scrollfiesta:local .

# THAUMATO_ARCH_LIST defaults to Turing, which is what the fleet host has. Every
# extra architecture multiplies the nvcc work.
build-thaumato:
	$(call require_digest,THAUMATO_CUDA_BASE_IMAGE)
	$(call require_source_tree,VILLA_SRC)
	docker build --file Containerfile.thaumato \
	  --build-context villa_src="$(VILLA_SRC)" \
	  --build-arg CUDA_BASE_IMAGE="$(THAUMATO_CUDA_BASE_IMAGE)" \
	  --build-arg TORCH_CUDA_ARCH_LIST="$(or $(THAUMATO_ARCH_LIST),7.5)" \
	  --build-arg VILLA_COMMIT="$(VILLA_COMMIT)" \
	  --build-arg BUILD_COMMIT="$(BUILD_COMMIT)" \
	  --build-arg SOURCE_DATE_EPOCH="$(SOURCE_DATE_EPOCH)" \
	  --tag helena-thaumato:local .
