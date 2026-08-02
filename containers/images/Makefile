SHELL := /bin/sh
.DEFAULT_GOAL := help

BUILD_COMMIT ?= $(shell git rev-parse HEAD 2>/dev/null || printf unknown)
SOURCE_DATE_EPOCH ?= $(shell git show -s --format=%ct HEAD 2>/dev/null || printf 0)

.PHONY: help check-base-images check-scrollfiesta-base-image check-surface-qc-images check-all-images build-vc3d build-scrollfiesta build-ink build-qc build-orchestrator build-surface-qc build-villa build-scrollfiesta-src build-thaumato build-worker

help:
	@printf '%s\n' 'Checks:  check-base-images check-scrollfiesta-base-image check-surface-qc-images check-all-images'
	@printf '%s\n' 'Assemble: build-vc3d build-scrollfiesta build-ink build-qc build-orchestrator build-surface-qc build-worker'
	@printf '%s\n' 'Compile:  build-villa build-scrollfiesta-src build-thaumato'
	@printf '%s\n' 'build-surface-qc requires INK_IMAGE and VC3D_IMAGE, both sha256-pinned.'
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
	$(call require_digest,VC3D_BASE_IMAGE)
	$(call require_digest,INK_BASE_IMAGE)
	$(call require_digest,QC_BASE_IMAGE)
	$(call require_digest,ORCHESTRATOR_BASE_IMAGE)

check-scrollfiesta-base-image:
	$(call require_digest,SCROLLFIESTA_BASE_IMAGE)

# Containerfile.surface-qc:6-9 takes two *parent images* rather than one base:
# INK_IMAGE (the frozen six-replica TimeSformer environment, staged as
# `ink_runtime`) and VC3D_IMAGE (the verified VC3D bundle). Neither was
# validated anywhere before FIX-16, so `helena-surface-qc` could be built on
# floating tags and the resulting QC image was not reproducible.
check-surface-qc-images:
	$(call require_digest,INK_IMAGE)
	$(call require_digest,VC3D_IMAGE)

check-all-images: check-base-images check-scrollfiesta-base-image check-surface-qc-images

define build_image
docker build --file Containerfile.$(1) --build-arg BASE_IMAGE="$($(2)_BASE_IMAGE)" --build-arg BUILD_COMMIT="$(BUILD_COMMIT)" --build-arg SOURCE_DATE_EPOCH="$(SOURCE_DATE_EPOCH)" --tag helena-$(1):local .
endef

define build_scrollfiesta_image
docker build --file Containerfile.scrollfiesta --build-context scrollfiesta_runtime="$(SCROLLFIESTA_RUNTIME_CONTEXT)" --build-arg BASE_IMAGE="$(SCROLLFIESTA_BASE_IMAGE)" --build-arg BUILD_COMMIT="$(BUILD_COMMIT)" --build-arg SOURCE_DATE_EPOCH="$(SOURCE_DATE_EPOCH)" --tag helena-scrollfiesta:local .
endef

define build_vc3d_image
docker build --file Containerfile.vc3d --build-context vc3d_runtime="$(VC3D_RUNTIME_CONTEXT)" --build-arg BASE_IMAGE="$(VC3D_BASE_IMAGE)" --build-arg CA_CERTIFICATES_VERSION="$(CA_CERTIFICATES_VERSION)" --build-arg BUILD_COMMIT="$(BUILD_COMMIT)" --build-arg SOURCE_DATE_EPOCH="$(SOURCE_DATE_EPOCH)" --tag campaignx-vc3d:local .
endef

define build_surface_qc_image
docker build --file Containerfile.surface-qc --build-arg INK_IMAGE="$(INK_IMAGE)" --build-arg VC3D_IMAGE="$(VC3D_IMAGE)" --build-arg BUILD_COMMIT="$(BUILD_COMMIT)" --build-arg SOURCE_DATE_EPOCH="$(SOURCE_DATE_EPOCH)" --tag helena-surface-qc:local .
endef

build-vc3d: check-base-images
	@test -n "$(VC3D_RUNTIME_CONTEXT)" || { printf '%s\n' 'VC3D_RUNTIME_CONTEXT is required' >&2; exit 2; }
	@test -f "$(VC3D_RUNTIME_CONTEXT)/vc3d-runtime.tgz" || { printf '%s\n' 'VC3D_RUNTIME_CONTEXT must contain vc3d-runtime.tgz' >&2; exit 2; }
	$(build_vc3d_image)

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
# existing: it was built by hand, and Containerfile.surface-qc plus
# requirements.surface-qc.txt were excluded by .dockerignore, so the image was
# not reproducible from the repository at all.
build-surface-qc: check-surface-qc-images
	@test -f Containerfile.surface-qc || { printf '%s\n' 'Containerfile.surface-qc is missing' >&2; exit 2; }
	@test -f requirements.surface-qc.txt || { printf '%s\n' 'requirements.surface-qc.txt is missing' >&2; exit 2; }
	$(build_surface_qc_image)

# The same omission as build-surface-qc above, made again: helena-worker:local
# was running the fleet on two hosts with no target that builds it. It was put
# together by hand, so what was in it could only be discovered by looking inside
# it -- and a code change reached the hosts only if somebody remembered the same
# command. VILLA_IMAGE is sha256-pinned because this image inherits VC3D's
# binaries from it.
build-worker:
	$(call require_digest,VILLA_IMAGE)
	@test -f "$(UV_CONTEXT)/uv" || { printf '%s\n' 'UV_CONTEXT must contain a uv binary' >&2; exit 2; }
	docker build --file Containerfile.worker \
	  --build-context repo="$(CURDIR)/../.." \
	  --build-context uv_context="$(UV_CONTEXT)" \
	  --build-arg BASE_IMAGE="$(VILLA_IMAGE)" \
	  --build-arg VILLA_IMAGE_DIGEST="$(VILLA_IMAGE)" \
	  --build-arg BUILD_COMMIT="$(BUILD_COMMIT)" \
	  --build-arg SOURCE_DATE_EPOCH="$(SOURCE_DATE_EPOCH)" \
	  --tag helena-worker:local .

# ---------------------------------------------------------------------------
# Compile-from-source targets.
#
# VILLA_SRC is a checkout of ScrollPrize/villa at the commit recorded in
# scrollfiesta/locks/source-lock.json. Nothing is fetched during the build: a
# build that clones is a build whose output depends on the day it ran.
# ---------------------------------------------------------------------------

VILLA_COMMIT ?= $(shell python3 -c "import json;print(json.load(open('scrollfiesta/locks/source-lock.json'))['volume_cartographer']['commit'])" 2>/dev/null)
SCROLLFIESTA_COMMIT ?= $(shell python3 -c "import json;print(json.load(open('scrollfiesta/locks/source-lock.json'))['scrollfiesta']['commit'])" 2>/dev/null)

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
	  --build-arg BASE_IMAGE="$(VC3D_BASE_IMAGE)" \
	  --build-arg VILLA_COMMIT="$(VILLA_COMMIT)" \
	  --build-arg VILLA_TREE="$$(git -C "$(VILLA_SRC)" rev-parse 'HEAD^{tree}')" \
	  --build-arg BUILD_COMMIT="$(BUILD_COMMIT)" \
	  --build-arg SOURCE_DATE_EPOCH="$(SOURCE_DATE_EPOCH)" \
	  --tag campaignx-villa:local .

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
