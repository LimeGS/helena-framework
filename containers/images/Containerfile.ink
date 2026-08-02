# syntax=docker/dockerfile:1.7
# Base image must contain the frozen CUDA/PyTorch/TimeSformer runtime used by
# run_ink_timesformer.py, including numpy, Pillow and its checkpoint
# loader dependencies.  The model checkpoint is mounted at runtime.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BUILD_COMMIT=unknown
ARG SOURCE_DATE_EPOCH=0
ARG FRAMEWORK_VERSION=0.2.0
LABEL org.opencontainers.image.title="helena-ink" \
      org.opencontainers.image.description="Helena Framework six-replica ink-screening runtime contract" \
      org.opencontainers.image.revision="${BUILD_COMMIT}" \
      org.opencontainers.image.version="${FRAMEWORK_VERSION}" \
      org.opencontainers.image.created="${SOURCE_DATE_EPOCH}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CAMPAIGN_X_STAGE=ink \
    CAMPAIGN_X_REPO=/workspace/campaign-x \
    CAMPAIGN_X_ARTIFACTS=/artifacts \
    CAMPAIGN_X_MODELS=/models
WORKDIR /workspace/campaign-x

# Keep the scientific Python layer explicit instead of assuming that an
# arbitrary PyTorch base happens to contain the TimeSformer implementation.
# The base image itself remains digest-pinned by the build contract.
COPY requirements.ink.txt /opt/campaignx/requirements.ink.txt
RUN python -m pip install --no-cache-dir -r /opt/campaignx/requirements.ink.txt

# /models and /artifacts are supplied by runtime mounts.  A completed
# high-recall receipt is required before any worker may delete replica .npy.
CMD ["sh"]
