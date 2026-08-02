# syntax=docker/dockerfile:1.7
# Base image must contain Python 3, numpy, scipy and Pillow.  It is used for
# stability analysis, high-recall routing, CT/fiber gates and static viewers.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BUILD_COMMIT=unknown
ARG SOURCE_DATE_EPOCH=0
ARG FRAMEWORK_VERSION=0.2.0
LABEL org.opencontainers.image.title="helena-qc" \
      org.opencontainers.image.description="Helena Framework evidence analysis and review runtime contract" \
      org.opencontainers.image.revision="${BUILD_COMMIT}" \
      org.opencontainers.image.version="${FRAMEWORK_VERSION}" \
      org.opencontainers.image.created="${SOURCE_DATE_EPOCH}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CAMPAIGN_X_STAGE=qc \
    CAMPAIGN_X_REPO=/workspace/campaign-x \
    CAMPAIGN_X_ARTIFACTS=/artifacts
WORKDIR /workspace/campaign-x

# QC consumes immutable upstream receipts and emits new evidence only.  It
# never embeds CT, models, secrets or generated results in this image.
CMD ["sh"]
