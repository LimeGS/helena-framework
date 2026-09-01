# Staging CI Runner Affinity Design

## Problem

The `staging` deployment belongs to `gpu-1`, but its prepare, unit-test, and panel-image build jobs are hard-pinned to `swisspost-1`. When `swisspost-1` is offline, a valid staging push cannot reach the existing `gpu-1` deploy and smoke jobs even though `gpu-1` is healthy.

## Decision

Select one CI runner affinity at pipeline creation time:

- `staging` uses `gpu-1` for the local CI image, unit tests, and panel image build.
- Every other pipeline keeps `swisspost-1` for those three Docker-daemon-coupled jobs.
- Deploy and smoke jobs keep their existing fixed host tags and host guards.

Use a pipeline variable in job `tags` rather than duplicate the prepare/test/build graph. The CI image remains local to one daemon because its producer and consumer resolve the same variable in the same pipeline.

## Rejected alternatives

- Manual image build/deploy: bypasses the protected pipeline and weakens provenance.
- Duplicate prepare, unit, and build jobs per branch: works but doubles graph definitions and creates drift risk.

## Safety and acceptance

- No queue, mission, surface, or database mutation is introduced.
- A regression test parses `.gitlab-ci.yml` and proves the branch mapping plus identical affinity for the local-image producer and consumer.
- Existing deployment-host guards remain unchanged.
- Acceptance requires the exact staging SHA to pass prepare, tests, panel build, gpu-1 deploy, and gpu-1 smoke, followed by runtime revision verification.
