# Framework contracts

The schemas in `schemas/` define the immutable interface of a stage manifest
and an execution receipt. A receipt is required for a success, a controlled
terminal failure, or an interrupted execution. It identifies the exact
framework commit, image digest, declared input hashes, command, hardware, and
output hashes; it is not a claim that ink, text, or a valid surface was found.

Stage 01 planner v2 is split into four auditable schemas:

- `segmentation-regional-attempt-history-v1.schema.json`: sanitized,
  geometry-only regional failures;
- `segmentation-planner-packet-v2.schema.json`: immutable model input;
- `segmentation-proposal-v2.schema.json`: bounded model output;
- `segmentation-locked-plan-v2.schema.json`: validator-approved VC3D input.

The Python validator remains authoritative for cross-object constraints such
as exact MCP seed identity, inclusive cell bounds, complete history citation,
parameter-envelope membership, and failed-recipe replay prevention.
