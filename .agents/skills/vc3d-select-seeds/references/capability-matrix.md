# Capability matrix

## Contents

1. Current deterministic mechanisms
2. Closed-loop seed probe
3. Agent responsibilities
4. Unimplemented extensions
5. Business value
6. Comparison questions

## Current deterministic mechanisms

| Mechanism | Inputs | Decision | Strength | Limitation |
|---|---|---|---|---|
| `score-cell-volume-clearance-v1` | Frozen m7 candidates and cell/volume clearance | Select the validator-forced argmax | Zero-provider, reproducible, cheap | Does not test whether GrowPatch follows a usable surface |
| `adaptive-geometry-history-v2` | Candidates plus regional geometry-only attempt history | Deterministically avoid repeated failure patterns and expose a bounded v2 decision space | Uses real local failures without ink | Still predicts before growth |
| `deterministic-v2` planner | Frozen v2 packet and parameter envelope | Select one legal seed/profile/parameter map | Replayable and zero-provider | Cannot observe counterfactual growth without probes |
| CT material-support gate | Frozen local CT neighborhood | Reject a candidate with insufficient material | Prevents obviously impossible compute | Does not identify the correct lamina |
| TIFXYZ geometry gate | Produced grid/mesh geometry | Certify, reject bridge/lamina-switch/fold/self-intersection, or remain unmeasured | Independent deterministic post-growth evidence | Does not use a normal grid and does not prove physical identity |
| Cost-Aware v2 | Packet history, cache, provider budget, legal envelope | Route obvious work deterministically and escalate difficult planning | Controls provider spend and preserves fallback | It is a router, not a seed experiment |
| Fusion v2 | Expensive model panel on selected planner cases | Produce a proposal validated by deterministic code | Useful canary for genuinely ambiguous planning | Expensive; no authority over geometry or finalization |

The historical planner-decision comparison lives at
`framework/stages/01-segmentation/scripts/helena_compare_planner_lanes.py`.
It compares proposal decisions only and explicitly cannot establish usable
surface yield per compute wall-hour.

## Closed-loop seed probe

`seed-probe-v1` adds evidence unavailable to pre-growth selectors:

1. Freeze the first one to three source-bound m7 candidates.
2. Run the same noncanonical 10–20-generation VC3D recipe for each.
3. Hash and retain every trial artifact and receipt.
4. Evaluate each TIFXYZ result with the frozen geometry gate.
5. Choose only a unique geometry-eligible winner; otherwise abstain or reject.
6. In `shadow`, leave the canonical path unchanged.
7. In authorized `select`, resume the exact winner into an ordinary full grow.

This is complementary to deterministic-v2 and Cost-Aware. It does not replace
them.

## Agent responsibilities

Use the agent for:

- choosing the correct documented workflow;
- running readiness and comparison scripts;
- inspecting structured receipts and explaining discrepancies;
- coordinating bounded shadow experiments;
- presenting exact evidence for human review.

Do not use the agent for:

- estimating numerical geometry from pixels;
- creating unlisted XYZ coordinates during a locked run;
- changing frozen thresholds after observing outcomes;
- overriding a deterministic rejection;
- promoting evidence without exact authority and content hashes.

## Unimplemented extensions

Do not imply these exist:

- normal-grid sampling or mesh-to-normal-grid agreement;
- normal-aligned multiplanar rendering;
- calibrated surface or sheet-switch probabilities;
- direction/Z-range adaptation during probe growth;
- CT-gradient structure tensors;
- overlap/Hausdorff/Chamfer stitching tests;
- flattening compatibility;
- a global segment-consistency graph;
- autonomous visual operation of VC3D.

Add them only as separately versioned evidence profiles with replay tests and
calibration. Never silently extend `unique-geometry-certified-v1`.

## Business value

Today, `shadow` adds compute and evidence but cannot improve canonical yield
because it never steers. Its immediate pipeline value is operational: replay,
failure localization, lower-ranked rescue opportunity, abstention, and cost
measurement before any production behavior changes.

An approved `select` rollout could avoid a full grow from a candidate whose
bounded micro-growth is geometrically rejected, while using zero provider calls
for a unique winner. That benefit is a hypothesis until the isolated paired
benchmark passes; approval remains restricted to its tested sample IDs. Do not
book savings or community yield before then.

For the community, the durable value already exists in the open contracts:
source-bound candidate IDs, noncanonical probe receipts, categorical
abstention, exact artifact lineage, and a reproducible outcome benchmark. These
make negative and ambiguous evidence shareable without presenting m7 intensity
or TIFXYZ geometry as physical-sheet truth.

## Comparison questions

Answer separately:

1. Does probing find a unique geometry-eligible lower-ranked candidate when
   the deterministic top-ranked candidate fails?
2. Does the probe decision replay byte-for-byte?
3. What incremental compute-wall cost does probing add?
4. In an isolated paired outcome experiment, does closed-loop selection improve
   usable nonduplicate single-lamina canonical area per compute wall-hour on a
   matched device tier?
5. Does reviewer time per usable square centimetre regress?
6. Are benefits consistent by scroll rather than driven by one source?
7. Does paired superiority reject chance, and did any cell introduce a new
   incorrect-lamina harm?
8. Did any lease, replay, budget, source, artifact, or canonical-output
   invariant fail?
