# First Letters discovery-recovery scientific contract

Status: frozen design for implementation; no real-data execution is authorized by this document.

## Objective

The next First Letters campaign must first show that the deployed system can recover a public known positive, measure whether its candidate source covers each target, and stop spending compute when that source is unproductive. Discovery remains content-blind. Only a new canonical full grow may enter the unchanged geometry, physical-QC, P3, P4, P5, P7 and human-review path.

The machine-readable contracts are:

- `campaignx.first_letters_control_manifest.v1` in `first-letters-control-policy-1.0.0.json`;
- `campaignx.first_letters_campaign_policy.v1` in `first-letters-campaign-decision-policy-1.0.0.json`.

Every consumer must fail closed on a missing source lock, expected outcome, versioned profile identity or hash; overlapping control and evaluation cohorts; non-finite coordinates; content-informed discovery input; or a positive control supported only by the synthetic probability-map test.

## Frozen control selection

The development control is PHerc0139 segment `20250108000000-w025_2025010863`. It is disjoint by scroll identity from all 13 entries in `workspace/catalog/eligible_volumes.json`.

The stable scholarly attribution is Angelotti et al., *Complete virtual unwrapping and reading of a rolled Herculaneum papyrus*, [arXiv:2606.29085v1](https://arxiv.org/abs/2606.29085v1), DOI `10.48550/arXiv.2606.29085`. The versioned paper PDF, including its Supplementary Information, is frozen at `https://arxiv.org/pdf/2606.29085v1`: SHA-256 `99d894c12970530d528d1b7559273bb783c0da4c67fabe12abe59710d321e77b` over 38,794,737 bytes. The paper reports visible recovered writing in PHerc. 139. Its supplementary phrase inventory includes w25; the vendored PHerc0139 atlas identifies that phrase as `ἀόρατα`. This establishes a public positive at wrap level, not the location of a unique autonomous seed. An execution profile without this versioned scholarly byte lock is invalid, not merely incomplete.

Vesuvius Challenge documents that its open bucket publishes CT volumes, TIFXYZ surfaces and derived prediction artifacts. The selected control binds the matching public 9.362 µm CT and M7 artifacts, the w025 TIFXYZ surface, and the official w025 positive map. Root OME-Zarr metadata and every finite TIFXYZ object are SHA-256 locked. The official positive map was independently downloaded and hashes to `d99dbd0698a41cc2106888eebc6bed76245cf5a9074a68d025066c89b982fa08` over 68,354,492 bytes.

Each positive-evidence role must appear exactly once. The scholarly attribution, TIFXYZ surface and official positive map relationships must resolve respectively to source locks declaring `SCHOLARLY_PAPER_WITH_SUPPLEMENTARY_INFORMATION`, `TIFXYZ_SURFACE` and `INK_PROBABILITY_MAP`; the scholarly relationship must also resolve to frozen identity `arXiv:2606.29085v1`. Duplicate roles, even when a later entry is valid, and role-to-lock kind mismatches invalidate the control.

CT and M7 are chunked stores for which this contract does not invent a whole-store digest. Their URI, shape and root metadata bytes are frozen for profile identity. At execution, each source must produce a canonical, lexicographically ordered read-set manifest containing `object_key`, `sha256` and `bytes` exactly once for every object actually read, plus the manifest's own SHA-256. The M7 provider exchange must additionally bind the request hash, response hash and response byte count. A missing object entry, missing digest, missing provider hash or changed object makes the control receipt `CONTROL_INCOMPLETE`; root metadata alone can never satisfy content binding.

This evidence is a public positive control, not independent validation. PHerc0139 was used in development and existing repository calibration. A passing control demonstrates liveness on a known positive under the exact frozen inputs; it does not estimate transfer performance.

## Source and coordinate contract

Discovery uses:

- CT `PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr`;
- M7 `PHerc0139/representations/predictions/surfaces/20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr`;
- coordinate frame `PHerc0139/20250728140407-9.362um/CT-L0/XYZ` at 9.362 µm isotropic voxels;
- the valid points of the byte-locked 364 by 340 w025 TIFXYZ as the known region;
- a 2.0 CT-L0 voxel nearest-surface tolerance.

The provenance-marked pipeline seed is TIFXYZ cell `[182, 170]`, CT-L0 XYZ `[4020.60107421875, 3377.9609375, 4968.8193359375]`. This is a finite point read from the locked surface. It is explicitly not presented as the location of the published phrase.

## Two controls, not one ambiguous result

`DISCOVERY_CONTROL` asks whether the frozen provider, M7 source, threshold, CT-material gate and clearance policy produce at least one survivor within tolerance of the locked public w025 surface. It performs no growth and accepts no P5, P7 or human signal as an input.

`PIPELINE_CONTROL` starts from the provenance-marked human seed on that surface. The receipt must preserve `seed_origin=human`; it may never relabel the seed autonomous. A pass requires a canonical full grow, geometry certification, downstream-compatible terminal physical QC, complete hash-bound P3 and P4 artifacts, a live hash-bound P5 output, P7 routing consistent with the known public positive, and a human-review packet. Automated routing is not letter acceptance.

The two checks are deliberately separate because the public evidence fixes a positive wrap and surface, not a unique M7 seed. Both checks must pass before target work.

## Outcome semantics

| State | Meaning |
|---|---|
| `CONTROL_PASS` | Every requirement for both controls is present and passes for the exact deployed revision and locks. |
| `CONTROL_INCOMPLETE` | A required stage, artifact, source response, hash or review-routing record is missing or nonterminal. |
| `CONTROL_FAILED` | Complete terminal evidence contradicts a required result. |

Any change to deployed code, CT/M7/surface bytes, provider configuration, model, threshold or bound profile makes a prior receipt stale. Missing evidence is never coerced to failure or success.

In particular, `CONTROL_PASS` requires materialized CT and M7 read-set manifest hashes and M7 provider request/response hashes. A declaration that these will be computed later is not a passing receipt.

## Campaign decisions

A complete preflight census with `K` usable candidate-bearing cells among `N` eligible cells uses sampling without replacement. The task budget is the smallest `n` satisfying:

`1 - C(N-K, n) / C(N, n) >= 0.95`.

If a complete census has `K=0`, the decision is `DO_NOT_QUEUE_CURRENT_SOURCE`. If only a sample exists, prevalence uses a one-sided 95% Clopper-Pearson lower bound. A sample with zero usable cells produces no task budget; it requires a larger preflight or a materially different source. The compute cap is a required frozen input, not a default invented by this policy. Clipping reports both requested tasks and achieved detection probability.

Candidate-starvation is evaluated after eight scientific-terminal attempts. Seven or more recorded raw-M7-empty attempts among the first eight pauses new queue creation, and each later block of eight is re-evaluated. Two consecutive scrolls completing frozen budgets with zero raw M7 also pause. Cancelled work, configuration blocks, lease exhaustion, publication failures, source failures and worker failures never enter the scientific denominator. Existing pending or running work is not cancelled.

Resumption requires a new policy version and a material causal change: M7 source, calibrated threshold, grid, provider, properly authorized seed-probe mode or evidence-backed clearance policy. A planner-only change cannot create a candidate the source did not offer.

## Discovery isolation and non-claims

Discovery artifacts remain noncanonical and cannot satisfy P2/P3 admission or trigger physical QC. Promotion consumes a byte-hashed discovery artifact and receipt plus one selected candidate whose identifier, evidence hash, provider-response hash and CT-L0 coordinate resolve inside that artifact. It creates a new normal full-grow attempt and immutably binds the parent artifact/candidate hashes, promotion receipt, control/preflight/budget receipts, source snapshot, CT/M7 read-set manifests, grid, M7 level set, material and clearance policies, discovery policy, growth profile and deployed revision. The discovery artifact itself can never be relabelled canonical. Missing or ambiguous lineage yields `CONTROL_INCOMPLETE_NO_RETRY_UNTIL_READBACK`; only the new attempt, after the normal admission gates, can enter acceptance. `allow_unvalidated` is false everywhere.

A discovery miss, zero-candidate census, pipeline failure or starvation pause describes the bounded method and frozen inputs. None is evidence that a scroll lacks a surface, ink, text or letters. A control pass is not independent validation, and no automated outcome satisfies First Letters without blinded human confirmation.
