# PHerc0826 golden run public dossier design

## Purpose

Publish an end-to-end, evidence-bound account of Helena's `golden-run` mission on
public PHerc0826 data. The dossier must let an external technical reader
understand what ran, what was measured, what failed, what was corrected, and what
the final result does and does not establish.

The report is an operational golden-run record. It is not a claim that the input
is a community-confirmed ink or letter positive control.

## Deliverables

The public record will live under
`docs/golden-runs/pherc0826-2026-08-01/` and contain:

- `README.md`: the narrative report, phase-by-phase result, incident ledger,
  rerun evidence, limitations, and final acceptance statement.
- `evidence.json`: a machine-readable projection of the public evidence,
  including profile versions, content hashes, lineage, parameters that affect
  scientific interpretation, and measured results.

The repository root `README.md` will link to the dossier.

## Information architecture

The narrative will follow the actual lineage rather than the order in which bugs
were fixed:

1. Scope, input and acceptance criteria.
2. P8 merge and independent synthetic merge control.
3. Geometry and physical CT quality control.
4. P3 flattening.
5. P4 surface-normal CT rendering.
6. P5 ink-probability inference and liveness.
7. P7 review outcome.
8. Failed attempts, defects found, fixes and deterministic reruns.
9. Final operational and scientific conclusions.

A compact Mermaid diagram will show the parent-to-child lineage and downstream
artifacts. Tables will be used only where exact mappings or comparisons are
clearer than prose.

## Public evidence boundary

The dossier will publish:

- the public sample and source-volume identity;
- phase and profile versions;
- scientifically meaningful parameters;
- artifact and evidence SHA-256 digests;
- file counts, byte counts and rehash results;
- measured geometry, QC, flattening, rendering and inference results;
- immutable-attempt history, including failed or cancelled attempts;
- the provenance discrepancy observed in the historical QC receipt and the
  subsequent correction, without rewriting the original evidence;
- the explicit `allow_unvalidated` P3 override and the fact that final acceptance
  waited for physical QC.

The dossier will not publish:

- hostnames, usernames, worker identifiers or infrastructure topology;
- private bucket names, internal paths or artifact-store URIs;
- authentication, deployment or account details;
- raw logs that could expose environment configuration.

Opaque content hashes remain public because they support integrity checking
without disclosing storage location. Internal job identifiers will be replaced by
stable public run labels such as `P8-official`, `P4-reproducibility-rerun` and
`P5-publication-rerun`.

## Accuracy rules

- `ALIVE` means the probability map was non-degenerate. It never means ink was
  detected.
- `CT_SUPPORTED` means the reconstructed surface is supported by CT evidence. It
  never means ink or text was found.
- Coverage and continuity metrics remain separate from seam-RANSAC metrics. No
  union-area percentage will be inferred.
- The final ink result is reported verbatim as
  `CT_SUPPORTED_NO_RETAINED_INK_SIGNAL` and paired with the facts that no
  candidate was retained for visual review and no automatic acceptance occurred.
- The report will distinguish the upstream synthetic merge test from the
  digest-pinned production runtime.
- Operational success, geometry acceptance, physical-QC acceptance and
  scientific ink outcome will be reported as separate verdicts.

## Verification

Before publication:

1. Re-read canonical database records, manifests and receipts in read-only mode.
2. Recompute or compare every published digest against the canonical evidence.
3. Validate `evidence.json` as JSON and check that required public fields exist.
4. Run a redaction scan for internal hosts, bucket names, private paths, user
   names and internal job IDs.
5. Check every hash and headline result in `README.md` against `evidence.json`.
6. Render the Markdown structure mentally and run the repository's relevant test
   suite before committing.

## Acceptance

The work is complete when an external reader can reconstruct the full logical
chain from two public parent surfaces to the final review outcome, see every
material failure and correction, verify the disclosed content identities, and
cannot infer Helena's private deployment topology or access paths.
