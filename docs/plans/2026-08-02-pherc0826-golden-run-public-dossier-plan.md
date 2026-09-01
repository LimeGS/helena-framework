# PHerc0826 golden run public dossier implementation plan

> **Goal:** Publish a sanitized, evidence-bound, end-to-end account of the
> PHerc0826 `golden-run` mission without exposing private infrastructure.

## Task 1: Extract the canonical public evidence projection

**Files:**

- Create: a temporary evidence projection outside the repository
- Reference: `framework/contracts/pipeline_phases.json`
- Reference: `framework/profiles/`

1. Query the production mission, surface, flattening and physical-QC records in
   read-only mode.
2. Read the canonical P8, QC, P3, P4, P5 and P7 manifests and receipts.
3. Capture only public fields: source identity, profile versions, scientific
   parameters, state transitions, metrics, file counts, byte counts and content
   hashes.
4. Record all failed/cancelled attempts and their normalized public causes.
5. Confirm the final rerun artifacts against their manifests and rehash reports.
6. Keep the extraction outside the repository; never commit raw production
   responses.

## Task 2: Write the machine-readable public evidence

**Files:**

- Create: `docs/golden-runs/pherc0826-2026-08-01/evidence.json`

1. Define a small self-describing public schema identifier and report version.
2. Encode scope, lineage, phase results, QC evidence, reproducibility reruns,
   incident history, final verdicts and limitations.
3. Use stable public labels instead of internal job or worker identifiers.
4. Replace private URIs with content identities, file counts and byte counts.
5. Validate syntax:

   ```sh
   python3 -m json.tool docs/golden-runs/pherc0826-2026-08-01/evidence.json >/dev/null
   ```

6. Run a semantic assertion script that checks key hashes, verdicts and counts.

## Task 3: Write the public narrative dossier

**Files:**

- Create: `docs/golden-runs/pherc0826-2026-08-01/README.md`

1. State the question the run answered and the questions it did not answer.
2. Add a compact Mermaid lineage diagram.
3. Document P8, QC, P3, P4, P5 and P7 in order, binding every headline claim to
   a value also present in `evidence.json`.
4. Add a complete attempt/incident ledger, including the P3 routing and merged
   schema failures, the original P5 publication gap, and the historical QC
   provenance discrepancy.
5. Explain which fixes were applied and what the deterministic reruns proved.
6. End with separate operational, geometry, CT-support and ink/letter verdicts.
7. Explicitly state that this was not a community-confirmed positive control.

## Task 4: Make the dossier discoverable

**Files:**

- Modify: `README.md`

1. Add one concise link under `Where to look next`.
2. Describe it as a public end-to-end evidence dossier, not a product claim or a
   positive scientific result.

## Task 5: Verify correctness and sanitization

**Files:**

- Verify: `docs/golden-runs/pherc0826-2026-08-01/README.md`
- Verify: `docs/golden-runs/pherc0826-2026-08-01/evidence.json`
- Verify: `README.md`

1. Validate JSON with `python3 -m json.tool`.
2. Search the two public files for private bucket names, hostnames, usernames,
   absolute internal paths, worker identifiers and internal job-ID patterns.
3. Search for unsupported claims such as `positive control`, `ink detected`,
   `letters found` and invented coverage percentages.
4. Cross-check every 64-character SHA-256 in the Markdown against
   `evidence.json` or a declared independent-test record.
5. Run focused truthfulness tests:

   ```sh
   python3 -m pytest tests/test_the_readme_is_true.py -q
   ```

6. Run the full local Python suite and frontend tests/build already used for this
   branch:

   ```sh
   python3 -m pytest tests/ -q --ignore=tests/e2e
   cd panel/web && npm test -- --run && npm run build
   ```

7. Review `git diff --check` and the complete diff.

## Task 6: Commit and publish

**Files:**

- Commit all approved documentation files.

1. Commit with a documentation-only message.
2. Push the feature branch.
3. Fast-forward `staging` only after its pipeline is green, preserving the
   repository's existing promotion flow.
