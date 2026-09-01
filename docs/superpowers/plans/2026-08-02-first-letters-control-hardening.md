# First Letters Control Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure a First Letters `CONTROL_PASS` is derived only from production-generated M7 reads, the current queued jobs, frozen profile/source/model locks, proven orientation, strict downstream artifacts, and append-only server-derived human routing.

**Architecture:** Extend existing MCP, fleet stores, job routes, and runner readbacks rather than creating a parallel control plane. Add one deterministic read-only orientation proof module/route, then bind its content hash into P4. Every boundary fails closed on missing, stale, contradictory, or client-asserted evidence.

**Tech Stack:** Python 3.11, FastAPI/Pydantic, SQLite/PostgreSQL fleet stores, Zarr/tifffile/numpy, pytest/TestClient, canonical JSON SHA-256.

## Global Constraints

- Work only in `/private/tmp/helena-qc-review-stack-fix`.
- Preserve `docs/superpowers/plans/2026-08-02-first-letters-hybrid-campaign.md` untracked and untouched.
- Tests replace only true remote provider, worker execution, and object-storage boundaries.
- No deployment, real campaign mutation, GPU work, or automatic mutation retry.
- Every production behavior change requires observed RED before implementation.
- A control pass remains a development-method check, never independent validation or letter acceptance.
- The committed real First Letters policy must remain `CONTROL_INCOMPLETE` while
  absolute-orientation or positive-control ROI evidence is `verified:false`.

---

### Task 1: Production MCP M7 read evidence

**Files:**
- Modify: `framework/stages/01-segmentation/mcp/server.py`
- Modify: `framework/stages/01-segmentation/mcp/seed_candidates.py`
- Modify: `framework/stages/01-segmentation/fleet/worker.py`
- Test: `tests/test_seed_mcp_server.py`
- Test: `tests/test_seed_candidates.py`
- Test: `tests/test_first_letters_control_contracts.py`

**Interfaces:**
- Produces: `open_prediction_with_read_set(uri, volume_root, level=0) -> (array, tracker)`.
- Produces: MCP structured result field `source_read_set: campaignx.first_letters_source_read_set.v1`.
- Consumes: `McpSeedProvider.discover()` verifies and merges the service result without fabricating objects.

- [ ] **Step 1: Write the failing real-service tests**

Create local and range-capable HTTP/FsspecStore Zarr v2 fixtures, invoke real
`Service.find_seed_candidates`, and assert root/group metadata, array metadata,
actual chunks, and canonical partial ranges are returned. Assert hashes/bytes,
cache coalescing, and `content_sha256(objects)`. Test explicit consolidated
`.zmetadata` locks and fail-closed separate-vs-consolidated mismatch. Through
`handler_for`, assert identical evidence crosses the JSON-RPC boundary.

- [ ] **Step 2: Run RED**

Run:

```text
python -m pytest -q tests/test_seed_candidates.py tests/test_seed_mcp_server.py tests/test_first_letters_control_contracts.py -k 'read_set or source_evidence'
```

Expected: fail because production `Service.find_seed_candidates` returns no `source_read_set`.

- [ ] **Step 3: Implement tracked Zarr reads**

Force Zarr v2. Add a Zarr `WrapperStore` that records exact bytes returned by
every `get`, `get_partial_values`, and metadata lookup under canonical object
keys plus `(offset,length)` for ranges, before cache reuse. Reject contradictory
repeated bytes/ranges and enforce size/range caps. Return:

```python
{
    "schema": "campaignx.first_letters_source_read_set.v1",
    "objects": sorted(objects, key=lambda row: row["object_key"]),
    "canonical_manifest_sha256": content_sha256(objects),
}
```

`Service.find_seed_candidates` must attach that tracker receipt after `find_candidates` completes. `McpSeedProvider` must reject missing/malformed evidence and only aggregate service-returned receipts.

- [ ] **Step 4: Run GREEN and affected MCP tests**

Run the RED command plus:

```text
python -m pytest -q tests/test_segment_search_fleet.py tests/test_vc3d_seed_skill.py
```

Expected: all pass (socket-binding tests may remain sandbox-blocked only in the full suite).

---

### Task 2: Frozen runtime/profile/source enforcement

**Files:**
- Modify: `panel/app.py`
- Modify: `framework/profiles/01-segmentation/first-letters-control-policy-1.0.0.json`
- Modify: `scripts/harness/run_first_letters_positive_control.py`
- Test: `tests/test_first_letters_control_contracts.py`
- Test: `tests/test_first_letters_positive_control.py`

**Interfaces:**
- Produces: `verified_control_runtime() -> dict` with actual referenced-profile hashes and `verified=True` per lock.
- Produces: server-normalized preflight request/evidence; client scientific fields are equality assertions only.

- [ ] **Step 1: Write failing drift/root-object tests**

Add TestClient tests that alter a referenced profile file after loading the manifest, send a threshold/provider/surface override, omit or change frozen CT/M7 root objects in the provider/sampler read sets, and assert `/api/state` reports verification failure or preflight returns 409/503. Add runner tests that reject `verified=False` runtime rows.

- [ ] **Step 2: Run RED**

```text
python -m pytest -q tests/test_first_letters_control_contracts.py tests/test_first_letters_positive_control.py -k 'profile_drift or policy_override or root_object'
```

Expected: current state trusts declarations and preflight trusts request policy.

- [ ] **Step 3: Implement server-owned locks**

Resolve every `profile_locks[].profile_id` to its repository JSON. Compare the
declared raw-file SHA to `actual_file_sha256`, separately expose
`actual_canonical_document_sha256`, and never conflate their semantics. Build
the preflight scientific body from the manifest; if a supplied compatibility
field differs, return 409. Require read-set object entries for every frozen
CT/M7 `metadata[].path` with the exact declared SHA before returning `COMPLETE`.

- [ ] **Step 4: Record normalized evidence in the runner**

Store the server-normalized preflight parameters, full preflight receipt SHA, and traversal `planned_region_count`, `visited_region_count`, `maximum_surface_probe_regions`, `coverage_fraction`, and `coverage_complete` in P1.

- [ ] **Step 5: Run GREEN**

Run:

```text
python -m pytest -q tests/test_first_letters_positive_control.py tests/test_first_letters_control_contracts.py tests/e2e/test_first_letters_positive_control.py
```

---

### Task 3: Current-job P2/QC and P3 immutable readbacks

**Files:**
- Modify: `framework/stages/01-segmentation/fleet/certifier.py`
- Modify: `framework/stages/01-segmentation/fleet/store.py`
- Modify: `framework/stages/01-segmentation/fleet/postgres_store.py`
- Modify: `framework/stages/01-segmentation/fleet/cli.py`
- Modify: `framework/stages/03-ink/fleet/job_store.py`
- Modify: `panel/app.py`
- Modify: `scripts/harness/run_first_letters_positive_control.py`
- Test: `tests/test_qc_job_diagnostics_api.py`
- Test: `tests/test_geometry_certification.py`
- Test: `tests/test_flattening.py`
- Test: `tests/test_first_letters_positive_control.py`
- Test: `tests/test_first_letters_control_contracts.py`

**Interfaces:**
- P2 receipt fields: `geometry_certified_by_job_id`, certification `profile_id`
  and file hash, `surface_id`, `surface_artifact_sha256`, `result_sha256`, `result`.
- QC receipt preserves its own `qc_job_id` and adds
  `unblocked_by_job_id=<P2 job>`, promotion/event link, QC profile/input/result.
- P3 flattening fields: `requested_by_job_id`, `surface_id`, `source_artifact_sha256`, `profile_id`, `artifact_uri`, `artifact_sha256`, `objects`, `receipt_sha256`, `receipt`.

- [ ] **Step 1: Write stale-evidence RED tests**

Queue job B after completed job A for the same surface. Return A's successful
QC/flattening row while B is current. Assert the runner stops P2/P3. Assert a QC
row whose own `qc_job_id` is correct but whose `unblocked_by_job_id` or promotion
link points at another P2 also stops. Assert incomplete stored parameters, wrong
events, profile drift, surface hash drift, missing full result, URI, receipt, or
inventory also stop.

- [ ] **Step 2: Run RED**

```text
python -m pytest -q tests/test_first_letters_positive_control.py tests/test_qc_job_diagnostics_api.py tests/test_flattening.py -k 'stale or current_job or immutable'
```

- [ ] **Step 3: Persist current-job lineage**

Pass the queue job ID into P2/P3 command execution through a fixed CLI
flag/environment field. Persist P2's geometry identity without overwriting the
physical `qc_job_id`; persist QC's causal unblocking/promotion link to that exact
P2. Store exact input surface/profile/hash/result for both. Extend SQLite/PostgreSQL
migrations compatibly and expose whitelisted immutable receipts from existing
QC/flattening endpoints. Do not expose raw exception/secret-bearing fields.

- [ ] **Step 4: Bind runner to just-enqueued jobs**

Require the exact returned job ID to be `succeeded`, exact stored parameters, exact lineage event, and exact QC/P3 immutable receipt. Ignore no earlier successful row; mismatches are `CONTROL_INCOMPLETE`.

- [ ] **Step 5: Run GREEN and fleet-store regression tests**

Run the RED command plus:

```text
python -m pytest -q tests/test_geometry_certification.py tests/test_geometry_certification_backlog.py tests/test_qc_job_diagnostics_api.py tests/test_flattening.py tests/test_flattening_gates.py
```

---

### Task 4: Deterministic orientation parity proof

**Files:**
- Create: `framework/stages/02-flattening/orientation_parity.py`
- Modify: `framework/stages/01-segmentation/fleet/finalizer.py`
- Modify: `framework/profiles/01-segmentation/first-letters-control-policy-1.0.0.json`
- Modify: `panel/app.py`
- Modify: `framework/stages/03-ink/fleet/job_store.py`
- Modify: `scripts/harness/run_first_letters_positive_control.py`
- Test: `tests/test_first_letters_orientation_parity.py`
- Test: `tests/test_first_letters_control_contracts.py`
- Test: `tests/test_first_letters_positive_control.py`

**Interfaces:**
- Produces shared exact TIFXYZ triangulation `(v00,v10,v01)` and
  `(v10,v11,v01)` with per-triangle validity.
- Produces: `prove_orientation(reference_xyz, grown_mesh_artifact, lineage, policy) -> dict` with schema `campaignx.first_letters_orientation_parity.v1`.
- Produces: `GET /api/geometry/orientation-proof?mission=...&sample=...&surface=...&p3_job=...` derived only from server-side artifacts.
- P4 parameter/event: `orientation_receipt_sha256`; `flip_normals` exists only
  when both parity and absolute-orientation evidence are verified.

- [ ] **Step 1: Write geometry RED tests**

Use synthetic triangulated grids/meshes for positive consensus, negative consensus,
63 correspondences, 94% consensus, median `0.89`, zero-area/non-finite normals,
anti-diagonal triangulation, spatial-bin boundaries, `1e-12` tie buckets,
candidate/index/vertex/time caps, and deterministic 1,024-point sampling. Assert
parity alone never selects a boolean. Assert the actual policy's
`verified:false` absolute lock yields `UNPROVEN`.

- [ ] **Step 2: Run RED**

```text
python -m pytest -q tests/test_first_letters_orientation_parity.py
```

Expected: module missing.

- [ ] **Step 3: Implement the frozen parity algorithm**

Extract the exact finalizer triangulation helper. Implement the bounded frozen
rule from the design: piecewise-planar normals, area-weighted grown normals,
4.0-voxel uniform-grid broad phase, exact float64 closest-point-on-triangle,
`1e-12` distance tie buckets, deterministic ordinal/barycentric ties, all
declared caps, deterministic 1,024 sampling, at least 64, 95% strict-sign
consensus, and median absolute dot at least 0.90. Canonical-hash the complete
per-sample receipt and reject non-finite values.

- [ ] **Step 4: Add server-derived proof route and P4 binding**

The route resolves reference/grown-mesh/P3 artifacts by mission/sample/surface/job;
`grown_faces` is never accepted as an unbound standalone input. Add parity and
absolute-orientation locks to the policy, with the latter explicitly
`verified:false` because no correct 9.362um w025 evidence exists. P4 must stop
`CONTROL_INCOMPLETE`; only a future immutable absolute receipt may unlock the
proof hash/boolean, and the worker event must repeat both.

- [ ] **Step 5: Run GREEN**

Run orientation and First Letters focused suites.

---

### Task 5: Strict P4/P5/P7 contracts

**Files:**
- Modify: `framework/stages/03-ink/fleet/ink_worker.py`
- Modify: `framework/stages/03-ink/fleet/job_store.py`
- Modify: `scripts/harness/run_first_letters_positive_control.py`
- Test: `tests/test_first_letters_positive_control.py`
- Test: `tests/test_job_parameter_domains.py`
- Test: `tests/test_p4_publishes_what_it_rendered.py`

**Interfaces:**
- P4 inventory resolves through `ordered_tiff_files(require_numeric=True)` to
  exactly distinct numeric indices `0..32`, binding actual names and all bytes.
- P5 result includes full 33-file source-manifest binding, exact P4 artifact/job,
  `source_slice_um = CT_voxel_um * P4.scale * P4.slice_step`, and a proven
  P3→P4 lateral metric/distortion receipt before `source_pixel_um` is allowed.
- P7 stored bbox equals the exact provenance-locked letterform ROI transformed
  into P5 coordinates; full-map bbox is forbidden. Stored `px_um` equals the
  independently verified training scale.

- [ ] **Step 1: Write downstream RED tests**

Add failures for 32 slices, nonnumeric/duplicate/missing indices, and acceptance
of actual unpadded `0.tif..32.tif`. Add P5 depth formula drift, missing/partial
33-file source inventory, wrong P4 job/artifact, missing lateral metric,
out-of-bound UV distortion, normalization/hash/training-lock failures. Add P7
full-map/synthetic/cropped/off-by-one bbox, missing ROI provenance/transform, and
`px_um` copied from an unverified result.

- [ ] **Step 2: Run RED**

```text
python -m pytest -q tests/test_first_letters_positive_control.py tests/test_job_parameter_domains.py tests/test_p4_publishes_what_it_rendered.py -k 'slice or normalization or bbox or px_um'
```

- [ ] **Step 3: Implement exact producer receipts and runner checks**

Publish exact numeric inventories via the shared slice-order contract. Derive
P3→P4 lateral metric and distortion bounds from the flattened coordinates and
raster transform; remain `UNPROVEN` rather than using CT voxel size when absent.
Compute depth using the exact formula and bind all 33 inputs/P4 lineage. Resolve
the downstream-only ROI provenance and transform server-side; remain
`CONTROL_INCOMPLETE` while the real ROI lock is `verified:false`. Include
normalized requests, stored params, profiles, events, and hashes in stage rows.

- [ ] **Step 4: Run GREEN**

Run the RED command plus:

```text
python -m pytest -q tests/test_a_remote_worker_publishes_through_the_panel.py tests/test_p4_reads_the_flattened_sheet.py
```

---

### Task 6: Append-only server-derived human routing

**Files:**
- Modify: `framework/stages/01-segmentation/fleet/store.py`
- Modify: `framework/stages/01-segmentation/fleet/postgres_store.py`
- Modify: `panel/app.py`
- Modify: `scripts/harness/run_first_letters_positive_control.py`
- Test: `tests/test_first_letters_control_contracts.py`
- Test: `tests/test_first_letters_positive_control.py`

**Interfaces:**
- Request: `{verdict: "INSPECT", note: str}` only.
- Stored event: immutable server-derived mission/sample/surface/P7 job/verdict/card/config/packet hashes, author/time, event SHA.
- Resource: `POST /api/jobs/{p7_job_id}/review`; the exact P7 job is in the path.
- Storage: dedicated INSERT-only `human_review_events` table/API in both SQLite
  and PostgreSQL, immutable event ID/hash, unique `(p7_job_id,intent)`.
- Readback: exact-job `human_reviews: list[dict]` in append order.

- [ ] **Step 1: Write forged/mutable review RED tests**

Assert client lineage fields are rejected, wrong mission/sample/surface/P7 state
or packet is rejected, distinct allowed intents append rather than overwrite,
same job+intent retries are idempotent, concurrent inserts return one immutable
event, update/delete is unavailable, and runner ignores older lookalikes.

- [ ] **Step 2: Run RED**

```text
python -m pytest -q tests/test_first_letters_control_contracts.py tests/test_first_letters_positive_control.py -k 'review'
```

- [ ] **Step 3: Implement append-only routing**

Add dedicated compatible SQLite/PostgreSQL tables and INSERT/read-only store
methods; mutable JSON arrays are forbidden. Resolve and verify the exact path P7
job/artifact server-side, construct/canonical-hash the event, insert atomically
with unique-constraint idempotency, and return it. Remove client lineage fields.

- [ ] **Step 4: Bind runner to new event**

Snapshot review event hashes before POST, submit only intent/note, then poll for one
new exact event derived from the P7 job. Record the event/resource identity.

- [ ] **Step 5: Run GREEN**

Run:

```text
python -m pytest -q tests/test_first_letters_control_contracts.py tests/test_first_letters_positive_control.py tests/test_qc_job_diagnostics_api.py
```

---

### Task 7: Successful real-route entire-chain E2E and final verification

**Files:**
- Modify: `tests/e2e/test_first_letters_positive_control.py`
- Modify: `tests/test_first_letters_positive_control.py`
- Modify: `.superpowers/sdd/2026-08-02-first-letters-discovery-recovery/task-2-report.md` (ignored report)

**Interfaces:**
- TestClient adapter calls only real FastAPI routes/request models/readbacks.
- Fake seams: remote MCP/provider, worker completion, and artifact-object storage only.

- [ ] **Step 1: Write entire-chain RED E2E**

Build real mission/artifact/store state and make external workers publish exact
immutable receipts. First, use the actual policy and assert it stops
`CONTROL_INCOMPLETE` without P4/P7/review mutation because absolute orientation
and the ROI/scale receipt are not verified. Separately use a clearly synthetic,
fully hash-bound fixture for those scientific artifacts to run application
plumbing through HUMAN_REVIEW and assert `CONTROL_PASS`, exact job IDs, receipts,
and no acceptance claim. Never report that synthetic contract pass as a real
First Letters scientific pass.

- [ ] **Step 2: Run RED then complete minimal adapters**

```text
python -m pytest -q tests/e2e/test_first_letters_positive_control.py -k entire_chain
```

Expected RED: first missing immutable route/store contract. Implement only the
minimal test adapter/external seams needed; do not replace application routes or
weaken the actual policy's fail-closed locks.

- [ ] **Step 3: Run focused GREEN**

```text
python -m pytest -q tests/test_first_letters_positive_control.py tests/test_first_letters_control_contracts.py tests/test_first_letters_orientation_parity.py tests/e2e/test_first_letters_positive_control.py
```

- [ ] **Step 4: Run affected and full verification**

Run the established 297-test affected command, then:

```text
python -m pytest -q --tb=short
git diff --check
python -m py_compile framework/stages/01-segmentation/mcp/server.py framework/stages/01-segmentation/mcp/seed_candidates.py framework/stages/01-segmentation/fleet/worker.py framework/stages/01-segmentation/fleet/certifier.py framework/stages/01-segmentation/fleet/store.py framework/stages/01-segmentation/fleet/postgres_store.py framework/stages/01-segmentation/fleet/cli.py framework/stages/02-flattening/orientation_parity.py framework/stages/03-ink/fleet/job_store.py framework/stages/03-ink/fleet/ink_worker.py panel/app.py scripts/harness/run_first_letters_positive_control.py
```

Document the exact pass/failure counts and distinguish only reproduced sandbox
DNS/socket failures.

- [ ] **Step 5: Update report and commit fix round**

Append RED commands/failures, GREEN commands/results, full-suite evidence,
orientation rule, immutable review semantics, and known limitations to the ignored
Task 2 report. Stage only fix-round files, preserve the unrelated hybrid plan, and
commit with `fix: harden First Letters control evidence`.
