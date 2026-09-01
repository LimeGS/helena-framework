# First Letters Control Hardening Design

## Objective

Make a `CONTROL_PASS` possible only when the production MCP service and every
queued Helena stage return immutable, content-addressed evidence for the exact
frozen First Letters control lineage. Stale jobs, client-asserted hashes,
orientation guesses, incomplete source reads, and mutable review state must all
produce `CONTROL_INCOMPLETE` or `CONTROL_FAILED`, never a pass.

The work remains a development positive control. It does not deploy, execute a
real campaign, accept ink or text, or establish independent validation.

## Architecture

### Production M7 evidence

The production `vc_find_seed_candidates` implementation owns M7 evidence. Its
Zarr opener wraps the actual store and records every metadata and chunk object
read by Zarr. Each record contains canonical object identity, byte count, and
SHA-256. The service returns a sorted, duplicate-free
`campaignx.first_letters_source_read_set.v1` plus its canonical manifest hash.
The fleet client may merge and verify service evidence but never invent it.

The integration tests invoke the real MCP `Service.find_seed_candidates` over
both a local Zarr fixture and a range-capable HTTP/FsspecStore fixture. Production
opens are forced to Zarr v2. The tracker records the canonical object key and the
exact returned byte range (`offset`, `length`, response bytes/hash) before cache
reuse; repeated equal ranges coalesce and contradictory bytes fail closed.
Separate metadata (`.zattrs`, `0/.zarray`) is the frozen production contract.
Consolidated `.zmetadata` is accepted only when the source lock explicitly names
and hashes it; it cannot silently substitute for separately locked metadata.
Tests cover cache hits, partial HTTP reads, and JSON-RPC transport of the same
structured evidence.

### Runtime and request locks

`/api/state` reads every profile referenced by the frozen control manifest. The
existing `profile_locks[].sha256` declaration means SHA-256 of the raw file bytes,
so the response labels and verifies `actual_file_sha256` against it. It also
reports a separately named `actual_canonical_document_sha256`; the two hash
semantics are never conflated. It similarly reports the frozen control manifest,
sources, and installed model identities.

The preflight route accepts operational intent only. It constructs provider,
threshold, candidate policy, CT gate, locked TIFXYZ URI/artifacts/bbox, and probe
cap from the server-side frozen manifest. If compatibility requires those fields
in a request, supplied values must equal the frozen values byte-for-byte or the
server returns 409. The resulting M7 read set must contain the frozen required
root metadata objects with their exact declared hashes; the CT read set has the
same requirement.

### Exact queued-stage lineage

Every mutation returns a job ID, and the runner accepts evidence only from that
job ID after it reaches `succeeded`.

- P2: stored parameters name the exact grown surface. Its event/result persist
  `geometry_certified_by_job_id`, exact surface artifact ID/hash, certification
  profile ID/file hash, and result hash. P2 does not pretend that its queue job
  physically performed QC.
- QC: the original `qc_job_id` remains the physical-QC identity. Its immutable
  receipt binds its own job/profile/input/result and records
  `unblocked_by_job_id=<exact P2 job>` (and the corresponding promotion/event
  link). A QC row can satisfy the runner only when that causal link and every
  identity match the just-enqueued P2 lineage.
- P3: stored parameters name the exact certified surface and locked flattening
  profile. The current P3 job result and event bind the output URI, receipt,
  immutable object inventory, and artifact SHA to that job. Earlier flattening
  rows cannot satisfy a later request.
- P4: `ordered_tiff_files(..., require_numeric=True)` must produce exactly 33
  distinct numeric indices `0..32`. Actual filenames (normally `0.tif` through
  `32.tif`) and every file hash/byte count are bound; zero-padding is neither
  required nor normalized away silently.
- P5: depth spacing is exactly
  `selected_CT_voxel_um * P4.scale * P4.slice_step`. Lateral spacing is not
  equated to CT voxel size: it comes from a frozen P3→P4 metric receipt computed
  from flattened UV-to-3D triangle Jacobians, P4 raster transform, and declared
  distortion bounds. Missing/unbounded metric evidence is `UNPROVEN`. P5 binds
  the exact P4 job/artifact and the complete 33-file manifest, not sample files.
- P7: a full-map bbox is never treated as a letterform ROI. The frozen policy
  must contain a downstream-only positive-control letterform ROI, its provenance
  artifact/hash, source coordinate system, and exact transform into the verified
  P5 map. Discovery is forbidden from reading it. P7 vets exactly that transformed
  bbox and verified `px_um`. The current repository has no defensible immutable
  ROI transform, so its lock is explicitly `verified:false` and production
  outcome remains `CONTROL_INCOMPLETE` until supplied.

Immutable job and artifact readbacks extend existing routes minimally; no
parallel control-only queue is introduced.

### Orientation proof

Normal direction is never inherited from the community mesh and never defaults
silently. Before P4, a deterministic geometry-only parity diagnostic compares
the exact grown mesh artifact against the content-locked community w025 TIFXYZ
and requires the P3 receipt to prove that the flattened artifact comes from that
same grown-mesh hash. Parity transfers winding only; it cannot establish which
absolute physical side should be rendered.

The frozen `first-letters-orientation-parity@1.0.0` rule is:

1. Reference coordinates are finite and non-negative. A shared helper extracted
   from `fleet/finalizer.py` triangulates every grid cell using exactly
   `(v00,v10,v01)` and `(v10,v11,v01)`, the `v10-v01` anti-diagonal, with
   per-triangle validity. This changes the earlier bilinear-reference proposal
   to the repository's frozen piecewise-planar TIFXYZ surface contract.
2. A grown-surface vertex normal is the normalized area-weighted sum of its
   incident oriented triangle cross products. A zero-area triangle, non-finite
   coordinate/normal, or zero-length summed normal makes the diagnostic
   `UNPROVEN`; it is never skipped silently.
3. The reference is capped at 250,000 triangles and the grown artifact at
   2,000,000 vertices. A deterministic uniform-grid index with 4.0-voxel cells
   stores each triangle in every cell intersecting its AABB expanded by 2.0
   voxels. Triangle insertion count is capped at 4,000,000 and candidates per
   vertex at 4,096. Exceeding a size/operation cap or 60 seconds monotonic elapsed
   time returns `UNPROVEN`; the receipt records the cap reached.
4. For each grown vertex, query its grid cell, visit candidate triangles in
   row-major triangle ordinal, and use a fixed float64 closest-point-on-triangle
   solver. Only unquantized squared distances `<=4.0` survive. Distances are
   quantized to `1e-12` squared-voxel buckets solely for equality; ties choose
   lower triangle ordinal then lexicographically smaller barycentric weights.
5. Correspondences are ordered by `(reference_triangle_ordinal,
   quantized_barycentric_weights, grown_vertex_index)`. If more than 1,024 survive, take indices
   `floor(i*(N-1)/1023)` for `i=0..1023`; repeated indices are forbidden. Fewer
   than 64 valid correspondences is `UNPROVEN`.
6. For every retained correspondence compute the normalized dot product between
   grown and reference normals. Every dot must be finite and non-zero. Require
   median absolute dot `>= 0.90` and one strict sign to represent at least 95% of
   correspondences. Exactly 95% passes; a sign tie or any lower consensus is
   `UNPROVEN`.
7. Positive/negative consensus proves only same/opposite winding. Selecting
   `flip_normals` additionally requires a content-locked absolute-orientation
   evidence receipt for this exact w025 9.362um reference. No such receipt is
   committed: the old 0.09→0.885 prose/scripts target a different 2.399um mesh.
   Therefore the frozen absolute lock is `verified:false`, no boolean is selected,
   and the real control remains `CONTROL_INCOMPLETE`. Future evidence must bind
   exact P4 argv/job, grown mesh-window artifact and faces, all 63 source TIFFs,
   P5 checkpoint/profile/map hashes, reference read set, comparison output, and
   separately proven parity to w025.

The content-bound receipt contains the rule/profile ID and actual profile hash;
locked reference URI/meta/x/y/z hashes; grown mesh artifact/object inventory
(including faces) and flattened artifact IDs/hashes;
P3 job/profile/receipt hashes; valid/retained correspondence counts; sampling
indices; per-sample reference triangle/barycentric weights, distance, grown
vertex index, dot and sign; sign counts/consensus; median absolute dot;
absolute-orientation evidence lock; selected `flip_normals` only when both parity
and absolute side are proven; and its canonical receipt SHA. P4 must store and event-bind this
exact receipt hash and use exactly the proven boolean. The rule never reads CT
intensity, rendered image quality, P5/P7 evidence, ink, or text.

### Append-only human routing

`POST /api/jobs/{p7_job_id}/review` accepts only an `INSPECT` intent and note.
The server loads the exact P7 job and mission artifact, verifies mission/sample/surface, successful
job state, PASS adjudication, verdict/card/config hashes, and packet hash, then
inserts a new immutable review event with server-derived lineage into dedicated
SQLite/PostgreSQL `human_review_events` tables. Each row has an immutable event
ID/hash and a unique `(p7_job_id,intent)` idempotency key. The API exposes INSERT
and read only—no update/delete path. Concurrent duplicate submits return the same
event. It rejects client-supplied lineage assertions. The runner accepts only its
new server-derived event for the exact P7 job.

### Survival receipt

Each stage row records its normalized mutation request, exact job/readback,
stored parameters, locked and actual profile identities, resource identity,
input/output hashes, and counts. P1 additionally records the complete preflight
receipt SHA plus planned/visited/cap/fraction/completeness traversal fields.
Receipt hashing remains canonical and fails on NaN/Infinity.

## Failure semantics

- Missing source objects, lock drift, stale jobs, missing current-job events,
  missing inventories, unproven orientation, capped incomplete traversal, and
  ambiguous mutations are `CONTROL_INCOMPLETE`.
- Exhaustive scientific contradictions from otherwise complete evidence are
  `CONTROL_FAILED`.
- Later stages are normalized to `NOT_RUN_PREREQUISITE` after the first non-pass.
- No mutating request is retried automatically.

## Test strategy

Tests are written and observed failing before each production change:

1. Real MCP service and JSON-RPC handler return actual Zarr metadata/chunk reads.
2. `/api/state` rejects profile declarations that differ from actual files;
   preflight rejects client lock/policy drift and missing CT/M7 root reads.
3. Runner rejects stale P2/QC and P3 evidence from earlier jobs.
4. Runner rejects 32/misnumbered P4 slices, physical-scale drift, partial P5
   normalization, P7 bbox/scale drift, and unproven orientation.
5. Review route rejects forged lineage and preserves append-only history.
6. One TestClient-backed entire-chain contract test uses real FastAPI routes,
   request models, stores, polling/readbacks, and runner logic while replacing
   only remote provider, worker execution, and object storage. A synthetic,
   fully hash-bound orientation/ROI fixture may prove plumbing to `CONTROL_PASS`;
   the actual First Letters policy must separately prove it stops
   `CONTROL_INCOMPLETE` at the missing absolute-orientation/ROI locks.
7. Focused, affected, and full repository suites plus diff and compile checks
   complete before the fix commit.

## Scope boundary

The positive control uses zero cell-interior clearance. Task 3 must separately
define “eligible in any covering cell” for coordinate duplicates under a positive
clearance threshold. This fix does not deploy, mutate external systems, or run
the control against real infrastructure.
