# QC Job Diagnostics API Design

**Status:** Approved

**Date:** 2026-08-02

**Goal:** Make the persisted cause of a surface-QC retry visible through Helena's authenticated, read-only HTTP API without exposing secrets or changing queue state.

## Context

The First Letters hybrid campaign produced surface `99fd9127-548b-52bd-991b-ad6e7277db0c` for PHerc0268. Geometry is certified, but physical QC remains `UNVALIDATED`. Aggregate API evidence shows both QC workers repeatedly claiming jobs and requeueing them as `RETRYABLE_QC_UNAVAILABLE` before either GPU is used. The PostgreSQL row in `segment_qc_jobs.result` contains the last retry receipt, including the exception text, but no existing authorized API exposes that row.

The campaign contract permits staging mutations only through Helena's HTTP API. Direct SQL, SSH inspection, queue reordering, cancellation, and validation bypasses are out of scope. Therefore diagnosis needs a narrowly scoped API read before any root-cause fix is justified.

## Chosen Approach

Add `GET /api/segmentation/qc-jobs` to the existing FastAPI panel. It reads the QC queue and associated surface state from PostgreSQL, applies the existing mission/sample isolation boundary, and returns a strict field whitelist. Error output is normalized, redacted, and bounded before serialization.

This is preferred over extending `/api/fleet`, because aggregate fleet counters cannot bind an error to a surface. It is preferred over direct SQL or SSH because those routes violate the API-only campaign contract and bypass the panel's human authentication and mission boundaries.

## HTTP Contract

### Request

```text
GET /api/segmentation/qc-jobs
    ?mission=<mission-id>
    &sample=<bucket-or-canonical-sample-id>
    &surface=<surface-id>
    &limit=<1..500>
```

- `mission` is required and constrained to `1..128` characters by FastAPI;
  every read is bound to that mission's P1/P8 lineage.
- `sample` and `surface` are optional filters within the required mission.
- `limit` defaults to `100` and is constrained to `1..500` by FastAPI.
- `surface` is constrained to `1..128` characters.
- The route uses the panel's existing human-session authentication. No new authentication mechanism is introduced.
- This endpoint performs no `INSERT`, `UPDATE`, `DELETE`, queue claim, cancellation, lease renewal, retry, or worker operation.

### Successful Response

```json
{
  "available": true,
  "schema": "campaignx.segment_qc_job_diagnostics.v1",
  "scope": {
    "mission": "first-letters-hybrid-20260802",
    "samples": ["PHerc268"],
    "surface": "99fd9127-548b-52bd-991b-ad6e7277db0c"
  },
  "summary": {
    "PENDING": 1
  },
  "jobs": [
    {
      "qc_job_id": "<id>",
      "surface_id": "99fd9127-548b-52bd-991b-ad6e7277db0c",
      "sample_id": "PHerc268",
      "profile_id": "<profile>",
      "state": "PENDING",
      "created_at": "<ISO-8601>",
      "updated_at": "<ISO-8601>",
      "retry_after": "<ISO-8601-or-null>",
      "claim_count": 2,
      "last_status": "RETRYABLE_QC_UNAVAILABLE",
      "error_type": "FileNotFoundError",
      "error": "<sanitized bounded message>",
      "geometry_qc_state": "GEOMETRY_CERTIFIED",
      "physical_qc_state": "UNVALIDATED"
    }
  ]
}
```

The response includes only these job fields. In particular, it must never include `payload`, raw `result`, `worker_id`, `lease_token_hash`, `lease_expires_at`, artifact URI, DSN, HTTP authorization material, cookies, or private filesystem locations.

`claim_count` is the count of persisted `QC_CLAIMED` events whose payload names the QC job. It is diagnostic evidence, not a retry limit or mutable attempt counter.

`summary` counts states only within the filtered and limited `jobs` result. It is not a global queue total and must not be presented as one.

### Empty Scope

If a mission contains no scrolls, or `sample` is outside the named mission, return HTTP 200 with `available: true`, an empty `jobs` list, and an empty `summary`. An empty mission-scoped read must never fall back to global data.

If `surface` does not exist inside the resolved mission/sample scope, return the same empty result rather than revealing whether that surface exists elsewhere.

### Degraded Response

If `CX_DB` is unset, the PostgreSQL driver is unavailable, or the query fails, return HTTP 200 with:

```json
{
  "available": false,
  "schema": "campaignx.segment_qc_job_diagnostics.v1",
  "reason_code": "DATABASE_UNAVAILABLE"
}
```

Allowed reason codes are `DATABASE_UNAVAILABLE`, `DRIVER_UNAVAILABLE`, and `DIAGNOSTICS_QUERY_FAILED`. A query failure may also include `error_type` equal to the Python exception class name. It must not include `str(exception)` because database exceptions can contain DSNs, SQL fragments, private hostnames, and values.

## Data Selection and Mission Isolation

The route resolves `mission` and `sample` with the existing `read_scope(mission, sample)` function. Bucket spelling such as `PHerc0268` is normalized to the control-plane spelling `PHerc268`.

Jobs are selected by joining `segment_qc_jobs q` to `segment_surfaces s` on `surface_id`. A mission owns a surface if either:

1. the surface artifact is linked through `segment_artifact_sets`, `segment_attempts`, and `segment_tasks` to the requested `mission_id`; or
2. the surface is a derived child linked through `surface_derivations` and `ink_jobs` to the requested `mission_id`.

This is the same P1/P8 ownership model used by the existing flattening read. Mission, sample, and surface predicates are combined with `AND`. Rows are ordered newest first by `q.updated_at`, then `q.created_at`, then `q.qc_job_id`, and limited in SQL.

The claim count is computed for the already filtered rows only. It uses `segment_events.event_type = 'QC_CLAIMED'` and `segment_events.payload ->> 'qc_job_id' = q.qc_job_id`.

## Error Normalization and Redaction

Only `result.status` and `result.error` are considered. All other result keys are ignored.

The serializer performs these operations in order:

1. Convert a string error to text; non-string error values are discarded.
2. Collapse all whitespace runs to one space.
3. Extract `error_type` only when the leading token matches `[A-Za-z_][A-Za-z0-9_.]{0,127}` followed by a colon. Otherwise use `null`.
4. Replace case-insensitive bearer/basic authorization values, cookie assignments, password/secret/token/access-key assignments, and AWS access-key identifiers with `<redacted>`.
5. Replace PostgreSQL DSNs with `<dsn>`, HTTP(S) URLs with `<url>`, `s3://` URIs with `<artifact-uri>`, and absolute Unix paths with `<path>`.
6. Truncate the final message to 500 Unicode code points. If truncation occurs, the last character is `…` and total length remains at most 500.
7. If no safe message remains, return `error: null`.

Redaction is defense in depth, not authorization. The database query still excludes payload, worker, lease, artifact, and unrelated result fields so those values never reach the response builder.

## Code Boundaries

- `panel/app.py`
  - Add a small pure helper that converts a QC `result` object into `last_status`, `error_type`, and sanitized `error` fields.
  - Add the new route next to the existing segmentation read endpoints.
  - Keep query construction and serialization local to the route, following the current panel structure; do not refactor unrelated endpoints.
- `tests/test_qc_job_diagnostics_api.py`
  - Exercise the public route with a fake PostgreSQL connection/cursor using the same monkeypatch style as existing panel endpoint tests.
  - Exercise the pure sanitizer directly for malicious and oversized inputs.

No database migration, frontend component, worker change, new dependency, or queue-store method is required.

## Testing Requirements

The implementation is accepted only when tests prove all of the following:

1. A mission-scoped request returns only QC jobs whose P1 or derived P8 lineage belongs to that mission.
2. Bucket-style sample IDs normalize correctly, and a sample outside the mission produces an empty result without querying globally.
3. The optional surface filter cannot reveal a surface outside the resolved scope.
4. Pending, claimed, completed, and retry-delayed rows serialize with the documented whitelist and ISO timestamps.
5. `claim_count` reflects matching persisted `QC_CLAIMED` events.
6. The state summary covers only returned jobs.
7. Raw `payload`, raw `result`, worker ID, lease data, artifact URI, private path, DSN, authorization value, cookie, token, secret, and AWS access-key identifiers do not appear anywhere in the encoded response.
8. Error messages are normalized and bounded to 500 code points.
9. Missing DB configuration, missing driver, and query failures degrade with the documented safe reason codes and do not echo exception text.
10. A request performs only `SELECT` statements and commits no database mutation.
11. FastAPI rejects `limit` outside `1..500` and an empty or oversized `surface` value.
12. Existing mission-isolation and segmentation endpoint tests remain green.

TDD is mandatory: each production behavior is introduced only after its focused test has been observed failing for the expected missing behavior.

## Deployment and Runtime Validation

After focused and relevant full-suite tests pass:

1. Commit and push the implementation branch.
2. Advance staging through the repository's existing deployment workflow.
3. Verify every Helena service reports the deployed revision and the staging smoke checks pass.
4. Authenticate through the normal human session endpoint using an ephemeral mode-0600 cookie jar.
5. Call the new endpoint with mission `first-letters-hybrid-20260802`, sample `PHerc0268`, and surface `99fd9127-548b-52bd-991b-ad6e7277db0c`.
6. Record only the sanitized response and its SHA-256 in the campaign ledger; remove the cookie jar.
7. State one root-cause hypothesis from that response, reproduce it, and create a separate focused TDD fix plan before editing the failing QC path.
8. Deploy the root-cause fix and allow the existing pending QC job to retry naturally. Do not cancel, duplicate, prioritize, reorder, or mark the surface validated manually.

## Non-Goals

- A general database browser or arbitrary job-debug endpoint.
- Exposing raw receipts, payloads, worker identities, lease material, artifact locations, credentials, or internal paths.
- Changing QC scheduling, retry timing, ordering, or terminal-state semantics.
- Adding a manual retry, cancellation, validation, or bypass control.
- Advancing the First Letters surface while physical QC is unvalidated.
- Claiming CT support, ink, or letters from geometry or automated output alone.

## Acceptance Criteria

- The endpoint is authenticated, read-only, mission-isolated, field-whitelisted, and safely degraded.
- A surface-specific query provides enough sanitized evidence to identify the failing QC boundary without direct infrastructure access.
- Automated tests demonstrate isolation, redaction, bounded output, read-only behavior, and degraded behavior.
- Staging deploy and smoke checks succeed on one exact revision.
- The original pending QC job remains intact and is neither cancelled nor reordered.
