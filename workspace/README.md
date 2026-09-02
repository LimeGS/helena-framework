# Workspace

`workspace/` holds run material only: source data, surfaces, plans, receipts,
evidence, findings and cache. It does not hold the framework — reusable code,
images, tests and documentation live at the repository root.

| Path | Canonical contents |
| --- | --- |
| `catalog/` | CT inventory, coordinate contract and public metadata |
| `surfaces/` | access to the public and private TIFXYZ libraries, without duplication |
| `campaigns/<id>/plans/` | manifests locked before a run |
| `campaigns/<id>/runs/` | per-stage receipts and their durable outputs |
| `campaigns/<id>/evidence/` | PNG, HTML, JSON and review evidence |
| `campaigns/<id>/findings/` | traceable candidates; never automatic acceptance |
| `campaigns/<id>/cache/` | CT staging, TIFFs and arrays that can be regenerated |

Only `catalog/` and these README files are tracked in this repository.
Everything a run produces — `campaigns/`, `surfaces/`, and the archive of past
campaigns — is deliberately untracked: it
is a data release with its own citation and its own licence, not a directory
people clone to read the code. See `NOTICE.md`.
