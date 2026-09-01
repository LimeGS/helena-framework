"""The terminal event has to name the surface the job flattened.

The control reached P3 for the first time and refused it:

    P3 -> INCOMPLETE | P3_CURRENT_JOB_EVIDENCE_MISSING

not because the flattening failed -- it succeeded, area ratio 0.949 against a
floor of 0.8, and published a hash-bound artifact to S3 -- but because one
conjunct of the evidence check had nothing to read. The control wants the job's
own terminal event as a second witness, independent of the flattening index:

    row.get("requested_by_job_id") == p3_job_id
    row.get("surface_id") == surface_artifact["artifact_id"]

and the `succeeded` payload carried only the runner's exit information:

    ['exit_code', 'liveness', 'output_dir', 'runtime_seconds', 'statistics',
     'stderr_tail', 'stdout_tail']

Everything else the check needs was already recorded. The flattening row had the
receipt, its digest, the objects, the artifact digest and uri. Only the witness
was missing, and the data for it was in the run receipt the job printed.

Why a second witness at all: the flattening index alone says a surface was
flattened, not that *this job* did it. Reading both is what makes a row that
some other run wrote unable to satisfy this run's boundary.

`P3` writes to a temp directory and publishes, so `output_dir` is empty and
`receipt_names()` finds no file -- the receipt exists only on stdout. That is
why this reads stdout, and why it reads the whole of it rather than the 4000
character tail the result stores: one surface fits, several do not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from ink_worker import flattening_lineage  # noqa: E402

JOB = "p3-050939d3a05646"
SURFACE = "7a2b066b-b2cd-5488-878c-5200a6cf8775"

# The shape the runner actually prints, trimmed to the fields under test.
RECEIPT = {
    "schema": "campaignx.surface_flattening_run.v1",
    "considered": 1,
    "flattened": {"FLATTENED": 1},
    "profile_id": "flatten-abf-v1@1.0.0",
    "surfaces": [{
        "surface_id": SURFACE,
        "requested_by_job_id": JOB,
        "receipt_sha256": "6540eed2bf14e73b24c0b0a095e4ea29fe56bda1469ab8cc94b06a8f431ea63c",
        "artifact_sha256": "30b0cb28c95630a2030b17e4e536e74c673a0460c308458b53125a9b3d62d831",
        "source_artifact_sha256": "5cf8e9ebac2527a882af52c08651d4a05901bdee2ebf9375840ef4cc63ce90a5",
        "artifact_uri": "s3://bucket/flattened-v1/surfaces/PHerc0139/flat/" + SURFACE,
        "artifact_id": "17ff9798-d86a-58ab-b0af-28c2a3da173d",
        "flattening_id": "17ff9798-d86a-58ab-b0af-28c2a3da173d",
        "profile_id": "flatten-abf-v1@1.0.0",
        "profile_file_sha256": "5771b2dc6465c8e59e530fa6410f5cdffafcda4b429ea3c5279b0e98834117a8",
        "state": "FLATTENED",
    }],
}

# What the geometry orientation proof reads off the same rows. It is a second
# consumer of this lineage, and the first version of `flattening_lineage`
# carried only what the control harness needed -- so P3 passed and the very next
# boundary refused with "P3 result lacks hash-bound flattened lineage".
ORIENTATION_PROOF_READS = (
    "surface_id", "requested_by_job_id", "profile_id", "profile_file_sha256",
    "source_artifact_sha256", "artifact_sha256", "receipt_sha256", "artifact_id",
)


def _stdout(receipt: dict, *, noise: str = "") -> str:
    return noise + json.dumps(receipt, indent=2)


def test_the_lineage_names_the_job_and_the_surface() -> None:
    """The two fields the control matches on, and the two digests it then
    compares against the flattening row."""
    rows = flattening_lineage(_stdout(RECEIPT), JOB)

    assert len(rows) == 1
    row = rows[0]
    assert row["requested_by_job_id"] == JOB
    assert row["surface_id"] == SURFACE
    assert row["receipt_sha256"] == RECEIPT["surfaces"][0]["receipt_sha256"]
    assert row["artifact_sha256"] == RECEIPT["surfaces"][0]["artifact_sha256"]


def test_the_control_s_own_check_passes_against_it() -> None:
    """Written as the consumer reads it, so this fails if the control's
    predicate and this producer ever drift apart."""
    result = {"surfaces": flattening_lineage(_stdout(RECEIPT), JOB)}

    candidates = [result, *(result.get("surfaces") or [])]
    lineage = next(
        (row for row in candidates
         if row.get("requested_by_job_id") == JOB
         and row.get("surface_id") == SURFACE), None)

    assert lineage is not None, (
        "the terminal event still gives the control nothing to match on")
    assert lineage["receipt_sha256"] == RECEIPT["surfaces"][0]["receipt_sha256"]


def test_the_lineage_carries_what_the_orientation_proof_reads() -> None:
    """The second consumer, which the first version of this forgot.

    P3 passed -- HASH_BOUND_FLATTENED_ARTIFACT -- and the geometry orientation
    proof immediately refused the same job with

        409 P3 result lacks hash-bound flattened lineage

    because it checks the profile file digest, the source artifact digest and an
    artifact id, and the lineage carried none of them. All three were sitting in
    the run receipt already; they simply were not copied. Fixing one reader and
    not the other is how the same defect gets found twice."""
    row = flattening_lineage(_stdout(RECEIPT), JOB)[0]

    missing = [field for field in ORIENTATION_PROOF_READS if not row.get(field)]
    assert not missing, f"the orientation proof reads {missing} and finds nothing"

    for field in ("profile_file_sha256", "source_artifact_sha256",
                  "artifact_sha256", "receipt_sha256"):
        value = row[field]
        assert len(value) == 64 and all(c in "0123456789abcdef" for c in value), (
            f"{field} must be a sha256; the proof matches it against [0-9a-f]{{64}}")


def test_another_job_s_surface_is_not_adopted() -> None:
    """The witness is only worth having if it cannot be borrowed. A receipt
    naming a different job must not become this job's evidence."""
    borrowed = json.loads(json.dumps(RECEIPT))
    borrowed["surfaces"][0]["requested_by_job_id"] = "p3-somebody-else"

    assert flattening_lineage(_stdout(borrowed), JOB) == []


def test_a_surface_that_did_not_flatten_is_not_reported() -> None:
    """A refused or failed surface is in the receipt too. Reporting it as
    lineage would say the job produced something it did not."""
    failed = json.loads(json.dumps(RECEIPT))
    failed["surfaces"][0]["state"] = "FLATTENING_FAILED"
    failed["surfaces"][0]["artifact_sha256"] = None

    assert flattening_lineage(_stdout(failed), JOB) == []


def test_runner_chatter_before_the_receipt_is_tolerated() -> None:
    """The binary logs before it prints the receipt, so the JSON is rarely the
    first thing on stdout."""
    noisy = _stdout(RECEIPT, noise="loading surface\nrunning abf\n")

    assert len(flattening_lineage(noisy, JOB)) == 1


def test_output_that_is_not_a_flattening_receipt_yields_nothing() -> None:
    """Every other phase goes through the same code path. None of them may
    crash the worker on the way to recording its result."""
    for stdout in ("", "not json at all", "{}", '{"schema": "something.else.v1"}',
                   '{"schema": "campaignx.surface_flattening_run.v1"}',
                   '{"schema": "campaignx.surface_flattening_run.v1", "surfaces": "no"}'):
        assert flattening_lineage(stdout, JOB) == []


def test_a_truncated_receipt_is_not_an_exception() -> None:
    """`stdout_tail` keeps 4000 characters and a multi-surface run overflows it.
    Reading a half receipt has to be empty evidence, not a failed job."""
    assert flattening_lineage(_stdout(RECEIPT)[:120], JOB) == []


# The half that makes the other half true: a helper nothing calls is not
# evidence, it is a function that reads like evidence.


class _RunStore:
    def __init__(self):
        self.finished = None

    def mark_running(self, *args, **kwargs):
        return None

    def heartbeat(self, *args, **kwargs):
        return None

    def note(self, *args, **kwargs):
        return None

    def finish(self, job_id, token, *, state, result):
        self.finished = {"state": state, "result": result}


def _p3_job() -> dict:
    return {"job_id": JOB, "lease_token": "lease", "phase": "P3",
            "sample_id": "PHerc0139", "parameters": {}}


def _run(monkeypatch, tmp_path, *, stdout: str, returncode: int = 0):
    from types import SimpleNamespace

    import ink_worker

    monkeypatch.setattr(ink_worker, "runner_for", lambda _job: "flatten")
    monkeypatch.setattr(ink_worker, "command_for",
                        lambda *_args, **_kwargs: ["flatten"])
    monkeypatch.setattr(ink_worker, "record_artifact",
                        lambda *_args, **_kwargs: None)
    # run_streaming, not subprocess.run: the worker echoes its child's output
    # as it arrives now, so that is the seam a stand-in has to take. Patching
    # subprocess.run left the real child to run and the job to fail.
    monkeypatch.setattr(ink_worker, "run_streaming", lambda *_args, **_kwargs:
                        SimpleNamespace(returncode=returncode, stdout=stdout,
                                        stderr=""))
    store = _RunStore()
    ink_worker.run_job(store, _p3_job(), runs_root=tmp_path, timeout=10)
    return store


def test_the_worker_puts_the_lineage_on_the_result(tmp_path, monkeypatch) -> None:
    """`result` is what `finish` stores as the terminal event's payload, so
    this is the boundary the control actually reads."""
    store = _run(monkeypatch, tmp_path, stdout=_stdout(RECEIPT))

    assert store.finished["state"] == "succeeded"
    surfaces = store.finished["result"].get("surfaces")
    assert surfaces, "the terminal event still carries only the runner's exit"
    assert surfaces[0]["surface_id"] == SURFACE
    assert surfaces[0]["requested_by_job_id"] == JOB


def test_a_failed_flattening_claims_no_lineage(tmp_path, monkeypatch) -> None:
    """A non-zero exit must not publish a witness saying the job produced a
    surface."""
    store = _run(monkeypatch, tmp_path, stdout=_stdout(RECEIPT), returncode=1)

    assert store.finished["state"] == "failed"
    assert not store.finished["result"].get("surfaces")
