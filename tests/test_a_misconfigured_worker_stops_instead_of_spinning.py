"""A failure that cannot resolve itself must not be retried.

On 2026-07-29 a rename changed the ink adapter's filename, which changed the ink
profile, which changed the surface-QC profile that pins it by hash. gpu-1's
HELENA_QC_PROFILE_SHA256 was never updated, so the adapter refused every job
with "surface-QC profile hash differs". The worker caught it as an outage and
requeued it. Both GPUs then claimed, failed in a second, requeued and reclaimed
for two days: 3118 receipts, zero surfaces measured.

Nothing was visibly wrong. The jobs sat PENDING, which is what a job waiting for
a free worker looks like, and the cards reported utilisation, which is what work
looks like. The queue could not distinguish "not started yet" from "started
three thousand times".

So the distinction is the fix, and it has three parts:

  * the adapter says which kind of failure it had, because the worker sees an
    exit code and nothing else;
  * a configuration failure is terminal -- claim_qc takes only PENDING, so
    BLOCKED_CONFIGURATION is never picked up again;
  * it is not a scientific verdict. A wrong hash says nothing about the papyrus,
    and the surface row must come through untouched.

The last one is the one worth being careful about: the easy implementation
routes this through finalize_qc, and then an operator's mistake is recorded as a
measurement of a scroll.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.qc_worker import (  # noqa: E402
    EX_CONFIG,
    QcConfigurationError,
    SurfaceQcWorker,
    _adapter_complaint,
)

ADAPTER = ROOT / "framework/stages/04-validation/scripts/campaignx_surface_qc_adapter.py"


# --------------------------------------------------------------------------
# The adapter says which kind of failure it had
# --------------------------------------------------------------------------

def test_the_adapter_exits_with_a_code_that_means_configuration(tmp_path):
    """End to end through a real process, because the exit code is the whole
    channel: everything the worker learns about a config error crosses here.

    Run with no HELENA_QC_* set at all, which is the simplest configuration
    error there is.
    """
    payload = {"schema": "campaignx.segment_qc_input.v1",
               "qc_job": {"surface_id": "s", "profile_id": "p@1.0.0",
                          "source": {"sample_id": "x"}, "surface": {}}}
    request = tmp_path / "QC_INPUT.json"
    request.write_text(json.dumps(payload))
    output = tmp_path / "out"
    output.mkdir()

    finished = subprocess.run(
        [sys.executable, str(ADAPTER), "--input", str(request), "--output", str(output)],
        capture_output=True, text=True, timeout=180,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert finished.returncode == EX_CONFIG, (
        f"exited {finished.returncode} with {finished.stderr[-400:]!r}; a "
        "configuration error is indistinguishable from a crash at any other code"
    )
    report = json.loads(finished.stderr.strip().splitlines()[-1])
    assert report["status"] == "BLOCKED_CONFIGURATION"
    assert report["no_scientific_conclusion"] is True
    assert "HELENA_QC" in report["error"], report["error"]


def test_the_worker_repeats_what_the_adapter_said_rather_than_the_exit_code():
    """"exit code 78" is not something anybody can act on."""
    said = _adapter_complaint(
        'Traceback (most recent call last):\n'
        '{"schema": "campaignx.segment_qc_configuration_error.v1", '
        '"status": "BLOCKED_CONFIGURATION", "error": "profile hash differs", '
        '"no_scientific_conclusion": true}')
    assert said == "profile hash differs"
    # And when the adapter says nothing useful, say that, rather than "".
    assert "did not say which" in _adapter_complaint("boom\n")


def _claim():
    return {"qc_job_id": "job-1", "surface_id": "surface-1",
            "lease_token": "token-1", "profile_id": "p@1.0.0"}


def test_the_executor_turns_exit_78_into_a_configuration_error(tmp_path):
    """The join between the two halves, and the piece nothing else covers.

    Every other test here either drives a fake executor or runs the adapter
    alone. If SubprocessQcExecutor stopped recognising the exit code, both would
    still pass and the fleet would be back to requeuing forever -- so this runs
    a stub adapter through the real executor.
    """
    from fleet.qc_worker import SubprocessQcExecutor

    stub = tmp_path / "adapter.py"
    stub.write_text(
        "import sys, json\n"
        "print(json.dumps({'status': 'BLOCKED_CONFIGURATION',\n"
        "                  'error': 'the pin and the profile disagree'}))\n"
        "sys.exit(78)\n")
    attempt = tmp_path / "attempt"
    attempt.mkdir()

    with pytest.raises(QcConfigurationError) as refused:
        SubprocessQcExecutor(stub).execute(_claim(), attempt)
    assert "the pin and the profile disagree" in str(refused.value)


def test_any_other_non_zero_exit_stays_an_ordinary_failure(tmp_path):
    """Only 78 means configuration. A crash is still a crash, and still retried."""
    from fleet.qc_worker import SubprocessQcExecutor

    stub = tmp_path / "adapter.py"
    stub.write_text("import sys\nsys.exit(1)\n")
    attempt = tmp_path / "attempt"
    attempt.mkdir()

    with pytest.raises(RuntimeError) as failed:
        SubprocessQcExecutor(stub).execute(_claim(), attempt)
    assert not isinstance(failed.value, QcConfigurationError)
    assert "exit code 1" in str(failed.value)


# --------------------------------------------------------------------------
# A configuration failure is terminal, and is not a verdict
# --------------------------------------------------------------------------

class Recording:
    """A store that records which finishing move the worker chose.

    It answers the routing and lineage questions because a control plane does.
    A double that stays silent about them would make this file green while the
    worker refused every real claim, which is the opposite of what it checks.
    """

    def routing_receipt(self, surface_id):
        from fleet import surface_routing

        return surface_routing.build_receipt(
            surface_id=surface_id, area_cm2=0.5,
            policy=surface_routing.load_policy(),
            measurement={}, read_set={},
        )

    def surface_artifact(self, surface_id, *, boundary):
        return {"surface_id": surface_id, "payload": {}}

    def __init__(self, claim):
        self._claim = claim
        self.calls: list[str] = []
        self.requeued_receipt: dict | None = None
        self.blocked_receipt: dict | None = None

    def claim_qc(self, *a, **k):
        return self._claim

    def heartbeat_qc(self, *a, **k):
        self.calls.append("heartbeat")

    def finalize_qc(self, *a, **k):
        self.calls.append("finalize")
        return {"status": "COMPLETED"}

    def requeue_qc_unavailable(self, *a, **k):
        self.calls.append("requeue")
        self.requeued_receipt = a[2]
        return {"status": "RETRYABLE_QC_UNAVAILABLE"}

    def block_qc_configuration(self, qc_job_id, lease_token, receipt):
        self.calls.append("block")
        self.blocked_receipt = receipt
        return {"status": "BLOCKED_CONFIGURATION", "qc_job_id": qc_job_id,
                "surface_id": receipt["surface_id"], "error": receipt["error"]}


class Raises:
    def __init__(self, error):
        self.error = error

    def execute(self, claim, attempt_dir):
        raise self.error


def _worker(store, error, tmp_path):
    return SurfaceQcWorker(store, "worker-1", Raises(error), tmp_path)


def test_a_configuration_error_blocks_the_job(tmp_path):
    """The regression itself. Reverting the fix requeues, and the fleet spins."""
    store = Recording(_claim())
    result = _worker(store, QcConfigurationError("profile hash differs"),
                     tmp_path).run_one()

    assert "requeue" not in store.calls, (
        "a configuration error was requeued; the job returns to PENDING and is "
        "claimed again, forever, because it will fail the same way every time"
    )
    assert "block" in store.calls
    assert result["status"] == "BLOCKED_CONFIGURATION"
    assert result["error"] == "profile hash differs", (
        "the job is blocked without saying what to change"
    )


def test_a_configuration_error_is_not_a_scientific_verdict(tmp_path):
    """finalize_qc writes a surface state and a physical_qc_state.

    Routing a misconfiguration through it would record an operator's mistake as
    a measurement of a scroll, which is worse than the retry loop it replaced.
    """
    store = Recording(_claim())
    _worker(store, QcConfigurationError("checkpoint hash differs"), tmp_path).run_one()
    assert "finalize" not in store.calls

    receipt = json.loads(
        next(tmp_path.rglob("BLOCKED_CONFIGURATION_RECEIPT.json")).read_text())
    assert receipt["no_scientific_conclusion"] is True
    assert receipt["status"] == "BLOCKED_CONFIGURATION"
    assert "outcome" not in receipt


def test_exit_78_sanitizes_adapter_detail_before_blocked_receipt_and_store(tmp_path):
    """A blocked adapter report must never serialize raw operational secrets.

    Removing the persistence-boundary sanitizer makes this test retain the
    adapter's Token credential, Cookie assignment, DSNs, artifact URI, Windows
    path, worker identity, and generic URI in the durable receipt.
    """
    from fleet.qc_worker import SubprocessQcExecutor

    raw = (
        "profile hash differs Authorization: Token config-token "
        "Authorization Token delimiter-free-token "
        "Cookie=session-cookie redis://alice:redis-secret@private-redis:6379/0 "
        "mysql://bob:mysql-secret@private-mysql/qc "
        "gs://private-bucket/checkpoint.bin C:\\private\\checkpoint.bin "
        "worker_id=blocked-underscore-worker worker-id=blocked-hyphen-worker "
        "worker identity=blocked-space-worker api_key=adapter-api-key "
        "client_secret=adapter-client-secret "
        "custom+artifact://private-authority/item"
    )
    stub = tmp_path / "adapter.py"
    stub.write_text(
        "import json, sys\n"
        f"print(json.dumps({{'status': 'BLOCKED_CONFIGURATION', 'error': {raw!r}}}))\n"
        "sys.exit(78)\n"
    )
    store = Recording(_claim())
    worker = SurfaceQcWorker(
        store, "worker-1", SubprocessQcExecutor(stub), tmp_path / "runs")

    result = worker.run_one()

    assert result["status"] == "BLOCKED_CONFIGURATION"
    assert store.calls[-1] == "block"
    assert "requeue" not in store.calls
    assert store.blocked_receipt is not None
    disk_receipt = json.loads(
        next(tmp_path.rglob("BLOCKED_CONFIGURATION_RECEIPT.json")).read_text())
    for receipt in (store.blocked_receipt, disk_receipt):
        assert receipt["status"] == "BLOCKED_CONFIGURATION"
        assert receipt["no_scientific_conclusion"] is True
        assert len(receipt["error"]) <= 500
        for forbidden in (
            "config-token", "session-cookie", "redis-secret", "private-redis",
            "mysql-secret", "private-mysql", "private-bucket", "C:\\private",
            "blocked-underscore-worker", "blocked-hyphen-worker",
            "blocked-space-worker", "adapter-api-key", "adapter-client-secret",
            "private-authority", "delimiter-free-token",
        ):
            assert forbidden not in json.dumps(receipt, sort_keys=True)


def test_exit_78_sanitizes_nonstring_configuration_mapping(tmp_path):
    """JSON error objects cross `_adapter_complaint` through their string form."""
    from fleet.qc_worker import SubprocessQcExecutor

    raw_error = {
        "Authorization": "Digest username=alice, realm=private-realm, response=digest-secret",
        "Cookie": "a=first-cookie, b=second-cookie",
    }
    stub = tmp_path / "adapter.py"
    stub.write_text(
        "import json, sys\n"
        f"print(json.dumps({{'status': 'BLOCKED_CONFIGURATION', "
        f"'error': {raw_error!r}}}))\n"
        "sys.exit(78)\n"
    )
    store = Recording(_claim())

    SurfaceQcWorker(
        store, "worker-1", SubprocessQcExecutor(stub), tmp_path / "runs"
    ).run_one()

    assert store.blocked_receipt is not None
    disk_receipt = json.loads(
        next(tmp_path.rglob("BLOCKED_CONFIGURATION_RECEIPT.json")).read_text()
    )
    for receipt in (store.blocked_receipt, disk_receipt):
        encoded = json.dumps(receipt, sort_keys=True)
        assert "<redacted>" in encoded
        for forbidden in (
            "alice", "private-realm", "digest-secret", "first-cookie",
            "second-cookie",
        ):
            assert forbidden not in encoded


def test_manually_raised_configuration_error_is_sanitized_at_persistence_boundary(
        tmp_path):
    """Direct callers cannot bypass the blocked-receipt redaction boundary."""
    raw = (
        "configuration worker_id=manual-worker Cookie=manual-cookie "
        + "x " * 600
    )
    store = Recording(_claim())

    _worker(store, QcConfigurationError(raw), tmp_path).run_one()

    assert store.blocked_receipt is not None
    error = store.blocked_receipt["error"]
    assert "manual-worker" not in error
    assert "manual-cookie" not in error
    assert len(error) <= 500


def test_a_genuine_outage_still_requeues(tmp_path):
    """The other half. S3 being down for a minute is exactly what retry is for,
    and narrowing that would trade one failure mode for its opposite."""
    store = Recording(_claim())
    result = _worker(store, OSError("connection reset by peer"), tmp_path).run_one()

    assert "requeue" in store.calls
    assert "block" not in store.calls
    assert result["status"] == "RETRYABLE_QC_UNAVAILABLE"
    assert next(tmp_path.rglob("RETRYABLE_QC_RECEIPT.json")).is_file()


def test_python_adapter_failure_persists_only_safe_exception_detail(tmp_path):
    """An adapter traceback is useful only after its sensitive detail is removed.

    Replacing this with the generic RuntimeError loses the actionable exception
    type; persisting the complete traceback leaks an internal path and token.
    """
    from fleet.qc_worker import SubprocessQcExecutor

    stub = tmp_path / "adapter.py"
    stub.write_text(
        "raise FileNotFoundError("
        "'/srv/private/checkpoint.bin token=do-not-persist')\n"
    )
    store = Recording(_claim())
    worker = SurfaceQcWorker(
        store,
        "worker-1",
        SubprocessQcExecutor(stub),
        tmp_path / "runs",
    )

    result = worker.run_one()

    assert result["status"] == "RETRYABLE_QC_UNAVAILABLE"
    assert store.calls[-1] == "requeue"
    assert store.requeued_receipt is not None
    error = store.requeued_receipt["error"]
    assert error.startswith("FileNotFoundError: ")
    assert "<path>" in error
    assert "<redacted>" in error
    assert "/srv/private" not in error
    assert "do-not-persist" not in error
    assert len(error) <= 500
    assert store.requeued_receipt["no_scientific_conclusion"] is True


def test_native_adapter_failure_persists_only_generic_exit_wrapper(tmp_path):
    """Native adapter output is never safe to extract or persist verbatim.

    Changing the generic wrapper to output-derived text would expose credentials
    and paths; the worker must retain only its fixed safe failure description.
    """
    from fleet.qc_worker import SubprocessQcExecutor

    stub = tmp_path / "adapter.sh"
    stub.write_text(
        "#!/bin/sh\n"
        "echo 'FatalError: node-worker-77 opaque-value' >&2\n"
        "exit 1\n"
    )
    stub.chmod(0o700)
    store = Recording(_claim())
    worker = SurfaceQcWorker(
        store,
        "worker-1",
        SubprocessQcExecutor(stub),
        tmp_path / "runs",
    )

    result = worker.run_one()

    assert result["status"] == "RETRYABLE_QC_UNAVAILABLE"
    assert store.requeued_receipt is not None
    disk_receipt = json.loads(
        next(tmp_path.rglob("RETRYABLE_QC_RECEIPT.json")).read_text()
    )
    for receipt in (store.requeued_receipt, disk_receipt):
        error = receipt["error"]
        assert error == (
            "QcAdapterExecutionError: scientific QC adapter failed with exit code 1"
        )
        assert len(error) <= 500
        encoded = json.dumps(receipt, sort_keys=True)
        for forbidden in ("FatalError", "node-worker-77", "opaque-value"):
            assert forbidden not in encoded
    attempt_log = next(tmp_path.rglob("scientific-executor.log")).read_text()
    assert attempt_log == "FatalError: node-worker-77 opaque-value\n"


def test_non_adapter_retry_error_is_sanitized_and_bounded_before_persistence(
        tmp_path):
    """The ordinary retry path must enforce the same safe error boundary.

    Bypassing redaction or length bounding here lets worker-local failures leak
    credentials and oversized messages into the durable retry receipt.
    """
    raw = (
        "outage /srv/private/model.bin "
        "postgresql://alice:hunter2@private-db/campaignx "
        "Authorization: Token retry-token Cookie=retry-cookie "
        "gs://private-bucket/checkpoint.bin C:\\private\\checkpoint.bin "
        "worker_id=retry-underscore-worker worker-id=retry-hyphen-worker "
        "worker id=retry-space-worker token=raw-token "
        "AWS_SECRET_ACCESS_KEY=raw-aws-secret "
        + "x" * 700
    )
    store = Recording(_claim())

    _worker(store, RuntimeError(raw), tmp_path).run_one()

    assert store.requeued_receipt is not None
    error = store.requeued_receipt["error"]
    assert len(error) <= 500
    assert "<path>" in error
    assert "<dsn>" in error
    assert "<artifact-uri>" in error
    assert "<redacted>" in error
    for forbidden in (
        "/srv/private", "hunter2", "private-db", "raw-token",
        "raw-aws-secret", "retry-token", "retry-cookie", "private-bucket",
        "C:\\private", "retry-underscore-worker", "retry-hyphen-worker",
        "retry-space-worker",
    ):
        assert forbidden not in error


@pytest.mark.parametrize(
    ("raw", "forbidden", "replacement"),
    [
        (
            "Authorization: Digest username=alice, realm=private-realm, "
            "response=digest-secret",
            ("alice", "private-realm", "digest-secret"),
            "<redacted>",
        ),
        (
            "Proxy-Authorization: Digest username=bob, realm=proxy-realm, "
            "response=proxy-secret",
            ("bob", "proxy-realm", "proxy-secret"),
            "<redacted>",
        ),
        (
            "Cookie: a=first-cookie, b=second-cookie",
            ("first-cookie", "second-cookie"),
            "<redacted>",
        ),
        (
            "Set-Cookie: a=first-set-cookie, b=second-set-cookie",
            ("first-set-cookie", "second-set-cookie"),
            "<redacted>",
        ),
        (
            "Authorization: Custom raw auth secret",
            ("Custom", "raw auth secret"),
            "<redacted>",
        ),
        (
            "Cookie: raw cookie secret",
            ("raw cookie secret",),
            "<redacted>",
        ),
        (
            str({"token": "raw-secret", "worker_id": "node-17"}),
            ("raw-secret", "node-17"),
            "<redacted>",
        ),
        (
            "'password': 'two word secret', "
            '"worker_id": "private worker 17"',
            ("two word secret", "private worker 17"),
            "<redacted>",
        ),
        (
            "token=raw unquoted secret phrase",
            ("unquoted secret phrase",),
            "<redacted>",
        ),
        (
            r"C:\Users\Alice Smith\secret model.bin",
            ("Alice Smith", "secret model.bin"),
            "<path>",
        ),
        (
            r'"C:\Users\Alice Smith\secret model.bin"',
            ("Alice Smith", "secret model.bin"),
            "<path>",
        ),
        (
            r"\\private-server\secret share\hidden model.bin",
            ("private-server", "secret share", "hidden model.bin"),
            "<path>",
        ),
        (
            '"/srv/private/Alice Smith/secret model.bin"',
            ("/srv/private", "Alice Smith", "secret model.bin"),
            "<path>",
        ),
        (
            "/srv/private/Alice Smith/secret model.bin",
            ("/srv/private", "Alice Smith", "secret model.bin"),
            "<path>",
        ),
    ],
)
def test_worker_retry_boundary_fully_redacts_sensitive_multitoken_forms(
    raw, forbidden, replacement, tmp_path,
):
    """Every worker fallback is safe before its receipt reaches any store."""
    store = Recording(_claim())

    _worker(store, RuntimeError(raw), tmp_path).run_one()

    assert store.requeued_receipt is not None
    disk_receipt = json.loads(
        next(tmp_path.rglob("RETRYABLE_QC_RECEIPT.json")).read_text()
    )
    for receipt in (store.requeued_receipt, disk_receipt):
        error = receipt["error"]
        assert replacement in error
        assert len(error) <= 500
        for value in forbidden:
            assert value not in error


@pytest.mark.parametrize("error", [
    KeyboardInterrupt(), MemoryError(), RuntimeError("adapter failed with exit code 1"),
])
def test_anything_that_is_not_a_configuration_error_keeps_the_old_behaviour(
        error, tmp_path):
    store = Recording(_claim())
    _worker(store, error, tmp_path).run_one()
    assert store.calls[-1] == "requeue"


# --------------------------------------------------------------------------
# The store: blocked means blocked
# --------------------------------------------------------------------------

def test_a_blocked_job_is_not_claimable_again():
    """claim_qc's predicate is the whole guarantee. If BLOCKED_CONFIGURATION
    were claimable this would be the retry loop with a new name."""
    for name in ("postgres_store", "store"):
        source = (ROOT / f"framework/stages/01-segmentation/fleet/{name}.py").read_text()
        claim = source.split("def claim_qc")[1].split("\n    def ")[0]
        assert "BLOCKED_CONFIGURATION" not in claim, (
            f"{name}.claim_qc mentions the blocked state"
        )

        blocking = source.split("def block_qc_configuration")[1].split("\n    def ")[0]
        # Past the docstring, which discusses the very things forbidden below --
        # the first version of this test matched its own prose, and the second
        # split on every triple quote and tore the SQL in half. Drop exactly the
        # signature and the docstring, keep the rest joined.
        parts = blocking.split('"""')
        statements = '"""'.join(parts[2:])
        assert "state='BLOCKED_CONFIGURATION'" in statements
        assert "retry_after=NULL" in statements, (
            "a blocked job keeps a retry time, which reads as pending-with-a-delay"
        )
        # The surface is somebody else's fact and must come through untouched.
        for forbidden in ("UPDATE surfaces", "UPDATE segment_surfaces"):
            assert forbidden not in statements, (
                f"{name} writes a surface state for a configuration error, which "
                "turns an operator's mistake into a measurement"
            )


def test_both_stores_offer_it():
    """The deployment runs PostgreSQL and the tests run SQLite. A worker calling
    a method one of them lacks fails at the moment it is most needed."""
    from fleet.postgres_store import PostgresFleetStore
    from fleet.store import FleetStore

    for store in (PostgresFleetStore, FleetStore):
        assert callable(getattr(store, "block_qc_configuration", None)), store.__name__


_RAW_STORE_ERROR = (
    "RuntimeError: Authorization: Digest username=store-user, "
    "realm=store-realm, response=store-digest Cookie: a=store-cookie-one, "
    "b=store-cookie-two 'token': 'store secret phrase', "
    r"'worker_id': 'store worker 17' C:\Users\Store User\secret model.bin "
    r"\\store-server\private share\hidden model.bin "
    "/srv/private/Store User/secret model.bin "
    + "x" * 700
)
_QUOTED_HEADER_STORE_ERRORS = (
    "RuntimeError: {'Authorization': 'Digest username=alice, realm=private-realm, response=digest-secret'}",
    "RuntimeError: {'Cookie': 'a=first-cookie, b=second-cookie'}",
)
_STORE_FORBIDDEN = (
    "store-user",
    "store-realm",
    "store-digest",
    "store-cookie-one",
    "store-cookie-two",
    "store secret phrase",
    "store worker 17",
    "Store User",
    "secret model.bin",
    "store-server",
    "private share",
    "/srv/private",
    "alice",
    "private-realm",
    "digest-secret",
    "first-cookie",
    "second-cookie",
)


def _qc_receipt(claim, status, error):
    return {
        "schema": "campaignx.test.raw_qc_receipt.v1",
        "status": status,
        "qc_job_id": claim["qc_job_id"],
        "surface_id": claim["surface_id"],
        "error": error,
        "no_scientific_conclusion": True,
    }


def _assert_store_error_is_safe(error):
    assert isinstance(error, str)
    assert len(error) <= 500
    for forbidden in _STORE_FORBIDDEN + ("mapping-secret", "mapping-worker"):
        assert forbidden not in error


def _claimed_sqlite_qc(tmp_path, suffix):
    from fleet.store import FleetStore

    store = FleetStore(tmp_path / f"fleet-{suffix}.sqlite")
    store.initialize()
    source_id = store.register_snapshot({
        "sample_id": "PHercSTORE",
        "ct_uri": "fixture://ct",
        "ct_sha256": "0" * 64,
        "m7_uri": "fixture://m7",
        "m7_sha256": "1" * 64,
        "shape_xyz": [32, 32, 32],
        "voxel_size_um": 1.0,
        "coordinate_frame": "ct_l0_xyz",
    })
    enqueued = store.enqueue_imported_surface_qc(
        {
            "surface_id": f"surface-{suffix}",
            "source_snapshot_id": source_id,
            "sample_id": "PHercSTORE",
            "artifact_sha256": "2" * 64,
            "artifact_uri": f"fixture://surface-{suffix}",
            "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
            # A measured area above the effort floor, because a QC enqueue now
            # requires a routing decision and an unmeasured surface has none.
            # The subject here is what a store error may say, not what a surface
            # may be, so the fixture supplies the measurement the boundary asks
            # for rather than the boundary being relaxed for the fixture.
            "area_cm2": 0.5,
            "geometry_qc_state": "GEOMETRY_CERTIFIED",
        },
        profile_id="store-boundary-test@1.0.0",
    )
    claim = store.claim_qc("store-worker", 60)
    assert claim is not None
    assert claim["qc_job_id"] == enqueued["qc_job_id"]
    return store, claim


@pytest.mark.parametrize("finish", ["block", "retry"])
@pytest.mark.parametrize(
    "raw_error",
    [
        _RAW_STORE_ERROR,
        *_QUOTED_HEADER_STORE_ERRORS,
        {"token": "mapping-secret", "worker_id": "mapping-worker"},
    ],
    ids=[
        "serialized-sensitive-forms",
        "quoted-authorization-mapping",
        "quoted-cookie-mapping",
        "nonstring-mapping",
    ],
)
def test_sqlite_qc_store_sanitizes_caller_receipt_at_durable_boundary(
    finish, raw_error, tmp_path,
):
    """SQLite persists a safe clone even when a future caller bypasses worker."""
    store, claim = _claimed_sqlite_qc(
        tmp_path, f"{finish}-{type(raw_error).__name__}"
    )
    status = (
        "BLOCKED_CONFIGURATION"
        if finish == "block"
        else "RETRYABLE_QC_UNAVAILABLE"
    )
    receipt = _qc_receipt(claim, status, raw_error)
    original = json.loads(json.dumps(receipt))

    if finish == "block":
        returned = store.block_qc_configuration(
            claim["qc_job_id"], claim["lease_token"], receipt
        )
        event_type = "QC_BLOCKED_CONFIGURATION"
    else:
        returned = store.requeue_qc_unavailable(
            claim["qc_job_id"],
            claim["lease_token"],
            receipt,
            retry_delay_seconds=0,
        )
        event_type = "QC_REQUEUED_UNAVAILABLE"

    assert receipt == original, "the store must sanitize a clone, not caller data"
    assert returned["status"] == status
    with store.connect() as connection:
        persisted = json.loads(connection.execute(
            "SELECT result_json FROM qc_jobs WHERE qc_job_id=?",
            (claim["qc_job_id"],),
        ).fetchone()["result_json"])
        event = json.loads(connection.execute(
            "SELECT payload_json FROM events WHERE event_type=? ORDER BY event_id DESC",
            (event_type,),
        ).fetchone()["payload_json"])
    _assert_store_error_is_safe(persisted["error"])
    _assert_store_error_is_safe(event["error"])
    if isinstance(raw_error, str):
        assert persisted["error"].startswith("RuntimeError: ")
        assert event["error"].startswith("RuntimeError: ")
    if raw_error in _QUOTED_HEADER_STORE_ERRORS:
        assert persisted["error"] == "RuntimeError: <redacted>"
        assert event["error"] == "RuntimeError: <redacted>"
    if finish == "block":
        _assert_store_error_is_safe(returned["error"])
        if raw_error in _QUOTED_HEADER_STORE_ERRORS:
            assert returned["error"] == "RuntimeError: <redacted>"


class _PostgresQcCursor:
    def __init__(self, job):
        self.job = job
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement, parameters=()):
        self.statements.append((" ".join(statement.split()), parameters))

    def fetchone(self):
        return self.job


class _PostgresQcConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def cursor(self):
        return self._cursor


@pytest.mark.parametrize("finish", ["block", "retry"])
@pytest.mark.parametrize(
    "raw_error",
    [
        _RAW_STORE_ERROR,
        *_QUOTED_HEADER_STORE_ERRORS,
        {"token": "mapping-secret", "worker_id": "mapping-worker"},
    ],
    ids=[
        "serialized-sensitive-forms",
        "quoted-authorization-mapping",
        "quoted-cookie-mapping",
        "nonstring-mapping",
    ],
)
def test_postgres_qc_store_sanitizes_caller_receipt_at_durable_boundary(
    finish, raw_error, monkeypatch,
):
    """PostgreSQL binds only safe receipt and event JSON at its SQL boundary."""
    from fleet.postgres_store import PostgresFleetStore, _token_hash

    lease_token = "lease-token"
    claim = {"qc_job_id": "job-1", "surface_id": "surface-1"}
    cursor = _PostgresQcCursor({
        **claim,
        "state": "CLAIMED",
        "lease_token_hash": _token_hash(lease_token),
        "lease_expires_at": datetime.now(timezone.utc) + timedelta(minutes=5),
    })
    store = PostgresFleetStore("postgresql://unused")
    monkeypatch.setattr(
        store, "connect", lambda: _PostgresQcConnection(cursor)
    )
    status = (
        "BLOCKED_CONFIGURATION"
        if finish == "block"
        else "RETRYABLE_QC_UNAVAILABLE"
    )
    receipt = _qc_receipt(claim, status, raw_error)
    original = json.loads(json.dumps(receipt))

    if finish == "block":
        returned = store.block_qc_configuration("job-1", lease_token, receipt)
        event_type = "QC_BLOCKED_CONFIGURATION"
    else:
        returned = store.requeue_qc_unavailable(
            "job-1", lease_token, receipt, retry_delay_seconds=0
        )
        event_type = "QC_REQUEUED_UNAVAILABLE"

    assert receipt == original, "the store must sanitize a clone, not caller data"
    assert returned["status"] == status
    update = next(
        parameters
        for statement, parameters in cursor.statements
        if statement.startswith("UPDATE segment_qc_jobs")
    )
    persisted = json.loads(update[0])
    event_parameters = next(
        parameters
        for statement, parameters in cursor.statements
        if statement.startswith("INSERT INTO segment_events")
        and parameters[2] == event_type
    )
    event = json.loads(event_parameters[3])
    _assert_store_error_is_safe(persisted["error"])
    _assert_store_error_is_safe(event["error"])
    if isinstance(raw_error, str):
        assert persisted["error"].startswith("RuntimeError: ")
        assert event["error"].startswith("RuntimeError: ")
    if raw_error in _QUOTED_HEADER_STORE_ERRORS:
        assert persisted["error"] == "RuntimeError: <redacted>"
        assert event["error"] == "RuntimeError: <redacted>"
    if finish == "block":
        _assert_store_error_is_safe(returned["error"])
        if raw_error in _QUOTED_HEADER_STORE_ERRORS:
            assert returned["error"] == "RuntimeError: <redacted>"


# --------------------------------------------------------------------------
# Failing every claim is loud
# --------------------------------------------------------------------------

def test_a_worker_that_never_succeeds_says_so(tmp_path, capsys):
    """For the failures that are genuinely retryable but never recover.

    Blocking covers what is knowably permanent. This covers the rest: a hundred
    requeues with no success between them is not an outage anybody is waiting
    out, and from outside it is indistinguishable from a busy worker.
    """
    store = Recording(_claim())
    worker = _worker(store, OSError("nope"), tmp_path)
    worker.run(max_jobs=6, idle_exit=False)

    alarms = [json.loads(line) for line in capsys.readouterr().err.splitlines()
              if line.startswith("{")]
    assert alarms, "six consecutive failures and the worker said nothing"
    assert alarms[0]["status"] == "NO_SURFACE_MEASURED"
    assert alarms[0]["consecutive_retryable_failures"] == 5
    assert alarms[0]["worker_id"] == "worker-1"


def test_the_alarm_resets_when_work_succeeds(tmp_path, capsys):
    """Otherwise it fires on any deployment that has ever had a bad afternoon."""
    store = Recording(_claim())
    worker = _worker(store, OSError("nope"), tmp_path)
    worker.alarm_after = frozenset({3})

    # Two failures, a success, then two more: never three in a row.
    outcomes = ["r", "r", "ok", "r", "r"]
    calls = iter(outcomes)

    def run_one():
        return ({"status": "COMPLETED"} if next(calls) == "ok"
                else {"status": "RETRYABLE_QC_UNAVAILABLE"})

    worker.run_one = run_one  # type: ignore[method-assign]
    worker.run(max_jobs=len(outcomes), idle_exit=False)
    assert not capsys.readouterr().err.strip(), "the alarm fired on a working worker"


# --------------------------------------------------------------------------
# Somebody is told
# --------------------------------------------------------------------------

def test_the_panel_counts_jobs_blocked_on_configuration():
    """A terminal state nobody looks at is the retry loop with better manners.

    The whole incident was invisible: the jobs sat in PENDING, which is what a
    healthy queue with no free worker looks like, and the GPUs reported
    utilisation. Blocking them stops the spin; counting them is what makes it
    somebody's problem. The count is only ever non-zero when a person has to
    change a setting.
    """
    app = (ROOT / "panel/app.py").read_text()
    status = app[app.index("def fleet_status("):]
    status = status[: status.index("\n\ndef ")]
    assert "BLOCKED_CONFIGURATION" in status, (
        "the fleet status does not count blocked QC jobs, so a misconfigured "
        "fleet looks exactly like an idle one"
    )
    assert "qc_blocked_on_configuration" in status
