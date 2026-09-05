"""The ink worker's decisions, without a GPU, a queue or a subprocess.

What is worth holding here is what the worker decides before and after the
process it starts: which runner a lane gets, what the renderer is allowed to
inherit, whether a stack is worth calling a success, where it goes afterwards
and what is cleaned up. Every one of those was wrong at least once this week,
and none of them needs a card to test.

The subprocess itself is deliberately not mocked. A test that replaces
vc_render_tifxyz with a stub proves the stub was called, and this pipeline's
failures have all been in the arguments, the inputs and the outputs -- never in
whether Python can start a process.
"""

from __future__ import annotations

import json
import os
import re
import hashlib
import sys
from pathlib import Path

import pytest

from control_manifest import IN_FORCE_ID, IN_FORCE_PATH  # noqa: E402

# The stack fixtures are real TIFFs, because what is verified is the content of
# a render and a stub would verify the stub. Skipped where the worker's own
# dependency is not installed, like the panel image.
pytest.importorskip("tifffile", reason="the ink worker's runtime carries this")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

import ink_worker  # noqa: E402
import job_store  # noqa: E402
from ink_worker import RenderNotUsable  # noqa: E402


# --------------------------------------------------------------------------
# Which runner, and what it may inherit
# --------------------------------------------------------------------------

def test_the_lane_profile_chooses_the_runner():
    """The queue built one argv for every ink job while the profiles named their
    own adapter, so every TimeSformer lane ran the ResNet runner's flags."""
    timesformer = ink_worker.runner_for(
        {"phase": "P5", "profile_id": "timesformer-gp-scroll1-screening@1.1.0"})
    canonical = ink_worker.runner_for(
        {"phase": "P5", "profile_id": "ink-canonical-2um-screening@1.1.0"})
    assert timesformer.name == "run_ink_timesformer.py"
    assert canonical.name == "run_ink_canonical2um.py"


def test_a_phase_with_no_runner_is_refused_rather_than_guessed():
    """The refusal now comes from the lane table, which is the thing that knows
    what a phase can run. JobRejected is the queue's own word for "this request
    does not describe a runnable job"."""
    from job_store import JobRejected

    with pytest.raises((RuntimeError, JobRejected)):
        ink_worker.runner_for({"phase": "P404"})


def test_the_renderer_does_not_inherit_the_private_bucket(monkeypatch):
    """The CT is public and served anonymously. Signing that request with keys
    for another bucket returns 400 one second into a render, on a URL that
    answers 200 to curl."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    assert "AWS_ACCESS_KEY_ID" not in ink_worker.runner_environment({"phase": "P4"})
    # And P5 no longer keeps them unconditionally. This asserted that it did,
    # because "its checkpoint may come from the private bucket" -- which
    # `checkpoint` being a PATH_PARAMETER rules out: it has to be an absolute
    # local path, so it is a mount and never an s3 URL. What P5 can reach over
    # the network is its artifact store, and it gets the keys when that store
    # is one.
    assert "AWS_ACCESS_KEY_ID" not in ink_worker.runner_environment(
        {"phase": "P5", "profile_id": "x", "parameters": {}})
    assert "AWS_ACCESS_KEY_ID" in ink_worker.runner_environment(
        {"phase": "P5", "profile_id": "x",
         "parameters": {"artifact_store": "s3://helena/ink-maps-v1"}})


def test_a_vendored_runner_gets_its_architecture_on_the_path(monkeypatch):
    """It imports the upstream model from beside itself, which is where that
    code sits in the recipe's own directory and not where it sits here.

    From the worker's environment, not from the job. This was a parameter, so
    the directory a GPU host inserted at sys.path[0] and imported a model from
    was whatever the request said -- which is code execution wearing a form
    field.
    """
    monkeypatch.setenv("HELENA_INK_UPSTREAM_ROOT", "/models/canonical")
    environment = ink_worker.runner_environment(
        {"phase": "P5", "profile_id": "ink-canonical-2um-screening@1.1.0",
         "parameters": {}})
    assert environment["PYTHONPATH"].startswith("/models/canonical:")


def test_a_job_cannot_say_where_the_architecture_comes_from(monkeypatch):
    """The host's answer is the only answer.

    A parameter named the same thing must not creep back in and win: the
    queue rejects the key outright, and nothing here reads it.
    """
    monkeypatch.setenv("HELENA_INK_UPSTREAM_ROOT", "/models/canonical")
    environment = ink_worker.runner_environment(
        {"phase": "P5", "profile_id": "ink-canonical-2um-screening@1.1.0",
         "parameters": {"upstream_dir": "/tmp/attacker"}})
    assert environment["PYTHONPATH"].startswith("/models/canonical:")
    assert "/tmp/attacker" not in environment["PYTHONPATH"]

    with pytest.raises(job_store.JobRejected) as refused:
        job_store.validate_parameters({"checkpoint": "/models/m.safetensors",
                                       "tiff_dir": "/runs/stack",
                                       "source_pixel_um": 2.399,
                                       "upstream_dir": "/tmp/attacker"}, "P5")
    assert "upstream_dir" in str(refused.value)


def test_a_worker_without_the_upstream_refuses_the_lane(monkeypatch):
    """Fail closed, and about the host rather than about the request.

    A worker that does not carry the vendored architecture must not fall back
    to importing from somewhere -- and the reason belongs to the machine, so
    another worker may still take the job.
    """
    monkeypatch.delenv("HELENA_INK_UPSTREAM_ROOT", raising=False)
    with pytest.raises(ink_worker.WorkerRefused) as refused:
        ink_worker.runner_environment(
            {"phase": "P5", "profile_id": "ink-canonical-2um-screening@1.1.0",
             "parameters": {}})
    assert "HELENA_INK_UPSTREAM_ROOT" in str(refused.value)


# --------------------------------------------------------------------------
# Whether the render is worth calling a success
# --------------------------------------------------------------------------

def stack(directory: Path, slices: int = 5, *, constant: bool = False) -> Path:
    import numpy
    import tifffile

    directory.mkdir(parents=True, exist_ok=True)
    for index in range(slices):
        plane = (numpy.zeros((4, 4), dtype=numpy.uint16) if constant
                 else numpy.arange(16, dtype=numpy.uint16).reshape(4, 4) + index)
        tifffile.imwrite(directory / f"{index:02d}.tif", plane)
    return directory


def control_binding() -> dict:
    policy = json.loads(IN_FORCE_PATH.read_text())
    canonical = lambda document: hashlib.sha256(json.dumps(  # noqa: E731
        document, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    locks = policy["source_locks"]
    source_lock = {
        "control_profile_id": policy["profile_id"],
        "control_profile_sha256": canonical(policy),
        "ct_lock_sha256": canonical(locks["ct"]),
        "m7_lock_sha256": canonical(locks["m7"]),
    }
    source_lock_sha = hashlib.sha256(json.dumps(
        source_lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "control_p0_artifact_id": "p0-control",
        "control_p0_artifact_sha256": "1" * 64,
        "control_p0_selection_version": "selection-control",
        "control_source_snapshot_id": "source-control",
        "control_source_content_lock": source_lock,
        "control_source_content_lock_sha256": source_lock_sha,
        "control_policy_sha256": canonical(policy),
    }


def test_a_stack_of_constants_is_refused(tmp_path):
    with pytest.raises(RenderNotUsable) as refused:
        ink_worker.verify_layer_stack(stack(tmp_path / "l", constant=True), {})
    assert "no signal" in str(refused.value)


def test_the_slice_count_must_be_the_one_that_was_asked_for(tmp_path):
    with pytest.raises(RenderNotUsable):
        ink_worker.verify_layer_stack(stack(tmp_path / "l", 5), {"num_slices": 33})


def test_a_real_stack_is_described(tmp_path):
    described = ink_worker.verify_layer_stack(stack(tmp_path / "l", 7), {"num_slices": 7})
    assert described["slices"] == 7 and described["bytes"] > 0
    assert described["middle_slice_range"][0] < described["middle_slice_range"][1]


# --------------------------------------------------------------------------
# What happens to it afterwards
# --------------------------------------------------------------------------

def test_publishing_verifies_what_arrives(tmp_path):
    published = ink_worker.publish_layer_stack(
        stack(tmp_path / "l", 3), store_spec=str(tmp_path / "store"),
        sample_id="PHerc826", job_id="p4-x")
    assert published["files"] == 3 and len(published["artifact_sha256"]) == 64
    fetched = ink_worker.fetch_artifact_set(published["artifact_uri"], tmp_path / "back")
    assert len(fetched["files"]) == 3

    # An artifact that arrives with a different digest is a different artifact.
    manifest = json.loads((Path(published["artifact_uri"]) / "ARTIFACT_SET.json").read_text())
    manifest["files"]["01.tif"]["sha256"] = "0" * 64
    (Path(published["artifact_uri"]) / "ARTIFACT_SET.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError) as refused:
        ink_worker.fetch_artifact_set(published["artifact_uri"], tmp_path / "again")
    assert "not that artifact" in str(refused.value)


# --------------------------------------------------------------------------
# The window the detector is given
# --------------------------------------------------------------------------

class Store:
    def __init__(self, render):
        self._render = render

    def job(self, job_id):
        return self._render


def test_a_map_is_not_computed_on_a_failed_render(tmp_path):
    store = Store({"job_id": "p4-1", "phase": "P4", "state": "failed", "result": {}})
    with pytest.raises(RuntimeError) as refused:
        ink_worker.resolve_layer_stack(
            store, {"parameters": {"layer_stack": "p4-1"}}, tmp_path / "stack")
    assert "failed" in str(refused.value)


def test_disjoint_pherc0139_job_may_consume_its_declared_three_slice_stack(tmp_path):
    """The scroll name alone must not opt an unrelated job into control rules."""
    published = ink_worker.publish_layer_stack(
        stack(tmp_path / "rendered", 3), store_spec=str(tmp_path / "published"),
        sample_id="PHerc0139", job_id="p4-disjoint")
    render = {
        "job_id": "p4-disjoint", "phase": "P4", "state": "succeeded",
        "sample_id": "PHerc0139", "parameters": {"num_slices": 3},
        "result": {"layer_stack": published},
    }
    job = {
        "job_id": "p5-disjoint", "phase": "P5", "sample_id": "PHerc0139",
        "parameters": {"layer_stack": "p4-disjoint"},
    }
    resolved = ink_worker.resolve_layer_stack(
        Store(render), job, tmp_path / "fetched")
    assert resolved == str(tmp_path / "fetched")
    assert len(list((tmp_path / "fetched").glob("*.tif"))) == 3


def test_worker_fails_closed_on_partial_or_changed_control_binding(tmp_path):
    binding = control_binding()
    with pytest.raises(RuntimeError, match="partial persisted control binding"):
        ink_worker.persisted_control_binding({
            "parameters": {"control_p0_artifact_id": "p0-control"}})
    with pytest.raises(RuntimeError, match="policy differs"):
        ink_worker.verified_control_binding({"parameters": {
            **binding, "control_policy_sha256": "f" * 64}})
    retained = {"parameters": dict(binding)}
    assert ink_worker.verified_control_binding(retained) == binding
    assert retained["_verified_control_policy"]["profile_id"] == \
        binding["control_source_content_lock"]["control_profile_id"]
    late_failure = ink_worker.worker_failure_result(
        "timed out after 10s", tmp_path / "output", binding)
    assert {field: late_failure[field] for field in
            ink_worker.CONTROL_BINDING_FIELDS} == binding

    published = ink_worker.publish_layer_stack(
        stack(tmp_path / "rendered-bound", 3),
        store_spec=str(tmp_path / "published-bound"),
        sample_id="PHerc0139", job_id="p4-bound")
    render = {
        "job_id": "p4-bound", "phase": "P4", "state": "succeeded",
        "sample_id": "PHerc0139", "parameters": {**binding, "num_slices": 3},
        "result": {"layer_stack": published, **binding},
    }
    changed = {**binding, "control_p0_artifact_sha256": "3" * 64}
    job = {
        "job_id": "p5-bound", "phase": "P5", "sample_id": "PHerc0139",
        "parameters": {**changed, "layer_stack": "p4-bound"},
    }
    with pytest.raises(RuntimeError, match="P5/P4 persisted control bindings disagree"):
        ink_worker.resolve_layer_stack(Store(render), job, tmp_path / "changed")


def test_the_depth_window_is_centred_on_the_stack_when_nothing_was_asked(tmp_path):
    """A lane's depth centres are written for the stack depth its author had.
    The GP Scroll1 lane says 25, 32 and 39, which are positions in a 62-layer
    volume, and on a 33-slice render they fall off the end."""
    job = {"profile_id": "timesformer-gp-scroll1-screening@1.1.0",
           "parameters": {"source_pixel_um": 9.362, "source_slice_um": 9.362}}
    ink_worker.fit_depth_to_stack(job, stack(tmp_path / "l", 33))
    assert job["parameters"]["depth_centers"] == "16"


def test_a_stack_too_shallow_for_the_model_is_refused(tmp_path):
    """Ten slices for a window that needs twenty-two is not a centring problem,
    and centring it silently would hand the model padding."""
    job = {"profile_id": "timesformer-gp-scroll1-screening@1.1.0",
           "parameters": {"source_pixel_um": 9.362, "source_slice_um": 9.362}}
    with pytest.raises(RuntimeError) as refused:
        ink_worker.fit_depth_to_stack(job, stack(tmp_path / "l", 10))
    assert "too shallow" in str(refused.value)


def test_an_integer_flag_is_never_given_a_half(tmp_path):
    """The canonical lane's --depth-center is an integer, and a 62-frame window
    fits a 62-slice stack at 30.5 and nowhere else. Rounding it would hand the
    runner a window half a slice past the end."""
    job = {"profile_id": "ink-canonical-2um-screening@1.1.0",
           "parameters": {"source_pixel_um": 2.399, "source_slice_um": 2.399}}
    ink_worker.fit_depth_to_stack(job, stack(tmp_path / "l", 63))
    assert job["parameters"]["depth_center"] == 31
    assert "depth_centers" not in job["parameters"]


def test_every_phase_is_given_a_directory_that_exists(tmp_path):
    """Some runners create their own output directory and some write a file into
    it. vet_map read the map, screened it, found the shapes and then died on
    FileNotFoundError for the verdict, because nothing had made the directory."""
    source = (ROOT / "framework/stages/03-ink/fleet/ink_worker.py").read_text()
    run_job = source[source.index("def run_job("):]
    run_job = run_job[:run_job.index("\ndef ")]
    assert "output.mkdir(parents=True, exist_ok=True)" in run_job
    # Before the command is built, not after: a builder that reads the directory
    # would see it missing.
    assert run_job.index("output.mkdir") < run_job.index("command_for(")


# --------------------------------------------------------------------------
# Which runners hold the campaign's keys
# --------------------------------------------------------------------------

def test_a_runner_that_never_opens_a_bucket_does_not_get_the_keys(monkeypatch):
    """P4 was the only phase credentials were taken from, for a reason about P4.

    That left every other runner holding keys to the campaign's private bucket
    whether or not it touches one -- and P5's destination used to be a request
    parameter, so those two facts together were an exfiltration path with the
    worker's own credentials paying for it.
    """
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    local = {"phase": "P5", "profile_id": "timesformer-gp-scroll1-screening@1.1.0",
             "parameters": {"artifact_store": "/artifacts/ink-maps-v1"}}
    assert not any(k.startswith("AWS_") for k in ink_worker.runner_environment(local))

    published = {**local, "parameters": {"artifact_store": "s3://helena/ink-maps-v1"}}
    assert ink_worker.runner_environment(published)["AWS_ACCESS_KEY_ID"] == "AKIAEXAMPLE"


@pytest.mark.parametrize("phase", ["P1", "P8"])
def test_the_phases_that_read_object_storage_keep_them(phase):
    """P1 fetches its lasagna volumes from the bucket and P8's default lane
    vendors a script that requires boto3, both regardless of where they
    publish. Taking the keys from them would be an outage, not a boundary."""
    assert ink_worker.runner_needs_object_storage(
        {"phase": phase, "parameters": {"artifact_store": "/artifacts/local"}})


def test_p4_never_gets_them_even_publishing_to_s3(monkeypatch):
    """Its reason is the opposite one: the renderer streams the CT from the
    public bucket anonymously, and a signed request against a bucket these keys
    do not own comes back 400 one second into a render."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    environment = ink_worker.runner_environment(
        {"phase": "P4", "parameters": {"artifact_store": "s3://helena/layer-stacks-v1"}})
    assert not any(key.startswith("AWS_") for key in environment)


# --------------------------------------------------------------------------
# Where a runner's output is allowed to land
# --------------------------------------------------------------------------

def a_job(**parameters):
    return {"job_id": "p9-1", "phase": "P9", "sample_id": "PHerc0139",
            "parameters": parameters}


def test_output_outside_the_run_directory_is_refused(tmp_path):
    """`absolute, no ..` is a shape, not a place.

    Every absolute path on the host satisfies it, so these named any location:
    another mission's published artifacts, or the worker's own checkout. For P9
    it is worse than an overwrite -- the mission's artifact record points at
    `out_dir`, so the index could be made to name anywhere.
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    with pytest.raises(ink_worker.WorkerRefused, match="outside this job"):
        ink_worker.refuse_writes_outside_the_job(
            a_job(out_dir="/etc/helena"), runs_root=runs)

    ink_worker.refuse_writes_outside_the_job(
        a_job(out_dir=str(runs / "pherc0139-p9-1")), runs_root=runs)


def test_where_the_job_publishes_counts_as_its_own(tmp_path):
    """A local artifact store is a place this job legitimately writes."""
    runs, store = tmp_path / "runs", tmp_path / "artifacts"
    runs.mkdir(); store.mkdir()
    ink_worker.refuse_writes_outside_the_job(
        a_job(out_dir=str(store / "plates"), artifact_store=str(store)),
        runs_root=runs)
    with pytest.raises(ink_worker.WorkerRefused):
        ink_worker.refuse_writes_outside_the_job(
            a_job(out_dir=str(tmp_path / "elsewhere"), artifact_store=str(store)),
            runs_root=runs)


def test_a_symlink_out_of_the_run_directory_does_not_count_as_inside(tmp_path):
    """Resolved, for the reason the artifact endpoints resolve: a runner that
    writes into its own run directory can put a symlink in it."""
    runs, outside = tmp_path / "runs", tmp_path / "outside"
    runs.mkdir(); outside.mkdir()
    (runs / "escape").symlink_to(outside)
    with pytest.raises(ink_worker.WorkerRefused):
        ink_worker.refuse_writes_outside_the_job(
            a_job(out_dir=str(runs / "escape" / "plates")), runs_root=runs)


def test_a_deployment_can_name_more_roots(tmp_path, monkeypatch):
    """Fail closed, with a named way to open it: a plate run with its own
    volume is a real case, and it should be declared rather than assumed."""
    runs, volume = tmp_path / "runs", tmp_path / "plates"
    runs.mkdir(); volume.mkdir()
    with pytest.raises(ink_worker.WorkerRefused):
        ink_worker.refuse_writes_outside_the_job(
            a_job(out_dir=str(volume / "run-1")), runs_root=runs)
    monkeypatch.setenv(ink_worker.WRITE_ROOTS_VARIABLE, str(volume))
    ink_worker.refuse_writes_outside_the_job(
        a_job(out_dir=str(volume / "run-1")), runs_root=runs)


# --------------------------------------------------------------------------
# Whether this worker can see what it was told to read
# --------------------------------------------------------------------------

def test_an_input_the_worker_cannot_see_is_named_before_the_lease(tmp_path):
    """A renderer pointed at a directory outside the worker's mounts said
    `Error loading`, exited 0, and the hour after that went on TIFF tag types
    and a bbox. The path existed on the host; the container mounts one
    directory and that was not it.

    The panel cannot answer this -- it is a different container, often a
    different host. This process is the only one that knows what it can reach.
    """
    present = tmp_path / "recorte"
    present.mkdir()

    ink_worker.refuse_unreadable_inputs(
        {"phase": "P4", "parameters": {"segmentation": str(present)}})

    with pytest.raises(ink_worker.WorkerRefused) as refused:
        ink_worker.refuse_unreadable_inputs(
            {"phase": "P4",
             "parameters": {"segmentation": "/mnt/bulk/helena/entrada/recorte"}})
    assert "does not exist on this worker" in str(refused.value)
    assert "container mounts" in str(refused.value)


@pytest.mark.skipif(os.geteuid() == 0,
                    reason="root reads a 0o000 directory, so there is no "
                           "permission here to observe -- and every container "
                           "in this fleet runs as root, CI included")
def test_a_path_that_is_there_and_unreadable_says_so_instead(tmp_path):
    """Different cause, different sentence: one is a mount, the other is a
    permission, and `Error loading` was standing in for both."""
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        with pytest.raises(ink_worker.WorkerRefused, match="permission, not a path"):
            ink_worker.refuse_unreadable_inputs(
                {"phase": "P4", "parameters": {"segmentation": str(locked)}})
    finally:
        locked.chmod(0o755)


def test_every_result_says_which_build_produced_it():
    """Four P4 jobs finished `succeeded` with zero layers, by a worker whose
    copy of this file predated the check that refuses exactly that. Nothing in
    the row said which build ran, so `the image is stale` was an inference from
    a second symptom rather than an answer the receipt could give."""
    ran_by = ink_worker.worker_code_revision()
    assert re.fullmatch(r"[0-9a-f]{64}", ran_by["worker_source_sha256"] or "")
    assert set(ran_by) == {"image", "build_revision", "worker_source_sha256"}


# --------------------------------------------------------------------------
# Telling a worker with nothing to do from one that cannot
# --------------------------------------------------------------------------

def test_a_silent_worker_is_not_an_idle_one():
    """`docker ps` said "Up 27 hours" for three workers that had claimed
    nothing in eighteen. The fleet page showed what it shows on a quiet
    Sunday, because the only liveness it had was ink_hosts.last_seen_at --
    written by the host-report timer, a separate branch of the same loop, which
    kept reporting while the claim beside it was blocked.

    The row this reads is written by the claim itself, so its absence is the
    signal rather than something else's presence.
    """
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)

    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def execute(self, *_args): pass
        def fetchall(self):
            return [
                ("gpu-1-ink0", "gpu-1", "helena-worker-gpu", ["P5"],
                 now - timedelta(seconds=4), now - timedelta(seconds=90), 4.0,
                 True),
                ("gpu-1-ink1", "gpu-1", "helena-worker-gpu", ["P5"],
                 now - timedelta(hours=18), now - timedelta(hours=18), 64800.0,
                 True),
            ]

    class Connection:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def cursor(self): return Cursor()

    store = job_store.InkJobStore("postgresql://unused")
    store._connect = lambda **_kwargs: Connection()

    polling, silent = store.workers()
    assert polling["state"] == "POLLING"
    assert silent["state"] == "SILENT"
    assert silent["seconds_since_poll"] == 64800.0


def test_a_claim_cannot_wait_forever():
    """FOR UPDATE SKIP LOCKED skips locked *rows* and waits like anything else
    for a lock on the table -- and the migrations take ACCESS EXCLUSIVE. One
    stalled migration held every claim on the fleet, silently, because a
    blocked call prints nothing while a failing one prints and retries."""
    assert job_store.InkJobStore.LOCK_TIMEOUT_MS > 0
    assert job_store.InkJobStore.STATEMENT_TIMEOUT_MS > 0


def test_the_migrations_are_applied_by_one_process_at_a_time():
    """Every worker calls initialize() at startup, so a host bringing up three
    at once had three processes applying the same ALTER TABLE. Each takes an
    AccessExclusiveLock on the same relation in an order nobody controls, and
    PostgreSQL kills one:

        DeadlockDetected: Process 33 waits for AccessExclusiveLock on relation
        26218; blocked by process 32. Process 32 waits for ... blocked by 33.

    A worker that dies half way out of its own migration has a schema in an
    unknown state, which is the best explanation anyone has for three of them
    going silent for eighteen hours afterwards.
    """
    import inspect

    source = inspect.getsource(job_store.InkJobStore.initialize)
    assert "pg_advisory_xact_lock" in source, (
        "the migration block is not serialised; concurrent workers will "
        "deadlock against each other on the same relation")
    # Transaction-scoped on purpose: released by the commit the `with` performs,
    # so a process that dies holding it does not wedge every later start.
    assert "pg_advisory_lock(" not in source
    # And the wait is given the migration budget rather than a claim's ten
    # seconds, or the second worker to arrive fails instead of waiting.
    assert "MIGRATION_STATEMENT_TIMEOUT_MS" in source
    assert "lock_timeout" in source


def test_volume_is_checked_when_it_is_an_input_and_not_a_cache(tmp_path):
    """`volume` means two things depending on `remote_url`.

    With one, the renderer streams the CT and uses that path as its own cache,
    so an absent one is the ordinary first run. Without one it is an input the
    renderer opens -- and leaving it out of the check left the failure this
    whole function exists for reachable by another door: `Error loading`, exit
    zero, nothing rendered.
    """
    absent = str(tmp_path / "not-here.zarr")

    # A cache path, with a remote source: absence is normal.
    ink_worker.refuse_unreadable_inputs(
        {"phase": "P4", "parameters": {"volume": absent,
                                       "remote_url": "https://example/v.zarr"}})

    # The same path as the only source: refused before the lease is spent.
    with pytest.raises(ink_worker.WorkerRefused, match="does not exist"):
        ink_worker.refuse_unreadable_inputs(
            {"phase": "P4", "parameters": {"volume": absent}})


# -- P7 adjudicates evidence, and an exploratory map is not that --------------
#
# The exclusion execution_mode exists for, at the one place a P5 map becomes
# evidence. Read off the source: the resolver is integration-shaped (a store, a
# job, a destination) and the contract it has to keep is two lines.


def test_p7_refuses_a_screening_that_certifies_nothing() -> None:
    from pathlib import Path as _Path
    source = (_Path(__file__).resolve().parents[1]
              / "framework/stages/03-ink/fleet/ink_worker.py").read_text()
    resolver = source[source.index('screening_id = str(job["parameters"]["screening_of"])'):]
    resolver = resolver[:resolver.index("fetch_artifact_set(")]

    assert "execution_mode.declares_uncertified(result)" in resolver
    # Before the ALIVE check, not after: an exploratory map that is ALIVE is
    # still not evidence.
    assert resolver.index("declares_uncertified(result)") < resolver.index('verdict != "ALIVE"')


def test_a_screening_from_before_stamps_existed_still_adjudicates() -> None:
    """Every P5 on the fleet from before this carries no stamp and came out of
    a pinned lane. Reading its silence as a refusal would block every existing
    map the day this deployed; the gate refuses a declaration, not an absence."""
    from framework.contracts import execution_mode

    legacy = {"liveness": {"verdict": "ALIVE"}, "statistics": {"p99": 0.7}}
    assert execution_mode.declares_uncertified(legacy) is False
    # ...while the reader's question keeps failing closed, as it must.
    assert execution_mode.is_certified(legacy) is False


def test_the_experimental_lanes_screening_is_refused_by_its_own_declaration() -> None:
    from framework.contracts import execution_mode

    trust = execution_mode.Trust(execution_mode.EXPLORATORY)
    trust.blocks("the checkpoint is not pinned")
    stamped = trust.stamp({"liveness": {"verdict": "ALIVE"}})

    assert execution_mode.declares_uncertified(stamped) is True
    for field in ("execution_mode", "certified", "uncertified_because"):
        # Each declaration alone is enough: a row that kept only one of the
        # three fields still refuses.
        alone = {"liveness": {"verdict": "ALIVE"}, field: stamped[field]}
        assert execution_mode.declares_uncertified(alone) is True, field


def test_a_certified_stamp_is_not_a_declaration_of_the_opposite() -> None:
    from framework.contracts import execution_mode

    stamped = execution_mode.Trust().stamp({"liveness": {"verdict": "ALIVE"}})
    assert execution_mode.declares_uncertified(stamped) is False
    assert execution_mode.is_certified(stamped) is True


def test_the_stamp_travels_from_the_receipt_to_the_job_row() -> None:
    """P7 reads the job row, not the receipt file, so the row has to carry it."""
    from pathlib import Path as _Path
    source = (_Path(__file__).resolve().parents[1]
              / "framework/stages/03-ink/fleet/ink_worker.py").read_text()
    for field in ("certified", "execution_mode", "uncertified_because", "shuffle_control"):
        assert f'"{field}": (receipt or {{}}).get("{field}")' in source, field
