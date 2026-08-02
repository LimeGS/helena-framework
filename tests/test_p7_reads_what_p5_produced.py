"""P7 consumes the canonical probability map emitted by its named P5 job."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

import ink_worker  # noqa: E402


class Store:
    def __init__(self, screening: dict):
        self._screening = screening

    def job(self, job_id: str) -> dict:
        return self._screening


def screening(output: Path) -> dict:
    return {
        "job_id": "p5-real",
        "phase": "P5",
        "state": "succeeded",
        "result": {
            "output_dir": str(output),
            "liveness": {"verdict": "ALIVE"},
        },
    }


def p7_job() -> dict:
    return {"phase": "P7", "parameters": {"screening_of": "p5-real"}}


def test_p7_reads_the_timesformer_aggregate_not_a_window_or_stability_map(tmp_path):
    """The production lane emits mean_probability.npy, centre windows and a
    stability map.  The aggregate is the map P7 must adjudicate."""
    aggregate = tmp_path / "mean_probability.npy"
    aggregate.write_bytes(b"aggregate")
    (tmp_path / "center-016_offset-00.npy").write_bytes(b"window")
    (tmp_path / "stability_std.npy").write_bytes(b"stability")

    resolved = ink_worker.resolve_screened_map(
        Store(screening(tmp_path)), p7_job(), tmp_path / "fetch")

    assert resolved == str(aggregate)


def test_p7_preserves_the_legacy_canonical_name_when_both_exist(tmp_path):
    legacy = tmp_path / "probability.npy"
    legacy.write_bytes(b"legacy")
    (tmp_path / "mean_probability.npy").write_bytes(b"aggregate")

    resolved = ink_worker.resolve_screened_map(
        Store(screening(tmp_path)), p7_job(), tmp_path / "fetch")

    assert resolved == str(legacy)


def test_p7_selects_the_timesformer_aggregate_after_verified_fetch(tmp_path):
    source = tmp_path / "published"
    source.mkdir()
    for name in ("mean_probability.npy", "center-016_offset-00.npy",
                 "stability_std.npy"):
        (source / name).write_bytes(name.encode())
    sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
    from fleet.common import file_sha256

    files = {
        name: {"sha256": file_sha256(source / name),
               "size_bytes": (source / name).stat().st_size}
        for name in ("mean_probability.npy", "center-016_offset-00.npy",
                     "stability_std.npy")
    }
    (source / "ARTIFACT_SET.json").write_text(json.dumps({
        "schema": "campaignx.ink_probability_map.v1", "files": files,
    }))
    upstream = screening(tmp_path / "gone")
    upstream["result"]["probability_map"] = {"artifact_uri": str(source)}

    destination = tmp_path / "fetched"
    resolved = ink_worker.resolve_screened_map(
        Store(upstream), p7_job(), destination)

    assert resolved == str(destination / "mean_probability.npy")
    assert (destination / "mean_probability.npy").read_bytes() == b"mean_probability.npy"


def test_p7_refuses_to_guess_between_unnamed_arrays(tmp_path):
    (tmp_path / "first.npy").write_bytes(b"first")
    (tmp_path / "second.npy").write_bytes(b"second")

    with pytest.raises(RuntimeError) as refused:
        ink_worker.resolve_screened_map(
            Store(screening(tmp_path)), p7_job(), tmp_path / "fetch")

    assert "no canonical probability map" in str(refused.value)


def test_p7_records_a_scientific_fail_as_an_explicit_valid_outcome(tmp_path):
    verdict = {
        "schema_version": "v1",
        "status": "ok",
        "overall": {"pass": False},
        "checks": {"structure": {"pass": False, "value": 0.003}},
    }
    (tmp_path / "verdict.json").write_text(json.dumps(verdict))
    (tmp_path / "VETTING_CARD.md").write_text("# Vetting Card\n\nFAIL\n")

    recorded = ink_worker.read_adjudication(tmp_path)

    assert recorded["verdict"] == "FAIL"
    assert recorded["overall"] == {"pass": False}
    assert recorded["checks"] == verdict["checks"]
    assert recorded["verdict_file"] == "verdict.json"
    assert recorded["card_file"] == "VETTING_CARD.md"


@pytest.mark.parametrize(
    "verdict",
    [
        None,
        {"schema_version": "v1", "status": "error", "overall": {"pass": False}},
        {"schema_version": "v1", "status": "ok", "overall": {}},
    ],
)
def test_p7_exit_zero_without_a_valid_verdict_is_not_a_success(tmp_path, verdict):
    if verdict is not None:
        (tmp_path / "verdict.json").write_text(json.dumps(verdict))
    (tmp_path / "VETTING_CARD.md").write_text("# Vetting Card\n")

    with pytest.raises(RuntimeError):
        ink_worker.read_adjudication(tmp_path)


class RunStore:
    def __init__(self):
        self.finished = None

    def mark_running(self, *args, **kwargs):
        return None

    def heartbeat(self, *args, **kwargs):
        return None

    def finish(self, job_id, token, *, state, result):
        self.finished = {"state": state, "result": result}


def test_worker_stores_a_p7_refutation_in_the_job_result(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    output = runs / "pherc826-p7-real"
    runner = tmp_path / "runner.py"
    runner.write_text(
        "from pathlib import Path\n"
        "import json\n"
        f"out = Path({str(output)!r})\n"
        "(out / 'verdict.json').write_text(json.dumps({"
        "'schema_version':'v1','status':'ok','overall':{'pass':False},"
        "'checks':{'structure':{'pass':False}}}))\n"
        "(out / 'VETTING_CARD.md').write_text('# Vetting Card\\nFAIL\\n')\n"
    )
    monkeypatch.setattr(ink_worker, "runner_for", lambda job: runner)
    monkeypatch.setattr(
        ink_worker, "command_for",
        lambda job, runner, output_dir: [sys.executable, str(Path(runner))],
    )
    store = RunStore()
    job = {
        "job_id": "p7-real",
        "lease_token": "lease",
        "phase": "P7",
        "sample_id": "PHerc826",
        "parameters": {"map_path": "/unused.npy", "bbox": "0,0,1,1", "px_um": 7.91},
    }

    ink_worker.run_job(store, job, runs_root=runs, timeout=10)

    assert store.finished["state"] == "succeeded"
    assert store.finished["result"]["adjudication"]["verdict"] == "FAIL"


def test_p5_is_not_successful_when_its_probability_map_cannot_be_published(
        tmp_path, monkeypatch):
    """A local map is not a fleet artifact and P7 may run elsewhere."""
    runs = tmp_path / "runs"
    output = runs / "pherc826-p5-publish"
    runner = tmp_path / "runner.py"
    runner.write_text(
        "from pathlib import Path\n"
        "import json\n"
        f"out = Path({str(output)!r})\n"
        "(out / 'probability.npy').write_bytes(b'map')\n"
        "(out / 'INK_SCREENING_RECEIPT.json').write_text(json.dumps({"
        "'liveness':{'verdict':'ALIVE'}}))\n"
    )
    monkeypatch.setattr(ink_worker, "runner_for", lambda job: runner)
    monkeypatch.setattr(
        ink_worker, "command_for",
        lambda job, runner, output_dir: [sys.executable, str(Path(runner))],
    )
    monkeypatch.setattr(ink_worker, "record_artifact", lambda *args, **kwargs: None)

    def publication_failed(*args, **kwargs):
        raise RuntimeError("durable store unavailable")

    monkeypatch.setattr(ink_worker, "publish_probability_map", publication_failed)
    store = RunStore()
    job = {
        "job_id": "p5-publish",
        "lease_token": "lease",
        "phase": "P5",
        "sample_id": "PHerc826",
        "profile_id": "timesformer-gp-scroll1-screening@1.1.0",
        "parameters": {"artifact_store": "/artifacts/ink-maps-v1"},
    }

    ink_worker.run_job(store, job, runs_root=runs, timeout=10)

    assert store.finished["state"] == "failed"
    assert "durable store unavailable" in store.finished["result"]["error"]
