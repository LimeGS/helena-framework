from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "framework/stages/04-validation/scripts/helena_cleanup_completed_qc_staging.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("helena_cleanup_qc", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_script()


def test_cleanup_requires_verified_durable_manifest_and_preserves_evidence(tmp_path: Path) -> None:
    run_root = tmp_path / "runtime"
    evidence_root = tmp_path / "evidence"
    output = run_root / "job-1/attempt-1/scientific-output"
    durable = evidence_root / "PHercX/surface-1/job-1/EVIDENCE_MANIFEST.json"
    output.mkdir(parents=True)
    durable.parent.mkdir(parents=True)
    manifest = b'{"schema":"evidence"}\n'
    source = output / "EVIDENCE_MANIFEST.json"
    source.write_bytes(manifest)
    durable.write_bytes(manifest)
    (output / "review.png").write_bytes(b"durable-review")
    (output / "mean.npy").write_bytes(b"regenerable-array")
    (output / "surface").mkdir()
    (output / "surface/x.tif").write_bytes(b"regenerable-surface")
    digest = hashlib.sha256(manifest).hexdigest()

    database = tmp_path / "fleet.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE qc_jobs (qc_job_id TEXT, surface_id TEXT, state TEXT, result_json TEXT)")
    connection.execute(
        "INSERT INTO qc_jobs VALUES (?,?,?,?)",
        (
            "job-1",
            "surface-1",
            "COMPLETED",
            json.dumps(
                {
                    "evidence_manifest_sha256": digest,
                    "evidence_uri": durable.as_uri(),
                    "executor_receipt": {"evidence_manifest_path": str(source)},
                }
            ),
        ),
    )
    connection.commit()
    connection.close()

    dry = MODULE.clean_completed(database, run_root, evidence_root, apply=False)
    assert dry["eligible_bytes"] > 0
    assert (output / "mean.npy").exists()

    applied = MODULE.clean_completed(database, run_root, evidence_root, apply=True)
    assert applied["removed_bytes"] == dry["eligible_bytes"]
    assert not (output / "mean.npy").exists()
    assert not (output / "surface").exists()
    assert (output / "review.png").read_bytes() == b"durable-review"
    assert source.read_bytes() == manifest
    assert durable.read_bytes() == manifest
    assert (output.parent / "POSTHOC_REGENERABLE_CLEANUP.json").is_file()
