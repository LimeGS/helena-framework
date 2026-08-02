from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "framework/stages/01-segmentation/scripts/run_federated_surface_qc_watch.sh"


def _executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_federated_qc_runs_each_database_sequentially(tmp_path: Path) -> None:
    db_a = tmp_path / "a.sqlite"
    db_b = tmp_path / "b.sqlite"
    db_a.touch()
    db_b.touch()
    database_list = tmp_path / "databases.txt"
    database_list.write_text(f"# frozen queues\n{db_a}\n\n{db_b}\n", encoding="utf-8")

    call_log = tmp_path / "calls.log"
    python_stub = _executable(
        tmp_path / "python-stub",
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\nexit 0\n',
    )
    renderer = _executable(tmp_path / "renderer")
    qc_adapter = tmp_path / "qc-adapter.py"
    checkpoint = tmp_path / "checkpoint.pt"
    qc_profile = tmp_path / "qc-profile.json"
    qc_adapter.touch()
    checkpoint.touch()
    qc_profile.touch()

    environment = os.environ.copy()
    environment.update(
        {
            "FLEET_DB_LIST_FILE": str(database_list),
            "QC_RUN_ROOT": str(tmp_path / "runs"),
            "SURFACE_QC_EXECUTABLE": str(qc_adapter),
            "HELENA_QC_RENDERER": str(renderer),
            "HELENA_QC_CHECKPOINT": str(checkpoint),
            "HELENA_QC_PROFILE": str(qc_profile),
            "HELENA_QC_PROFILE_SHA256": "0" * 64,
            "SURFACE_QC_PROFILE_ID": "fixture-surface-qc@1.0.0",
            "HELENA_QC_EVIDENCE_ROOT": str(tmp_path / "evidence"),
            "HELENA_QC_CODE_COMMIT": "fixture",
            "HELENA_REPO_ROOT": str(ROOT),
            "PYTHON_BIN": str(python_stub),
            "CALL_LOG": str(call_log),
            "QC_FEDERATED_MAX_CYCLES": "1",
        }
    )
    completed = subprocess.run(
        ["sh", str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "completed 1 cycle(s)" in completed.stdout
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert f"--db {db_a}" in calls[0]
    assert f"--db {db_b}" in calls[1]
    assert all("--max-jobs 1" in call for call in calls)


def test_federated_qc_rejects_invalid_max_cycles(tmp_path: Path) -> None:
    database_list = tmp_path / "databases.txt"
    qc_adapter = tmp_path / "qc-adapter.py"
    checkpoint = tmp_path / "checkpoint.pt"
    qc_profile = tmp_path / "qc-profile.json"
    renderer = _executable(tmp_path / "renderer")
    database_list.touch()
    qc_adapter.touch()
    checkpoint.touch()
    qc_profile.touch()
    environment = os.environ.copy()
    environment.update(
        {
            "FLEET_DB_LIST_FILE": str(database_list),
            "QC_RUN_ROOT": str(tmp_path / "runs"),
            "SURFACE_QC_EXECUTABLE": str(qc_adapter),
            "HELENA_QC_RENDERER": str(renderer),
            "HELENA_QC_CHECKPOINT": str(checkpoint),
            "HELENA_QC_PROFILE": str(qc_profile),
            "HELENA_QC_PROFILE_SHA256": "0" * 64,
            "SURFACE_QC_PROFILE_ID": "fixture-surface-qc@1.0.0",
            "HELENA_QC_EVIDENCE_ROOT": str(tmp_path / "evidence"),
            "HELENA_QC_CODE_COMMIT": "fixture",
            "QC_FEDERATED_MAX_CYCLES": "forever",
        }
    )
    completed = subprocess.run(
        ["sh", str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert "must be a non-negative integer" in completed.stderr
