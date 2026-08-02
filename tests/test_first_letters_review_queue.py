from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "framework/stages/06-discovery/scripts/helena_build_first_letters_review_queue.py"


def load_module():
    spec = importlib.util.spec_from_file_location("helena_first_letters_queue", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Body:
    def __init__(self, value: bytes):
        self.value = value

    def read(self) -> bytes:
        return self.value


class MemoryS3:
    def __init__(self, objects: dict[tuple[str, str], bytes]):
        self.objects = objects

    def get_object(self, *, Bucket: str, Key: str):
        return {"Body": Body(self.objects[(Bucket, Key)])}


def retained_fixture(module):
    files = {
        "review/candidate.png": b"not-a-real-png",
        "review/index.html": b"<html>review</html>",
    }
    inventory = [
        {
            "path": name,
            "size_bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        for name, body in sorted(files.items())
    ]
    manifest = {
        "schema": module.EVIDENCE_SCHEMA,
        "qc_job_id": "qc-1",
        "surface_id": "surface-1",
        "sample_id": "PHercTEST",
        "outcome": module.RETAINED_OUTCOME,
        "files": inventory,
        "non_claims": ["not a First Letters submission"],
    }
    manifest_body = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    objects = {
        ("bucket", "root/PHercTEST/surface-1/qc-1/EVIDENCE_MANIFEST.json"): manifest_body,
        **{
            ("bucket", f"root/PHercTEST/surface-1/qc-1/{name}"): body
            for name, body in files.items()
        },
    }
    row = {
        "qc_job_id": "qc-1",
        "surface_id": "surface-1",
        "profile_id": "geometry-screen-v1",
        "state": "COMPLETED",
        "updated_at": "2026-07-21T00:00:00Z",
        "result_json": json.dumps(
            {
                "schema": module.RESULT_SCHEMA,
                "surface_id": "surface-1",
                "outcome": module.RETAINED_OUTCOME,
                "evidence_uri": "s3://bucket/root/PHercTEST/surface-1/qc-1/EVIDENCE_MANIFEST.json",
                "evidence_manifest_sha256": hashlib.sha256(manifest_body).hexdigest(),
                "executor_receipt": {
                    "retained_for_visual_review_count": 2,
                    "no_automatic_acceptance": True,
                },
            }
        ),
    }
    return row, MemoryS3(objects)


def test_retained_evidence_is_verified_materialized_and_idempotent(tmp_path: Path):
    module = load_module()
    row, s3 = retained_fixture(module)

    first = module.build_queue([row], tmp_path, s3_client=s3, code_commit="a" * 40)
    first_queue_bytes = (tmp_path / "FIRST_LETTERS_REVIEW_QUEUE.json").read_bytes()
    first_receipt_bytes = (tmp_path / "DISCOVERY_RECEIPT.json").read_bytes()
    second = module.build_queue([row], tmp_path, s3_client=s3, code_commit="a" * 40)

    queue = json.loads((tmp_path / "FIRST_LETTERS_REVIEW_QUEUE.json").read_text())
    assert first["queue_id"] == second["queue_id"] == queue["queue_id"]
    assert (tmp_path / "FIRST_LETTERS_REVIEW_QUEUE.json").read_bytes() == first_queue_bytes
    assert (tmp_path / "DISCOVERY_RECEIPT.json").read_bytes() == first_receipt_bytes
    assert first["status"] == "READY_FOR_HUMAN_REVIEW"
    assert queue["candidate_count"] == 1
    assert queue["retained_component_count"] == 2
    assert queue["candidates"][0]["review_state"] == "UNREVIEWED"
    assert queue["candidates"][0]["claim_state"] == "NO_FIRST_LETTERS_CLAIM"
    assert (tmp_path / "evidence/surface-1/qc-1/review/candidate.png").is_file()
    assert "Not accepted ink" in (tmp_path / "FIRST_LETTERS_REVIEW_QUEUE.html").read_text()


def test_no_retained_signal_produces_honest_waiting_queue(tmp_path: Path):
    module = load_module()
    row, _ = retained_fixture(module)
    result = json.loads(row["result_json"])
    result["outcome"] = "CT_SUPPORTED_NO_RETAINED_INK_SIGNAL"
    row["result_json"] = json.dumps(result)

    receipt = module.build_queue([row], tmp_path, code_commit="b" * 40)

    assert receipt["status"] == "WAITING_FOR_RETAINED_SIGNALS"
    queue = json.loads((tmp_path / "FIRST_LETTERS_REVIEW_QUEUE.json").read_text())
    assert queue["candidate_count"] == 0
    assert "No retained signals yet" in (tmp_path / "FIRST_LETTERS_REVIEW_QUEUE.html").read_text()


def test_manifest_hash_mismatch_fails_closed(tmp_path: Path):
    module = load_module()
    row, s3 = retained_fixture(module)
    result = json.loads(row["result_json"])
    result["evidence_manifest_sha256"] = "0" * 64
    row["result_json"] = json.dumps(result)

    try:
        module.build_queue([row], tmp_path, s3_client=s3, code_commit="c" * 40)
    except RuntimeError as error:
        assert "manifest hash mismatch" in str(error)
    else:
        raise AssertionError("corrupted manifest was routed into discovery")


def test_stage_registry_resolves_discovery_router():
    from scripts.harness.stage_script_registry import resolve_stage_script

    assert resolve_stage_script(ROOT, SCRIPT.name).resolve() == SCRIPT.resolve()


def test_review_minimal_materializes_decisive_assets_only(tmp_path: Path):
    module = load_module()
    row, _ = retained_fixture(module)
    manifest_uri = json.loads(row["result_json"])["evidence_uri"]
    manifest = {
        "schema": module.EVIDENCE_SCHEMA,
        "qc_job_id": "qc-1",
        "surface_id": "surface-1",
        "sample_id": "PHercTEST",
        "outcome": module.RETAINED_OUTCOME,
        "files": [],
        "non_claims": ["not a First Letters submission"],
    }
    files = {
        "high-recall/ct_application/gate/CT_FIBER_GATE_DECISIONS.json": json.dumps(
            [{"candidate_id": "C003", "retained": True}]
        ).encode(),
        "high-recall/ct_application/review_v1/surface-1/C003-model-context.png": b"model",
        "high-recall/ct_application/review_v1/surface-1/C003-orthogonal-ct.png": b"ct",
        "robust/analysis/comparison_layers/mean_probability.png": b"probability",
        "robust/mean_probability.npy": b"large-regenerable-array",
    }
    manifest["files"] = [
        {
            "path": name,
            "size_bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        for name, body in sorted(files.items())
    ]
    manifest_body = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    result = json.loads(row["result_json"])
    result["evidence_manifest_sha256"] = hashlib.sha256(manifest_body).hexdigest()
    row["result_json"] = json.dumps(result)
    s3 = MemoryS3(
        {
            ("bucket", "root/PHercTEST/surface-1/qc-1/EVIDENCE_MANIFEST.json"): manifest_body,
            **{
                ("bucket", f"root/PHercTEST/surface-1/qc-1/{name}"): body
                for name, body in files.items()
                if not name.endswith(".npy")
            },
        }
    )

    module.build_queue(
        [row],
        tmp_path,
        s3_client=s3,
        code_commit="d" * 40,
        materialization_profile=module.MATERIALIZATION_REVIEW_MINIMAL,
    )

    candidate = json.loads(
        (tmp_path / "FIRST_LETTERS_REVIEW_QUEUE.json").read_text()
    )["candidates"][0]
    assert candidate["retained_candidate_ids"] == ["C003"]
    assert candidate["materialized_file_count"] == 4
    assert not (tmp_path / "evidence/surface-1/qc-1/robust/mean_probability.npy").exists()
    assert candidate["review_image_paths"][0].endswith("C003-model-context.png")
    assert candidate["review_image_paths"][1].endswith("C003-orthogonal-ct.png")


def test_ledger_normalization_overlay_is_additive_and_hash_bound(tmp_path: Path):
    module = load_module()
    ledger_path = tmp_path / "ledger.json"
    ledger = {
        "schema": module.LEDGER_SCHEMA,
        "no_automatic_acceptance": True,
        "surfaces": [
            {
                "sample_id": "PHercTEST",
                "surface_id": "surface-1",
                "qc_job_id": "qc-1",
                "qc_profile_id": "geometry-screen-v1",
                "qc_state": "COMPLETED",
                "qc_updated_at": "2026-07-22T00:00:00Z",
                "outcome": module.RETAINED_OUTCOME,
                "retained_for_visual_review_count": 1,
                "evidence_uri": "s3://bucket/root/manifest.json",
                "evidence_manifest_sha256": "a" * 64,
            }
        ],
    }
    ledger_path.write_text(json.dumps(ledger) + "\n")
    overlay_path = tmp_path / "overlay.json"
    overlay = {
        "schema": module.NORMALIZATION_SCHEMA,
        "status": "COMPLETED",
        "source_ledger_sha256": module.sha256(ledger_path),
        "surfaces": [
            {
                "surface_id": "surface-1",
                "effective_outcome": "CT_SUPPORTED_NO_RETAINED_INK_SIGNAL",
                "retained_count": 0,
                "source_gate": "v2",
                "normalized_gate": "v3",
                "evaluation_sha256": "b" * 64,
                "decisions_sha256": "c" * 64,
            }
        ],
    }
    overlay_path.write_text(json.dumps(overlay) + "\n")

    rows = module.read_completed_ledger(ledger_path, overlay_path)
    result = rows[0]["result_json"]
    assert result["outcome"] == "CT_SUPPORTED_NO_RETAINED_INK_SIGNAL"
    assert result["executor_receipt"]["retained_for_visual_review_count"] == 0
    assert rows[0]["normalization"]["normalized_gate"] == "v3"
    assert json.loads(ledger_path.read_text())["surfaces"][0]["outcome"] == module.RETAINED_OUTCOME

    overlay["source_ledger_sha256"] = "0" * 64
    overlay_path.write_text(json.dumps(overlay) + "\n")
    try:
        module.read_completed_ledger(ledger_path, overlay_path)
    except RuntimeError as error:
        assert "source ledger hash mismatch" in str(error)
    else:
        raise AssertionError("unbound normalization overlay was accepted")
