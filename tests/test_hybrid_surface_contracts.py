from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.contracts.hybrid_surface_contracts import (  # noqa: E402
    HybridContractValidationError,
    SCHEMA_FILES,
    validate_hybrid_contract,
    validate_hybrid_contract_file,
)


FIXTURES = ROOT / "tests/fixtures/hybrid_contracts"
SCHEMAS = ROOT / "framework/contracts/schemas"


def fixture(name: str) -> dict:
    value = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize("filename", SCHEMA_FILES.values())
def test_hybrid_json_schemas_are_valid_draft_2020_12(filename: str) -> None:
    schema = json.loads((SCHEMAS / filename).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    ("filename", "contract"),
    [
        ("surface_artifact.valid.json", "campaignx.surface_artifact.v2"),
        ("backend_run.valid.json", "campaignx.segmentation_backend_run.v1"),
        ("backend_comparison.valid.json", "campaignx.surface_backend_comparison.v1"),
    ],
)
def test_valid_hybrid_contract_fixtures_pass(filename: str, contract: str) -> None:
    validate_hybrid_contract_file(FIXTURES / filename, expected_contract=contract)


@pytest.mark.parametrize(
    "filename",
    [
        "surface_artifact.invalid.json",
        "backend_run.invalid.json",
        "backend_comparison.invalid.json",
    ],
)
def test_invalid_hybrid_contract_fixtures_fail_closed(filename: str) -> None:
    with pytest.raises(HybridContractValidationError, match="contract validation failed"):
        validate_hybrid_contract_file(FIXTURES / filename)


def test_unknown_contract_and_unexpected_fields_fail_closed() -> None:
    artifact = fixture("surface_artifact.valid.json")
    artifact["schema"] = "campaignx.surface_artifact.v999"
    with pytest.raises(HybridContractValidationError, match="unknown contract"):
        validate_hybrid_contract(artifact)

    artifact = fixture("surface_artifact.valid.json")
    artifact["silent_override"] = True
    with pytest.raises(HybridContractValidationError, match="silent_override"):
        validate_hybrid_contract(artifact)


def test_expected_contract_prevents_cross_stage_substitution() -> None:
    comparison = fixture("backend_comparison.valid.json")
    with pytest.raises(HybridContractValidationError, match="expected"):
        validate_hybrid_contract(
            comparison, expected_contract="campaignx.surface_artifact.v2"
        )


def test_failed_run_requires_error_and_forbids_surface_artifact() -> None:
    run = fixture("backend_run.valid.json")
    run["status"] = "FAILED"
    run["surface_artifact"] = None
    run["error"] = {
        "class": "BackendFailure",
        "message": "fixture failure",
        "log_uri": "s3://helena-fixture/logs/backend.log",
        "log_sha256": "f" * 64,
    }
    validate_hybrid_contract(run)

    invalid = copy.deepcopy(run)
    invalid["error"] = None
    with pytest.raises(HybridContractValidationError, match="error: required"):
        validate_hybrid_contract(invalid)


def test_receipt_hashes_and_secret_names_are_semantically_checked() -> None:
    run = fixture("backend_run.valid.json")
    run["command"]["argv_sha256"] = "0" * 64
    with pytest.raises(HybridContractValidationError, match="argv_sha256"):
        validate_hybrid_contract(run)

    run = fixture("backend_run.valid.json")
    run["environment_non_sensitive"] = [{"name": "AWS_SECRET_ACCESS_KEY", "value": "x"}]
    with pytest.raises(HybridContractValidationError, match="sensitive variable"):
        validate_hybrid_contract(run)


def test_comparison_declared_totals_must_match_region_evidence() -> None:
    comparison = fixture("backend_comparison.valid.json")
    comparison["summary"]["area_by_class_cm2"]["CONSENSUS"] = 9.0
    with pytest.raises(HybridContractValidationError, match="regions sum"):
        validate_hybrid_contract(comparison)
