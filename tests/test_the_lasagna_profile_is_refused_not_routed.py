"""Selecting the lasagna flattening profile used to run vc_flatten anyway.

`command_for`'s P3 branch names `--binary` unconditionally -- `vc_flatten`, the
allowlist's one entry -- with no branch on which profile was chosen. Handing
it `flatten-lasagna-v1@1.0.0` did not run lasagna's own GPU script; it ran
vc_flatten against a profile file `vc_flatten` was never written to read, and
nothing said so. The queue now refuses the job by the profile's filename
instead, until this lane's builder can actually dispatch a second engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from job_store import JobRejected, command_for, validate_parameters  # noqa: E402


def _p3_job(profile: str) -> dict:
    parameters = validate_parameters({
        "sample": "PHerc0139",
        "profile": profile,
        "artifact_store": "s3://helena/flatten-v1",
    }, "P3", server_owned=("artifact_store",))
    return {"job_id": "p3-test", "phase": "P3", "profile_id": "flatten-abf-v1@1.0.0",
            "sample_id": "PHerc0139", "parameters": parameters}


def test_the_lasagna_profile_is_refused():
    lasagna = ("/workspace/campaign-x/framework/profiles/02-flattening"
              "/flatten-lasagna-v1-1.0.0.json")
    with pytest.raises(JobRejected) as refused:
        command_for(_p3_job(lasagna), runner="fleet.py", output_dir="/runs/p3-job")
    assert "vc_flatten" in str(refused.value)


def test_the_default_abf_profile_still_runs():
    abf = ("/workspace/campaign-x/framework/profiles/02-flattening"
          "/flatten-abf-v1-1.0.0.json")
    argv = command_for(_p3_job(abf), runner="fleet.py", output_dir="/runs/p3-job")
    assert argv[argv.index("--profile") + 1] == abf


def test_the_profile_file_declaring_a_different_engine_is_the_one_refused():
    """The check reads the filename, not the path prefix, so it survives the
    workspace root moving -- but it has to name the file lasagna's own profile
    actually is, or a rename anywhere silently stops refusing it."""
    import json
    profile = json.loads((ROOT / "framework/profiles/02-flattening"
                          "/flatten-lasagna-v1-1.0.0.json").read_text())
    assert profile["engine"] == "lasagna"

    from job_store import LASAGNA_FLATTEN_PROFILE_NAME
    assert Path(profile["profile_id"]).name != LASAGNA_FLATTEN_PROFILE_NAME
    assert LASAGNA_FLATTEN_PROFILE_NAME == "flatten-lasagna-v1-1.0.0.json"
