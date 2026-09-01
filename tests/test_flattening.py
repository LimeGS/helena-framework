"""P3, which until now did not exist.

The phase registry said "nothing here flattens" and P4 was being fed a PPM built
straight from a TIFXYZ instead. vc_flatten was in the same VC3D build the fleet
already grows with the whole time.

What is tested here is the part that is not vc_flatten: refusing an uncertified
surface, refusing to believe an exit code over the output on disk, and building
an argv entirely from the frozen profile.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/02-flattening"))

from flatten import (  # noqa: E402
    FlatteningFailed,
    SurfaceNotCertified,
    flatten_command,
    flatten_surface,
    load_profile,
)

PROFILE = ROOT / "framework/profiles/02-flattening/flatten-abf-v1-1.0.0.json"


def tifxyz(directory: Path) -> Path:
    """A real, tiny TIFXYZ: the input is measured before vc_flatten is called,
    so a placeholder byte string is not enough to reach the code under test."""
    import numpy
    import tifffile

    directory.mkdir(parents=True, exist_ok=True)
    grid = numpy.arange(16, dtype=numpy.float32).reshape(4, 4)
    for name, plane in zip(("x.tif", "y.tif", "z.tif"),
                           (grid, grid * 2.0, grid + 100.0)):
        tifffile.imwrite(directory / name, plane)
    (directory / "meta.json").write_text(json.dumps({"format": "tifxyz"}))
    return directory


def exits_zero_writing_nothing(directory: Path) -> Path:
    """A stand-in for vc_flatten that succeeds and produces no output.

    Not /bin/true: it is /usr/bin/true on macOS, and a test that skips on the
    developer's laptop is a test that does not run.
    """
    script = directory / "pretend-vc-flatten"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)
    return script


def test_the_shipped_profile_is_a_valid_profile():
    profile = load_profile(PROFILE)
    assert profile["profile_id"] == "flatten-abf-v1@1.0.0"


def test_every_parameter_comes_from_the_profile():
    """A flag typed at a prompt is a setting no receipt records and no second
    run reproduces."""
    argv = flatten_command("vc_flatten", Path("in"), Path("out"), {
        "iterations": 20, "downsample": 2, "lscm_only": True,
        "scale_to_3d_area": False})
    assert argv[:5] == ["vc_flatten", "--input", "in", "--output", "out",
                        ][:5]
    assert "--iterations" in argv and "20" in argv
    assert "--downsample" in argv and "2" in argv
    assert "--lscm-only" in argv
    assert "--no-scale" in argv


def test_abf_and_scaling_are_left_off_the_command_when_the_profile_wants_them():
    """Absence of a flag is the setting, so the shipped profile must not send
    --lscm-only or --no-scale at all."""
    argv = flatten_command("vc_flatten", Path("in"), Path("out"),
                           load_profile(PROFILE))
    assert "--lscm-only" not in argv
    assert "--no-scale" not in argv


def test_an_uncertified_surface_is_refused(tmp_path):
    """Flattening an uncertified surface produces a flat sheet of whatever the
    surface was -- including one that crossed a lamina -- with the seam that
    would have shown it smoothed out of view."""
    with pytest.raises(SurfaceNotCertified):
        flatten_surface(tifxyz(tmp_path / "in"), tmp_path / "out",
                        binary=exits_zero_writing_nothing(tmp_path),
                        profile=load_profile(PROFILE), voxel_size_um=7.91,
                        geometry_qc_state="GEOMETRY_UNMEASURED")


def test_an_exit_code_is_not_believed_over_the_output(tmp_path):
    """/bin/true succeeds and writes nothing. A phase that publishes on the
    strength of a zero exit publishes an empty directory."""
    with pytest.raises(FlatteningFailed) as failure:
        flatten_surface(tifxyz(tmp_path / "in"), tmp_path / "out",
                        binary=exits_zero_writing_nothing(tmp_path),
                        profile=load_profile(PROFILE), voxel_size_um=7.91,
                        geometry_qc_state="GEOMETRY_CERTIFIED",
                        physical_qc_state="CT_SUPPORTED")
    assert "wrote no" in str(failure.value)


def test_a_non_empty_destination_is_refused(tmp_path):
    """Two runs into one directory leave a mix of both, and the receipt hashes
    whatever happens to be there afterwards."""
    occupied = tifxyz(tmp_path / "out")
    with pytest.raises(FlatteningFailed) as failure:
        flatten_surface(tifxyz(tmp_path / "in"), occupied,
                        binary=exits_zero_writing_nothing(tmp_path),
                        profile=load_profile(PROFILE), voxel_size_um=7.91,
                        geometry_qc_state="GEOMETRY_CERTIFIED",
                        physical_qc_state="CT_SUPPORTED")
    assert "non-empty" in str(failure.value)


# --------------------------------------------------------------------------
# The two axes, and the floor
#
# P3 read the geometry verdict and nothing else for as long as it existed. Ten
# surfaces whose CT support was never measured were unrolled and rendered into
# the detector on the strength of their shape alone.
# --------------------------------------------------------------------------


def copies_the_input(directory: Path) -> Path:
    """A stand-in for vc_flatten that writes a real TIFXYZ, so the receipt is
    built and the area comparison is reached."""
    script = directory / "pretend-vc-flatten-that-works"
    script.write_text(
        "#!/bin/sh\n"
        'while [ $# -gt 0 ]; do case "$1" in --input) src="$2";; --output) dst="$2";; esac; shift; done\n'
        'mkdir -p "$dst" && cp "$src"/x.tif "$src"/y.tif "$src"/z.tif "$src"/meta.json "$dst"/\n')
    script.chmod(0o755)
    return script


def test_a_surface_the_ct_never_supported_is_refused(tmp_path):
    from flatten import SurfaceNotAdmissible

    with pytest.raises(SurfaceNotAdmissible) as refused:
        flatten_surface(tifxyz(tmp_path / "in"), tmp_path / "out",
                        binary=copies_the_input(tmp_path),
                        profile=load_profile(PROFILE), voxel_size_um=7.91,
                        geometry_qc_state="GEOMETRY_CERTIFIED",
                        physical_qc_state="UNVALIDATED")
    assert "UNVALIDATED" in str(refused.value)


def test_the_override_is_explicit(tmp_path):
    """Comparing against what the old gate admitted has to stay possible, and
    it has to be something someone typed."""
    receipt = flatten_surface(tifxyz(tmp_path / "in"), tmp_path / "out",
                              binary=copies_the_input(tmp_path),
                              profile=load_profile(PROFILE), voxel_size_um=7.91,
                              geometry_qc_state="GEOMETRY_CERTIFIED",
                              physical_qc_state="UNVALIDATED",
                              require_physical_qc=False)
    assert receipt["status"] == "FLATTENED"
    assert receipt["physical_qc_state"] == "UNVALIDATED"


def test_a_surface_retained_for_review_is_still_ct_supported(tmp_path):
    receipt = flatten_surface(tifxyz(tmp_path / "in"), tmp_path / "out",
                              binary=copies_the_input(tmp_path),
                              profile=load_profile(PROFILE), voxel_size_um=7.91,
                              geometry_qc_state="GEOMETRY_CERTIFIED",
                              physical_qc_state="CT_SUPPORTED_REVIEW")
    assert receipt["status"] == "FLATTENED"


def test_a_sheet_that_lost_most_of_its_area_is_not_published():
    """Not a failure -- vc_flatten ran and produced a sheet -- but not something
    to hand the detector either. The receipt keeps the measurement; the status
    keeps it out of the queue P4 reads.

    Measured ratios on this corpus are 0.944 to 0.957, so 0.80 is far outside
    anything observed rather than a tuned threshold."""
    from flatten import AREA_RATIO_FLOOR, flattening_status

    assert flattening_status(0.95, AREA_RATIO_FLOOR) == "FLATTENED"
    assert flattening_status(0.62, AREA_RATIO_FLOOR) == "FLATTENING_REJECTED_AREA"
    assert flattening_status(AREA_RATIO_FLOOR, AREA_RATIO_FLOOR) == "FLATTENED"
    # No measurement is not a rejection: rejecting on a missing number rejects
    # on the absence of evidence.
    assert flattening_status(None, AREA_RATIO_FLOOR) == "FLATTENED"


def test_the_floor_reaches_the_receipt(tmp_path):
    receipt = flatten_surface(tifxyz(tmp_path / "in"), tmp_path / "out",
                              binary=copies_the_input(tmp_path),
                              profile=load_profile(PROFILE), voxel_size_um=7.91,
                              geometry_qc_state="GEOMETRY_CERTIFIED",
                              physical_qc_state="CT_SUPPORTED")
    from flatten import AREA_RATIO_FLOOR

    assert receipt["area_ratio_floor"] == AREA_RATIO_FLOOR


def test_panel_flattening_readback_returns_full_current_job_receipt(monkeypatch):
    import panel.app as app

    receipt = {
        "schema": "campaignx.surface_flattening_receipt.v1",
        "state": "FLATTENED", "surface_id": "surface-control",
        "requested_by_job_id": "p3-current", "source_artifact_sha256": "3" * 64,
        "profile_id": "flatten-abf-v1@1.0.0", "profile_file_sha256": "5" * 64,
        "artifact_uri": "s3://flat/current", "artifact_sha256": "6" * 64,
        "files": 1, "objects": [
            {"object_key": "x.tif", "sha256": "7" * 64, "bytes": 42}],
    }
    payload = {**receipt, "receipt_sha256": app._canonical_document_sha256(receipt)}

    class Cursor:
        calls = 0

        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def execute(self, _statement, _parameters=()):
            self.calls += 1
        def fetchone(self): return (1,)
        def fetchall(self):
            if self.calls == 3:
                return [("FLATTENED", 1)]
            return [("flat-id", "surface-control", "PHerc0139",
                     "flatten-abf-v1@1.0.0", "FLATTENED", 0.95,
                     "s3://flat/current", "6" * 64,
                     datetime(2026, 8, 2, tzinfo=timezone.utc), payload)]

    class Connection:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def cursor(self): return Cursor()

    monkeypatch.setattr(app, "DSN", "configured")
    monkeypatch.setattr(app, "read_scope", lambda mission, sample: {"PHerc0139"})
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(
        connect=lambda *_args, **_kwargs: Connection()))
    body = json.loads(app.api_flattening(sample="PHerc0139", mission="control").body)
    row = body["rows"][0]
    assert row["artifact_id"] == "flat-id"
    assert row["requested_by_job_id"] == "p3-current"
    assert row["source_artifact_sha256"] == "3" * 64
    assert row["receipt"] == receipt
    assert row["receipt_sha256"] == app._canonical_document_sha256(receipt)


# --------------------------------------------------------------- lasagna --
#
# A second flattener, selectable rather than substituted. vc_flatten stays the
# default: its profile is frozen and its hash is in receipts already written,
# so `engine` is absent from that profile and absent must keep meaning
# vc_flatten forever.

LASAGNA_PROFILE = (ROOT
                   / "framework/profiles/02-flattening/flatten-lasagna-v1-1.0.0.json")


def test_a_profile_with_no_engine_is_still_vc_flatten():
    """The shipped profile predates the choice and cannot be edited: its
    content hash is in every receipt P3 has ever written."""
    profile = load_profile(PROFILE)
    assert "engine" not in profile, (
        "adding a field to the frozen profile changes its hash and breaks the "
        "comparability the profile exists to provide")
    argv = flatten_command("vc_flatten", Path("in"), Path("out"), profile)
    assert argv[0] == "vc_flatten"
    assert "--iterations" in argv


def test_the_lasagna_profile_is_a_valid_profile():
    profile = load_profile(LASAGNA_PROFILE)
    assert profile["profile_id"] == "flatten-lasagna-v1@1.0.0"
    assert profile["engine"] == "lasagna"


def test_lasagna_builds_its_own_command_from_its_own_profile():
    """Lasagna is config-file driven, not flag driven: bare .json paths are
    positional and everything else is a flag. An argv shaped like
    vc_flatten's would simply not run."""
    profile = load_profile(LASAGNA_PROFILE)
    argv = flatten_command("/opt/villa/lasagna/fit.py", Path("in"), Path("out"),
                           profile, volume="s3://bucket/scroll.zarr")
    assert argv[0] == "/opt/villa/lasagna/fit.py"
    # Every config the profile names, as a bare positional .json.
    for config in profile["configs"]:
        assert config in argv, f"{config} never reached the command"
    assert "--tifxyz-init" in argv and "in" in argv
    assert "--out-dir" in argv and "out" in argv
    assert "--input" in argv and "s3://bucket/scroll.zarr" in argv
    assert "--device" in argv and profile["device"] in argv
    # vc_flatten's flags must not leak into a lasagna run.
    assert "--iterations" not in argv
    assert "--downsample" not in argv


def test_lasagna_without_a_volume_is_refused_rather_than_guessed():
    """vc_flatten reads only the TIFXYZ; lasagna samples the CT and cannot be
    given a default for which scan that is."""
    profile = load_profile(LASAGNA_PROFILE)
    with pytest.raises(ValueError, match="volume"):
        flatten_command("fit.py", Path("in"), Path("out"), profile)


def test_an_unknown_engine_is_refused_rather_than_defaulted():
    """Silently falling back to vc_flatten would write a receipt naming an
    engine that never ran."""
    with pytest.raises(ValueError, match="engine"):
        flatten_command("x", Path("in"), Path("out"),
                        {"profile_id": "p@1.0.0", "engine": "not-a-flattener"})
