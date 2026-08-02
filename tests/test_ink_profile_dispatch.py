"""The ink lane contract: profiles decide the model, and the clip is the divisor.

Two things are guarded here.

The dispatch: a profile names its model_type and the runner refuses anything it
does not implement, with a message that points at the right entry point. Adding
a detector must stay a matter of adding a profile.

The normalization: every published villa checkpoint expects
``clip(0, max_clip_value)`` followed by a divide by *that same value*. The
project shipped ``clamp(0, 200).div_(255)`` for 95 screening runs, which fed the
model 78.4% of the contrast range it was trained on. That is a silent defect --
the output stays in range and merely goes flat -- so it needs a test, not a
comment.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "framework" / "profiles" / "03-ink"
SCRIPT_DIR = ROOT / "framework" / "stages" / "03-ink" / "scripts"

RUNNER = SCRIPT_DIR / "run_ink.py"


def load_runner():
    import importlib.util
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("run_ink", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ink_profiles() -> list[Path]:
    return [
        p
        for p in sorted(PROFILE_DIR.glob("*.json"))
        if json.loads(p.read_text()).get("schema") == "campaignx.ink_lane_profile.v1"
    ]


def test_every_ink_profile_is_routable():
    """Routing must be resolvable from the profile alone.

    Either the profile declares a model_type the unified runner dispatches, or
    it names an adapter that exists. The frozen timesformer profiles take the
    second form on purpose: 95 receipts bind their sha256, so they are not
    edited to gain a field they do not need.
    """
    unroutable = []
    for path in ink_profiles():
        profile = json.loads(path.read_text())
        has_type = bool(profile["input_contract"].get("model_type"))
        adapter = profile.get("adapter")
        has_adapter = bool(adapter) and (ROOT / adapter).exists()
        if not (has_type or has_adapter):
            unroutable.append(profile["profile_id"])
    assert not unroutable, f"profiles with neither model_type nor a live adapter: {unroutable}"


def test_unified_runner_profiles_declare_a_model_type():
    """Anything pointed at the unified runner must say which model it is."""
    missing = [
        json.loads(p.read_text())["profile_id"]
        for p in ink_profiles()
        if Path(json.loads(p.read_text()).get("adapter", "")).name == RUNNER.name
        and not json.loads(p.read_text())["input_contract"].get("model_type")
    ]
    assert not missing, f"unified-runner profiles without model_type: {missing}"


def test_runner_rejects_a_model_type_it_does_not_implement():
    module = load_runner()
    timesformer = [
        p
        for p in ink_profiles()
        if json.loads(p.read_text())["input_contract"].get("model_type") == "timesformer"
    ]
    if not timesformer:
        pytest.skip("no timesformer profile present")
    with pytest.raises(RuntimeError, match="run_ink_timesformer"):
        module.read_profile(timesformer[0])


def test_profile_drives_frames_tile_and_clip():
    """The resnet50 profile would be the other example, but the registry now
    disqualifies it, so reading it raises -- which is itself the point."""
    module = load_runner()
    spec = module.read_profile(PROFILE_DIR / "ink-canonical-2um-screening-1.0.0.json")
    assert spec["model_type"] == "resnet3d-152-3d-decoder"
    assert spec["frames"] == 62
    assert spec["tile_size"] == 256
    assert spec["training_pixel_um"] == 2.399
    assert spec["max_clip_value"] == 200


def test_profile_id_resolves_without_a_path():
    module = load_runner()
    found = module.resolve_profile("ink-canonical-2um-screening@1.0.0")
    assert json.loads(found.read_text())["profile_id"] == "ink-canonical-2um-screening@1.0.0"


def test_depth_window_spans_the_physical_thickness_the_model_trained_on():
    """The depth axis needs the same physical match as the spatial axes.

    18 consecutive slices of a 2.399 um stack are 43 um of papyrus; the model
    trained at 7.9 um saw 142 um. Feeding it the thin slab produced a
    near-constant map, and nothing in the receipt said so.
    """
    module = load_runner()
    frames, training_slice_um = 18, 7.9
    for source_slice_um in (2.399, 8.64, 9.362):
        index = module.depth_positions(
            centre=54.0, frames=frames, step=training_slice_um / source_slice_um
        )
        assert len(index) == frames
        span_um = (index[-1] - index[0]) * source_slice_um
        assert span_um == pytest.approx((frames - 1) * training_slice_um, rel=1e-9)
        assert index.mean() == pytest.approx(54.0)


def test_depth_window_refuses_to_run_off_the_end_of_the_stack():
    module = load_runner()
    files = [Path(f"{i:02d}.tif") for i in range(28)]
    index = module.depth_positions(centre=14.0, frames=18, step=3.293)
    with pytest.raises(RuntimeError, match="outside the 28 available layers"):
        module.read_interpolated_depth(files, index)


def test_no_ink_adapter_clips_to_one_value_and_divides_by_another():
    """The upstream contract is clip(0,V) then /V. Any other divisor is a bug."""
    pattern = re.compile(
        r"clamp_?\(\s*0\s*,\s*([A-Za-z_0-9.\[\]'\"]+)\s*\)\s*"
        r"\.div_\(\s*(?:float\()?([A-Za-z_0-9.\[\]'\"]+)"
    )
    offenders = []
    for script in sorted(SCRIPT_DIR.glob("*.py")):
        for clip, divisor in pattern.findall(script.read_text()):
            if clip.rstrip(")") != divisor.rstrip(")"):
                offenders.append(f"{script.name}: clip={clip} divisor={divisor}")
    assert not offenders, "clip value and divisor disagree: " + "; ".join(offenders)


def test_declared_normalization_matches_the_clip_value():
    """A receipt that misreports its own preprocessing is worse than none."""
    for path in ink_profiles():
        profile = json.loads(path.read_text())
        contract = profile["input_contract"]
        clip = contract.get("max_clip_value")
        if clip is None:
            continue
        prose = " ".join(contract.get("normalization_provenance", []))
        assert "255" not in prose or "not 255" in prose, (
            f"{profile['profile_id']} declares max_clip_value={clip} but its "
            "normalization prose still endorses 255"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
