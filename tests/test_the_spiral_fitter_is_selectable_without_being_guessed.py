"""The spiral fitter as a P1 backend, and what it actually lets a caller choose.

Upstream calls this "currently our most powerful method" for recovering a
surface and now recommends it. At the locked commit it is a research script
rather than a tool, and its two halves are selected two different ways:

  * There is no argparse. Everything runs off nine FIT_SPIRAL_* environment
    variables, and `FIT_SPIRAL_CONFIG_OVERRIDES` is validated by upstream
    against the keys of its own `default_config`, raising KeyError on anything
    else. That half is this module's job, and it refuses a bad key before a GPU
    is claimed rather than after.
  * The scroll is not among those keys. `dataset_path`, `scroll_name`,
    `z_begin`/`z_end`, `voxel_size_um` and `spiral_outward_sense` are
    module-level constants, so selecting a scroll is a source rewrite. That
    half is `repin.py`, tested beside this file.

What is checked here is the first half, plus the profile telling the truth
about both: the frozen 0.2.0 declares a scroll binding and is runnable, and the
0.1.0 that predates the rebind still parses and refuses to run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation/backends/spiral"))

from adapter import (  # noqa: E402
    UnknownOverrideKey,
    load_spiral_profile,
    require_runnable_profile,
    scroll_binding_for,
    spiral_config_keys,
    spiral_environment,
)

PROFILE = (ROOT
           / "framework/profiles/01-segmentation/spiral-fitter-v1-0.3.0.json")
# Two generations behind it, each superseded for its own reason: 0.1.0 could
# not select a scroll, 0.2.0 required four inputs the fit treats as optional.
SUPERSEDED = [
    ROOT / "framework/profiles/01-segmentation/spiral-fitter-v1-0.1.0.json",
    ROOT / "framework/profiles/01-segmentation/spiral-fitter-v1-0.2.0.json",
]
SCROLL = {"dataset_path": "/artifacts/spiral/PHerc0172",
          "scroll_name": "PHerc0172", "z_begin": 500, "z_end": 9000,
          "voxel_size_um": 7.91, "spiral_outward_sense": "CCW"}


def fake_fit_spiral(directory: Path) -> Path:
    """Upstream's shape, not upstream's file: a module-level `default_config`
    dict, and constants that no environment variable reaches."""
    script = directory / "fit_spiral.py"
    script.write_text(
        "import os\n"
        "dataset_path = '/ephemeral/paul/spiral/dataset'\n"
        "scroll_name = 's1'\n"
        "z_begin, z_end = 4000, 17000\n"
        "default_config = {\n"
        "    'random_seed': 1,\n"
        "    'disable_patches': False,\n"
        "    'num_training_steps': 20000,\n"
        "}\n",
        encoding="utf-8")
    return script


def test_the_override_keys_are_read_from_the_script_not_restated(tmp_path):
    """A second copy of a 105-key list is a second thing to keep true. The
    keys are parsed out of the script itself, without importing it -- import
    would execute module-level code that opens datasets."""
    keys = spiral_config_keys(fake_fit_spiral(tmp_path))
    assert keys == frozenset({"random_seed", "disable_patches",
                              "num_training_steps"})


def test_an_override_upstream_would_reject_is_refused_here_first(tmp_path):
    """Upstream raises KeyError deep inside a run that has already claimed a
    GPU and loaded a dataset. This is the same refusal, before any of that."""
    script = fake_fit_spiral(tmp_path)
    with pytest.raises(UnknownOverrideKey, match="z_begin"):
        spiral_environment({"profile_id": "p@0.1.0",
                            "config_overrides": {"z_begin": 10000}},
                           script=script, out_dir=tmp_path / "out",
                           cache_dir=tmp_path / "cache")


def test_the_environment_carries_the_overrides_as_the_json_upstream_parses(tmp_path):
    script = fake_fit_spiral(tmp_path)
    env = spiral_environment(
        {"profile_id": "p@0.1.0",
         "config_overrides": {"disable_patches": True, "random_seed": 7}},
        script=script, out_dir=tmp_path / "out", cache_dir=tmp_path / "cache",
        run_tag="control-a")
    assert json.loads(env["FIT_SPIRAL_CONFIG_OVERRIDES"]) == {
        "disable_patches": True, "random_seed": 7}
    assert env["FIT_SPIRAL_OUT_DIR"] == str(tmp_path / "out")
    assert env["FIT_SPIRAL_CACHE_DIR"] == str(tmp_path / "cache")
    assert env["FIT_SPIRAL_RUN_TAG"] == "control-a"


def test_no_overrides_means_no_override_variable_at_all(tmp_path):
    """Upstream treats an empty string as "no overrides", but sending "{}"
    would put an empty object in the receipt as though a choice was made."""
    env = spiral_environment({"profile_id": "p@0.1.0", "config_overrides": {}},
                             script=fake_fit_spiral(tmp_path),
                             out_dir=tmp_path / "out", cache_dir=tmp_path / "c")
    assert "FIT_SPIRAL_CONFIG_OVERRIDES" not in env


def test_the_shipped_profile_is_a_valid_profile():
    profile = load_spiral_profile(PROFILE)
    assert profile["profile_id"] == "spiral-fitter-v1@0.3.0"
    assert profile["backend"] == "spiral"
    assert require_runnable_profile(profile) is profile


def test_the_profile_still_records_that_the_six_are_constants(tmp_path):
    """They are selectable now, and they are still not options: the rebind is a
    source rewrite, and a profile that stopped saying so would read as though
    upstream had an environment variable for the scroll."""
    profile = load_spiral_profile(PROFILE)
    constants = profile["upstream_module_constants"]
    for constant in ("dataset_path", "scroll_name", "z_begin", "voxel_size_um"):
        assert constant in constants["names"]
        assert constant in constants["upstream_values"]
    assert set(profile["scroll_binding"]["selectable"]) == set(constants["names"])


def test_a_run_that_names_a_scroll_gets_all_six():
    binding = scroll_binding_for(load_spiral_profile(PROFILE), SCROLL)
    assert binding == SCROLL


def test_the_winding_sense_is_the_only_one_with_a_default():
    """CW is upstream's, and a fit made with the wrong sense is visibly wrong.
    The other five have none, so a forgotten field cannot silently fit Scroll 1
    under another scroll's name."""
    without = {name: value for name, value in SCROLL.items()
               if name != "spiral_outward_sense"}
    assert scroll_binding_for(
        load_spiral_profile(PROFILE), without)["spiral_outward_sense"] == "CW"

    with pytest.raises(ValueError, match="does not say which scroll"):
        scroll_binding_for(load_spiral_profile(PROFILE), {"scroll_name": "x"})


@pytest.mark.parametrize("path", SUPERSEDED, ids=lambda p: p.stem)
def test_every_superseded_profile_still_parses_and_will_not_run(path):
    """A frozen profile that stops parsing takes its own record with it. Each is
    still refused as a way to run one: 0.1.0 names no scroll, so a fit made
    against it would use whatever dataset sat at upstream's path, and 0.2.0
    refuses every dataset without the Paris 4 winding annotations."""
    superseded = load_spiral_profile(path)

    assert superseded["superseded_by"].startswith("spiral-fitter-v1@")
    with pytest.raises(ValueError, match="superseded"):
        require_runnable_profile(superseded)


def test_the_profile_splits_its_inputs_by_what_an_absence_costs():
    """Umbilicus, tracks and the lasagna volumes are opened directly and raise;
    the four point-collection files are read through a loader that swallows
    every exception, so their absence degrades a fit rather than stopping one.

    0.2.0 demanded all nine, which is what the tutorial lists, and that made the
    fitter unrunnable on every scroll without the Paris 4 winding annotations.
    """
    inputs = load_spiral_profile(PROFILE)["inputs"]
    required = [entry["path"] for entry in inputs["required"]]
    optional = {entry["path"] for entry in inputs["optional"]}

    assert "umbilicus.json" in required
    assert any("tracks" in name for name in required)
    assert sum("lasagna_inputs" in name for name in required) == 3
    assert optional == {"abs_winding.json", "patch-overlap-pcls.json",
                        "relative_windings.json", "same_windings.json"}
    # Every optional input says what its absence costs, because "optional" is
    # not "free": without abs_winding.json the fit's windings are relative.
    assert all(entry.get("absence_costs") for entry in inputs["optional"])


def test_the_dataset_layout_is_selectable_and_defaults_to_upstreams():
    layout = load_spiral_profile(PROFILE)["dataset_layout"]

    selectable = layout["selectable"]
    assert selectable["lasagna_volume_name"]["default"] == "las_008_{array}.ome.zarr"
    assert selectable["normal_zarr_group"]["default"] == "4"
    assert selectable["tracks_file"]["default"].endswith(".dbm")
    assert layout["defaults_are_upstreams"] is True
    # The three lasagna settings are one decision, and the profile has to say
    # so: prepare_lasagna_volume checks the group it opened against the scale.
    assert "coupled" in layout
    assert "derived" in selectable["lasagna_scale"]
