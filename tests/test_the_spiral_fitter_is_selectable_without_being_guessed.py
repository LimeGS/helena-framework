"""The spiral fitter as a P1 backend, and what it actually lets a caller choose.

Upstream calls this "currently our most powerful method" for recovering a
surface and now recommends it. At 23adee04 it has a real three-flag headless
CLI (--dataset, --scroll-spec, --cache), and its two halves are still selected
two different ways:

  * ~120 keys -- the z range and the optimizer seed among them now, alongside
    everything the old default_config dict held -- are validated by upstream
    against `config.Config().as_dict()` and ride FIT_SPIRAL_CONFIG_OVERRIDES.
    That half is this module's job, and it refuses a bad key before a GPU is
    claimed rather than after.
  * The scroll's name, voxel size and winding sense are not among those keys.
    They are `spiral-scroll.json` fields now, so selecting a scroll is a
    manifest write rather than a source rewrite. That half is tested in
    test_the_spiral_scroll_is_named_in_a_manifest_not_rebound_in_source.py.

What is checked here is the first half, plus the profile telling the truth
about all four generations: the frozen 0.4.0 writes a manifest and is
runnable, and 0.1.0-0.3.0 -- which respectively could not select a scroll,
required four inputs the fit treats as optional, and drove an AST rebind this
commit removed the target of -- still parse and refuse to run.
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
           / "framework/profiles/01-segmentation/spiral-fitter-v1-0.4.0.json")
# Three generations behind it, each superseded for its own reason: 0.1.0 could
# not select a scroll, 0.2.0 required four inputs the fit treats as optional,
# 0.3.0 drove an AST rebind of six module constants 23adee04 removed.
SUPERSEDED = [
    ROOT / "framework/profiles/01-segmentation/spiral-fitter-v1-0.1.0.json",
    ROOT / "framework/profiles/01-segmentation/spiral-fitter-v1-0.2.0.json",
    ROOT / "framework/profiles/01-segmentation/spiral-fitter-v1-0.3.0.json",
]
SCROLL = {"dataset_path": "/artifacts/spiral/PHerc0172",
          "scroll_name": "PHerc0172", "z_begin": 500, "z_end": 9000,
          "voxel_size_um": 7.91, "spiral_outward_sense": "CCW"}


def fake_fitter_root(directory: Path) -> Path:
    """Upstream's shape at 23adee04, not upstream's file: a lightweight
    config.py with a Config class, importable without touching a dataset."""
    (directory / "config.py").write_text(
        "class Config:\n"
        "    def __init__(self, overrides=None):\n"
        "        self.z_begin = 4000\n"
        "        self.z_end = 17000\n"
        "        self.optimizer_random_seed = 1\n"
        "        self.input_disable_patches = False\n"
        "        self.num_training_steps = 20000\n"
        "        for key, value in (overrides or {}).items():\n"
        "            setattr(self, key, value)\n"
        "\n"
        "    def as_dict(self):\n"
        "        return vars(self).copy()\n",
        encoding="utf-8")
    return directory


def test_the_override_keys_are_read_from_config_not_restated(tmp_path):
    """A second copy of a ~120-key list is a second thing to keep true. The
    keys come from importing config.py directly -- cheap and side-effect-free,
    unlike fit_spiral.py, whose module level opens a dataset."""
    keys = spiral_config_keys(fake_fitter_root(tmp_path))
    assert keys == frozenset({"z_begin", "z_end", "optimizer_random_seed",
                              "input_disable_patches", "num_training_steps"})


def test_an_override_upstream_would_reject_is_refused_here_first(tmp_path):
    """Upstream raises KeyError deep inside a run that has already claimed a
    GPU and loaded a dataset. This is the same refusal, before any of that."""
    root = fake_fitter_root(tmp_path)
    with pytest.raises(UnknownOverrideKey, match="dataset_path"):
        spiral_environment({"profile_id": "p@0.1.0",
                            "config_overrides": {"dataset_path": "/x"}},
                           fitter_root=root, out_dir=tmp_path / "out",
                           cache_dir=tmp_path / "cache")


def test_the_environment_carries_the_overrides_as_the_json_upstream_parses(tmp_path):
    root = fake_fitter_root(tmp_path)
    env = spiral_environment(
        {"profile_id": "p@0.1.0",
         "config_overrides": {"input_disable_patches": True,
                              "optimizer_random_seed": 7}},
        fitter_root=root, out_dir=tmp_path / "out", cache_dir=tmp_path / "cache",
        run_tag="control-a")
    assert json.loads(env["FIT_SPIRAL_CONFIG_OVERRIDES"]) == {
        "input_disable_patches": True, "optimizer_random_seed": 7}
    assert env["FIT_SPIRAL_OUT_DIR"] == str(tmp_path / "out")
    assert env["FIT_SPIRAL_RUN_DIR"] == str(tmp_path / "out")
    assert env["FIT_SPIRAL_CACHE_DIR"] == str(tmp_path / "cache")
    assert env["FIT_SPIRAL_RUN_TAG"] == "control-a"


def test_no_overrides_means_no_override_variable_at_all(tmp_path):
    """Upstream treats an empty string as "no overrides", but sending "{}"
    would put an empty object in the receipt as though a choice was made."""
    env = spiral_environment({"profile_id": "p@0.1.0", "config_overrides": {}},
                             fitter_root=fake_fitter_root(tmp_path),
                             out_dir=tmp_path / "out", cache_dir=tmp_path / "c")
    assert "FIT_SPIRAL_CONFIG_OVERRIDES" not in env


def test_wandb_mode_is_forced_disabled_even_if_the_base_environment_disagrees(tmp_path):
    """Not merely defaulted: this platform has no wandb credentials and must
    never attempt the network call that would find that out."""
    root = fake_fitter_root(tmp_path)
    env = spiral_environment({"config_overrides": {}}, fitter_root=root,
                             out_dir=tmp_path / "out", cache_dir=tmp_path / "c",
                             base_env={"WANDB_MODE": "online", "WANDB_API_KEY": "x"})
    assert env["WANDB_MODE"] == "disabled"


def test_rank_world_size_and_local_rank_never_survive_into_the_run(tmp_path):
    """DistributedContext.from_env() reads exactly these three and defaults to
    single-process when they are absent. A worker is one GPU per container,
    and this makes that a guarantee rather than an accident of what the host
    happened not to export."""
    root = fake_fitter_root(tmp_path)
    env = spiral_environment(
        {"config_overrides": {}}, fitter_root=root, out_dir=tmp_path / "out",
        cache_dir=tmp_path / "c",
        base_env={"RANK": "1", "WORLD_SIZE": "4", "LOCAL_RANK": "1",
                 "PATH": "/usr/bin"})
    assert "RANK" not in env and "WORLD_SIZE" not in env and "LOCAL_RANK" not in env
    assert env["PATH"] == "/usr/bin"


def test_the_shipped_profile_is_a_valid_profile():
    profile = load_spiral_profile(PROFILE)
    assert profile["profile_id"] == "spiral-fitter-v1@0.4.0"
    assert profile["backend"] == "spiral"
    assert require_runnable_profile(profile) is profile


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
    """A frozen profile that stops parsing takes its own record with it. Each
    is still refused as a way to run one: 0.1.0 names no scroll, 0.2.0
    refuses every dataset without the Paris 4 winding annotations, and 0.3.0's
    six-constant rebind has nothing left to rebind at this commit."""
    superseded = load_spiral_profile(path)

    assert superseded["superseded_by"].startswith("spiral-fitter-v1@")
    with pytest.raises(ValueError, match="superseded"):
        require_runnable_profile(superseded)


def test_the_profile_splits_its_inputs_by_what_an_absence_costs():
    """Umbilicus, tracks and the lasagna volumes are opened directly and raise;
    the point-collection files are read through a loader that swallows every
    exception, so their absence degrades a fit rather than stopping one.

    0.2.0 demanded all nine, which is what the tutorial lists, and that made
    the fitter unrunnable on every scroll without the Paris 4 winding
    annotations. The required set is unchanged at this commit: see the
    profile's own notes.config_overrides for how that was verified rather
    than assumed.
    """
    inputs = load_spiral_profile(PROFILE)["inputs"]
    required = [entry["path"] for entry in inputs["required"]]
    optional = {entry["path"] for entry in inputs["optional"]}

    assert "umbilicus.json" in required
    assert any("tracks" in name for name in required)
    assert sum("lasagna_inputs" in name for name in required) == 3
    assert {"abs_winding.json", "patch-overlap-pcls.json",
            "relative_windings.json", "same_windings.json"} <= optional
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


def test_the_profile_names_where_the_scroll_spec_mechanism_lives():
    """Not "scroll_binding" any more (0.3.0's field, for the AST rebind); the
    manifest mechanism has its own section, and the required-keys check in
    load_spiral_profile depends on the schema saying which section a v4
    profile must carry."""
    profile = load_spiral_profile(PROFILE)
    assert "scroll_binding" not in profile
    assert profile["scroll_spec"]["filename"] == "spiral-scroll.json"
    assert profile["scroll_spec"]["defaults"] == {"spiral_outward_sense": "CW"}
