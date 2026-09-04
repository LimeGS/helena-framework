"""Turning `input_disable_patches` off is not a one-line flip.

spiral-fitter-v1@0.4.0 fits with `input_disable_patches: true` -- the minimal
route, no verified or unverified patches -- because upstream's own splice
tolerance means a patch-less fit can sit a winding or more away from the
actual sheet. spiral-fitter-v1@0.4.1 exists to make an A/B against that
possible, by turning patches back on. Read naively, that is one config key.
Read against spiral-fitting/fit_session.py's own FIT_INPUT_CATALOG at the
pinned commit, it is not:

  * `verified_patches`'s FitInputSpec sets `required=_verified_patches_enabled`
    -- the only optional-looking input in the whole catalog whose required-ness
    equals its enabled-ness. The instant patches are not force-disabled,
    `input_use_verified_patches`'s own True default makes verified_patches
    required, and fit_spiral.py's own loader does `os.listdir()` on it
    unconditionally -- which raises on a directory that is not there. This
    platform holds no verified_patches for any target scroll (that absence is
    the whole reason grow-track-patches exists), so leaving
    input_use_verified_patches at its default would make 0.4.1 crash on every
    dataset it could otherwise run against.
  * `unverified_patches` -- the input 0.4.1 actually wants -- has no
    conventional_relative at all in FIT_INPUT_CATALOG. It is reachable only
    through a spiral-scroll.json `paths` override, and adapter.py's own
    mirrored SCROLL_SPEC_PATH_OVERRIDE_KEYS was missing exactly that key (and
    verified_patches) before this change -- harmless while every profile ran
    with patches disabled, and a silent dead end for 0.4.1 otherwise: the
    override would have been rejected as unknown, unverified_patches would
    resolve to no path, and the fit would run exactly as if patches were still
    off, with no error at all.

Every test below is one of those two findings, or the fix for them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation/backends/spiral"))

import adapter  # noqa: E402

PHERC0172 = {"dataset_path": "/artifacts/spiral/PHerc0172",
             "scroll_name": "PHerc0172", "z_begin": 500, "z_end": 9000,
             "voxel_size_um": 7.91, "spiral_outward_sense": "CCW"}

PROFILE_DIR = ROOT / "framework/profiles/01-segmentation"

# A minimal stand-in for spiral-fitting/config.py's Config class, carrying
# only the keys these tests need to see accepted or rejected. Verified
# separately against the real pinned commit (not asserted here, since this
# repository does not vendor villa's source): 0.4.1's five config_overrides
# keys all resolve against the real Config().as_dict().
FAKE_CONFIG = '''\
class Config:
    def __init__(self, overrides=None):
        self.input_disable_patches = False
        self.input_use_verified_patches = True
        self.input_use_unverified_patches = True
        self.input_use_tracks = False
        self.input_use_outer_shell = True
        self.dense_spacing_mode = "phase"
        for key, value in (overrides or {}).items():
            setattr(self, key, value)

    def as_dict(self):
        return vars(self).copy()
'''


def fitter(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.py").write_text(FAKE_CONFIG, encoding="utf-8")
    return root


# -- the adapter's own path-override-key list --------------------------------

def test_the_mirrored_override_keys_now_include_both_patch_directories():
    """Before this change the tuple had 10 of upstream's 12
    SCROLL_SPEC_PATH_OVERRIDE_KEYS entries -- fit_session.py's own
    construction is `tuple(spec.key for spec in FIT_INPUT_CATALOG if
    spec.kind != "pcl-set")`, and verified_patches/unverified_patches are
    both kind="directory", not "pcl-set"."""
    assert "verified_patches" in adapter.SCROLL_SPEC_PATH_OVERRIDE_KEYS
    assert "unverified_patches" in adapter.SCROLL_SPEC_PATH_OVERRIDE_KEYS
    assert len(adapter.SCROLL_SPEC_PATH_OVERRIDE_KEYS) == 12


# -- the new dataset_layout key -----------------------------------------------

def test_unverified_patches_dir_defaults_to_not_asked_for():
    assert adapter.LAYOUT_DEFAULTS["unverified_patches_dir"] == ""
    assert adapter.validate_layout(None)["unverified_patches_dir"] == ""
    assert adapter.validate_layout({})["unverified_patches_dir"] == ""


def test_0_4_0s_own_manifest_is_unchanged_by_this_key_existing():
    """The regression this default guards against: adding a fifth layout key
    must not put a new path in every profile's spiral-scroll.json, only in
    one that actually asks for it."""
    document, _layout = adapter.scroll_spec_document(PHERC0172, None)
    assert "paths" not in document


@pytest.mark.parametrize("bogus", [
    "{unverified_patches}", "sub/dir", "../../etc", ".", "..",
])
def test_a_malformed_unverified_patches_dir_is_refused(bogus):
    with pytest.raises(adapter.ScrollSpecRefused):
        adapter.validate_layout({"unverified_patches_dir": bogus})


def test_a_plain_directory_name_becomes_one_path_override():
    document, layout = adapter.scroll_spec_document(
        PHERC0172, {"unverified_patches_dir": "unverified_patches"})
    assert layout["unverified_patches_dir"] == "unverified_patches"
    assert document["paths"] == {
        "unverified_patches": "/artifacts/spiral/PHerc0172/unverified_patches"}


def test_naming_unverified_patches_dir_does_not_also_move_verified_patches():
    """The two keys are independent: this profile never writes a
    verified_patches override at all (see notes.config_overrides -- it is
    disabled, not relocated)."""
    document, _layout = adapter.scroll_spec_document(
        PHERC0172, {"unverified_patches_dir": "unverified_patches"})
    assert "verified_patches" not in document["paths"]


def test_every_path_override_key_this_layout_can_write_is_one_upstream_accepts():
    document, _layout = adapter.scroll_spec_document(
        PHERC0172, {"unverified_patches_dir": "unverified_patches",
                    "tracks_file": "other.dbm"})
    assert set(document["paths"]) <= set(adapter.SCROLL_SPEC_PATH_OVERRIDE_KEYS)


# -- the two profiles ----------------------------------------------------------

def test_both_spiral_fitter_profiles_load_and_are_runnable():
    for name in ("spiral-fitter-v1-0.4.0.json", "spiral-fitter-v1-0.4.1.json"):
        profile = adapter.load_spiral_profile(PROFILE_DIR / name)
        adapter.require_runnable_profile(profile)


def test_0_4_1_is_a_sibling_not_a_successor():
    """The two exist to be A/B compared, not to replace one another -- unlike
    every earlier version bump in this family, neither carries supersedes/
    superseded_by pointing at the other."""
    v040 = adapter.load_spiral_profile(PROFILE_DIR / "spiral-fitter-v1-0.4.0.json")
    v041 = adapter.load_spiral_profile(PROFILE_DIR / "spiral-fitter-v1-0.4.1.json")
    assert v040.get("superseded_by") != "spiral-fitter-v1@0.4.1"
    assert v041.get("supersedes") != "spiral-fitter-v1@0.4.0"


def test_0_4_1_disables_patches_it_cannot_supply_and_enables_the_one_it_can():
    """The two decisions this profile's config_overrides has to make,
    verified against fit_session.input_source_enabled's own coupling: turn
    verified_patches off explicitly (it would otherwise become required with
    nothing to satisfy it) while leaving unverified_patches at its True
    default (the one this platform can actually supply, through
    --unverified-patches-dir)."""
    v041 = adapter.load_spiral_profile(PROFILE_DIR / "spiral-fitter-v1-0.4.1.json")
    overrides = v041["config_overrides"]
    assert overrides["input_disable_patches"] is False
    assert overrides["input_use_verified_patches"] is False
    assert "input_use_unverified_patches" not in overrides  # already True upstream


def test_0_4_1s_config_overrides_are_keys_the_fitters_own_config_accepts(tmp_path):
    v041 = adapter.load_spiral_profile(PROFILE_DIR / "spiral-fitter-v1-0.4.1.json")
    root = fitter(tmp_path / "spiral")
    accepted = adapter.spiral_config_keys(root)
    unknown = sorted(set(v041["config_overrides"]) - accepted)
    assert unknown == []


def test_0_4_1_declares_verified_patches_not_applicable_and_unverified_optional():
    v041 = adapter.load_spiral_profile(PROFILE_DIR / "spiral-fitter-v1-0.4.1.json")
    not_applicable = " ".join(v041["inputs"]["not_applicable_under_this_profile"])
    assert "verified_patches" in not_applicable
    optional_paths = {entry["path"] for entry in v041["inputs"]["optional"]}
    assert "{unverified_patches_dir}" in optional_paths


def test_0_4_0_and_0_4_1_disagree_on_exactly_the_patch_related_overrides():
    """Everything unrelated to patches carries over unchanged -- input_use_
    tracks and input_use_outer_shell and dense_spacing_mode are identical, so
    the A/B measures patches, not some other drift between the two."""
    v040 = adapter.load_spiral_profile(PROFILE_DIR / "spiral-fitter-v1-0.4.0.json")
    v041 = adapter.load_spiral_profile(PROFILE_DIR / "spiral-fitter-v1-0.4.1.json")
    for shared_key in ("input_use_tracks", "input_use_outer_shell",
                       "dense_spacing_mode"):
        assert v040["config_overrides"][shared_key] == v041["config_overrides"][shared_key]
    assert v040["config_overrides"]["input_disable_patches"] is True
    assert v041["config_overrides"]["input_disable_patches"] is False
