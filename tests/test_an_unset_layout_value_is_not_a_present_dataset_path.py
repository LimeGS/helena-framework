"""A dataset root is not a present copy of everything you forgot to name.

Found while writing spiral-fitter-v1@0.4.1's own `inputs.optional` entry for
`{unverified_patches_dir}` -- a layout key whose default is deliberately
empty (adapter.DEFAULT_UNVERIFIED_PATCHES_DIR: there is no upstream default
directory the way tracks_file or lasagna_volume_name have one).
`survey_inputs`'s own `resolve()` formats a profile's `{template}` path
against the resolved dataset layout, so an unset `unverified_patches_dir`
resolves `"{unverified_patches_dir}"` to `""`. `present()` then checked
`(dataset / "").exists()` -- and `Path("/a/b") / ""` is `Path("/a/b")` in
pathlib, not a nonexistent child, so that check reads "yes, present" for
every dataset, whether or not the input was ever named.

This is not specific to unverified_patches_dir: any optional templated path
built the same way, from a layout key whose value can be empty, would have
had the same false positive. The fix is in `present()` itself, not in the
one caller that happened to notice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation/scripts"))
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation/backends/spiral"))

import run_spiral_fit as runner  # noqa: E402
import adapter  # noqa: E402


def test_an_empty_resolved_path_is_absent_not_present(tmp_path):
    """The bug itself, isolated: a dataset that is a real, existing directory
    must not make `present("")` true."""
    tmp_path.mkdir(exist_ok=True)
    profile = {"inputs": {"optional": [
        {"path": "{unverified_patches_dir}", "absence_costs": "nothing to grow from"}]}}
    layout = adapter.validate_layout(None)  # unverified_patches_dir defaults to ""
    survey = runner.survey_inputs(tmp_path, profile, layout)
    assert survey["absent_optional"] == [
        {"path": "{unverified_patches_dir}", "costs": "nothing to grow from"}]
    assert survey["degraded"] is True


def test_a_named_directory_that_exists_is_present(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "unverified_patches").mkdir()
    profile = {"inputs": {"optional": [
        {"path": "{unverified_patches_dir}", "absence_costs": "nothing to grow from"}]}}
    layout = adapter.validate_layout({"unverified_patches_dir": "unverified_patches"})
    survey = runner.survey_inputs(tmp_path, profile, layout)
    assert survey["absent_optional"] == []
    assert survey["degraded"] is False


def test_a_named_directory_that_does_not_exist_is_still_absent(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    profile = {"inputs": {"optional": [
        {"path": "{unverified_patches_dir}", "absence_costs": "nothing to grow from"}]}}
    layout = adapter.validate_layout({"unverified_patches_dir": "unverified_patches"})
    survey = runner.survey_inputs(tmp_path, profile, layout)
    assert survey["absent_optional"] == [
        {"path": "unverified_patches", "costs": "nothing to grow from"}]


def test_0_4_1_run_against_a_dataset_with_no_patches_says_so_in_the_receipt(tmp_path):
    """The end-to-end shape of the bug: a real spiral-fit dry-run receipt,
    against spiral-fitter-v1@0.4.1's own profile, over a dataset with every
    required input but no grown patches at all."""
    root = tmp_path / "ds"
    root.mkdir(parents=True, exist_ok=True)
    (root / "umbilicus.json").write_text("{}")
    (root / "tracks").mkdir()
    (root / "tracks/2um_ds2_ps256_surf_v2.dbm").write_text("")
    for array in ("nx", "ny", "grad_mag"):
        (root / "lasagna_inputs" / f"las_008_{array}.ome.zarr").mkdir(parents=True)

    profile_path, profile = runner.resolve_profile("spiral-fitter-v1@0.4.1")
    layout = adapter.validate_layout({})  # no --unverified-patches-dir given
    survey = runner.survey_inputs(root, profile, layout)
    assert survey["missing_required"] == []
    absent_paths = {entry["path"] for entry in survey["absent_optional"]}
    assert "{unverified_patches_dir}" in absent_paths
    costs = next(entry["costs"] for entry in survey["absent_optional"]
                if entry["path"] == "{unverified_patches_dir}")
    assert "0.4.0" in costs  # names what silently running like 0.4.0 means
