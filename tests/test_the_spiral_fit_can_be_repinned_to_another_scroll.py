"""Pointing the spiral fitter at a scroll that is not Scroll 1.

The fitter selects its scroll with six module-level assignments evaluated at
import, none of which is one of upstream's 105 config keys. So a rebind is a
source rewrite, and the thing a source rewrite can do wrong is the thing this
guards: change five of six, match a name inside a comment, or leave a constant
untouched and report success. Any of those produces a fit that runs, costs a
GPU-day, and files windings of one scroll under the name of another.

Every test below is a way that could happen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation/backends/spiral"))

from repin import (  # noqa: E402
    SCROLL_CONSTANTS, ScrollNotRebindable, binding_sha256, module_constants,
    rebind, repin, validate_binding,
)

UPSTREAM = '''\
import os

# A comment mentioning dataset_path, and a docstring below that mentions it too.
"""z_begin is discussed here: résumé, ñ, and other bytes above ASCII."""

dataset_path = '/ephemeral/paul/spiral/dataset'
scroll_name = 's1'
z_begin, z_end = 4000, 17000
voxel_size_um = 9.6
spiral_outward_sense = 'CW'

# The dataset layout, as upstream writes it: f-strings over dataset_path,
# which is why rebinding them regenerates a template instead of a literal.
normal_nx_zarr_path = f'{dataset_path}/lasagna_inputs/las_008_nx.ome.zarr'
normal_ny_zarr_path = f'{dataset_path}/lasagna_inputs/las_008_ny.ome.zarr'
grad_mag_zarr_path = f'{dataset_path}/lasagna_inputs/las_008_grad_mag.ome.zarr'
tracks_dbm_path = f'{dataset_path}/tracks/2um_ds2_ps256_surf_v2.dbm'
normal_zarr_group = '4'
lasagna_scale = 4

default_config = {'disable_patches': False, 'random_seed': 1}


def elsewhere(scroll_name='not this one', z_begin=0):
    """A function whose parameters share the names. Not module level."""
    return scroll_name, z_begin
'''

PHERC0172 = {"dataset_path": "/artifacts/spiral/PHerc0172",
             "scroll_name": "PHerc0172", "z_begin": 500, "z_end": 9000,
             "voxel_size_um": 7.91, "spiral_outward_sense": "CCW"}


def values_of(source: str) -> dict[str, object]:
    return {name: value for name, (_node, value) in module_constants(source).items()}


# -- reading what is there -------------------------------------------------

def test_the_six_are_found_where_upstream_assigns_them():
    assert values_of(UPSTREAM) == {
        "dataset_path": "/ephemeral/paul/spiral/dataset", "scroll_name": "s1",
        "z_begin": 4000, "z_end": 17000, "voxel_size_um": 9.6,
        "spiral_outward_sense": "CW"}


def test_a_tuple_assignment_is_two_constants_not_one():
    """`z_begin, z_end = 4000, 17000` is one statement. A rewrite that treated
    it as one value would put a scalar where a pair belongs."""
    constants = module_constants(UPSTREAM)
    assert constants["z_begin"][1] == 4000 and constants["z_end"][1] == 17000


# -- rewriting -------------------------------------------------------------

def test_every_one_of_the_six_moves():
    assert values_of(rebind(UPSTREAM, PHERC0172)) == PHERC0172


def test_nothing_else_moves():
    """The comment, the docstring and the function default all name these
    constants. A textual substitution would have hit all three."""
    rewritten = rebind(UPSTREAM, PHERC0172)
    assert "A comment mentioning dataset_path" in rewritten
    assert "z_begin is discussed here" in rewritten
    assert "scroll_name='not this one'" in rewritten
    assert "'disable_patches': False" in rewritten
    # And the file is still a file, not a plausible-looking string.
    assert values_of(rewritten)["scroll_name"] == "PHerc0172"


def test_a_constant_before_a_non_ascii_line_still_lands_in_the_right_place():
    """Columns in a parse tree are byte offsets into utf-8. Counting
    characters puts the substitution a few bytes early on any line after an
    accented one, which produces a file that still parses."""
    rewritten = rebind(UPSTREAM, PHERC0172)
    assert "résumé, ñ" in rewritten
    assert values_of(rewritten)["voxel_size_um"] == 7.91


def test_a_missing_constant_stops_the_run_rather_than_leaving_scroll_1():
    """The failure this exists to prevent: upstream moves the selection
    somewhere else, the rewrite silently changes five of six, and the fit runs
    on a dataset that is half one scroll and half another."""
    moved = UPSTREAM.replace("spiral_outward_sense = 'CW'\n", "")
    with pytest.raises(ScrollNotRebindable, match="spiral_outward_sense"):
        rebind(moved, PHERC0172)


def test_a_computed_constant_is_refused_rather_than_replaced():
    """Replacing `voxel_size_um = native / 4` with a literal drops the
    computation, and nothing downstream would know."""
    computed = UPSTREAM.replace("voxel_size_um = 9.6",
                                "voxel_size_um = round(2.4 * 4, 3)")
    with pytest.raises(ScrollNotRebindable, match="computed"):
        rebind(computed, PHERC0172)


# -- refusing a binding that is not one -----------------------------------

@pytest.mark.parametrize("binding,expected", [
    ({**PHERC0172, "spiral_outward_sense": "clockwise"}, "CW"),
    ({**PHERC0172, "z_end": 500, "z_begin": 9000}, "above z_begin"),
    ({**PHERC0172, "voxel_size_um": 0}, "positive"),
    ({**PHERC0172, "scroll_name": "  "}, "must not be empty"),
    ({**PHERC0172, "z_begin": "500"}, "must be int"),
    ({name: value for name, value in PHERC0172.items() if name != "z_end"},
     "missing"),
    ({**PHERC0172, "num_training_steps": 20000}, "not module constants"),
])
def test_a_binding_that_would_fit_nothing_is_refused(binding, expected):
    with pytest.raises(ScrollNotRebindable, match=expected):
        validate_binding(binding)


def test_the_six_are_the_six():
    assert set(SCROLL_CONSTANTS) == set(PHERC0172)


# -- the receipt -----------------------------------------------------------

def test_the_receipt_names_both_scripts_by_digest(tmp_path):
    """"Which script actually ran" has to be answerable from the record: the
    image holds the bytes its source lock verified, and the run reads a copy."""
    script = tmp_path / "fit_spiral.py"
    script.write_text(UPSTREAM, encoding="utf-8")

    receipt = repin(script, PHERC0172, tmp_path / "run/fit_spiral.py")

    assert len(receipt["upstream_sha256"]) == 64
    assert len(receipt["rebound_sha256"]) == 64
    assert receipt["upstream_sha256"] != receipt["rebound_sha256"]
    assert receipt["binding_sha256"] == binding_sha256(PHERC0172)
    assert receipt["replaced"]["scroll_name"] == {"was": "s1", "now": "PHerc0172"}
    assert receipt["is_upstream_default"] is False
    # The upstream file is never edited: two scrolls can be fitted at once on
    # one host, and the lock still verifies what the image holds.
    assert script.read_text(encoding="utf-8") == UPSTREAM


def test_a_binding_equal_to_upstream_says_so(tmp_path):
    """The fact a cross-scroll comparison turns on: this run is the pinned
    Scroll 1 fit, unchanged."""
    script = tmp_path / "fit_spiral.py"
    script.write_text(UPSTREAM, encoding="utf-8")

    receipt = repin(script, {
        "dataset_path": "/ephemeral/paul/spiral/dataset", "scroll_name": "s1",
        "z_begin": 4000, "z_end": 17000, "voxel_size_um": 9.6,
        "spiral_outward_sense": "CW"}, tmp_path / "run/fit_spiral.py")

    assert receipt["is_upstream_default"] is True
    assert all(change["was"] == change["now"]
               for change in receipt["replaced"].values())
    # A claim about the six values, never about the bytes -- and now visibly so.
    # The dataset layout is re-emitted on every run as `dataset_path + '...'`,
    # because the f-string it used to write was an interpolation context a
    # caller value could escape into. So an unchanged binding still produces a
    # different file, and both digests are in the receipt to say which is which.
    assert receipt["upstream_sha256"] != receipt["rebound_sha256"]
    assert receipt["layout_is_upstream_default"] is True
