"""Pointing the spiral fitter at a scroll that is not Scroll 1, at 23adee04.

Before this commit the fitter selected its scroll with six module-level
assignments evaluated at import, and Helena rebound them with an AST rewrite
into a private copy of the script -- repin.py, deleted alongside this test's
predecessor. Upstream restructured the scroll's identity into a JSON manifest,
spiral-scroll.json, and there is no more source to rewrite: dataset_path
becomes a CLI flag, and the scroll's name, voxel size and winding sense become
manifest fields written fresh for every run.

What a rebind could once get wrong -- change five of six, match a name inside
a comment, leave a constant untouched and report success -- is not a failure
mode JSON writing has. What replaces it: writing the wrong value, writing a
manifest that reads back as something other than what was asked for, or
translating the winding sense to the wrong upstream spelling. Every test below
is one of those.
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


# -- the CW/CCW -> CW/ACW boundary ------------------------------------------

def test_cw_stays_cw():
    assert adapter.translate_winding_sense("CW") == "CW"


def test_ccw_becomes_acw():
    """The one deliberate decision this integration makes: Helena's own
    vocabulary (CCW) is not upstream's (ACW), and this is the one place they
    meet."""
    assert adapter.translate_winding_sense("CCW") == "ACW"


@pytest.mark.parametrize("bogus", ["ACW", "clockwise", "ccw", "cw", "", None])
def test_anything_outside_cw_ccw_is_refused(bogus):
    """In particular: upstream's own spelling, ACW, is not something a caller
    of this platform's API may type. Accepting it here would mean Helena's
    public vocabulary silently absorbed upstream's rename."""
    with pytest.raises(adapter.ScrollSpecRefused, match="CW or CCW"):
        adapter.translate_winding_sense(bogus)


# -- the scroll binding ------------------------------------------------------

def test_a_full_binding_validates():
    checked = adapter.validate_binding(PHERC0172)
    assert checked == PHERC0172


@pytest.mark.parametrize("binding,expected", [
    ({**PHERC0172, "spiral_outward_sense": "clockwise"}, "CW or CCW"),
    ({**PHERC0172, "z_end": 500, "z_begin": 9000}, "above z_begin"),
    ({**PHERC0172, "voxel_size_um": 0}, "positive"),
    ({**PHERC0172, "scroll_name": "  "}, "must not be empty"),
    ({**PHERC0172, "z_begin": "500"}, "must be int"),
    ({name: value for name, value in PHERC0172.items() if name != "z_end"},
     "missing"),
    ({**PHERC0172, "num_training_steps": 20000}, "not part of the scroll binding"),
])
def test_a_binding_that_would_fit_nothing_is_refused(binding, expected):
    with pytest.raises(adapter.ScrollSpecRefused, match=expected):
        adapter.validate_binding(binding)


# -- the dataset layout -------------------------------------------------------

def test_the_layout_defaults_reproduce_upstream():
    assert adapter.validate_layout(None) == adapter.LAYOUT_DEFAULTS
    assert adapter.validate_layout({}) == adapter.LAYOUT_DEFAULTS


def test_lasagna_scale_is_derived_when_the_layout_moves_and_scale_is_not_stated():
    layout = adapter.validate_layout({"normal_zarr_group": "3"})
    assert layout["lasagna_scale"] == 8  # 2 ** 3, not upstream's own 4-at-group-4


def test_an_explicit_scale_is_not_overridden_by_the_derivation():
    layout = adapter.validate_layout(
        {"normal_zarr_group": "3", "lasagna_scale": 4})
    assert layout["lasagna_scale"] == 4


@pytest.mark.parametrize("payload", [
    # A brace-injection payload class that mattered when this was rewritten
    # into an f-string; kept as a test because the same values are still not
    # ones this platform should accept into a filename, even though writing
    # JSON has no interpolation context left for them to escape into.
    "las_{array}_{{__import__('os').system('id')}}",
    "las_{array}_{{open('/tmp/x','w')}}",
    "{array}}{",
    "x{array}{oops}",
    "las_{array}/../../etc",
])
def test_a_volume_name_that_could_carry_an_expression_is_refused(payload):
    with pytest.raises(adapter.ScrollSpecRefused):
        adapter.validate_layout({"lasagna_volume_name": payload})


def test_a_volume_name_naming_no_array_is_refused():
    with pytest.raises(adapter.ScrollSpecRefused, match="array"):
        adapter.validate_layout({"lasagna_volume_name": "las_008_nx.ome.zarr"})


def test_a_tracks_name_carries_no_braces():
    with pytest.raises(adapter.ScrollSpecRefused, match="no braces"):
        adapter.validate_layout({"tracks_file": "{__import__('os')}.dbm"})


def test_a_tracks_name_that_is_a_path_is_refused():
    with pytest.raises(adapter.ScrollSpecRefused, match="not a path"):
        adapter.validate_layout({"tracks_file": "../../etc/passwd.dbm"})


def test_an_unknown_layout_key_is_refused():
    with pytest.raises(adapter.ScrollSpecRefused, match="not part of the dataset layout"):
        adapter.validate_layout({"lasagna_bucket": "somewhere-else"})


# -- the manifest -------------------------------------------------------------

def test_the_manifest_carries_the_translated_sense_and_no_z_range():
    document, _layout = adapter.scroll_spec_document(PHERC0172, None)
    assert document["schema_version"] == 1
    assert document["name"] == "PHerc0172"
    assert document["voxel_size_um"] == 7.91
    assert document["spiral_outward_sense"] == "ACW"
    # z_begin/z_end are not scroll-spec fields at this commit: they ride
    # FIT_SPIRAL_CONFIG_OVERRIDES instead. A manifest that carried them would
    # be describing a fact this file no longer owns.
    assert "z_begin" not in document and "z_end" not in document


def test_upstream_default_layout_needs_no_path_overrides():
    """A run against upstream's own naming should ask the fitter to resolve
    its own conventional paths, not restate them."""
    document, _layout = adapter.scroll_spec_document(PHERC0172, None)
    assert "paths" not in document


def test_a_moved_lasagna_volume_name_becomes_three_path_overrides():
    document, _layout = adapter.scroll_spec_document(
        PHERC0172, {"lasagna_volume_name": "PHerc0172_{array}.ome.zarr"})
    assert document["paths"] == {
        "normal_x": "/artifacts/spiral/PHerc0172/lasagna_inputs/PHerc0172_nx.ome.zarr",
        "normal_y": "/artifacts/spiral/PHerc0172/lasagna_inputs/PHerc0172_ny.ome.zarr",
        "gradient_magnitude":
            "/artifacts/spiral/PHerc0172/lasagna_inputs/PHerc0172_grad_mag.ome.zarr",
    }


def test_a_moved_tracks_file_becomes_one_path_override():
    document, _layout = adapter.scroll_spec_document(
        PHERC0172, {"tracks_file": "m7_ds2_z3000_18000_surf.dbm"})
    assert document["paths"] == {
        "tracks_dbm":
            "/artifacts/spiral/PHerc0172/tracks/m7_ds2_z3000_18000_surf.dbm",
    }


def test_every_path_override_key_is_one_upstream_actually_accepts():
    document, _layout = adapter.scroll_spec_document(
        PHERC0172, {"lasagna_volume_name": "PHerc0172_{array}.ome.zarr",
                    "tracks_file": "other.dbm"})
    assert set(document["paths"]) <= set(adapter.SCROLL_SPEC_PATH_OVERRIDE_KEYS)


# -- writing it, and reading it back ------------------------------------------

def test_write_scroll_spec_writes_valid_json_that_reads_back(tmp_path):
    destination = tmp_path / "run" / "spiral-scroll.json"
    written = adapter.write_scroll_spec(PHERC0172, None, destination)

    assert destination.is_file()
    on_disk = json.loads(destination.read_text(encoding="utf-8"))
    assert on_disk == written["document"]
    assert on_disk["spiral_outward_sense"] == "ACW"
    assert len(written["sha256"]) == 64


def test_write_scroll_spec_leaves_no_temp_file_behind(tmp_path):
    destination = tmp_path / "spiral-scroll.json"
    adapter.write_scroll_spec(PHERC0172, None, destination)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["spiral-scroll.json"]


def test_write_scroll_spec_records_whether_the_layout_is_upstreams():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        destination = Path(tmp) / "spiral-scroll.json"
        default_layout = adapter.write_scroll_spec(PHERC0172, None, destination)
        assert default_layout["layout_is_upstream_default"] is True

        destination2 = Path(tmp) / "moved" / "spiral-scroll.json"
        moved_layout = adapter.write_scroll_spec(
            PHERC0172, {"normal_zarr_group": "3"}, destination2)
        assert moved_layout["layout_is_upstream_default"] is False


def test_a_readback_mismatch_is_refused(tmp_path, monkeypatch):
    """The one failure mode a write-then-trust approach cannot catch: a write
    that landed wrong. Simulated by corrupting the file between the write and
    the read-back this function performs."""
    destination = tmp_path / "spiral-scroll.json"
    real_read_text = Path.read_text

    def corrupting_read_text(self, *args, **kwargs):
        text = real_read_text(self, *args, **kwargs)
        if self == destination:
            document = json.loads(text)
            document["name"] = "not-what-was-asked-for"
            return json.dumps(document)
        return text

    monkeypatch.setattr(Path, "read_text", corrupting_read_text)
    with pytest.raises(adapter.ScrollSpecRefused, match="did not read back"):
        adapter.write_scroll_spec(PHERC0172, None, destination)
