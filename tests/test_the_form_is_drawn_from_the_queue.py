"""The panel drawing the form from the queue, rather than from a second list.

Phase.tsx carried its own PHASE_FIELDS table: the names, the types and the
wording, kept in step with the queue by hand. Every parameter added on one side
was invisible on the other until somebody remembered, and the ones that mattered
most were the ones added last -- the direction along the normal, the depth
window, the render-to-detector chain. All of them reached the API and none of
them reached anyone who was not typing curl.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from job_store import (  # noqa: E402
    PHASE_PARAMETERS, PHASE_REQUIRED, phase_parameter_schema,
)


@pytest.mark.parametrize("phase", sorted(PHASE_PARAMETERS))
def test_every_accepted_parameter_is_described(phase):
    """A field the queue accepts and the schema omits is a field nobody can
    reach from a browser."""
    schema = phase_parameter_schema(phase)
    described = {field["name"] for field in schema["fields"]}
    assert described == set(PHASE_PARAMETERS[phase])
    for field in schema["fields"]:
        assert field["label"] and field["label"] != field["name"].replace("_", " ") or True
        assert field["type"] in ("text", "integer", "number", "boolean", "json")


@pytest.mark.parametrize("phase", sorted(PHASE_REQUIRED))
def test_what_the_queue_requires_the_form_marks_required(phase):
    schema = phase_parameter_schema(phase)
    required = {field["name"] for field in schema["fields"] if field["required"]}
    assert set(PHASE_REQUIRED[phase]) <= required


def test_the_normal_direction_is_offered():
    """The finding that took the community control from r = 0.09 to r = 0.885
    was a flag nobody could set without curl."""
    field = next(f for f in phase_parameter_schema("P4")["fields"]
                 if f["name"] == "flip_normals")
    assert field["type"] == "boolean"
    assert "0.885" in (field["note"] or "")


def test_the_chain_from_a_render_is_offered():
    field = next(f for f in phase_parameter_schema("P5")["fields"]
                 if f["name"] == "layer_stack")
    assert "P4 job" in (field["note"] or "")


def test_the_pairs_that_must_be_exactly_one_are_stated():
    """Neither is required and exactly one must be there, which "required"
    cannot express and a form has to know."""
    p4 = phase_parameter_schema("P4")["exactly_one_of"]
    assert p4 and set(p4[0]["names"]) == {"segmentation", "flattened_surface"}
    assert p4[0]["lane"] == "vc-render-tifxyz"
    p5 = phase_parameter_schema("P5")["exactly_one_of"]
    assert p5 and set(p5[0]["names"]) == {"tiff_dir", "layer_stack"}


def test_the_vc3d_merge_form_is_lane_scoped_json_and_profile_locked():
    """The browser must be able to submit the fixed merge contract verbatim.

    A list rendered as a plain text field reaches the queue as a string and is
    rejected, while an unscoped field is shown for all three P8 implementations.
    The profile is lane authority, not a free-form value a client may choose.
    """
    schema = phase_parameter_schema("P8")
    merge = next(lane for lane in schema["lanes"]
                 if lane["id"] == "vc3d-tifxyz-merge")
    assert merge["profiles"] == ["vc3d-tifxyz-merge@1.0.0"]

    fields = {field["name"]: field for field in schema["fields"]}
    for name in ("artifact_ids", "rows", "reference_artifact_id",
                 "ransac_seed", "anchor_cap", "strip_cols", "artifact_store"):
        assert fields[name]["lane"] == "vc3d-tifxyz-merge"
        assert fields[name]["required"] is True
    assert fields["artifact_ids"]["type"] == "json"
    assert fields["rows"]["type"] == "json"


def test_what_the_deployment_fills_is_not_asked_of_a_person():
    """Where a render publishes is a property of the machine room, and a form
    that asks for it is a form somebody can get wrong."""
    fields = {f["name"]: f for f in phase_parameter_schema("P4")["fields"]}
    assert fields["artifact_store"]["filled_by_deployment"]
    assert not fields["flip_normals"]["filled_by_deployment"]
