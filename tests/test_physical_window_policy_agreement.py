"""The benchmark path and the production path must resolve one half-window.

`ct-fiber-supported-window-router@4.1.0` declares an adaptive depth-window
policy.  Only the benchmark executor honoured it; the production extractor read
`half_window_um` directly and ignored both `window_policy` and
`minimum_supported_half_window_um`.  The observable consequence was that
`MULTISCROLL_TRANSFER_V2`/`V3` measured `physical_window_half_width_um = 72.0`
on all 300 controls while the production run measured 120.0 — so the benchmark
that accepted v4.1 never exercised the code path production used.

These tests pin the two consumers to one resolver and to the frozen profile.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "framework/stages/04-validation/scripts"
sys.path.insert(0, str(SCRIPTS))

from extract_ct_fiber_features_physical import (  # noqa: E402
    ADAPTIVE_WINDOW_POLICY,
    resolve_half_window_um,
)

PROFILE = (
    ROOT
    / "framework/profiles/validation/ct-fiber-supported-window-router-v4.1.json"
)
PRODUCTION = SCRIPTS / "extract_ct_fiber_features_physical.py"
BENCHMARK = SCRIPTS / "helena_execute_multiscroll_gates_once.py"
BENCHMARK_CENTRAL_SLICE = 32


@pytest.fixture(scope="module")
def physical_config() -> dict:
    return json.loads(PROFILE.read_text(encoding="utf-8"))["physical_depth_sampling"]


def test_the_frozen_profile_still_declares_the_adaptive_policy(physical_config) -> None:
    """If any of these change, every number below is about a different policy."""

    assert physical_config["window_policy"] == ADAPTIVE_WINDOW_POLICY
    assert physical_config["half_window_um"] == 120.0
    assert physical_config["minimum_supported_half_window_um"] == 64.0
    assert physical_config["canonical_step_um"] == 8.0


@pytest.mark.parametrize(
    "depth_slices,voxel_um",
    [
        (65, 2.4),  # benchmark controls, canonically resampled
        (65, 8.64),  # production, 8.64 um cohort
        (65, 9.362),  # production, 9.362 um cohort
        (26, 2.4),  # shallow stack: the policy actually binds
        (20, 2.4),  # shallower still: the 64 um floor binds
        (128, 9.362),  # deep stack: the declared 120 um caps it
    ],
)
def test_both_consumers_resolve_the_same_half_window(
    physical_config, depth_slices: int, voxel_um: float
) -> None:
    """One resolver, so agreement is structural rather than coincidental.

    The guard against regression is the source check below: this asserts the
    value is well-formed and inside the declared envelope for every shape both
    consumers actually see.
    """

    resolved = resolve_half_window_um(
        physical_config,
        depth_slices=depth_slices,
        central_slice=BENCHMARK_CENTRAL_SLICE,
        voxel_um=voxel_um,
    )
    assert resolved <= physical_config["half_window_um"]
    assert resolved >= physical_config["minimum_supported_half_window_um"]
    assert resolved % physical_config["canonical_step_um"] == 0 or (
        resolved == physical_config["half_window_um"]
    )


def test_neither_consumer_reads_half_window_um_directly() -> None:
    """The divergence was a direct read.  Forbid it in both files.

    A direct `config["half_window_um"]` in either consumer is exactly how the
    production path drifted away from the validated one.
    """

    for path in (PRODUCTION, BENCHMARK):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        direct_reads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "half_window_um"
        ]
        if path is PRODUCTION:
            # The resolver itself legitimately reads the declared value once.
            inside_resolver = source.index("def resolve_half_window_um")
            end_of_resolver = source.index("def extract_physical_depth_features")
            offending = [
                node
                for node in direct_reads
                if not (
                    inside_resolver
                    <= _offset_of(source, node.lineno)
                    < end_of_resolver
                )
            ]
            assert not offending, (
                f"{path.name} reads half_window_um outside the shared resolver "
                f"at lines {[n.lineno for n in offending]}"
            )
        else:
            assert not direct_reads, (
                f"{path.name} must resolve through resolve_half_window_um, "
                f"found direct reads at lines {[n.lineno for n in direct_reads]}"
            )


def _offset_of(source: str, lineno: int) -> int:
    return sum(len(line) + 1 for line in source.splitlines()[: lineno - 1])


def test_the_policy_binds_only_when_support_is_short(physical_config) -> None:
    """A deep stack keeps the declared window; a shallow one is clamped down.

    This is the property v4.1 was created for: its predecessor's fixed 120 um
    window produced coverage 0.6 on all 300 controls and failed transfer V1.
    """

    deep = resolve_half_window_um(
        physical_config,
        depth_slices=200,
        central_slice=BENCHMARK_CENTRAL_SLICE,
        voxel_um=9.362,
    )
    assert deep == physical_config["half_window_um"]

    shallow = resolve_half_window_um(
        physical_config,
        depth_slices=40,
        central_slice=BENCHMARK_CENTRAL_SLICE,
        voxel_um=2.4,
    )
    assert shallow < physical_config["half_window_um"]
    assert shallow >= physical_config["minimum_supported_half_window_um"]


def test_a_non_adaptive_policy_returns_the_declared_value(physical_config) -> None:
    """Without the policy the resolver is a passthrough, as v4.0 behaved."""

    fixed = {**physical_config, "window_policy": "FIXED"}
    assert (
        resolve_half_window_um(
            fixed,
            depth_slices=26,
            central_slice=BENCHMARK_CENTRAL_SLICE,
            voxel_um=2.4,
        )
        == physical_config["half_window_um"]
    )
