"""A worker is ephemeral; what it produces must not be.

A surface published to a directory on the worker carries that path as its
artifact_uri, and every phase downstream resolves it against its own
filesystem. Surface QC on another host requeues it forever, P3 cannot fetch it,
and when the host goes away so does the only copy.

This happened: four surfaces grown on one machine had to be copied to another by
hand before QC could measure them. The check exists so the next one fails at
startup, where it is one message, instead of five minutes into a phase nobody is
watching.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.cli import refuse_stranded_artifacts  # noqa: E402


def test_object_storage_is_accepted():
    refuse_stranded_artifacts("s3://bucket/surfaces", allowed=False)


def test_a_local_directory_is_refused_by_default():
    with pytest.raises(RuntimeError) as refused:
        refuse_stranded_artifacts("/artifacts/surfaces", allowed=False)
    assert "ephemeral" in str(refused.value)
    # The message has to name the way out, or it is a wall.
    assert "--allow-local-artifacts" in str(refused.value)


def test_a_single_machine_run_can_still_ask_for_it():
    refuse_stranded_artifacts("/artifacts/surfaces", allowed=True)


def test_the_refusal_names_the_path_it_refused():
    with pytest.raises(RuntimeError) as refused:
        refuse_stranded_artifacts("/mnt/campaignx/artifacts/surfaces", allowed=False)
    assert "/mnt/campaignx/artifacts/surfaces" in str(refused.value)
