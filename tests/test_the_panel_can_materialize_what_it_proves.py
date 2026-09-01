"""The panel must be able to fetch the surface it is asked to prove.

The control reached P4 and the geometry orientation proof refused:

    409 S3 surface materialization requires boto3

The panel image ships s3fs and, through it, botocore -- but not boto3. Four
places in this repository gate real work behind `import boto3`: surface
materialization and QC evidence publication in stage 04, artifact storage in
stage 01, and the review queue in stage 06. The workers carry it. The panel did
not, and the orientation proof is the first panel request path that reaches one
of them.

This does not contradict the note beside s3fs in panel/requirements.txt. That
note explains reading a TIFXYZ through fsspec so one open() handles s3:// and
local paths alike, and it is still right for reading. Materializing a published
artifact set is a different job, and the whole repository does it with boto3 --
deliberately, because s3transfer retries a dropped download and a bare fsspec
read does not.

Kept as a test rather than a comment because the failure it prevents costs a
two-hour run to observe: the panel starts, serves every other route, and refuses
only at the one boundary that needs the dependency.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "panel/requirements.txt"


def _declared() -> set[str]:
    names = set()
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(re.split(r"[><=\[]", line, maxsplit=1)[0].strip().lower())
    return names


def test_the_panel_declares_boto3() -> None:
    assert "boto3" in _declared(), (
        "the orientation proof materializes an S3 surface through code that "
        "imports boto3; without it the panel serves everything except the "
        "boundary that needs it")


def test_the_panel_still_declares_the_fsspec_reader() -> None:
    """Adding boto3 must not become a reason to drop s3fs: reading a TIFXYZ
    through one open() that handles s3:// and local paths alike is a separate,
    still-correct decision."""
    declared = _declared()
    assert "s3fs" in declared and "fsspec" in declared


def test_the_gated_call_sites_still_name_boto3() -> None:
    """If these ever stop needing boto3, this test should be the thing that
    notices -- not a dependency nobody can explain."""
    gated = [
        "framework/stages/04-validation/scripts/campaignx_surface_qc_adapter.py",
        "framework/stages/01-segmentation/fleet/artifact_store.py",
        "framework/stages/06-discovery/scripts/helena_build_first_letters_review_queue.py",
    ]
    still_gated = [path for path in gated
                   if "requires boto3" in (ROOT / path).read_text(encoding="utf-8")]
    assert still_gated, (
        "nothing gates work behind boto3 any more; the panel dependency can go")
