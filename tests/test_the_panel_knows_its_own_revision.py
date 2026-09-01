"""The panel has to be able to say which revision it is, from inside itself.

`_deployed_revision()` reads CX_DEPLOYED_REVISION and otherwise shells out to
`git rev-parse HEAD`. The panel image has no git and no such variable, so on the
deployment the call raised FileNotFoundError, which FastAPI turned into a bare
HTTP 500. Creating any First Letters campaign mission -- including the control
mission whose receipt gates the whole campaign -- failed that way, and the
message said nothing about revisions.

Two separate defects, and the second is the one that matters:

* the crash. A missing binary is an expected condition in a container, and it
  has to become the refusal this function already knows how to give, not a 500.
* the gap. The image carries its revision in an OCI label, which a process
  cannot read about itself. It was never put anywhere the code could see, so
  there was nothing for the function to find even when it worked.

The same shape was fixed in CI months earlier -- HELENA_QC_CODE_COMMIT exists
because the QC adapter took the commit from a Git checkout and the job container
refused it. That fix did not reach here.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONTAINERFILE = (ROOT / "containers/images/Containerfile.panel").read_text()
CI = (ROOT / ".gitlab-ci.yml").read_text()


def test_the_image_carries_its_revision_where_the_process_can_read_it() -> None:
    """A label is for whoever inspects the image; ENV is for the code inside."""
    assert "CX_DEPLOYED_REVISION" in CONTAINERFILE, (
        "the panel image records its revision only in an OCI label, which the "
        "running process cannot read about itself"
    )
    # And it must be the full hash: _deployed_revision requires 40 hex, while
    # BUILD_COMMIT is the short sha because the label is for humans.
    assert re.search(r"ARG\s+BUILD_REVISION", CONTAINERFILE), (
        "there is no full-length revision build argument"
    )
    assert re.search(r"ENV\s+CX_DEPLOYED_REVISION=\$\{BUILD_REVISION\}", CONTAINERFILE)


def test_the_pipeline_passes_the_full_hash() -> None:
    build = CI[CI.index("build the panel image:"):]
    build = build[: build.index("\n# ---")]
    assert "--build-arg BUILD_REVISION=$CI_COMMIT_SHA" in build, (
        "the image is built without the full revision, so the panel inside it "
        "cannot name the commit the deploy asked for"
    )


def test_a_missing_git_is_a_refusal_and_not_a_crash(monkeypatch) -> None:
    """The condition is expected in a container, so it gets the same 409 the
    function already gives for an unusable answer."""
    from fastapi import HTTPException

    import panel.app as app

    monkeypatch.delenv("CX_DEPLOYED_REVISION", raising=False)

    def absent(*_args, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'git'")

    monkeypatch.setattr(app.subprocess, "run", absent)

    with pytest.raises(HTTPException) as raised:
        app._deployed_revision()
    assert raised.value.status_code == 409
    assert "revision" in str(raised.value.detail).lower()


def test_the_declared_revision_is_used_when_present(monkeypatch) -> None:
    import panel.app as app

    monkeypatch.setenv("CX_DEPLOYED_REVISION", "a" * 40)
    assert app._deployed_revision() == "a" * 40


def test_a_malformed_declared_revision_is_refused(monkeypatch) -> None:
    """Short hashes are the trap: BUILD_COMMIT is one, and a caller that let it
    through would bind evidence to an ambiguous prefix."""
    from fastapi import HTTPException

    import panel.app as app

    for bad in ("62446c6a", "", "z" * 40, "A" * 40):
        monkeypatch.setenv("CX_DEPLOYED_REVISION", bad)
        if not bad:
            continue
        with pytest.raises(HTTPException) as raised:
            app._deployed_revision()
        assert raised.value.status_code == 409
