"""One place says what version this is.

There was no such place. HELENA_VERSION defaulted to 0.10.0 in seven compose
files, containers/build-images.sh documented the local tag as helena-panel:0.11.0,
and the README promised Semantic Versioning without saying where to read the
number. Two of those drifted apart and nothing noticed, because nothing compared
them.

Deliberately not covered here:

  * Component image tags -- helena-vc3d:0.3.2, helena-surface-qc:0.1.1. Those are
    pulled from a registry by tag. Renaming them to match the platform names an
    image that does not exist, and the deploy fails at the pull.
  * Scientific profile versions. They are immutable identities that receipts
    already quote; renaming one to look tidy falsifies provenance, which is the
    thing this project exists to prevent.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = (ROOT / "VERSION").read_text().strip()


def test_the_version_file_is_a_version() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", VERSION), (
        f"VERSION holds {VERSION!r}, which is not a semantic version"
    )


def test_every_compose_default_matches_it() -> None:
    """The default is what an unconfigured host runs -- the installer's case."""
    for compose in sorted((ROOT / "containers/compose").glob("*.yaml")):
        for found in re.findall(r"HELENA_VERSION:-([0-9][^}\s]*)", compose.read_text()):
            assert found == VERSION, (
                f"{compose.name} defaults HELENA_VERSION to {found}, and VERSION "
                f"says {VERSION}"
            )


def test_the_build_script_agrees() -> None:
    """It names the local tag in its own documentation, and that drifted once."""
    script = (ROOT / "containers/build-images.sh").read_text()
    for found in re.findall(r"helena-panel:([0-9]+\.[0-9]+\.[0-9]+)", script):
        assert found == VERSION, (
            f"build-images.sh names helena-panel:{found}; VERSION says {VERSION}"
        )
