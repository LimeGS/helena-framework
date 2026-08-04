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


def test_the_coverage_badge_is_a_number_ci_enforces() -> None:
    """A badge nobody enforces goes green while the thing it claims falls.

    There is no coverage service and no token here, and the workflow does not
    commit anything -- the public mirror is one squashed commit per release, so
    a job writing a percentage into the repository would fight that. What the
    badge says instead is a floor, and --cov-fail-under is what makes it true:
    the run goes red before the badge can lie.
    """
    import re

    readme = (ROOT / "README.md").read_text()
    claimed = re.search(r"coverage-%E2%89%A5(\d+)%25", readme)
    if not claimed:
        return  # no badge, nothing to keep honest

    workflow = (ROOT / ".github/workflows/audit.yml").read_text()
    enforced = re.search(r"--cov-fail-under=(\d+)", workflow)
    assert enforced, (
        f"the README claims coverage of at least {claimed.group(1)}% and nothing "
        "in the workflow enforces it"
    )
    assert int(enforced.group(1)) >= int(claimed.group(1)), (
        f"the badge claims ≥{claimed.group(1)}% and CI only requires "
        f"{enforced.group(1)}%"
    )
    assert "pytest-cov" in (ROOT / "tests/requirements.txt").read_text(), (
        "the workflow measures coverage with a plugin it does not install"
    )


def test_the_workflow_gives_the_database_tests_a_database() -> None:
    """Eighty-five tests skipped for want of one, and they cover the queue.

    postgres_store.py -- leases, SKIP LOCKED, claim expiry, which is the
    README's headline guarantee -- sat at 11%. Not because nobody tested it:
    thirteen test files exercise it and every one skipped, because CI had no
    Postgres. The coverage was written. It never ran.
    """
    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/audit.yml").read_text())
    job = next(iter(workflow["jobs"].values()))
    assert "postgres" in job.get("services", {}), (
        "the workflow has no database, so the tests that need one still skip"
    )

    steps = " ".join(str(s) for s in job["steps"])
    for variable in ("HELENA_TEST_DSN", "SEGMENT_FLEET_DATABASE_URL"):
        assert variable in steps, f"{variable} is never set, so those tests skip anyway"

    assert "psycopg2" in (ROOT / "tests/requirements.txt").read_text(), (
        "a Postgres the tests cannot reach: the driver is not installed, which "
        "surfaces as 'PostgreSQL mode requires psycopg2' rather than as a skip"
    )
