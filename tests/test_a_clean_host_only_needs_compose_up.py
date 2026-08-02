"""A clean machine, a checkout, `docker compose up`. Nothing else.

This is a property, not a preference, and it decays the moment somebody adds a
service with one required variable -- because the person who adds it has that
variable set, so it works for them and for nobody else. That is how the file got
to five of them: HELENA_POSTGRES_ENV, HELENA_POSTGRES_DATA, HELENA_PANEL_STATE
and HELENA_BACKUP_S3 all read as obviously-necessary to whoever added them, and
together they meant a second host could not be brought up without reconstructing
four files from a shell somebody had closed.

Three things are checked, each of which broke a real deployment:

  * no variable is required, including in a service behind a profile -- compose
    interpolates the whole file before it applies profiles, so a `:?` in the
    backup service made a bare `up` fail on a service it would never start;
  * nothing defaults to a host path, because a host path has to be created,
    owned and sized by somebody first;
  * the schema initialises before the panel, because nothing else creates it.

Static. The point is what the file promises somebody who has not run it yet.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
PLATFORM = ROOT / "containers/compose/platform.compose.yaml"
TEXT = PLATFORM.read_text()
COMPOSE = yaml.safe_load(TEXT)


def _mount_source(mount: str) -> str:
    """The source half of `source:target[:mode]`.

    Not `mount.split(":")[0]`: `${HELENA_PANEL_RUNS:-/ssd/campaignx/runs}` has
    two colons of its own, so splitting naively returned `${HELENA_PANEL_RUNS`
    and this test passed on exactly the host paths it exists to reject.
    """
    if mount.startswith("${"):
        close = mount.index("}") + 1
        return mount[:close]
    return mount.split(":")[0]


def test_no_variable_is_required_to_bring_the_platform_up() -> None:
    """`${VAR:?message}` is a refusal to start. There must be none.

    Including inside `backup`, which is behind a profile: compose interpolates
    every service before it decides which ones to run, so a required variable
    there is a required variable everywhere.
    """
    required = re.findall(r"\$\{([A-Z_]+):\?", TEXT)
    assert not required, (
        f"{sorted(set(required))} must be set before this file will parse, so "
        "`docker compose up` on a clean host fails before it starts anything"
    )


def test_nothing_is_stored_on_a_path_the_host_has_to_provide() -> None:
    """Every default is a named volume.

    A host path has to exist, be owned by the right user and be on a disk with
    room, and none of that is true of a machine somebody just cloned onto. Two
    of these used to default to /ssd/campaignx, which exists on one machine.

    Bind mounts are still allowed -- the variables take a path -- but the
    default cannot be one.
    """
    for name, service in COMPOSE["services"].items():
        for mount in service.get("volumes") or []:
            source = _mount_source(str(mount))
            default = re.match(r"\$\{[A-Z_]+:-([^}]*)\}$", source)
            if default is None:
                # A literal source. Only a named volume is acceptable.
                assert not source.startswith(("/", ".", "~")), (
                    f"{name} mounts the host path {source}"
                )
                continue
            assert not default.group(1).startswith(("/", ".", "~")), (
                f"{name} mounts {source}, which defaults to a host path that "
                "has to exist before the platform will work"
            )


def test_the_images_can_be_built_from_the_checkout() -> None:
    """Otherwise `up` needs an image from a registry nobody has credentials for.

    Every service that runs first-party code declares a build context, so the
    image is either already present or made from the repository in front of you.
    """
    for name in ("panel", "init", "backup"):
        service = COMPOSE["services"][name]
        assert service.get("build"), f"{name} can only run a pre-built image"
        dockerfile = ROOT / "containers/compose" / service["build"]["context"] \
            / service["build"]["dockerfile"]
        assert dockerfile.resolve().exists(), f"{name} names a missing {dockerfile}"


def test_the_panel_image_builds_its_own_frontend() -> None:
    """panel/web/dist is gitignored.

    So a Containerfile that COPYs it can only be built on a machine that has
    node and has remembered to run npm -- which is not a clean checkout, and was
    a hard build failure on any host that had never built the panel before.
    """
    containerfile = (ROOT / "containers/images/Containerfile.panel").read_text()
    assert "COPY panel/web/dist" not in containerfile, (
        "the image copies a build output that is not in the repository"
    )
    assert re.search(r"FROM \$\{NODE_IMAGE[^}]*\} AS web", containerfile), (
        "there is no stage that builds the frontend"
    )
    assert "npm run build" in containerfile
    assert "COPY --from=web" in containerfile

    ignored = (ROOT / "panel/web/.gitignore").read_text()
    assert "dist/" in ignored, (
        "dist is committed now, so this test is guarding a rule that changed"
    )
    # npm ci needs it, and it is the only thing that makes the build repeatable.
    assert (ROOT / "panel/web/package-lock.json").exists()


def test_the_schema_exists_before_the_panel_reads_it() -> None:
    """Nothing else creates it.

    `initialize()` is called by the fleet CLI and by the worker. A first
    deployment has neither, so the panel came up against an empty database and
    every page reported no data without saying why.
    """
    init = COMPOSE["services"]["init"]
    assert init["depends_on"]["postgres"]["condition"] == "service_healthy"

    command = " ".join(str(part) for part in init["command"])
    assert "open_fleet_store" in command and "InkJobStore" in command, (
        "the init step does not create both schemas; the panel reads both"
    )
    assert command.count("initialize()") == 2

    panel = COMPOSE["services"]["panel"]
    assert panel["depends_on"]["init"]["condition"] == "service_completed_successfully", (
        "the panel does not wait for the schema, so a first `up` races it"
    )


def test_the_panel_and_the_initialiser_are_the_same_build() -> None:
    """A schema created by one build and read by another is how a migration
    lands in a database that a different version of the code then queries."""
    services = COMPOSE["services"]
    assert services["init"]["image"] == services["panel"]["image"]


def test_the_credentials_are_stated_once() -> None:
    """They were in two files, and a password changed in one produced a panel
    that started cleanly and answered every page with "no control plane"."""
    services = COMPOSE["services"]
    dsn = services["panel"]["environment"]["CX_DB"]
    assert services["init"]["environment"]["CX_DB"] == dsn
    for part in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
        assert part in dsn, f"the connection string does not derive {part}"
    # And the same defaults on both sides, or the panel authenticates with a
    # password the database was never created with.
    for key, default in (("POSTGRES_USER", "campaignx"), ("POSTGRES_DB", "campaignx")):
        assert f"${{{key}:-{default}}}" in dsn
        assert services["postgres"]["environment"][key] == f"${{{key}:-{default}}}"


def test_an_optional_env_file_is_marked_optional() -> None:
    """A missing env_file is a hard error unless it says otherwise, and these
    all point at /etc/helena, which a clean host does not have."""
    for name, service in COMPOSE["services"].items():
        for entry in service.get("env_file") or []:
            assert isinstance(entry, dict), (
                f"{name} names {entry} as a plain string, so the file has to "
                "exist or compose refuses to start"
            )
            assert entry["required"] is False


def test_the_database_is_not_reachable_from_off_the_host() -> None:
    """The default password is defensible only because of this line.

    It ships in the repository, so if the port were ever published on 0.0.0.0
    the control plane would be open to anybody who read the file.
    """
    for published in COMPOSE["services"]["postgres"]["ports"]:
        assert str(published).startswith("127.0.0.1:"), (
            f"postgres is published as {published}, and the password is public"
        )
