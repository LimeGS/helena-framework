"""The README is the first thing a stranger reads, so it has to be true.

It is short on purpose -- the panel carries a tutorial, a user guide, a
developer reference and an API reference, all generated partly from the same
contracts the code runs on. What stays here is what someone needs before they
have a deployment to look at: what this is, how it works, and how to start it.

Everything it asserts is checked below. A README that has drifted is worse than
a short one, because it is the file people trust without being able to verify.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()


def test_every_path_it_names_exists() -> None:
    """A layout section that lists a directory nobody has is a reader's first
    impression that this project does not build."""
    for path in re.findall(r"`(framework/[\w./-]+|panel/[\w./-]+|containers/[\w./-]+|tests/[\w./-]+)`", README):
        assert (ROOT / path.rstrip("/")).exists(), f"the README names {path}, which is not here"
    # The layout block is plain text rather than backticked.
    layout = README[README.index("## Repository layout"):]
    layout = layout[layout.index("```") + 3: layout.index("```", layout.index("```") + 3)]
    for line in layout.strip().splitlines():
        directory = line.split()[0]
        assert (ROOT / directory).exists(), f"the layout names {directory}, which is not here"


def test_the_phase_table_matches_the_contract() -> None:
    """Ten phases, in order, named as the contract names them.

    The panel, the queue and the docs are all generated from that file. A README
    that renumbers them sends somebody to the wrong phase on their first day --
    which is exactly what an earlier draft of the in-app tutorial did.
    """
    contract = json.loads((ROOT / "framework/contracts/pipeline_phases.json").read_text())
    phases = contract if isinstance(contract, list) else contract["phases"]

    rows = re.findall(r"\|\s*\*\*(P\d)\*\*\s*\|\s*([^|]+?)\s*\|", README)
    assert [p for p, _ in rows] == [p["id"] for p in phases], (
        "the README's phase list does not match the contract"
    )
    for (phase_id, name), contracted in zip(rows, phases):
        assert name.lower() == contracted["name"].lower(), (
            f"{phase_id} is {contracted['name']!r} in the contract and {name!r} here"
        )


def test_the_commands_it_gives_are_real() -> None:
    """Somebody pastes these. They have to exist and take those arguments."""
    assert (ROOT / "containers/compose/platform.compose.yaml").is_file()

    deploy = (ROOT / "containers/deploy-platform.sh").read_text()
    for profile in ("nogpu", "gpu"):
        assert f"deploy-platform.sh {profile}" in README or f"deploy-platform.sh gpu|nogpu" in deploy
    # Both profiles are things the script actually accepts.
    assert "gpu|nogpu)" in deploy, "the deploy script no longer takes these profiles"

    # The quick start really is configuration-free: a compose file with a
    # required variable would stop at the first `${VAR:?}`.
    platform = (ROOT / "containers/compose/platform.compose.yaml").read_text()
    required = re.findall(r"\$\{([A-Z_]+):\?", platform)
    assert not required, (
        f"the README promises `docker compose up` with no configuration, but "
        f"platform.compose.yaml requires {required}"
    )


def test_it_does_not_publish_anything_private() -> None:
    """This repository is meant to be public.

    Internal addresses and host names are not secrets, but they are noise to a
    reader and a map to anyone else. Credentials are neither.
    """
    hazards = {
        "an internal IP": r"\b10\.\d+\.\d+\.\d+\b",
        "an AWS key": r"\bAKIA[A-Z0-9]{16}\b",
        "a runner token": r"\bglrt-[A-Za-z0-9_-]{10,}",
        "a private host": r"\b(work-3|gpu-1)\b",
        "an internal registry": r"registry\.\w+\.com",
    }
    for what, pattern in hazards.items():
        found = re.findall(pattern, README)
        assert not found, f"the README contains {what}: {found[:2]}"


def test_it_stays_short_enough_to_read() -> None:
    """It was 1240 lines, most of it duplicating what the panel now serves from
    the contracts themselves. Length is the failure mode here: nobody reads a
    README that is a manual, so the manual and the introduction both go unread."""
    lines = README.strip().splitlines()
    # 260, then 250, now 270, and each move happened the same way: the number
    # was hit while adding something a reader needs, and shaving prose to fit is
    # optimising for the threshold rather than for the reader. This time it was
    # a month of platform -- lanes, the installer's question, and a contributing
    # command that does not silently skip eighty tests.
    #
    # What this guards is the return to a 1240-line manual. Anything in that
    # region is unambiguously that; 250 versus 270 is not. Raise it again for
    # content that earns it, and record why here, so the next person can see
    # whether the moves were reasonable or whether the limit stopped meaning
    # anything.
    assert len(lines) < 270, (
        f"the README is {len(lines)} lines. The tutorial, user guide, developer "
        "reference and API reference all live in the panel -- if something needs "
        "saying at length, it probably belongs there."
    )


def test_it_points_at_the_documentation_that_replaced_it() -> None:
    """The reason it can stay short."""
    for section in ("Tutorial", "User guide", "Developer reference", "API reference"):
        assert section in README, f"the README does not mention the {section}"


def test_the_opening_claims_are_implemented() -> None:
    """The first screen is where a reader decides whether to trust the rest.

    It names specific tools and specific behaviour, to an audience that can
    check both.
    """
    opening = README[: README.index("## The pipeline")]

    if "ScrollFiesta" in opening:
        assert (ROOT / "containers/images/scrollfiesta").exists(), (
            "the README claims ScrollFiesta is integrated; nothing here builds it"
        )
    if "Volume Cartographer" in opening:
        assert (ROOT / "vendor/villa/volume-cartographer").exists() or list(
            (ROOT / "framework/vendored").glob("*")), (
            "the README claims the Volume Cartographer flatteners; none are vendored"
        )
    if "repository@sha256" in opening:
        build = (ROOT / "containers/build-worker.sh").read_text()
        assert "exit 2" in build and "@sha256:" in build, (
            "the README says a build refuses without a resolved digest; it does not"
        )
    if "does not hand the same row to two workers" in opening:
        claims = [p.read_text() for p in (ROOT / "framework/stages").rglob("*.py")]
        assert any("SKIP LOCKED" in c for c in claims), (
            "the README promises no double-claim and no query takes the lock that "
            "provides it"
        )
    if "four distinct judgements" in opening:
        # geometry, CT support, model response, human review
        certification = (ROOT / "panel/web/src/routes/Certification.tsx").read_text()
        assert "CT_SUPPORTED" in certification, (
            "the README separates CT support from geometry; the panel does not"
        )


def test_the_guarantees_it_advertises_are_implemented() -> None:
    """The README now makes specific claims to an audience that will check them.

    Each is anchored to the thing that implements it. The phase numbers in
    particular: an earlier draft credited P7 with rejecting a missing liveness
    verdict, and it is P5 that does it to itself.
    """
    guarantees = README[README.index("## How that is enforced"):README.index("## Deploying")]

    if "SHA-256" in guarantees:
        profiles = list((ROOT / "framework/profiles").rglob("*.json"))
        pinned = [p for p in profiles if "checkpoint_sha256" in p.read_text()]
        assert pinned, "the README claims profiles pin checkpoints by hash; none do"

    if "ALIVE, DEGENERATE, EMPTY" in guarantees:
        liveness = (ROOT / "framework/contracts/lane_liveness.py").read_text()
        for verdict in ("ALIVE", "DEGENERATE", "EMPTY"):
            assert verdict in liveness, f"{verdict} is not a verdict this code produces"

    if "P5 job that finishes without one" in guarantees:
        worker = (ROOT / "framework/stages/03-ink/fleet/ink_worker.py").read_text()
        assert 'job.get("phase") == "P5"' in worker, (
            "the README says a P5 job without liveness is failed; nothing does that"
        )

    if "SKIP LOCKED" in guarantees:
        claims = [p.read_text() for p in (ROOT / "framework/stages").rglob("*.py")]
        assert any("SKIP LOCKED" in c for c in claims), (
            "the README describes the claim as FOR UPDATE SKIP LOCKED and no query is"
        )

    if "image IDs rather than tag strings" in guarantees:
        deploy = (ROOT / "containers/deploy-platform.sh").read_text()
        assert "{{.Image}}" in deploy and "{{.Id}}" in deploy, (
            "the README says the deploy compares image IDs; it compares names"
        )


def test_it_does_not_promise_bit_identical_images() -> None:
    """Building per host is fine; claiming byte-equality for it is not.

    Docker builds are not reproducible by default, and this repository has the
    evidence: one commit built twice produced two different image IDs, and the
    second push re-pointed the tag under a container still running the first.

    What survives building per host is everything a result is attributed to --
    the base resolved to a digest, the checkpoint, the profile, the artifact,
    the commit -- because those are content-addressed. The README has to say
    which of the two it means.
    """
    opening = README[: README.index("## The pipeline")]
    if "digest-pinned containers" in opening:
        raise AssertionError(
            "'digest-pinned containers' reads as bit-identical images, which is "
            "not what building from a checkout gives you. The base image is "
            "digest-pinned; the wrapper layers are not."
        )
    if "builds from the checkout" in opening or "needs a published image" in opening:
        assert "not \"same bytes\"" in opening or "not reproducible by default" in opening, (
            "the README says images are built locally without saying what that "
            "costs, which is the first question a reader asks"
        )

    # And the base really is pinned, or the paragraph is describing nothing.
    build = (ROOT / "containers/build-worker.sh").read_text()
    assert "repository@sha256" in build or "@sha256:" in build
    assert "SOURCE_DATE_EPOCH" in (ROOT / "containers/images/Containerfile.worker-cpp").read_text()


def test_the_bootstrap_command_is_the_one_the_panel_answers() -> None:
    """Somebody pastes this on a fresh host with no other way in.

    A wrong endpoint, a wrong field name or a password under the minimum all
    fail at the one moment the reader has nothing else to try.
    """
    app = (ROOT / "panel/app.py").read_text()
    deploying = README[README.index("## Deploying"):]

    endpoint = re.search(r"https://localhost:8800(/api/[\w/-]+)", deploying)
    assert endpoint, "the README gives no bootstrap request"
    assert f'@app.post("{endpoint.group(1)}")' in app, (
        f"the README posts to {endpoint.group(1)}, which the panel does not serve"
    )

    body = re.search(r"-d '(\{.*?\})'", deploying)
    assert body, "the bootstrap command sends no body"
    import json
    fields = json.loads(body.group(1))
    assert set(fields) == {"username", "password"}, (
        f"the panel takes username and password; the README sends {sorted(fields)}"
    )

    from framework.contracts import auth
    assert len(fields["password"]) >= auth.MINIMUM_PASSWORD, (
        f"the example password is shorter than the {auth.MINIMUM_PASSWORD} "
        "characters the panel requires, so pasting it returns 422"
    )
    # Self-signed on first boot, so a plain curl fails on the certificate.
    assert "-sk" in deploying or "-k" in deploying, (
        "the certificate is self-signed and the command does not allow for it"
    )


def test_the_screenshots_it_shows_are_in_the_repository() -> None:
    """A README whose images 404 on GitHub looks abandoned before it is read."""
    for src in re.findall(r'src="([^"]+)"', README):
        # Badges are served by GitHub and shields.io and are supposed to be
        # remote; what must exist locally is everything this repository ships.
        if src.startswith("http"):
            continue
        assert (ROOT / src).is_file(), f"the README shows {src}, which is not here"

    # And each says what it is, for anyone who cannot see it.
    for tag in re.findall(r"<img [^>]+>", README, re.S):
        assert 'alt="' in tag, f"an image has no alt text: {tag[:70]}"


def test_the_worker_section_promises_only_what_the_deploy_does() -> None:
    """The README now says the two commands are the whole of it.

    That is a claim about a chain, not a file: deploy-platform.sh has to invoke
    build-worker.sh, and build-worker.sh has to build the villa base rather than
    require one. It was untrue until both were, and it is the kind of sentence
    that stays on a page long after the behaviour under it has moved.

    Verified end to end on a clean host once: install, then
    `deploy-platform.sh nogpu`, which built helena-villa from the pinned commit,
    built the worker on top and started the stack with no registry present.
    """
    workers = README[README.index("### Workers"):]
    if "the deploy builds the worker images" not in workers:
        return                       # the claim was withdrawn; nothing to hold

    deploy = (ROOT / "containers/deploy-platform.sh").read_text()
    assert "build-worker.sh" in deploy, (
        "the README says the deploy builds the workers, and it does not call "
        "the script that would")

    build = (ROOT / "containers/build-worker.sh").read_text()
    assert "Containerfile.villa" in build, (
        "the deploy would stop at the villa base, which the README says it "
        "builds -- that was true only after build-worker.sh learned to")

    if "No registry needed" in workers:
        assert "could not pull it; building" in deploy, (
            "the README promises the deploy works without a registry; the "
            "deploy has no fallback for a pull that fails")
