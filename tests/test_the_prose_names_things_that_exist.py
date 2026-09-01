"""Comments and docs are checked against the tree, because they drift silently.

A stale comment is not a cosmetic problem. It sends a reader to a file that is
not there, or tells them to run a target that was deleted, and the time they
spend finding that out is the cost. This repository writes long comments on
purpose; the price of that is a check that they still describe the code.

Three kinds of name are cheap to verify and expensive to get wrong: the
Containerfiles a text points at, the `make` targets it tells somebody to run,
and the compose files it names. Each of those either exists or does not.

What this cannot check is whether a comment's *reasoning* is still true -- that
`helena-vc3d` was removed because its only consumer was one build, say. Those
are read by people. This catches the mechanical half, which is the half that
rots fastest.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "containers/images"


# Directories whose contents are not ours to hold to this.
NOT_OURS = {".git", ".venv", "venv", "node_modules", "__pycache__",
            "vendored", ".claude", "dist", "build", "tests", "workspace"}


def texts() -> list[Path]:
    """Prose and scripts in the tree. Not the tests, which name things on purpose.

    Walked off disk rather than asked of `git ls-files`, which exits 128 inside
    the CI container: the checkout belongs to another user there and git calls
    it dubious ownership. That is the second test in this repository to be
    written against git and fail only on the machine whose opinion decides
    whether anything ships.
    """
    out = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        parts = set(path.relative_to(ROOT).parts)
        if NOT_OURS & parts:
            continue
        if path.suffix in {".md", ".sh", ".yaml", ".yml", ".py", ".example"} \
           or path.name.startswith("Containerfile") or path.name == "Makefile":
            out.append(path)
    return out


def test_every_containerfile_named_in_prose_exists() -> None:
    missing: dict[str, list[str]] = {}
    for path in texts():
        try:
            body = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for named in set(re.findall(r"Containerfile\.[a-z0-9][a-z0-9.-]*", body)):
            named = named.rstrip(".")
            if not (IMAGES / named).is_file():
                missing.setdefault(named, []).append(str(path.relative_to(ROOT)))
    assert not missing, (
        "these name a Containerfile that is not in containers/images:\n" +
        "\n".join(f"  {n}: {', '.join(sorted(set(w)))}" for n, w in sorted(missing.items())))


def test_every_make_target_named_in_prose_exists() -> None:
    makefile = (IMAGES / "Makefile").read_text()
    declared = set(re.findall(r"^([a-z][a-z0-9-]*):", makefile, re.MULTILINE))
    missing: dict[str, list[str]] = {}
    for path in texts():
        try:
            body = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        # `make build-x`, not "make it" -- only names shaped like targets here.
        for named in set(re.findall(r"\bmake ((?:build|check)-[a-z0-9-]+)", body)):
            if named not in declared:
                missing.setdefault(named, []).append(str(path.relative_to(ROOT)))
    assert not missing, (
        "these tell somebody to run a target the images Makefile does not "
        "declare:\n" +
        "\n".join(f"  {n}: {', '.join(sorted(set(w)))}" for n, w in sorted(missing.items())))


def test_every_compose_file_named_in_prose_exists() -> None:
    missing: dict[str, list[str]] = {}
    for path in texts():
        try:
            body = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        # Dots allowed inside the name: host-report.gpu.compose.yaml is one
        # file, and a pattern that stops at the dot reports a `gpu.compose.yaml`
        # nobody wrote.
        for named in set(re.findall(r"[a-z0-9][a-z0-9.-]*\.compose\.yaml", body)):
            if not (ROOT / "containers/compose" / named).is_file():
                missing.setdefault(named, []).append(str(path.relative_to(ROOT)))
    assert not missing, (
        "these name a compose file that is not in containers/compose:\n" +
        "\n".join(f"  {n}: {', '.join(sorted(set(w)))}" for n, w in sorted(missing.items())))


def test_the_runtime_topology_asks_for_steps_the_runner_has() -> None:
    """A service whose `--step` is not a choice cannot start, quietly.

    runtime.compose.yaml carried a `scrollfiesta` service calling
    `--step scrollfiesta-mesh`. run_contract_step.py has four steps and that was
    never one of them, so argparse refused before anything ran -- a service in a
    committed topology file that could not have worked on any day.
    """
    import yaml  # noqa: PLC0415

    runner = (ROOT / "scripts/container/run_contract_step.py").read_text()
    declared = set(re.search(r"^STEPS = \{([^}]*)\}", runner, re.M)
                   .group(1).replace('"', "").replace("'", "").split(", "))
    assert declared, "run_contract_step.py no longer declares STEPS"

    topology = yaml.safe_load((ROOT / "containers/compose/runtime.compose.yaml").read_text())
    for name, service in (topology.get("services") or {}).items():
        command = service.get("command") or []
        if "--step" not in command:
            continue
        step = command[command.index("--step") + 1]
        assert step in declared, (
            f"the {name} service asks for --step {step}, which "
            f"run_contract_step.py does not offer: {sorted(declared)}")


def test_the_runtime_topology_pins_images_by_digest() -> None:
    """The reason the deploys page points at this file.

    Every image here is a `:?Pin ... by digest` reference, not a tag: a tag is a
    request and a digest is the bytes. Losing that is losing the only thing this
    file demonstrates.
    """
    body = (ROOT / "containers/compose/runtime.compose.yaml").read_text()
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped.startswith("image:"):
            continue
        assert "by digest" in stripped, (
            f"an image here is not pinned by digest: {stripped}")
