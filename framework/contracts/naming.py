"""The system's name, and why its schema identifiers do not carry it.

The framework is **Helena Framework**. It used to be called Campaign X, and
every artefact it has ever written declares a schema beginning ``campaignx.``.

Those identifiers are deliberately *not* renamed, and the reason is worth
stating because the instinct is to rename everything:

A schema identifier lives inside frozen artefacts -- profiles, registries,
calibration declarations -- whose sha256 is hash-locked in tests and bound into
the receipt of every run that used them. Changing one byte inside such a file
changes its hash, which breaks the binding retroactively. When this rename was
first attempted as a blind replace it broke 22 tests, all of them hash locks
doing exactly the job they exist for.

So ``campaignx.`` is now a frozen namespace: a historical name for a contract,
the way a package keeps its import path after the product is renamed. It says
nothing about what the system is called today.
"""

from __future__ import annotations

NAME = "Helena Framework"
SHORT = "Helena"

# The namespace artefacts declare. Frozen at its historical value on purpose;
# see the module docstring.
SCHEMA_NAMESPACE = "campaignx."


def schema(suffix: str) -> str:
    """The identifier for a schema, e.g. schema('mission.v1')."""
    return f"{SCHEMA_NAMESPACE}{suffix.lstrip('.')}"


def is_ours(value: str | None) -> bool:
    """Does this artefact belong to this framework?"""
    return bool(value) and value.startswith(SCHEMA_NAMESPACE)
