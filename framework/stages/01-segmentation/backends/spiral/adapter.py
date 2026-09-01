#!/usr/bin/env python3
"""The spiral fitter as a selectable P1 backend.

Upstream calls this "currently our most powerful method" for recovering a
surface. It is an alternative to `vc3d-m7-seed-grow`, not a replacement: the
seed-grow backend produces local surfaces from an m7 prior and a seed, and
this one fits a global spiral through tracks, point collections and a
lasagna-derived normal field.

What the caller can actually choose
-----------------------------------
At the locked commit, `fit_spiral.py` is a research script rather than a tool.
It has no argparse. Nine `FIT_SPIRAL_*` environment variables exist, and the
only one that carries settings is `FIT_SPIRAL_CONFIG_OVERRIDES`, a JSON object
whose keys upstream validates against its own `default_config` -- 105 of them
at this commit -- raising `KeyError` on anything else.

The scroll is not among those keys, and never was: `dataset_path`,
`scroll_name`, `z_begin`, `z_end`, `voxel_size_um` and `spiral_outward_sense`
are module-level constants assigned at import. So the two halves of a run are
selected by two different mechanisms, and this module owns both ends of the
first:

  * settings, through `spiral_environment` -- refused here if upstream's own
    validator would refuse them, before a GPU is claimed rather than after;
  * the scroll, through `repin.py` beside this file, which rewrites those six
    assignments into a copy of the script and records both digests.

`upstream_module_constants` in the profile still records that the six are
constants rather than options, because that is why a rebind is a source
rewrite instead of a flag.

Non-claims
----------
* A fitted spiral is not physical-QC acceptance, ink, or text. It is geometry,
  and it enters the same certification gate as any other surface.
* Nothing here has been compared against seed-grow on this corpus.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any, Mapping

SPIRAL_PROFILE_SCHEMA = "campaignx.segmentation_backend_profile.v1"
SPIRAL_PROFILE_SCHEMA_V2 = "campaignx.segmentation_backend_profile.v2"
SPIRAL_PROFILE_SCHEMA_V3 = "campaignx.segmentation_backend_profile.v3"
SPIRAL_PROFILE_SCHEMAS = (SPIRAL_PROFILE_SCHEMA, SPIRAL_PROFILE_SCHEMA_V2,
                          SPIRAL_PROFILE_SCHEMA_V3)
# The two a run may be made against. v1 could not select a scroll; v2 could,
# and required nine input paths the code does not require.
RUNNABLE_SCHEMAS = (SPIRAL_PROFILE_SCHEMA_V3,)


class UnknownOverrideKey(KeyError):
    """An override upstream's own validator would reject.

    Raised here, before a GPU is claimed and a dataset is opened, rather than
    deep inside a run that has already paid for both.
    """


def spiral_config_keys(script: Path) -> frozenset[str]:
    """The override keys upstream accepts, read from upstream's own source.

    Parsed rather than imported: importing `fit_spiral` executes module-level
    code that resolves dataset paths and opens Zarr groups. Parsed rather than
    restated: a second copy of a 105-key list is a second thing to keep true,
    and it would drift silently the first time the pinned commit moved.
    """
    tree = ast.parse(Path(script).read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "default_config":
                if not isinstance(node.value, ast.Dict):
                    break
                return frozenset(
                    key.value for key in node.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str))
    raise ValueError(
        f"{script} has no module-level default_config; this is not the spiral "
        "fitter, or upstream changed shape and the profile's pinned commit is stale")


def load_spiral_profile(path: Path) -> dict[str, Any]:
    """A frozen declaration of how this backend runs, with its own hash.

    Two schemas are read. v1 is the integration that could not select a scroll
    and says so in `not_selectable_here`; v2 declares a `scroll_binding` and is
    what a run uses. The older one is still loadable because a frozen profile
    that stops parsing takes its own record with it -- but it carries
    `superseded_by`, and `require_runnable_profile` refuses to run it.
    """
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = profile.get("schema")
    if schema not in SPIRAL_PROFILE_SCHEMAS:
        raise ValueError(f"not a segmentation backend profile: {path}")
    required = ("profile_id", "backend", "config_overrides", "source")
    required += {
        # v3 splits the inputs by what an absence costs, because four of the
        # nine v2 demanded are optional in the code, and names the dataset
        # layout the fit reads.
        SPIRAL_PROFILE_SCHEMA_V3: ("scroll_binding", "inputs", "dataset_layout"),
        SPIRAL_PROFILE_SCHEMA_V2: ("scroll_binding", "required_inputs"),
        SPIRAL_PROFILE_SCHEMA: ("not_selectable_here", "required_inputs"),
    }[schema]
    for key in required:
        if key not in profile:
            raise ValueError(f"spiral profile is missing {key}: {path}")
    return profile


def require_runnable_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Refuse a profile that records a scroll it cannot select.

    A v1 profile is a readable record of an integration, not a way to run one.
    Queueing against it would fit whatever dataset happened to be at
    upstream's path and file the surfaces under the scroll that was asked for.
    """
    if profile.get("schema") not in RUNNABLE_SCHEMAS:
        raise ValueError(
            f"{profile.get('profile_id')} is superseded and must not be run: "
            f"use {profile.get('superseded_by') or 'the current profile'}. "
            "v1 could not select a scroll at all; v2 required four input files "
            "the fit treats as optional, and refused every dataset without the "
            "Paris 4 winding annotations.")
    return profile


def scroll_binding_for(profile: dict[str, Any],
                       requested: Mapping[str, Any]) -> dict[str, Any]:
    """The six constants this run fits with, from the profile and the request.

    The profile supplies defaults -- only the winding sense has one -- and the
    request supplies the rest. A value left empty on both sides is reported by
    name here, because the alternative is upstream's Scroll 1 constant standing
    in for it silently.
    """
    # Both import forms, because this module is reached both ways: as
    # `backends.spiral.adapter` from the fleet, and as a top-level `adapter`
    # from a test that puts this directory on the path.
    try:
        from .repin import SCROLL_CONSTANTS, validate_binding  # noqa: PLC0415
    except ImportError:  # no parent package: this directory is on the path
        from repin import SCROLL_CONSTANTS, validate_binding  # type: ignore # noqa: PLC0415

    require_runnable_profile(profile)
    defaults = (profile.get("scroll_binding") or {}).get("defaults") or {}
    binding = {}
    unset = []
    for name in SCROLL_CONSTANTS:
        value = requested.get(name)
        if value is None:
            value = defaults.get(name)
        if value is None:
            unset.append(name)
        else:
            binding[name] = value
    if unset:
        raise ValueError(
            f"this run does not say which scroll to fit: {sorted(unset)} are "
            "set neither on the request nor as a profile default. The fitter "
            "would use upstream's Scroll 1 values for them.")
    return validate_binding(binding)


# The optimizer's seed, which upstream renamed. At the pinned commit 05dcf034 it
# is `random_seed`; by 6847063f (2026-08-26) it is `optimizer_random_seed`. Both
# are default_config keys, so neither needs a rebind -- but naming one in a
# profile pins the profile to a commit range nobody wrote down.
SEED_KEYS = ("optimizer_random_seed", "random_seed")


def seed_key(script: Path) -> str:
    """Which name this commit's `default_config` gives the optimizer's seed.

    Resolved from the script rather than chosen, for the reason the whole
    adapter parses that dict instead of restating it: a key that is right for
    one commit is a KeyError on the next, and this particular one fails in the
    worst available way. A seed override that reaches nothing produces two
    identical fits and a seed agreement of zero, which reads as perfect
    reproducibility rather than as a run that never varied.
    """
    accepted = spiral_config_keys(script)
    for name in SEED_KEYS:
        if name in accepted:
            return name
    raise UnknownOverrideKey(
        f"this commit's default_config has none of {list(SEED_KEYS)}; the "
        "optimizer's seed is named something else here, and guessing would "
        "produce a pair of fits that are the same fit twice")


def spiral_environment(profile: dict[str, Any], *, script: Path,
                       out_dir: Path, cache_dir: Path,
                       run_tag: str | None = None,
                       base_env: dict[str, str] | None = None) -> dict[str, str]:
    """The exact environment, so the receipt can carry it and a run repeats.

    Every setting comes from the profile. A value typed at a prompt is a
    setting no receipt records and no second run reproduces -- the same rule
    P3 applies to vc_flatten's flags, applied to the only interface this
    script has.
    """
    overrides = dict(profile.get("config_overrides") or {})
    unknown = sorted(set(overrides) - spiral_config_keys(script))
    if unknown:
        raise UnknownOverrideKey(
            f"upstream's default_config has no {unknown}; it would raise "
            "KeyError once the run had already claimed a GPU and opened the "
            "dataset. Note that the z range is not an override key: z_begin "
            "and z_end are module-level constants at this commit.")

    env = dict(base_env if base_env is not None else os.environ)
    env["FIT_SPIRAL_OUT_DIR"] = str(out_dir)
    env["FIT_SPIRAL_CACHE_DIR"] = str(cache_dir)
    if run_tag:
        env["FIT_SPIRAL_RUN_TAG"] = run_tag
    # Only when there is something to say. Upstream reads an empty string as
    # "no overrides", but sending "{}" would put an empty object in the receipt
    # as though a choice had been made and had been to change nothing.
    if overrides:
        env["FIT_SPIRAL_CONFIG_OVERRIDES"] = json.dumps(overrides, sort_keys=True)
    else:
        env.pop("FIT_SPIRAL_CONFIG_OVERRIDES", None)
    # Off unless a profile deliberately turns it on: a fitter that reports to a
    # third-party service from inside an evidence run is an exfiltration path,
    # not a diagnostic.
    env.setdefault("WANDB_MODE", "disabled")
    return env
