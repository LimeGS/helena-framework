#!/usr/bin/env python3
"""The spiral fitter as a selectable P1 backend.

Upstream calls this "currently our most powerful method" for recovering a
surface. It is an alternative to `vc3d-m7-seed-grow`, not a replacement: the
seed-grow backend produces local surfaces from an m7 prior and a seed, and
this one fits a global spiral through tracks, point collections and a
lasagna-derived normal field.

What changed at 23adee04
-------------------------
At the previously pinned commit (05dcf034), `fit_spiral.py` was a research
script with no argparse: the scroll was selected by six module-level
constants assigned at import (`dataset_path`, `scroll_name`, `z_begin`,
`z_end`, `voxel_size_um`, `spiral_outward_sense`), and everything else moved
through `FIT_SPIRAL_CONFIG_OVERRIDES`, a JSON object validated against a
105-key `default_config` dict. Selecting a scroll meant rewriting those six
assignments into a private copy of the script -- `repin.py`, an AST rewrite
chosen over a regex specifically to survive upstream moving the constants
around without silently fitting the wrong scroll under the right name.

Upstream restructured the fitter into a top-level `spiral-fitting/` package
(fit_session.py, config.py, a service layer, a nanobind C++ extension) and
gave it a real three-flag CLI: `--dataset` (required), `--scroll-spec`
(defaults to `<dataset>/spiral-scroll.json`), `--cache`. The six module
constants are gone. In their place:

  * `spiral_outward_sense`, the scroll's `name` and its `voxel_size_um` are
    now read from a JSON manifest at the dataset root -- `spiral-scroll.json`,
    schema_version 1, `fit_session.parse_scroll_spec` -- rather than assigned
    at import. `write_scroll_spec` below builds and writes that manifest; no
    source is rewritten at all, because there is no longer any source to
    rewrite.
  * `z_begin`/`z_end` moved onto `config.Config`, alongside the 105-odd keys
    the old `default_config` dict held (renamed but not restructured: the
    seed is `optimizer_random_seed` there just as it already was in this
    adapter's `SEED_KEYS`, `input_disable_patches` replaces `disable_patches`,
    and several inputs -- tracks, patches -- default to a different
    enabled/disabled state than before). They ride the same
    `FIT_SPIRAL_CONFIG_OVERRIDES` channel the 105 keys always used; that
    channel did not change shape, only what lives on it.
  * `dataset_path` is now the `--dataset` CLI argument, not a rebound
    constant.

One deliberate, explicit decision this file makes: `spiral_outward_sense`
must now be `"CW"` or `"ACW"` at the boundary the fitter reads, where it used
to be `"CW"` or `"CCW"`. Helena keeps `CCW` as its own public vocabulary --
the word every operator, profile and existing job already types -- and
`translate_winding_sense` is the one place it becomes `ACW` on the way out.
The alternative (renaming Helena's own vocabulary to match upstream's) would
make an existing caller's `CCW` either an error or, worse, something that
silently stops meaning what it used to mean; a vendored tool renaming its own
internal spelling is not a reason for this platform's public API to rename
itself under callers who never asked for the change.

What the caller can actually choose
-----------------------------------
  * the scroll's identity and the dataset layout, through `write_scroll_spec`
    -- refused here if the layout is not one this platform's staged datasets
    can satisfy, before a GPU is claimed;
  * everything else, through `spiral_environment` -- refused here if
    upstream's own `Config` would refuse it, for the same reason.

Non-claims
----------
* A fitted spiral is not physical-QC acceptance, ink, or text. It is
  geometry, and it enters the same certification gate as any other surface.
* Nothing here has been compared against seed-grow on this corpus.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Mapping

SPIRAL_PROFILE_SCHEMA = "campaignx.segmentation_backend_profile.v1"
SPIRAL_PROFILE_SCHEMA_V2 = "campaignx.segmentation_backend_profile.v2"
SPIRAL_PROFILE_SCHEMA_V3 = "campaignx.segmentation_backend_profile.v3"
SPIRAL_PROFILE_SCHEMA_V4 = "campaignx.segmentation_backend_profile.v4"
SPIRAL_PROFILE_SCHEMAS = (SPIRAL_PROFILE_SCHEMA, SPIRAL_PROFILE_SCHEMA_V2,
                          SPIRAL_PROFILE_SCHEMA_V3, SPIRAL_PROFILE_SCHEMA_V4)
# The one a run may be made against. v1 could not select a scroll; v2 required
# nine input paths the code does not require; v3 drove the six-constant AST
# rebind against a fitter that no longer has module constants to rebind.
RUNNABLE_SCHEMAS = (SPIRAL_PROFILE_SCHEMA_V4,)

# Helena's own public vocabulary for the winding direction, and the one
# boundary where it becomes upstream's. See the module docstring for why this
# is a translation and not a rename.
WINDING_SENSES = ("CW", "CCW")
_UPSTREAM_WINDING_SENSE = {"CW": "CW", "CCW": "ACW"}


class UnknownOverrideKey(KeyError):
    """An override upstream's own Config would reject.

    Raised here, before a GPU is claimed and a dataset is opened, rather than
    deep inside a run that has already paid for both.
    """


class ScrollSpecRefused(ValueError):
    """The scroll cannot be described to the fitter, said before it costs
    anything.

    The successor to `repin.ScrollNotRebindable`: there is no more source to
    rebind, but a caller can still ask for a spiral-scroll.json this platform
    should refuse to write -- an unresolvable dataset layout, a winding sense
    outside CW/CCW, a scroll name that is empty.
    """


def _load_config_module(fitter_root: Path):
    """`spiral-fitting/config.py`, imported in isolation.

    Not `fit_spiral.py`: importing that module resolves dataset paths and
    opens Zarr groups. `config.py` does neither -- `Config.__init__` only
    sets defaults and validates a mapping against them -- so importing it is
    the same kind of cheap, side-effect-free read the old adapter got from
    parsing `default_config` out of the AST, without needing to parse
    anything: the keys this platform validates overrides against are exactly
    `Config().as_dict()`.

    Loaded as a private module rather than via `sys.path`, so this never
    shadows (or is shadowed by) anything Helena imports under the same name.
    """
    path = Path(fitter_root) / "config.py"
    if not path.is_file():
        raise ScrollSpecRefused(
            f"{path} does not exist; this is not the spiral fitter, or "
            "upstream moved config.py and the profile's pinned commit is stale")
    spec = importlib.util.spec_from_file_location("_villa_spiral_config", path)
    if spec is None or spec.loader is None:
        raise ScrollSpecRefused(f"{path} could not be loaded as a module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Config"):
        raise ScrollSpecRefused(
            f"{path} has no Config class; the adapter validates every "
            "FIT_SPIRAL_CONFIG_OVERRIDES key against it")
    return module


def spiral_config_keys(fitter_root: Path) -> frozenset[str]:
    """The override keys upstream's `Config` accepts, read from the source
    this run will actually invoke.

    A second, hand-maintained list of ~120 keys would drift silently the
    first time the pinned commit moved; reading `Config().as_dict()` cannot.
    """
    module = _load_config_module(fitter_root)
    return frozenset(module.Config().as_dict())


# The optimizer's seed. At 23adee04 `config.Config` names it
# `optimizer_random_seed`, the same name this adapter already looked for
# first at the previous pin -- upstream had already renamed it once, from
# `random_seed`, before this restructure. Kept as a tuple and resolved from
# the source rather than hardcoded so a future rename fails loudly here
# instead of producing two identical fits and a seed agreement of zero, which
# reads as perfect reproducibility rather than as a run that never varied.
SEED_KEYS = ("optimizer_random_seed", "random_seed")


def seed_key(fitter_root: Path) -> str:
    """Which name this commit's `Config` gives the optimizer's seed."""
    accepted = spiral_config_keys(fitter_root)
    for name in SEED_KEYS:
        if name in accepted:
            return name
    raise UnknownOverrideKey(
        f"this commit's Config has none of {list(SEED_KEYS)}; the optimizer's "
        "seed is named something else here, and guessing would produce a "
        "pair of fits that are the same fit twice")


def load_spiral_profile(path: Path) -> dict[str, Any]:
    """A frozen declaration of how this backend runs, with its own hash.

    Older schemas stay loadable because a frozen profile that stops parsing
    takes its own record with it -- but they carry `superseded_by`, and
    `require_runnable_profile` refuses to run any of them.
    """
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = profile.get("schema")
    if schema not in SPIRAL_PROFILE_SCHEMAS:
        raise ValueError(f"not a segmentation backend profile: {path}")
    required = ("profile_id", "backend", "config_overrides", "source")
    required += {
        # v4 drops scroll_binding (there is no more source to rebind) for
        # dataset_layout plus a description of the spiral-scroll.json it
        # writes instead.
        SPIRAL_PROFILE_SCHEMA_V4: ("inputs", "dataset_layout", "scroll_spec"),
        SPIRAL_PROFILE_SCHEMA_V3: ("scroll_binding", "inputs", "dataset_layout"),
        SPIRAL_PROFILE_SCHEMA_V2: ("scroll_binding", "required_inputs"),
        SPIRAL_PROFILE_SCHEMA: ("not_selectable_here", "required_inputs"),
    }[schema]
    for key in required:
        if key not in profile:
            raise ValueError(f"spiral profile is missing {key}: {path}")
    return profile


def require_runnable_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Refuse a profile that records a mechanism this commit no longer has.

    v1 could not select a scroll at all. v2 required four input files the fit
    treats as optional. v3 drove an AST rebind of six module constants that
    23adee04 removed -- running it against this commit would fit whichever
    dataset happened to sit at upstream's own hardcoded path, under whatever
    scroll name that run's `spiral-scroll.json` (if any) named.
    """
    if profile.get("schema") not in RUNNABLE_SCHEMAS:
        raise ValueError(
            f"{profile.get('profile_id')} is superseded and must not be run: "
            f"use {profile.get('superseded_by') or 'the current profile'}. "
            "v1 could not select a scroll at all; v2 required four input "
            "files the fit treats as optional; v3's six-constant rebind has "
            "nothing left to rebind now that the scroll is a JSON manifest.")
    return profile


# --------------------------------------------------------------------------
# spiral-scroll.json: the manifest that replaced the six-constant rebind
# --------------------------------------------------------------------------
#
# Upstream's own path-override keys (fit_session.SCROLL_SPEC_PATH_OVERRIDE_KEYS,
# built there as `tuple(spec.key for spec in FIT_INPUT_CATALOG if spec.kind !=
# "pcl-set")`), mirrored here rather than imported: the runner invokes the
# fitter as a subprocess in its own interpreter, so this adapter never imports
# fit_session either. Re-derive this tuple from the pinned commit's
# fit_session.py if it is ever out of date -- the module docstring there
# names the exact source (FIT_INPUT_CATALOG keys whose kind is not
# "pcl-set").
#
# Corrected for spiral-fitter-v1@0.4.1: this tuple carried only 10 of
# upstream's 12 keys, missing "verified_patches" and "unverified_patches" --
# both are FIT_INPUT_CATALOG entries of kind "directory" (not "pcl-set"), so
# upstream's own SCROLL_SPEC_PATH_OVERRIDE_KEYS has always included them.
# Harmless while every profile ran with input_disable_patches: true (nothing
# ever asked to override either path), but load-bearing the moment a profile
# turns patches on: "unverified_patches" carries no conventional_relative at
# all in FIT_INPUT_CATALOG (fit_session.conventional_input_paths resolves it
# as `spec.path_override("unverified_patches")` and nothing else), so without
# this key a scroll-spec `paths` override naming it was rejected by
# parse_scroll_spec as unknown -- there was no way to ever point the fitter at
# an unverified-patches directory. Verified directly against fit_session.py's
# own FIT_INPUT_CATALOG at the pinned commit (see the profile's own
# `notes.config_overrides` for the full trace).
SCROLL_SPEC_PATH_OVERRIDE_KEYS = (
    "umbilicus", "verified_patches", "unverified_patches", "fibers",
    "fiber_directions", "outer_shell", "tracks_dbm", "normal_x", "normal_y",
    "gradient_magnitude", "surf_sdt", "winding_inference",
)

SCROLL_SPEC_SCHEMA_VERSION = 1

# Upstream's own conventional dataset layout at this commit
# (fit_session._CONVENTIONAL_INPUT_RELATIVES / PCL_ROLE_CONVENTIONS): the
# defaults below reproduce it exactly, so a caller who states nothing about
# the layout gets the layout upstream ships.
DEFAULT_LASAGNA_VOLUME_NAME = "las_008_{array}.ome.zarr"
DEFAULT_TRACKS_FILE = "2um_ds2_ps256_surf_v2.dbm"
# Unlike the four above, upstream has no default for this one at all --
# "unverified_patches" is the one FIT_INPUT_CATALOG entry with an empty
# conventional_relative, so there is no dataset-relative path a caller who
# says nothing would get. Empty here means exactly that: no override is
# written, matching every profile before 0.4.1 byte-for-byte (see
# test_upstream_default_layout_needs_no_path_overrides). A profile that wants
# grown patches to participate has to say where they are, the same as
# lasagna_volume_name or tracks_file, except there is no "upstream's own
# value" to fall back to -- only "not asked for".
DEFAULT_UNVERIFIED_PATCHES_DIR = ""
# The same absence for verified patches: upstream's other patch role, with
# the same empty conventional_relative and the same "not asked for" default.
# Upstream's own docstring for the winding model says the raw spiral surface
# can sit a whole turn off the sheet and is spliced with verified patches for
# that reason; a profile that wants them has to say where they are.
DEFAULT_VERIFIED_PATCHES_DIR = ""
LAYOUT_DEFAULTS: dict[str, Any] = {
    "lasagna_volume_name": DEFAULT_LASAGNA_VOLUME_NAME,
    "normal_zarr_group": "4",
    "lasagna_scale": 4,
    "tracks_file": DEFAULT_TRACKS_FILE,
    "unverified_patches_dir": DEFAULT_UNVERIFIED_PATCHES_DIR,
    "verified_patches_dir": DEFAULT_VERIFIED_PATCHES_DIR,
}

# Which array each lasagna path override key names, mirroring the three-way
# split fit_session.py's FIT_INPUT_CATALOG makes.
_LASAGNA_PATH_KEYS = {"nx": "normal_x", "ny": "normal_y", "grad_mag": "gradient_magnitude"}


def translate_winding_sense(sense: Any) -> str:
    """Helena's CW/CCW, translated to the CW/ACW the fitter reads.

    The one function this integration's CW/CCW decision runs through. See
    the module docstring for why this is a translation, not a rename.
    """
    if sense not in WINDING_SENSES:
        raise ScrollSpecRefused(
            f"spiral_outward_sense must be CW or CCW, got {sense!r}. The "
            "winding direction is a property of the scroll, and the wrong "
            "one fits the spiral backwards.")
    return _UPSTREAM_WINDING_SENSE[sense]


def validate_layout(layout: Mapping[str, Any] | None) -> dict[str, Any]:
    """The dataset-layout binding, defaulted to upstream's own values.

    Three of the four are one decision: which lasagna volumes the fit reads,
    at which zarr group, and what scale that group is. They are coupled
    because upstream's `prepare_lasagna_volume` opens `root[normal_zarr_group]`
    and then checks its shape against `ceil(scroll_shape / lasagna_scale)`;
    getting the pair apart produces a volume that opens and a warning nobody
    reads.

    So `lasagna_scale` is *derived* from the group when it is not stated, as
    `2 ** group` -- the relation a downloaded pyramid level satisfies.
    Upstream's own trio (group "4", scale 4) does not satisfy it, which is
    why it stays the default rather than being "corrected": reproducing
    upstream is a meaningful run.

    Unlike the scroll's identity, this one defaults rather than refusing:
    leaving it out reproduces upstream exactly.

    A fifth key, `unverified_patches_dir`, joined at spiral-fitter-v1@0.4.1
    and is unrelated to the other four: it names where grown patches live,
    not how the fit reads lasagna or tracks, and its default is empty rather
    than upstream's -- see DEFAULT_UNVERIFIED_PATCHES_DIR. A sixth,
    `verified_patches_dir`, is its twin for upstream's other patch role.
    """
    given = dict(layout or {})
    unknown = sorted(set(given) - set(LAYOUT_DEFAULTS))
    if unknown:
        raise ScrollSpecRefused(
            f"{unknown} are not part of the dataset layout; it holds "
            f"{sorted(LAYOUT_DEFAULTS)}")
    checked = {**LAYOUT_DEFAULTS, **given}

    name = str(checked["lasagna_volume_name"])
    if "{array}" not in name:
        raise ScrollSpecRefused(
            f"lasagna_volume_name is {name!r} and names no {{array}}; the fit "
            "reads normal_x, normal_y and gradient_magnitude from one set "
            "and this template is how the three paths stay one decision")
    if name != Path(name).name:
        raise ScrollSpecRefused(
            f"lasagna_volume_name is {name!r}; it is one name under "
            "lasagna_inputs/, not a path")
    import string as _string
    try:
        fields = list(_string.Formatter().parse(name))
    except ValueError as malformed:
        raise ScrollSpecRefused(
            f"lasagna_volume_name is {name!r} and is not a template: "
            f"{malformed}") from malformed
    unresolved = sorted({field for _, field, _, _ in fields
                         if field and field != "array"})
    if unresolved:
        raise ScrollSpecRefused(
            f"lasagna_volume_name is {name!r} and still carries {unresolved}; "
            "only {array} is filled here, and a field nobody resolves reaches "
            "the fitter as a directory that does not exist")
    if any(brace in name.replace("{array}", "") for brace in "{}"):
        raise ScrollSpecRefused(
            f"lasagna_volume_name is {name!r}; the only brace it may carry is "
            "{array}")
    checked["lasagna_volume_name"] = name

    group = str(checked["normal_zarr_group"])
    if not group.isdigit():
        raise ScrollSpecRefused(
            f"normal_zarr_group is {group!r}; it indexes a level inside the "
            "zarr and upstream writes it as a digit string")
    checked["normal_zarr_group"] = group

    if "lasagna_scale" not in given:
        if "lasagna_volume_name" in given or "normal_zarr_group" in given:
            checked["lasagna_scale"] = 2 ** int(group)
    scale = checked["lasagna_scale"]
    if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
        raise ScrollSpecRefused(
            f"lasagna_scale is {scale!r}; it divides the scroll's shape and "
            "must be a positive integer")

    tracks = str(checked["tracks_file"])
    if any(brace in tracks for brace in "{}"):
        raise ScrollSpecRefused(
            f"tracks_file is {tracks!r}; a .dbm name carries no braces")
    if tracks != Path(tracks).name or not tracks.endswith(".dbm"):
        raise ScrollSpecRefused(
            f"tracks_file is {tracks!r}; it is one .dbm name under tracks/, "
            "not a path")
    checked["tracks_file"] = tracks

    # Empty means "not asked for" -- see DEFAULT_UNVERIFIED_PATCHES_DIR above.
    # Non-empty must be one directory name directly under the dataset root,
    # the same discipline tracks_file gets for the same reason: this becomes
    # an f-string path in scroll_spec_document, and a caller-controlled path
    # is a way to point the fitter (and, upstream, an unverified-patches
    # loader that does `os.listdir(path)`) somewhere this platform did not
    # mean to.
    for key in ("unverified_patches_dir", "verified_patches_dir"):
        value = str(checked[key])
        if not value:
            continue
        if any(brace in value for brace in "{}"):
            raise ScrollSpecRefused(
                f"{key} is {value!r}; a directory name, not a template")
        if value != Path(value).name or value in (".", ".."):
            raise ScrollSpecRefused(
                f"{key} is {value!r}; it is one directory directly under the "
                "dataset root, not a path")
    unverified_patches_dir = str(checked["unverified_patches_dir"])
    if unverified_patches_dir:
        if any(brace in unverified_patches_dir for brace in "{}"):
            raise ScrollSpecRefused(
                f"unverified_patches_dir is {unverified_patches_dir!r}; a "
                "directory name carries no braces")
        if (unverified_patches_dir != Path(unverified_patches_dir).name
                or unverified_patches_dir in (".", "..")):
            raise ScrollSpecRefused(
                f"unverified_patches_dir is {unverified_patches_dir!r}; it is "
                "one directory name directly under the dataset root, not a "
                "path")
    checked["unverified_patches_dir"] = unverified_patches_dir
    return checked


def validate_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """The scroll's identity and slab, checked before anything is written.

    A partial binding is refused rather than merged with a default: fitting
    PHerc0172's dataset at Scroll 1's z range is a run that produces
    surfaces, costs a GPU-day and means nothing. dataset_path is here too,
    even though it becomes a CLI flag rather than part of spiral-scroll.json:
    it is exactly as load-bearing as the four that are, and a partial binding
    should refuse for the same reason regardless of which mechanism a field
    ends up reaching the fitter through.
    """
    fields: dict[str, tuple[type, ...]] = {
        "dataset_path": (str,), "scroll_name": (str,), "z_begin": (int,),
        "z_end": (int,), "voxel_size_um": (float, int),
        "spiral_outward_sense": (str,),
    }
    missing = sorted(set(fields) - set(binding))
    if missing:
        raise ScrollSpecRefused(
            f"a scroll binding names all six; missing {missing}. Half a "
            "binding fits a new dataset over Scroll 1's z range at Scroll "
            "1's voxel size, which produces surfaces and means nothing.")
    unknown = sorted(set(binding) - set(fields))
    if unknown:
        raise ScrollSpecRefused(
            f"{unknown} are not part of the scroll binding; it holds "
            f"{sorted(fields)}")
    checked: dict[str, Any] = {}
    for name, kinds in fields.items():
        value = binding[name]
        if isinstance(value, bool) or not isinstance(value, kinds):
            raise ScrollSpecRefused(
                f"{name} must be {' or '.join(k.__name__ for k in kinds)}, "
                f"got {value!r}")
        checked[name] = value
    translate_winding_sense(checked["spiral_outward_sense"])
    if checked["z_end"] <= checked["z_begin"]:
        raise ScrollSpecRefused(
            f"z_end ({checked['z_end']}) must be above z_begin "
            f"({checked['z_begin']}); an empty slab fits nothing")
    if checked["voxel_size_um"] <= 0:
        raise ScrollSpecRefused(
            "voxel_size_um must be positive: every micron figure the fit "
            "produces is derived from it")
    if not str(checked["scroll_name"]).strip():
        raise ScrollSpecRefused("scroll_name must not be empty")
    if not str(checked["dataset_path"]).strip():
        raise ScrollSpecRefused("dataset_path must not be empty")
    return checked


_BINDING_FIELDS = ("scroll_name", "dataset_path", "z_begin", "z_end",
                   "voxel_size_um", "spiral_outward_sense")


def scroll_binding_for(profile: dict[str, Any],
                       requested: Mapping[str, Any]) -> dict[str, Any]:
    """The scroll binding this run fits with, from the profile and the
    request.

    The profile supplies defaults -- only the winding sense has one -- and the
    request supplies the rest. A value left empty on both sides is reported by
    name here, because the alternative is a run that never says which scroll
    it means.
    """
    defaults = (profile.get("scroll_spec") or {}).get("defaults") or {}
    binding: dict[str, Any] = {}
    unset = []
    for name in _BINDING_FIELDS:
        value = requested.get(name)
        if value is None:
            value = defaults.get(name)
        if value is None:
            unset.append(name)
        else:
            binding[name] = value
    if unset:
        raise ScrollSpecRefused(
            f"this run does not say which scroll to fit: {sorted(unset)} are "
            "set neither on the request nor as a profile default.")
    return validate_binding(binding)


def scroll_spec_document(binding: Mapping[str, Any],
                         layout: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """The `spiral-scroll.json` object this run writes, and the layout it
    resolved. Pure, so a test can check the document without touching disk.
    """
    checked_binding = validate_binding(binding)
    checked_layout = validate_layout(layout)
    document: dict[str, Any] = {
        "schema_version": SCROLL_SPEC_SCHEMA_VERSION,
        "name": checked_binding["scroll_name"],
        "voxel_size_um": float(checked_binding["voxel_size_um"]),
        "spiral_outward_sense": translate_winding_sense(
            checked_binding["spiral_outward_sense"]),
        "normal_zarr_group": checked_layout["normal_zarr_group"],
        "lasagna_scale": checked_layout["lasagna_scale"],
    }
    paths: dict[str, str] = {}
    dataset = str(checked_binding["dataset_path"])
    if checked_layout["lasagna_volume_name"] != DEFAULT_LASAGNA_VOLUME_NAME:
        for array, key in _LASAGNA_PATH_KEYS.items():
            resolved = checked_layout["lasagna_volume_name"].format(array=array)
            paths[key] = f"{dataset}/lasagna_inputs/{resolved}"
    if checked_layout["tracks_file"] != DEFAULT_TRACKS_FILE:
        paths["tracks_dbm"] = f"{dataset}/tracks/{checked_layout['tracks_file']}"
    if checked_layout["unverified_patches_dir"]:
        # Upstream has no conventional_relative for this key at all -- see
        # DEFAULT_UNVERIFIED_PATCHES_DIR -- so unlike the two overrides above,
        # this one is never "moved from upstream's own default"; it is either
        # asked for or it is not written, which is exactly what the empty
        # default and this `if` express.
        paths["unverified_patches"] = (
            f"{dataset}/{checked_layout['unverified_patches_dir']}")
    if checked_layout["verified_patches_dir"]:
        paths["verified_patches"] = (
            f"{dataset}/{checked_layout['verified_patches_dir']}")
    if paths:
        unknown = sorted(set(paths) - set(SCROLL_SPEC_PATH_OVERRIDE_KEYS))
        if unknown:  # pragma: no cover - defensive; _LASAGNA_PATH_KEYS is closed
            raise ScrollSpecRefused(f"{unknown} are not overridable dataset paths")
        document["paths"] = paths
    return document, checked_layout


def write_scroll_spec(binding: Mapping[str, Any], layout: Mapping[str, Any] | None,
                      destination: Path) -> dict[str, Any]:
    """Write `spiral-scroll.json` for this run, and describe what it says.

    Written into the run's own output directory and named on the fitter's
    `--scroll-spec` flag, never into the shared staged dataset: the dataset
    cache is shared per scroll and reused across runs (see
    stage_spiral_dataset.py), while a scroll's identity as this run states it
    -- and the receipt that has to prove what was asked for -- belongs to the
    run that asked. Atomic write plus a read-back, the same shape
    repin.repin() used for its rewritten script: a write that landed wrong is
    the one failure mode no receipt written from the in-memory document could
    catch afterwards.
    """
    document, checked_layout = scroll_spec_document(binding, layout)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, destination)
    read_back = json.loads(destination.read_text(encoding="utf-8"))
    if read_back != document:
        raise ScrollSpecRefused(
            f"{destination} did not read back as what was written; refusing "
            "to run a fit against a scroll specification nobody can trust")
    return {
        "schema": "campaignx.spiral_scroll_spec_write.v1",
        "path": str(destination),
        "document": document,
        "layout": checked_layout,
        "layout_is_upstream_default": checked_layout == dict(LAYOUT_DEFAULTS),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def binding_sha256(binding: Mapping[str, Any]) -> str:
    """One digest for the whole scroll selection, for a receipt to carry."""
    return hashlib.sha256(
        json.dumps(dict(binding), sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Everything else: FIT_SPIRAL_CONFIG_OVERRIDES and the run environment
# --------------------------------------------------------------------------

def spiral_environment(profile: dict[str, Any], *, fitter_root: Path,
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
    unknown = sorted(set(overrides) - spiral_config_keys(fitter_root))
    if unknown:
        raise UnknownOverrideKey(
            f"upstream's Config has no {unknown}; it would raise KeyError "
            "once the run had already claimed a GPU and opened the dataset. "
            "Note that the scroll's name, voxel size and winding sense are "
            "not override keys any more: they are spiral-scroll.json fields.")

    env = dict(base_env if base_env is not None else os.environ)
    # Single-process, explicitly. DistributedContext.from_env() reads exactly
    # RANK/WORLD_SIZE/LOCAL_RANK and defaults to rank=0/world_size=1 when they
    # are absent -- verified against ddp_helpers.py at the pinned commit --
    # but Helena's worker is one GPU per container, and popping them here
    # makes that a guarantee this environment gives rather than a fact that
    # happens to hold because nothing upstream of this call ever set them.
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        env.pop(name, None)
    env["FIT_SPIRAL_OUT_DIR"] = str(out_dir)
    # A deterministic checkpoint location. Left unset, self.out_path is
    # `{out_base_dir}/{today}_{scroll_name}_slice-{z_begin}-{z_end}_{n}-patch`
    # -- n is the verified-patch count, known only after the dataset loads --
    # so the runner could not name the checkpoint it needs to export
    # afterwards without first parsing that directory back out. Naming the
    # run directory explicitly is the same trade repin.py made by writing a
    # private copy rather than trusting upstream's own naming.
    env["FIT_SPIRAL_RUN_DIR"] = str(out_dir)
    env["FIT_SPIRAL_CACHE_DIR"] = str(cache_dir)
    if run_tag:
        env["FIT_SPIRAL_RUN_TAG"] = run_tag
    if overrides:
        env["FIT_SPIRAL_CONFIG_OVERRIDES"] = json.dumps(overrides, sort_keys=True)
    else:
        env.pop("FIT_SPIRAL_CONFIG_OVERRIDES", None)
    # Forced, not defaulted. Upstream's own CLI already reads WANDB_MODE=
    # disabled when nothing sets it, but this platform has no wandb
    # credentials and must never attempt the network call that finds that
    # out -- a profile or a base environment asking for anything else is
    # overridden rather than honoured.
    env["WANDB_MODE"] = "disabled"
    return env
