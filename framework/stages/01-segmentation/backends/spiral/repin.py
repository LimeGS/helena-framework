#!/usr/bin/env python3
"""Pointing the spiral fitter at a scroll that is not Scroll 1.

At the pinned commit `fit_spiral.py` decides which scroll it fits with six
module-level assignments, evaluated at import:

    dataset_path         where the inputs live
    scroll_name          which scroll they are
    z_begin, z_end       the slab of it to fit
    voxel_size_um        the scale everything downstream is in
    spiral_outward_sense which way the winding goes

None of them is a `default_config` key, so `FIT_SPIRAL_CONFIG_OVERRIDES` cannot
reach them -- upstream's own validator raises `KeyError` on anything outside
that dict. Which left two ways to fit another scroll: present a different
dataset at the one path the script names, or edit the script.

This is the edit, made a mechanism instead of a habit.

Why rewriting the source rather than patching at build time
-----------------------------------------------------------
A patch in `containers/patches/` bakes one scroll into an image, and the source
lock exists to prove the image holds upstream's bytes. Rewriting at launch
keeps both: the image ships exactly what upstream wrote and the lock still
verifies it, and the run writes a second file whose digest the receipt carries
beside the first. A reader can hash both and see precisely which six values
moved.

Why an AST rewrite rather than a regular expression
---------------------------------------------------
`z_begin, z_end = 4000, 17000` is one statement with two targets, and a
substitution that matched `z_begin` textually would also match it inside a
comment, a docstring, or a keyword argument further down. Every constant here
is located as a module-level assignment whose value is a literal, and a
constant that is missing, shadowed, or computed is refused rather than left at
Scroll 1's value -- a rebind that silently did nothing would produce a fit of
the wrong scroll under the new scroll's name, which is the one outcome no
receipt could catch afterwards.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

# The six, and what each one must be. The type is checked here because these
# are substituted into source: `voxel_size_um = "9.6"` would parse, import, and
# produce micron arithmetic on a string several minutes into a GPU run.
SCROLL_CONSTANTS: dict[str, tuple[type, ...]] = {
    "dataset_path": (str,),
    "scroll_name": (str,),
    "z_begin": (int,),
    "z_end": (int,),
    "voxel_size_um": (float, int),
    "spiral_outward_sense": (str,),
}

WINDING_SENSES = ("CW", "CCW")

# A second group, and a different mechanism. These are not all literals:
# upstream writes the four paths as f-strings over `dataset_path`, so replacing
# them with literals would freeze the dataset root into four more places. What
# is rebound is the piece inside the template, and the template is regenerated.
#
# The lasagna trio is one decision, not three: the fit reads nx, ny and grad_mag
# from the same volume set and `prepare_lasagna_volume` refuses shapes that
# differ between them.
TEMPLATE_CONSTANTS: dict[str, str] = {
    "normal_nx_zarr_path": "{dataset_path}/lasagna_inputs/{lasagna_volume_name}",
    "normal_ny_zarr_path": "{dataset_path}/lasagna_inputs/{lasagna_volume_name}",
    "grad_mag_zarr_path": "{dataset_path}/lasagna_inputs/{lasagna_volume_name}",
    "tracks_dbm_path": "{dataset_path}/tracks/{tracks_file}",
}

# Which array each of the three paths names, so one template covers them.
TEMPLATE_ARRAYS: dict[str, str] = {
    "normal_nx_zarr_path": "nx",
    "normal_ny_zarr_path": "ny",
    "grad_mag_zarr_path": "grad_mag",
}

# Two more that are plain literals, and they are here rather than with the
# scroll because they describe the dataset rather than the specimen.
LAYOUT_LITERALS = ("normal_zarr_group", "lasagna_scale")

# Upstream's own values. Note that upstream's trio -- a volume named for scale
# 8, read at group '4', with lasagna_scale 4 -- does not satisfy the pyramid
# relation below, because that file is prepared rather than downloaded as a
# level. It is the default because reproducing upstream is a meaningful run.
LAYOUT_DEFAULTS: dict[str, Any] = {
    "lasagna_volume_name": "las_008_{array}.ome.zarr",
    "normal_zarr_group": "4",
    "lasagna_scale": 4,
    "tracks_file": "2um_ds2_ps256_surf_v2.dbm",
}



class ScrollNotRebindable(ValueError):
    """Upstream no longer assigns one of the six the way this rewrites them.

    Raised before a GPU is claimed. The alternative is a run that fits Scroll 1
    and files the surfaces under whichever scroll was asked for.
    """


def _literal(node: ast.expr) -> Any:
    """The value of a literal node, or a refusal.

    `ast.literal_eval` on purpose: it accepts constants and simple containers
    and nothing else, so a value that is a call, a name or an f-string is
    reported here rather than replaced with something that drops it.
    """
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError) as exc:
        raise ScrollNotRebindable(
            "this constant is computed rather than written out, so replacing it "
            f"would drop the computation: {ast.dump(node)[:120]}") from exc


def module_constants(source: str,
                     names: set[str] | None = None) -> dict[str, tuple[ast.expr, Any]]:
    """Every one of the six that this source assigns at module level.

    Returns the value node -- which carries its own position in the text -- and
    the value it currently holds. A name assigned more than once maps to the
    last assignment, because that is the one the import leaves behind.
    """
    wanted = set(SCROLL_CONSTANTS) if names is None else set(names)
    found: dict[str, tuple[ast.expr, Any]] = {}
    for statement in ast.parse(source).body:
        if not isinstance(statement, ast.Assign):
            continue
        for target in statement.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                found[target.id] = (statement.value, _literal(statement.value))
            elif isinstance(target, ast.Tuple):
                # `z_begin, z_end = 4000, 17000`. Paired element-wise, and only
                # when the right-hand side is itself a tuple of the same length:
                # unpacking a call has no per-name node to replace.
                if not isinstance(statement.value, ast.Tuple):
                    continue
                if len(target.elts) != len(statement.value.elts):
                    continue
                for name_node, value_node in zip(target.elts, statement.value.elts):
                    if (isinstance(name_node, ast.Name)
                            and name_node.id in wanted):
                        found[name_node.id] = (value_node, _literal(value_node))
    return found


def _render(value: Any) -> str:
    """A literal this platform is prepared to substitute.

    `repr` handles str, int, float and bool correctly and quotes and escapes
    strings the way Python reads them back. Anything else is refused rather
    than formatted hopefully.
    """
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ScrollNotRebindable(
            f"{value!r} is not a value this rewrites into source")
    return repr(value)


def validate_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """The six, checked before anything is written.

    A partial binding is refused rather than merged with upstream's defaults:
    fitting PHerc0172's dataset at Scroll 1's z range is a run that produces
    surfaces, costs a GPU-day and means nothing.
    """
    missing = sorted(set(SCROLL_CONSTANTS) - set(binding))
    if missing:
        raise ScrollNotRebindable(
            f"a scroll binding names all six constants; missing {missing}. "
            "Half a binding leaves the other half at Scroll 1's values.")
    unknown = sorted(set(binding) - set(SCROLL_CONSTANTS))
    if unknown:
        raise ScrollNotRebindable(
            f"{unknown} are not module constants of the spiral fitter; "
            f"the six are {sorted(SCROLL_CONSTANTS)}")
    checked: dict[str, Any] = {}
    for name, kinds in SCROLL_CONSTANTS.items():
        value = binding[name]
        if isinstance(value, bool) or not isinstance(value, kinds):
            raise ScrollNotRebindable(
                f"{name} must be {' or '.join(k.__name__ for k in kinds)}, "
                f"got {value!r}")
        checked[name] = value
    if checked["spiral_outward_sense"] not in WINDING_SENSES:
        raise ScrollNotRebindable(
            f"spiral_outward_sense is {checked['spiral_outward_sense']!r}; "
            f"upstream reads {list(WINDING_SENSES)}. The winding direction is a "
            "property of the scroll and getting it wrong reverses the fit.")
    if checked["z_end"] <= checked["z_begin"]:
        raise ScrollNotRebindable(
            f"z_end ({checked['z_end']}) must be above z_begin "
            f"({checked['z_begin']}); an empty slab fits nothing")
    if checked["voxel_size_um"] <= 0:
        raise ScrollNotRebindable(
            "voxel_size_um must be positive: every micron figure the fit "
            "produces is derived from it")
    if not str(checked["scroll_name"]).strip():
        raise ScrollNotRebindable("scroll_name must not be empty")
    return checked


def rebind(source: str, binding: Mapping[str, Any] | None,
           only: Mapping[str, Any] | None = None) -> str:
    """`source` with named module constants replaced, and nothing else touched.

    Replacements are applied from the end of the file backwards so that each
    one's recorded position is still valid when it is made.

    `only` rewrites a named set instead of the scroll's six -- the dataset
    layout's two literals use it. It takes the same route on purpose: one
    positioner, one read-back, one refusal when a constant has moved.
    """
    if only is not None:
        checked = dict(only)
        wanted = set(checked)
    else:
        checked = validate_binding(binding)
        wanted = set(SCROLL_CONSTANTS)
    present = module_constants(source, names=wanted)
    absent = sorted(wanted - set(present))
    if absent:
        raise ScrollNotRebindable(
            f"this script assigns no module-level {absent}. Either it is not "
            "the spiral fitter, or upstream moved past the pinned commit and "
            "the scroll is now selected somewhere this does not look.")

    # ast reports a column as a byte offset into the utf-8 encoding of its
    # line, so the substitution is made on bytes and decoded afterwards. On a
    # file with a non-ASCII character before a constant -- upstream's comments
    # have several -- character offsets would land in the wrong place.
    encoded = source.encode("utf-8")
    line_starts = [0]
    for line in source.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line.encode("utf-8")))

    def byte_offset(line_number: int, column: int) -> int:
        return line_starts[line_number - 1] + column

    edits = []
    for name, (node, current) in present.items():
        if name not in wanted:
            continue
        if node.end_lineno is None or node.end_col_offset is None:
            raise ScrollNotRebindable(f"{name} has no position in the source")
        edits.append((byte_offset(node.lineno, node.col_offset),
                      byte_offset(node.end_lineno, node.end_col_offset),
                      name, current))
    result = encoded
    for start, end, name, _current in sorted(edits, reverse=True):
        result = result[:start] + _render(checked[name]).encode("utf-8") + result[end:]
    rewritten = result.decode("utf-8")

    # The rewrite is checked by reading it back rather than trusted. A slice
    # that landed one character off produces source that still parses and holds
    # a value nobody asked for.
    readback = {name: value for name, (_node, value)
                in module_constants(rewritten, names=wanted).items()}
    if readback != checked:
        differing = sorted(name for name in checked
                           if readback.get(name) != checked[name])
        raise ScrollNotRebindable(
            f"the rewrite did not take on {differing}: asked for "
            f"{ {k: checked[k] for k in differing} }, the file now says "
            f"{ {k: readback.get(k) for k in differing} }")
    return rewritten


def validate_layout(layout: Mapping[str, Any] | None) -> dict[str, Any]:
    """The dataset-layout binding, defaulted to upstream's own values.

    Four decisions, and three of them are one: which lasagna volumes the fit
    reads, at which group inside them, and what scale that group is. They are
    coupled because `lasagna_data.prepare_lasagna_volume` opens
    `root[normal_zarr_group]` and then checks its shape against
    `ceil(scroll_shape / lasagna_scale)`. Getting the pair apart produces a
    volume that opens and a warning nobody reads.

    So `lasagna_scale` is *derived* from the group when it is not stated, as
    `2 ** group`, which is the relation a pyramid level satisfies and the one a
    lookup table got wrong -- level 3 came out as scale 4 instead of 8. Stating
    it explicitly is still allowed, because upstream's own trio does not satisfy
    the relation and reproducing upstream is a meaningful run.

    Unlike the scroll binding, this one defaults rather than refusing: leaving
    it out reproduces upstream exactly. The scroll cannot default because there
    is no such thing as the right scroll.
    """
    given = dict(layout or {})
    unknown = sorted(set(given) - set(LAYOUT_DEFAULTS))
    if unknown:
        raise ScrollNotRebindable(
            f"{unknown} are not part of the dataset layout; it holds "
            f"{sorted(LAYOUT_DEFAULTS)}")
    checked = {**LAYOUT_DEFAULTS, **given}

    name = str(checked["lasagna_volume_name"])
    if "{array}" not in name:
        raise ScrollNotRebindable(
            f"lasagna_volume_name is {name!r} and names no {{array}}; the fit "
            "reads nx, ny and grad_mag from one set and this template is how "
            "the three paths stay one decision")
    if "/" in name:
        raise ScrollNotRebindable(
            f"lasagna_volume_name is {name!r}; it is one name under "
            "lasagna_inputs/, not a path")
    import string as _string

    try:
        fields = list(_string.Formatter().parse(name))
    except ValueError as malformed:
        # An unbalanced brace is a refusal, not a traceback out of a validator.
        raise ScrollNotRebindable(
            f"lasagna_volume_name is {name!r} and is not a template: "
            f"{malformed}") from malformed
    unresolved = sorted({field for _, field, _, _ in fields
                         if field and field != "array"})
    if unresolved:
        raise ScrollNotRebindable(
            f"lasagna_volume_name is {name!r} and still carries {unresolved}; "
            "only {array} is filled here, and a field nobody resolves reaches "
            "the fit as a directory that does not exist")
    # Nothing but `{array}` may carry a brace. `{{` and `}}` are escapes rather
    # than fields, so `Formatter().parse` reports no unresolved name for them --
    # and `.format()` turns them back into single braces. That mattered when
    # this value was substituted into an f-string, where a surviving brace is an
    # expression: `las_{array}_{{__import__('os').system('id')}}` produced
    # `f'...las_nx_{__import__("os").system("id")}'` and ran at import. The
    # emitted form is a concatenation now and cannot interpolate at all, but a
    # filename has no use for a brace either way, so both doors are shut.
    if any(brace in name.replace("{array}", "") for brace in "{}"):
        raise ScrollNotRebindable(
            f"lasagna_volume_name is {name!r}; the only brace it may carry is "
            "{array}. A filename needs no others, and a brace that reaches the "
            "rewrite is a brace somebody can put an expression in.")
    checked["lasagna_volume_name"] = name

    group = str(checked["normal_zarr_group"])
    if not group.isdigit():
        raise ScrollNotRebindable(
            f"normal_zarr_group is {group!r}; it indexes a level inside the "
            "zarr and upstream writes it as a digit string")
    checked["normal_zarr_group"] = group

    if "lasagna_scale" not in given:
        if "lasagna_volume_name" in given or "normal_zarr_group" in given:
            # The caller moved the volumes or the level and said nothing about
            # the scale, so it is derived rather than left at upstream's, which
            # would be a shape check against the wrong divisor.
            checked["lasagna_scale"] = 2 ** int(group)
    scale = checked["lasagna_scale"]
    if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
        raise ScrollNotRebindable(
            f"lasagna_scale is {scale!r}; it divides the scroll's shape and "
            "must be a positive integer")

    tracks = str(checked["tracks_file"])
    if any(brace in tracks for brace in "{}"):
        raise ScrollNotRebindable(
            f"tracks_file is {tracks!r}; a .dbm name carries no braces")
    if tracks != Path(tracks).name or not tracks.endswith(".dbm"):
        raise ScrollNotRebindable(
            f"tracks_file is {tracks!r}; it is one .dbm name under tracks/, not "
            "a path")
    checked["tracks_file"] = tracks
    return checked


def rebind_layout(source: str, layout: Mapping[str, Any]) -> str:
    """`source` with the four templated paths regenerated at this layout.

    Rewritten as f-strings rather than literals so `{dataset_path}` keeps
    meaning what it means: freezing the dataset root into three more places is
    the bug this exists beside, not a shortcut for it.
    """
    import ast as _ast

    checked = validate_layout(layout)
    tree = _ast.parse(source)
    edits = []
    seen = set()
    for statement in tree.body:
        if not isinstance(statement, _ast.Assign):
            continue
        for target in statement.targets:
            if not (isinstance(target, _ast.Name)
                    and target.id in TEMPLATE_CONSTANTS):
                continue
            node = statement.value
            if not isinstance(node, _ast.JoinedStr):
                raise ScrollNotRebindable(
                    f"{target.id} is not an f-string any more; upstream changed "
                    "how the dataset layout is built and this rewrite would "
                    "produce a path nobody asked for")
            filled = dict(checked)
            if target.id in TEMPLATE_ARRAYS:
                filled["lasagna_volume_name"] = str(
                    checked["lasagna_volume_name"]).format(
                        array=TEMPLATE_ARRAYS[target.id])
            # A concatenation, not an f-string. This used to emit
            # `f"{replacement!r}"`, which put a caller-supplied value inside an
            # interpolation context -- and `repr()` quotes a string, it does not
            # neutralise braces that the surrounding `f` prefix then treats as
            # expressions. `las_{array}_{{...}}` survived validation as an
            # escape, `.format()` turned it back into a single brace, and the
            # emitted line ran arbitrary code at import time on a GPU worker.
            #
            # `dataset_path + '<literal>'` has no interpolation context at all.
            # The name stays a name, the rest is a plain literal, and `repr()`
            # is then doing the only job it is good at.
            tail = TEMPLATE_CONSTANTS[target.id].format(
                dataset_path="", **filled)
            edits.append((node, f"dataset_path + {tail!r}"))
            seen.add(target.id)

    absent = sorted(set(TEMPLATE_CONSTANTS) - seen)
    if absent:
        raise ScrollNotRebindable(
            f"this script assigns no module-level {absent}; the dataset layout "
            "is built somewhere this does not look")

    encoded = source.encode("utf-8")
    line_starts = [0]
    for line in source.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line.encode("utf-8")))
    positioned = sorted(
        ((line_starts[node.lineno - 1] + node.col_offset,
          line_starts[node.end_lineno - 1] + node.end_col_offset, text)
         for node, text in edits), reverse=True)
    result = encoded
    for start, end, text in positioned:
        result = result[:start] + text.encode("utf-8") + result[end:]
    rewritten = result.decode("utf-8")

    # Read the rewrite back as a shape, not as a substring. The check used to be
    # `wanted in unparse(node)`, which a payload satisfies while carrying
    # anything else beside it; this asserts the node IS `dataset_path + <str>`
    # and that the literal is exactly the intended one. Nothing else parses to
    # that shape, so an injected call cannot hide inside a passing check.
    produced: dict[str, _ast.expr] = {}
    for statement in _ast.parse(rewritten).body:
        if isinstance(statement, _ast.Assign):
            for target in statement.targets:
                if isinstance(target, _ast.Name) and target.id in TEMPLATE_CONSTANTS:
                    produced[target.id] = statement.value
    for name, template in TEMPLATE_CONSTANTS.items():
        filled = dict(checked)
        if name in TEMPLATE_ARRAYS:
            filled["lasagna_volume_name"] = str(
                checked["lasagna_volume_name"]).format(array=TEMPLATE_ARRAYS[name])
        wanted = template.format(dataset_path="", **filled)
        node = produced.get(name)
        if not (isinstance(node, _ast.BinOp) and isinstance(node.op, _ast.Add)
                and isinstance(node.left, _ast.Name)
                and node.left.id == "dataset_path"
                and isinstance(node.right, _ast.Constant)
                and node.right.value == wanted):
            raise ScrollNotRebindable(
                f"the layout rewrite did not take on {name}: the file now says "
                f"{_ast.unparse(node) if node is not None else None!r}, and the "
                "only shape accepted here is `dataset_path + <literal>`")

    # The two literals travel with the paths, because the shape check inside
    # prepare_lasagna_volume compares the group it opened against this scale.
    rewritten = rebind(rewritten, None, only={
        "normal_zarr_group": checked["normal_zarr_group"],
        "lasagna_scale": checked["lasagna_scale"]})
    return rewritten


def binding_sha256(binding: Mapping[str, Any]) -> str:
    """One digest for the whole scroll selection, for a receipt to carry."""
    return hashlib.sha256(
        json.dumps(dict(binding), sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()


def repin(script: Path, binding: Mapping[str, Any], destination: Path, *,
          layout: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write a rebound copy of the fitter and describe what changed.

    The upstream file is never modified: the image keeps the bytes its source
    lock verified, and the run reads the copy. Both digests go in the receipt,
    so "which script actually ran" is answerable from the record alone.
    """
    script, destination = Path(script), Path(destination)
    original = script.read_text(encoding="utf-8")
    was = {name: value for name, (_node, value) in module_constants(original).items()}
    rewritten = rebind(original, binding)
    checked_layout = validate_layout(layout)
    # Applied second and always: the defaults are upstream's own values, so a
    # run that says nothing about the layout gets the layout upstream ships.
    rewritten = rebind_layout(rewritten, checked_layout)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rewritten, encoding="utf-8")
    checked = validate_binding(binding)
    return {
        "schema": "campaignx.spiral_scroll_repin.v1",
        "upstream_script": str(script),
        "upstream_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
        "rebound_script": str(destination),
        "rebound_sha256": hashlib.sha256(rewritten.encode("utf-8")).hexdigest(),
        "binding": checked,
        "binding_sha256": binding_sha256(checked),
        "replaced": {name: {"was": was.get(name), "now": checked[name]}
                     for name in sorted(SCROLL_CONSTANTS)},
        "layout": checked_layout,
        "layout_is_upstream_default": checked_layout == dict(LAYOUT_DEFAULTS),
        # Recorded because it is the fact a comparison across scrolls turns on:
        # a run whose binding equals upstream's is the pinned Scroll 1 fit.
        "is_upstream_default": was == checked,
    }
