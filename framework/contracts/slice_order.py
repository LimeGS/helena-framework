"""Single shared contract for the order of a rendered CT TIFF stack.

Every reader of a rendered stack must agree on what "slice ``n``" means.
Before this module the repository carried four incompatible policies at the
same time:

* ``sorted(directory.glob("*.tif"))`` — plain lexicographic order.  For an
  unpadded ``0.tif … 64.tif`` render this yields
  ``0, 1, 10, 11, …, 19, 2, 20, …`` so ``stack[32]`` is ``38.tif``, six
  physical slices away from the requested depth.
* numeric order with a silent reserve bucket (``else 1_000_000``, ``10_000``,
  ``10**9``).  A stem that is not an integer does not fail; it is quietly
  appended at the end and every subsequent index is off by one.
* the correct numeric order, copied verbatim into three scripts and therefore
  impossible to fix in one place.
* an ad-hoc ``int(path.stem)`` sort key that raises ``ValueError`` on a
  non-numeric stem.

This module replaces all four.  It fails closed: an ambiguous or mixed set of
names raises instead of picking an order, and the applied policy is returned so
that it can be recorded in the receipt of any artifact that consumed a stack.

Nothing here changes any scientific threshold.  It changes only which file the
index ``n`` refers to, and makes that choice auditable.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence


__all__ = [
    "EMPTY_TIFF_DIRECTORY",
    "LEXICOGRAPHIC_FILENAME",
    "NUMERIC_STEM_ASCENDING",
    "NUMERIC_STEM_CONTIGUOUS_ASCENDING",
    "NUMERIC_STEM_INDEX",
    "SLICE_ORDERING_POLICIES",
    "SliceOrderError",
    "ordered_tiff_files",
    "ordered_tiff_stack_position",
    "resolve_tiff_slice",
]


#: Files are ordered by ``int(stem)``; ties are impossible because duplicate
#: normalized integers are rejected.
NUMERIC_STEM_ASCENDING = "NUMERIC_STEM_ASCENDING"

#: As above, and the integers additionally form one contiguous interval.
NUMERIC_STEM_CONTIGUOUS_ASCENDING = "NUMERIC_STEM_CONTIGUOUS_ASCENDING"

#: No stem is numeric, so filename order is the only defensible order.
LEXICOGRAPHIC_FILENAME = "LEXICOGRAPHIC_FILENAME"

#: A single slice was addressed through ``int(stem) -> Path`` without ordering
#: anything.  Immune to zero padding and to lexicographic accidents.
NUMERIC_STEM_INDEX = "NUMERIC_STEM_INDEX"

#: The directory holds no TIFF at all and the caller opted into that.
EMPTY_TIFF_DIRECTORY = "EMPTY_TIFF_DIRECTORY"

SLICE_ORDERING_POLICIES = frozenset(
    {
        NUMERIC_STEM_ASCENDING,
        NUMERIC_STEM_CONTIGUOUS_ASCENDING,
        LEXICOGRAPHIC_FILENAME,
        NUMERIC_STEM_INDEX,
        EMPTY_TIFF_DIRECTORY,
    }
)

DEFAULT_SUFFIXES: tuple[str, ...] = (".tif",)

_NUMERIC_STEM = re.compile(r"[0-9]+")


class SliceOrderError(RuntimeError):
    """A TIFF stack cannot be ordered without guessing.

    Derived from :class:`RuntimeError` so that the callers migrated onto this
    module keep the failure mode they already declared.
    """


def _is_numeric_stem(stem: str) -> bool:
    # ``str.isdigit`` accepts superscripts and other non-ASCII digits that
    # ``int`` either rejects or normalizes surprisingly; anchor on ASCII.
    return _NUMERIC_STEM.fullmatch(stem) is not None


def _collect(directory: Path, suffixes: Sequence[str]) -> list[Path]:
    if not suffixes:
        raise ValueError("at least one TIFF suffix is required")
    found: dict[Path, Path] = {}
    for suffix in suffixes:
        if not suffix.startswith("."):
            raise ValueError(f"suffix must start with a dot: {suffix!r}")
        for path in directory.glob(f"*{suffix}"):
            if path.is_file():
                found[path] = path
    return list(found.values())


def _describe(paths: Iterable[Path], limit: int = 6) -> str:
    names = sorted(path.name for path in paths)
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f", … (+{len(names) - limit})"


def ordered_tiff_files(
    directory: Path,
    *,
    suffixes: Sequence[str] = DEFAULT_SUFFIXES,
    require_numeric: bool = False,
    require_contiguous: bool = False,
    allow_empty: bool = False,
) -> tuple[list[Path], str]:
    """Return ``(files, slice_ordering)`` for one rendered TIFF stack.

    ``slice_ordering`` is one of :data:`SLICE_ORDERING_POLICIES` and belongs in
    the receipt of every artifact that consumed the stack.

    Fails closed on: an empty directory (unless ``allow_empty``), stems that
    collide after integer normalization (``1.tif`` and ``01.tif``), and a
    directory that mixes numeric and non-numeric stems.  A non-numeric stem is
    never pushed into a reserve bucket at the end of the stack.
    """

    directory = Path(directory)
    files = _collect(directory, suffixes)
    if not files:
        if allow_empty:
            return [], EMPTY_TIFF_DIRECTORY
        raise SliceOrderError(f"no TIFF slices found in {directory}")

    stems = [path.stem for path in files]
    numeric_count = sum(1 for stem in stems if _is_numeric_stem(stem))
    if numeric_count and numeric_count != len(stems):
        offenders = [path for path in files if not _is_numeric_stem(path.stem)]
        raise SliceOrderError(
            f"{directory} mixes numeric and non-numeric TIFF stems "
            f"({_describe(offenders)}); refusing to guess a slice order"
        )

    if not numeric_count:
        if require_numeric:
            raise SliceOrderError(
                f"{directory} requires numeric TIFF stems, found "
                f"{_describe(files)}"
            )
        return sorted(files, key=lambda path: path.name), LEXICOGRAPHIC_FILENAME

    values = [int(stem) for stem in stems]
    if len(set(values)) != len(values):
        raise SliceOrderError(
            "numeric TIFF names are ambiguous after integer normalization: "
            f"{_describe(files)} in {directory}"
        )
    ordered = sorted(files, key=lambda path: (int(path.stem), path.name))
    if require_contiguous:
        sequence = [int(path.stem) for path in ordered]
        if any(right != left + 1 for left, right in zip(sequence, sequence[1:])):
            raise SliceOrderError(
                "numeric TIFF stems must form one contiguous interval: "
                f"{_describe(ordered)} in {directory}"
            )
        return ordered, NUMERIC_STEM_CONTIGUOUS_ASCENDING
    return ordered, NUMERIC_STEM_ASCENDING


def resolve_tiff_slice(
    directory: Path,
    index: int,
    *,
    suffixes: Sequence[str] = DEFAULT_SUFFIXES,
) -> Path:
    """Resolve a physical slice index without ordering anything.

    Builds ``int(stem) -> Path`` and looks the index up directly, so the answer
    is immune to zero padding (``032.tif`` and ``32.tif`` both resolve index
    ``32``) and to lexicographic accidents.  This is the preferred entry point
    wherever one slice is selected by index; the applied policy to record is
    :data:`NUMERIC_STEM_INDEX`.
    """

    directory = Path(directory)
    files = _collect(directory, suffixes)
    if not files:
        raise SliceOrderError(f"no TIFF slices found in {directory}")
    numeric: dict[int, Path] = {}
    for path in files:
        if not _is_numeric_stem(path.stem):
            raise SliceOrderError(
                f"{directory} holds a non-numeric TIFF stem {path.name}; "
                "a physical slice index is not resolvable"
            )
        value = int(path.stem)
        if value in numeric:
            raise SliceOrderError(
                "numeric TIFF names are ambiguous after integer normalization: "
                f"{numeric[value].name} and {path.name} in {directory}"
            )
        numeric[value] = path
    if index not in numeric:
        raise SliceOrderError(
            f"missing TIFF slice index {index} in {directory}"
        )
    return numeric[index]


def ordered_tiff_stack_position(files: Sequence[Path], index: int) -> int:
    """Return the position of physical slice ``index`` inside an ordered stack.

    ``ordered_tiff_files`` gives the array that was stacked; this maps a
    physical slice number onto the axis-0 offset of that array, so a caller can
    address ``stack[position]`` by slice number instead of by list position.
    Padding-immune and ambiguity-intolerant, exactly like
    :func:`resolve_tiff_slice`.
    """

    positions: dict[int, int] = {}
    for position, path in enumerate(files):
        path = Path(path)
        if not _is_numeric_stem(path.stem):
            raise SliceOrderError(
                f"stack holds a non-numeric TIFF stem {path.name}; "
                "a physical slice index is not resolvable"
            )
        value = int(path.stem)
        if value in positions:
            raise SliceOrderError(
                "numeric TIFF names are ambiguous after integer normalization: "
                f"duplicate slice index {value}"
            )
        positions[value] = position
    if index not in positions:
        raise SliceOrderError(f"missing TIFF slice index {index} in stack")
    return positions[index]
