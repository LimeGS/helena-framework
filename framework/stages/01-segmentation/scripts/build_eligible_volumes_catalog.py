#!/usr/bin/env python3
"""Build ``workspace/catalog/eligible_volumes.json`` from the open-data bucket.

The catalogue was hand-maintained from an old Campania filter and held 13 of the
45 scrolls the bucket exposes. A scroll it does not carry cannot be intaken by P0
at all, and in P1 it fails as ``NO_M7_CANDIDATES`` in about a second -- a message
that means "I searched the prediction and nothing qualified" being returned for a
scroll whose prediction was never looked up. That misdiagnosis cost a day.

So the source of truth moves to the bucket. Every scroll the bucket lists is
either catalogued or named in the skip report with the reason it was not; there
is no third outcome, and that accounting is the thing the old process lacked.

What is derived, and from where:

* the scroll set, from the top-level common prefixes;
* the eligible CT, from the one volume the m7 surface prediction is named after;
* voxel size and beam energy, from the tokens of the volume directory name
  (``9.362um``, ``113keV``) -- by token, never by position, because the distance
  field is optional and some scans carry no energy token at all;
* the shape, from the CT's own ``0/.zarray``;
* the prediction threshold, from the ``th0.2`` token of the prediction name;
* whether the prediction is at CT resolution, by comparing the two shapes rather
  than by believing the ``L0`` in its name. The coordinate contract
  (``phase0/coordinate_contracts/ct_l0_xyz.json``) forbids scaling in P1, so a
  prediction at a reduced level is not usable, and the shape is the fact.

What is *not* derived, and is therefore written as ``null``:
``target_allowed`` and ``training_allowed``. Those are prize-rule and licensing
claims. The bucket states neither, and a catalogue that quietly asserts a right
it cannot see is worse than one that admits it does not know.

This script only prints unless ``--output`` is given: regenerating the catalogue
is a decision somebody reviews as a diff, not a side effect of running a tool.

Anonymous HTTPS throughout. A generator that needs a key is a generator nobody
runs, and the catalogue goes stale again.

Usage::

    framework/stages/01-segmentation/scripts/build_eligible_volumes_catalog.py \
        --report - --compare workspace/catalog/eligible_volumes.json

    framework/stages/01-segmentation/scripts/build_eligible_volumes_catalog.py \
        --output workspace/catalog/eligible_volumes.json \
        --report workspace/catalog/eligible_volumes.report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
DEFAULT_CATALOG = "workspace/catalog/eligible_volumes.json"

#: The model whose predictions this pipeline screens. The bucket also publishes
#: `surface-recto-*` predictions, and PHercParis4 carries one at L0 -- taking it
#: because the level matched would hand P1 a prediction screened at a threshold
#: from a different model.
PREDICTION_MODEL = "m7"

#: A prediction directory: `<scan>-surface-<model-run>-surface-<model>-L<level>-th<threshold>.zarr`.
PREDICTION_NAME = re.compile(
    r"^(?P<scan>\d{14})-surface-(?P<run>\d{14})-surface-"
    r"(?P<model>[a-z0-9]+)-L(?P<level>\d+)-th(?P<threshold>\d+(?:\.\d+)?)\.zarr$"
)

#: A scroll directory whose number carries leading zeros in the bucket and none
#: in the catalogue: the bucket says PHerc0125, every frozen plan says PHerc125,
#: and the catalogue spelling is the one hashed into those plans. Only the zeros
#: immediately after the prefix are stripped, so PHerc0009B becomes PHerc9B and
#: PHercMAN5 and PHerc1203 are left exactly as the bucket spells them.
BUCKET_NUMBER = re.compile(r"^(PHerc)0+(\d)")

SCAN_SUFFIX = ".zarr"
MASK_SUFFIX = "-masked"


class SourceUnreachable(RuntimeError):
    """The bucket did not answer. Never downgraded into an empty catalogue."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """A listing does not redirect; following one leaves the source we named."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, msg, headers, newurl
        raise SourceUnreachable(f"the source redirected ({code}); it is read at the URL given")


class BucketClient:
    """Anonymous reads against one S3-compatible source.

    Split out from the build so the tests can drive the whole derivation from a
    recorded listing. The alternative -- monkeypatching urlopen -- tests the
    mock; this tests the code that decides what a scroll is.
    """

    def __init__(self, source: str = DEFAULT_SOURCE, timeout: int = 30):
        self.source = source.rstrip("/")
        self.timeout = timeout
        self._opener = urllib.request.build_opener(NoRedirect)
        #: The newest object timestamp seen, which dates the catalogue against
        #: bucket state instead of against the clock. See `frozen_at`.
        self.newest_modified: datetime | None = None

    def _open(self, url: str):
        request = urllib.request.Request(url, headers={"User-Agent": "helena-eligible-catalog/1.0"})
        return self._opener.open(request, timeout=self.timeout)

    def list_prefixes(self, prefix: str = "") -> list[str]:
        """Every immediate child directory of ``prefix``, deepest component only.

        Paginated. A truncated first page silently dropping a scroll is exactly
        the failure this whole script exists to end, and 1000 keys is not a
        number the bucket is guaranteed to stay under.
        """
        found: list[str] = []
        token = ""
        depth = prefix.count("/")
        while True:
            query = (f"?list-type=2&delimiter=/&max-keys=1000"
                     f"&prefix={urllib.parse.quote(prefix)}")
            if token:
                query += f"&continuation-token={urllib.parse.quote(token)}"
            try:
                with self._open(f"{self.source}/{query}") as response:
                    root = ET.fromstring(response.read())
            except (urllib.error.URLError, OSError, ET.ParseError) as error:
                raise SourceUnreachable(f"listing {prefix or '/'}: {error}") from error
            found += [
                element.text.rstrip("/").split("/")[-1]
                for element in root.iter()
                if element.tag.endswith("Prefix")
                and element.text
                and element.text.rstrip("/").count("/") == depth
            ]
            truncated = next((e.text for e in root.iter() if e.tag.endswith("IsTruncated")), "false")
            token = next((e.text for e in root.iter() if e.tag.endswith("NextContinuationToken")), "")
            if str(truncated).lower() != "true" or not token:
                return [name for name in found if name]

    def read_json(self, key: str) -> dict[str, Any]:
        try:
            with self._open(f"{self.source}/{urllib.parse.quote(key)}") as response:
                body = json.loads(response.read())
                self._note_modified(response.headers.get("Last-Modified"))
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise SourceUnreachable(f"reading {key}: {error}") from error
        if not isinstance(body, dict):
            raise SourceUnreachable(f"reading {key}: not a JSON object")
        return body

    def _note_modified(self, header: str | None) -> None:
        if not header:
            return
        try:
            stamp = parsedate_to_datetime(header).astimezone(timezone.utc)
        except (TypeError, ValueError):
            return
        if self.newest_modified is None or stamp > self.newest_modified:
            self.newest_modified = stamp

    def frozen_at(self) -> str | None:
        """When the bucket state this catalogue describes was last written.

        Not the wall clock. Two runs over an unchanged bucket have to produce
        byte-identical files, and a generation timestamp inside the file makes
        every run a diff -- which trains a reviewer to skim exactly the artifact
        they are there to read. The newest `Last-Modified` of the objects
        actually read answers the same question and answers it about the source.
        """
        if self.newest_modified is None:
            return None
        return self.newest_modified.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def catalog_sample_id(bucket_directory: str) -> str:
    """The catalogue's spelling of a scroll the bucket calls ``bucket_directory``."""
    return BUCKET_NUMBER.sub(r"\1\2", bucket_directory)


def volume_id(directory: str) -> str:
    """``20250821151825-9.362um-1.2m-113keV`` out of the volume directory name."""
    return directory.removesuffix(SCAN_SUFFIX).removesuffix(MASK_SUFFIX)


def parse_volume_name(directory: str) -> dict[str, Any]:
    """Scan id, voxel size and beam energy out of a volume directory name.

    By token, not by position. The distance field is optional (PHerc0172 has
    none) and at least one volume carries no energy token at all
    (``20250821110041-0.500um-masked.zarr``); both come back as ``None`` rather
    than as a default, because a defaulted energy is a wrong number that looks
    like a measurement.
    """
    identity = volume_id(directory)
    tokens = identity.split("-")
    voxel = next((t.removesuffix("um") for t in tokens if t.endswith("um")), "")
    energy = next((t.removesuffix("keV") for t in tokens if t.endswith("keV")), "")
    scan = tokens[0] if tokens and len(tokens[0]) == 14 and tokens[0].isdigit() else None
    return {
        "volume_id": identity,
        "scan_id": scan,
        "voxel_size_um": _float_or_none(voxel),
        "energy_kev": _int_or_float_or_none(energy),
    }


def parse_prediction_name(directory: str) -> dict[str, Any] | None:
    """Model, declared level and threshold out of a prediction directory name."""
    match = PREDICTION_NAME.match(directory)
    if not match:
        return None
    return {
        "directory": directory,
        "scan_id": match["scan"],
        "model": match["model"],
        "declared_level": int(match["level"]),
        "threshold": float(match["threshold"]),
    }


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _int_or_float_or_none(value: str) -> int | float | None:
    """``113keV`` is an int; the catalogue has published ints since it existed."""
    if not value:
        return None
    if value.isdigit():
        return int(value)
    return _float_or_none(value)


class Skip:
    """One scroll the catalogue does not carry, and the reason it does not.

    A code as well as a sentence: the sentence is for the person reading the
    report, the code is so a later run can be diffed against this one without
    matching prose.
    """

    __slots__ = ("sample", "bucket_directory", "code", "detail")

    def __init__(self, sample: str, bucket_directory: str, code: str, detail: str):
        self.sample = sample
        self.bucket_directory = bucket_directory
        self.code = code
        self.detail = detail

    def as_dict(self) -> dict[str, str]:
        return {"sample_id": self.sample, "bucket_directory": self.bucket_directory,
                "code": self.code, "detail": self.detail}


def _shape_of(client: BucketClient, uri_key: str) -> list[int]:
    metadata = client.read_json(f"{uri_key}/0/.zarray")
    shape = metadata.get("shape")
    if not (isinstance(shape, list) and len(shape) == 3 and all(isinstance(v, int) for v in shape)):
        raise SourceUnreachable(f"reading {uri_key}/0/.zarray: shape is not three integers")
    return [int(v) for v in shape]


def examine(client: BucketClient, directory: str, *, excluded: dict[str, str]) -> tuple[dict[str, Any] | None, Skip | None]:
    """Decide what one scroll directory is, and say why when it is nothing.

    Every ``return`` on the failure side names a cause. There is deliberately no
    branch that drops a scroll without one.
    """
    sample = catalog_sample_id(directory)

    if directory in excluded or sample in excluded:
        return None, Skip(sample, directory, "EXCLUDED_BY_POLICY",
                          excluded.get(directory) or excluded[sample])

    volumes = [v for v in client.list_prefixes(f"{directory}/volumes/") if v.endswith(SCAN_SUFFIX)]
    if not volumes:
        return None, Skip(sample, directory, "NO_VOLUMES_LISTED",
                          "the bucket lists this prefix but it holds no volumes/*.zarr; "
                          "nothing has been scanned, or not published")

    listed = client.list_prefixes(f"{directory}/representations/predictions/surfaces/")
    predictions = [p for p in (parse_prediction_name(name) for name in listed if name.endswith(SCAN_SUFFIX)) if p]
    if not predictions:
        return None, Skip(sample, directory, "NO_SURFACE_PREDICTION",
                          "no surface prediction is published for any of this scroll's "
                          f"{len(volumes)} volume(s), so P1 has nothing to screen")

    m7 = [p for p in predictions if p["model"] == PREDICTION_MODEL]
    if not m7:
        others = sorted({p["model"] for p in predictions})
        return None, Skip(sample, directory, "NO_M7_SURFACE_PREDICTION",
                          f"the published surface prediction(s) are from {', '.join(others)}, "
                          f"not {PREDICTION_MODEL}; their thresholds are not this pipeline's")

    by_volume = {parse_volume_name(v)["scan_id"]: v for v in volumes}
    matched = [p for p in m7 if p["scan_id"] in by_volume]
    if not matched:
        return None, Skip(sample, directory, "PREDICTION_NAMES_NO_LISTED_VOLUME",
                          "every m7 prediction is named after a scan this scroll does not "
                          f"list under volumes/ ({', '.join(sorted(p['scan_id'] for p in m7))})")

    # The level in the name is a claim; the shape is the fact. P1 may not scale
    # coordinates, so the prediction has to be indexable in the CT's own frame.
    full_resolution: list[tuple[dict[str, Any], str, list[int], list[int]]] = []
    reduced: list[str] = []
    for prediction in sorted(matched, key=lambda p: p["directory"]):
        volume = by_volume[prediction["scan_id"]]
        ct_key = f"{directory}/volumes/{volume}"
        m7_key = f"{directory}/representations/predictions/surfaces/{prediction['directory']}"
        ct_shape = _shape_of(client, ct_key)
        m7_shape = _shape_of(client, m7_key)
        if ct_shape == m7_shape:
            full_resolution.append((prediction, volume, ct_shape, m7_shape))
        else:
            reduced.append(f"{prediction['directory']} is {m7_shape} against a CT of {ct_shape} "
                           f"(named L{prediction['declared_level']})")

    if not full_resolution:
        return None, Skip(sample, directory, "PREDICTION_NOT_AT_CT_RESOLUTION",
                          "no m7 prediction has the shape of the CT it was run on, and P1 is "
                          "forbidden to scale coordinates: " + "; ".join(reduced))

    if len(full_resolution) > 1:
        names = ", ".join(sorted(p["directory"] for p, _v, _c, _m in full_resolution))
        return None, Skip(sample, directory, "SEVERAL_PREDICTIONS_AT_CT_RESOLUTION",
                          "more than one m7 prediction is at CT resolution and the bucket does "
                          f"not say which is the target: {names}")

    prediction, volume, ct_shape, _m7_shape = full_resolution[0]
    scan = parse_volume_name(volume)
    if scan["voxel_size_um"] is None:
        return None, Skip(sample, directory, "NO_VOXEL_SIZE_IN_SCAN_NAME",
                          f"{volume} carries no `<n>um` token, and every micron figure "
                          "downstream is computed from that number")
    if scan["energy_kev"] is None:
        return None, Skip(sample, directory, "NO_BEAM_ENERGY_IN_SCAN_NAME",
                          f"{volume} carries no `<n>keV` token; the energy is not guessed")

    ct_uri = f"{client.source}/{directory}/volumes/{volume}"
    m7_uri = (f"{client.source}/{directory}/representations/predictions/surfaces/"
              f"{prediction['directory']}")

    # The finest sibling, not the first one listed. The field exists so a
    # contamination firewall can name the scan that must not be read, and the
    # one worth naming is the one that would most obviously leak the answer.
    siblings = [(parse_volume_name(v)["voxel_size_um"], v) for v in volumes if v != volume]
    higher = sorted((s for s in siblings if s[0] is not None and s[0] < scan["voxel_size_um"]))
    higher_uri = f"{client.source}/{directory}/volumes/{higher[0][1]}" if higher else None

    entry = {
        "sample_id": sample,
        "eligible_scan_id": scan["volume_id"],
        "eligible_volume_id": scan["volume_id"],
        "ct_uri": ct_uri,
        "voxel_size_um": scan["voxel_size_um"],
        "energy_kev": scan["energy_kev"],
        "shape_zyx": ct_shape,
        "surface_prediction_uri": m7_uri,
        "surface_prediction_threshold": prediction["threshold"],
        "higher_resolution_sibling_uri": higher_uri,
        # Null rather than dropped, so the silence is visible in the file. The
        # bucket publishes one LICENSE.txt at its root and no per-scroll rights
        # of any kind (the single per-scroll copy, under PHerc0009B, is that same
        # file byte for byte), so there is nothing here to read these from.
        #
        # Null and not absent because both shapes of consumer then refuse rather
        # than proceed: one asks `entry["target_allowed"] is True` and gets
        # False, the other asks `row.get("target_allowed", True)` and gets None.
        # Omitting the key would make the second read "allowed" for a scroll
        # whose rights nobody has established, which is the guess this avoids.
        "target_allowed": None,
        "training_allowed": None,
        "reason": (
            f"Derived from the open-data bucket: {directory}/volumes/{volume} is the one "
            f"volume carrying an {PREDICTION_MODEL} surface prediction at CT resolution "
            f"(threshold {prediction['threshold']}). The bucket does not state prize "
            "eligibility or training rights; those two fields are null."
        ),
    }
    return entry, None


def build(client: BucketClient, *, excluded: dict[str, str] | None = None,
          workers: int = 12) -> dict[str, Any]:
    """Catalogue the bucket, or raise rather than return part of it.

    A failure reading one scroll is not turned into "that scroll is not
    eligible". A partial catalogue that looks complete is the failure mode this
    replaces, so an unreachable scroll aborts the build and the caller decides.
    """
    excluded = dict(excluded or {})
    directories = sorted(d for d in client.list_prefixes("") if d and not d.startswith("_"))
    if not directories:
        raise SourceUnreachable("the source listed no scroll prefixes at all")

    entries: list[dict[str, Any]] = []
    skips: list[Skip] = []
    failures: list[str] = []
    order: dict[str, str] = {}

    def one(directory: str):
        return directory, examine(client, directory, excluded=excluded)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(one, directory) for directory in directories]
        for future in as_completed(futures):
            try:
                directory, (entry, skip) = future.result()
            except SourceUnreachable as error:
                failures.append(str(error))
                continue
            if entry is not None:
                order[entry["sample_id"]] = directory
                entries.append(entry)
            if skip is not None:
                skips.append(skip)

    if failures:
        raise SourceUnreachable(
            f"{len(failures)} scroll(s) could not be read, so this run cannot say whether "
            "they are eligible; refusing to write a catalogue that would look complete: "
            + "; ".join(sorted(failures)[:5])
        )

    # Sorted by the bucket's own spelling, whose zero padding orders PHerc0800
    # before PHerc1203 the way a reader expects -- and the way the catalogue this
    # replaces was already ordered, so the first regeneration is a diff about
    # content and not about line order.
    entries.sort(key=lambda e: order[e["sample_id"]])
    skips.sort(key=lambda s: s.bucket_directory)

    catalog = {
        "schema_version": 1,
        "frozen_at_utc": client.frozen_at(),
        "entries": entries,
    }
    report = {
        "source": client.source,
        "scroll_prefixes_listed": len(directories),
        "catalogued": len(entries),
        "skipped": len(skips),
        "skipped_scrolls": [skip.as_dict() for skip in skips],
        "not_derivable_from_the_bucket": {
            "target_allowed": "prize-rule eligibility; the bucket does not state it",
            "training_allowed": "licence to train on this scroll; the bucket does not state it",
        },
        # The identity the whole thing turns on: nothing is dropped in silence.
        "accounted_for": len(entries) + len(skips) == len(directories),
    }
    return {"catalog": catalog, "report": report}


def compare(previous: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    """What regenerating would change, so the reviewer reads a diff and not a file."""
    def by_sample(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
        rows = document.get("entries") or []
        return {str(row.get("sample_id")): row for row in rows if isinstance(row, dict)}

    old, new = by_sample(previous), by_sample(catalog)
    changed = {}
    for sample in sorted(set(old) & set(new)):
        fields = sorted(
            field for field in set(old[sample]) | set(new[sample])
            if old[sample].get(field) != new[sample].get(field)
        )
        if fields:
            changed[sample] = fields
    return {"added": sorted(set(new) - set(old)),
            "removed": sorted(set(old) - set(new)),
            "changed": changed}


def default_exclusions(root: Path) -> dict[str, str]:
    """Scrolls the repository has already decided are not campaign targets.

    Only one today: PHerc0139 is the development-only public positive control,
    and `framework/contracts/physical_scale.py` resolves its scale from the
    frozen control manifest precisely *because* the catalogue does not carry it.
    Catalogueing it would make the control a target of the campaign it validates.

    Read from the control policy rather than written here, so the exclusion has
    one definition and moves when the policy moves.
    """
    policy = root / "framework/profiles/01-segmentation/first-letters-control-policy-1.3.0.json"
    try:
        cohort = json.loads(policy.read_text(encoding="utf-8")).get("control_cohort") or {}
    except (OSError, ValueError):
        return {}
    scroll = cohort.get("scroll_id")
    if not isinstance(scroll, str) or not scroll:
        return {}
    return {scroll: (f"{scroll} is the {cohort.get('role', 'control')} named by "
                     f"{cohort.get('control_id', 'the control policy')}; it is a control, "
                     "not a campaign target, and its sources are locked by that policy")}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render(report: dict[str, Any], stream) -> None:
    print(f"source              {report['source']}", file=stream)
    print(f"scroll prefixes     {report['scroll_prefixes_listed']}", file=stream)
    print(f"catalogued          {report['catalogued']}", file=stream)
    print(f"skipped             {report['skipped']}", file=stream)
    for skip in report["skipped_scrolls"]:
        print(f"  {skip['bucket_directory']:<18} {skip['code']}", file=stream)
        print(f"  {'':<18} {skip['detail']}", file=stream)
    print("not derivable from the bucket, written as null:", file=stream)
    for field, why in sorted(report["not_derivable_from_the_bucket"].items()):
        print(f"  {field:<18} {why}", file=stream)
    against = report.get("against")
    if against and "unreadable" not in against:
        print(f"against {against['path']}", file=stream)
        print(f"  added   {len(against['added'])}: {', '.join(against['added']) or '-'}", file=stream)
        print(f"  removed {len(against['removed'])}: {', '.join(against['removed']) or '-'}", file=stream)
        for sample, fields in sorted(against["changed"].items()):
            print(f"  changed {sample}: {', '.join(fields)}", file=stream)
    if not report["accounted_for"]:
        print("WARNING: catalogued + skipped does not equal the scrolls listed", file=stream)


def main(argv: Iterable[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="S3-compatible base URL, read anonymously")
    parser.add_argument("--output", type=Path,
                        help="write the catalogue here; without it nothing is written")
    parser.add_argument("--report", type=Path,
                        help="write the machine-readable skip report here ('-' for stdout)")
    parser.add_argument("--compare", type=Path, default=root / DEFAULT_CATALOG,
                        help="an existing catalogue to diff against")
    parser.add_argument("--include-control", action="store_true",
                        help="catalogue the control scroll too; it is excluded by default")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args(list(argv) if argv is not None else None)

    client = BucketClient(args.source, timeout=args.timeout)
    excluded = {} if args.include_control else default_exclusions(root)
    try:
        built = build(client, excluded=excluded, workers=args.workers)
    except SourceUnreachable as error:
        print(f"the catalogue was not built: {error}", file=sys.stderr)
        return 2

    catalog, report = built["catalog"], built["report"]
    if args.compare and Path(args.compare).is_file():
        try:
            report["against"] = {"path": str(args.compare),
                                 **compare(json.loads(Path(args.compare).read_text(encoding="utf-8")),
                                           catalog)}
        except ValueError as error:
            report["against"] = {"path": str(args.compare), "unreadable": str(error)}

    render(report, sys.stderr)
    if args.report and str(args.report) == "-":
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.report:
        write_json(Path(args.report), report)

    if args.output:
        write_json(Path(args.output), catalog)
        print(f"wrote {args.output}: {len(catalog['entries'])} entries", file=sys.stderr)
    else:
        print("nothing written: pass --output to regenerate the catalogue", file=sys.stderr)
        print(json.dumps(catalog, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
