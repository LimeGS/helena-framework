"""A scroll the bucket holds is in the catalogue, or the report says why not.

`workspace/catalog/eligible_volumes.json` was hand-maintained and held 13 of the
45 scrolls the open-data bucket exposes. There was no third state: a scroll that
was neither catalogued nor rejected simply was not there, and nothing in the
repository could tell the two apart.

That gap surfaced as a lie downstream. P0 cannot intake a scroll the catalogue
does not carry, and P1 answers for one in about a second with
`NO_M7_CANDIDATES` -- which reads as "I screened the prediction and nothing
qualified" and actually meant "this scroll was never catalogued". A day went
into the first reading before somebody checked the second.

So the check these tests hold is an accounting identity, not a count:

    scrolls the bucket lists == scrolls catalogued + scrolls skipped with a reason

A scroll present in the bucket and absent from both sides fails here. The rest
of the file pins the individual judgements that identity is made of -- what
makes a prediction usable, what may not be guessed when the bucket is silent,
and what the generator does when the bucket does not answer at all.

Everything runs against `tests/fixtures/open_data_bucket_2026-09-01.json`, a
recorded walk of the real bucket. Offline on purpose: a test that needs the
network is a test that is skipped in CI and then deleted.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "framework/stages/01-segmentation/scripts/build_eligible_volumes_catalog.py"
FIXTURE = ROOT / "tests/fixtures/open_data_bucket_2026-09-01.json"
CATALOG = ROOT / "workspace/catalog/eligible_volumes.json"

_spec = importlib.util.spec_from_file_location("build_eligible_volumes_catalog", SCRIPT)
assert _spec and _spec.loader
builder = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = builder
_spec.loader.exec_module(builder)


class RecordedBucket(builder.BucketClient):
    """The recorded bucket, answering the same two questions the real one does.

    A subclass rather than a patched `urlopen`, so the timestamp handling, the
    JSON shape checking and the failure type are the production ones. Only the
    transport is replaced.
    """

    def __init__(self, document: dict, *, unreadable: set[str] | None = None):
        super().__init__(document["recorded_from"])
        self.listings = document["listings"]
        self.objects = document["objects"]
        self.unreadable = set(unreadable or ())

    def list_prefixes(self, prefix: str = "") -> list[str]:
        if prefix in self.unreadable:
            raise builder.SourceUnreachable(f"listing {prefix}: recorded outage")
        if prefix not in self.listings:
            raise builder.SourceUnreachable(f"listing {prefix}: not in the recording")
        return [name for name in self.listings[prefix] if name]

    def read_json(self, key: str) -> dict:
        if key in self.unreadable:
            raise builder.SourceUnreachable(f"reading {key}: recorded outage")
        if key not in self.objects:
            raise builder.SourceUnreachable(f"reading {key}: not in the recording")
        self._note_modified(self.objects[key]["last_modified"])
        return {"shape": self.objects[key]["shape"]}


@pytest.fixture
def recording() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def built(recording: dict) -> dict:
    return builder.build(RecordedBucket(recording), excluded={}, workers=4)


def samples(built: dict) -> set[str]:
    return {entry["sample_id"] for entry in built["catalog"]["entries"]}


def skips(built: dict) -> dict[str, dict]:
    return {row["bucket_directory"]: row for row in built["report"]["skipped_scrolls"]}


# --------------------------------------------------------------------------
# The identity itself.
# --------------------------------------------------------------------------

def test_every_scroll_the_bucket_lists_is_catalogued_or_skipped_with_a_reason(
    recording: dict, built: dict
) -> None:
    """The bug, stated as an assertion: no scroll may fall between the two.

    The old catalogue carried 13 of 45 and said nothing about the other 32.
    """
    listed = {builder.catalog_sample_id(name)
              for name in recording["listings"][""]
              if name and not name.startswith("_")}
    catalogued = samples(built)
    skipped = {row["sample_id"] for row in built["report"]["skipped_scrolls"]}
    missing = listed - catalogued - skipped
    assert not missing, f"{len(missing)} scroll(s) are in the bucket and in neither list: {sorted(missing)}"
    assert not (catalogued & skipped), "a scroll cannot be both catalogued and skipped"
    assert len(catalogued) + len(skipped) == len(listed)
    assert built["report"]["accounted_for"] is True


def test_the_report_gives_every_skip_a_code_and_a_sentence(built: dict) -> None:
    """`NO_M7_CANDIDATES` was misread because nothing said what was actually true.

    A code so two runs can be diffed, a sentence so the person reading it does
    not have to go and look the code up.
    """
    for row in built["report"]["skipped_scrolls"]:
        assert row["code"] and row["code"].isupper(), row
        assert len(row["detail"]) > 30, row
        assert row["bucket_directory"] and row["sample_id"], row


def test_a_scroll_the_bucket_gains_needs_no_edit_to_the_generator(recording: dict) -> None:
    """The 13 were a hardcoded tuple, so every new scan needed somebody to notice.

    Here a scroll appears in the bucket with a volume and a matching prediction
    at CT resolution, and is catalogued because the bucket says so.
    """
    grown = copy.deepcopy(recording)
    grown["listings"][""] = sorted(grown["listings"][""] + ["PHerc0777"])
    grown["listings"]["PHerc0777/volumes/"] = ["20260901120000-9.362um-1.2m-113keV-masked.zarr"]
    grown["listings"]["PHerc0777/representations/predictions/surfaces/"] = [
        "20260901120000-surface-20260413222639-surface-m7-L0-th0.2.zarr"
    ]
    grown["objects"]["PHerc0777/volumes/20260901120000-9.362um-1.2m-113keV-masked.zarr/0/.zarray"] = {
        "shape": [1000, 500, 500], "last_modified": "Mon, 01 Sep 2026 12:00:00 GMT"}
    grown["objects"]["PHerc0777/representations/predictions/surfaces/"
                     "20260901120000-surface-20260413222639-surface-m7-L0-th0.2.zarr/0/.zarray"] = {
        "shape": [1000, 500, 500], "last_modified": "Mon, 01 Sep 2026 12:00:00 GMT"}

    built = builder.build(RecordedBucket(grown), excluded={}, workers=4)
    entry = next(e for e in built["catalog"]["entries"] if e["sample_id"] == "PHerc777")
    assert entry["voxel_size_um"] == 9.362
    assert entry["energy_kev"] == 113
    assert entry["shape_zyx"] == [1000, 500, 500]
    assert entry["surface_prediction_threshold"] == 0.2


# --------------------------------------------------------------------------
# What makes a prediction usable, and why a name is not enough.
# --------------------------------------------------------------------------

def test_a_prediction_below_ct_resolution_is_skipped_and_its_shapes_are_shown(
    built: dict
) -> None:
    """Nine scrolls have an m7 prediction that P1 still cannot use.

    P1 may not scale coordinates -- `phase0/coordinate_contracts/ct_l0_xyz.json`
    says the prediction is always the matching L0 Zarr -- so a prediction at a
    quarter of the CT's shape is not a candidate source. The skip names both
    shapes, because "no candidates" without them is the message that started
    this.
    """
    row = skips(built)["PHerc0009B"]
    assert row["code"] == "PREDICTION_NOT_AT_CT_RESOLUTION"
    assert "[7278, 7065, 7065]" in row["detail"]
    assert "[29112, 28259, 28259]" in row["detail"]


def test_resolution_is_decided_by_shape_and_not_by_the_L_in_the_name(
    recording: dict
) -> None:
    """A prediction named L0 whose shape is not the CT's is still refused.

    The level in a directory name is the publisher's claim about the file. The
    two `.zarray` shapes are the fact, and the fact is what the coordinate
    contract is about. Trusting the name would put a quarter-scale array behind
    a full-scale coordinate frame and produce seeds in the wrong place -- which
    is worse than finding none, because it looks like a result.
    """
    lying = copy.deepcopy(recording)
    key = ("PHerc0125/representations/predictions/surfaces/"
           "20250821151825-surface-20260413222639-surface-m7-L0-th0.2.zarr/0/.zarray")
    lying["objects"][key] = {"shape": [5210, 2097, 2097], "last_modified": None}

    built = builder.build(RecordedBucket(lying), excluded={}, workers=4)
    assert "PHerc125" not in samples(built)
    assert skips(built)["PHerc0125"]["code"] == "PREDICTION_NOT_AT_CT_RESOLUTION"


def test_a_prediction_from_another_model_is_not_taken_because_its_level_matched(
    built: dict
) -> None:
    """PHercParis4 publishes a `surface-recto` prediction at L0 and m7 at L2.

    Taking the recto one because L0 matched would hand P1 an array screened at
    another model's threshold (th0.45 there against m7's th0.2), under a field
    called `surface_prediction_threshold` that the whole seed screen reads. The
    scroll is skipped instead.
    """
    assert "PHercParis4" not in samples(built)
    assert skips(built)["PHercParis4"]["code"] == "PREDICTION_NOT_AT_CT_RESOLUTION"
    catalogued_predictions = [e["surface_prediction_uri"] for e in built["catalog"]["entries"]]
    assert all("-m7-" in uri for uri in catalogued_predictions)


def test_a_scroll_with_no_prediction_at_all_says_that_rather_than_nothing(
    built: dict
) -> None:
    """PHerc0172, PHerc1667 and PHercParis3 have volumes and no surface model run.

    Distinct from "the prediction is at the wrong level": one is waiting on a
    model run, the other is waiting on a rerun at L0, and the operator's next
    move differs.
    """
    for directory in ("PHerc0172", "PHerc1667", "PHercParis3"):
        assert skips(built)[directory]["code"] == "NO_SURFACE_PREDICTION"


def test_a_prefix_with_no_volumes_is_not_a_scroll_that_failed_screening(
    built: dict
) -> None:
    """Six prefixes carry photos and segments and no CT at all."""
    for directory in ("PHerc51Cr4Fr8", "PHercParis1Fr34", "PHercParis2Fr47"):
        assert skips(built)[directory]["code"] == "NO_VOLUMES_LISTED"


# --------------------------------------------------------------------------
# What may not be guessed.
# --------------------------------------------------------------------------

def test_a_scan_name_without_an_energy_token_is_skipped_not_defaulted(
    recording: dict
) -> None:
    """A catalogue that quietly guesses an energy is worse than one omitting it.

    The bucket really does hold such a name -- `20250821110041-0.500um-masked.zarr`
    under PHerc0500P2 -- so the parser cannot assume the token is there. Here the
    only predicted volume of a scroll lacks it, and the scroll is dropped with
    the reason said out loud rather than carrying a 113 that nobody measured.
    """
    silent = copy.deepcopy(recording)
    old = "20250521115057-8.640um-1.2m-116keV-masked.zarr"
    new = "20250521115057-8.640um-masked.zarr"
    silent["listings"]["PHerc0175A/volumes/"] = [new]
    silent["objects"][f"PHerc0175A/volumes/{new}/0/.zarray"] = (
        silent["objects"].pop(f"PHerc0175A/volumes/{old}/0/.zarray"))

    built = builder.build(RecordedBucket(silent), excluded={}, workers=4)
    assert "PHerc175A" not in samples(built)
    assert skips(built)["PHerc0175A"]["code"] == "NO_BEAM_ENERGY_IN_SCAN_NAME"


def test_the_two_rights_fields_the_bucket_cannot_state_are_null_everywhere(
    built: dict
) -> None:
    """Prize eligibility and training rights are not properties of an S3 listing.

    They are written as null rather than omitted so that a consumer asking
    `entry["target_allowed"] is True` fails closed instead of raising, and so the
    absence is visible in the file rather than inferred from a missing key.
    """
    for entry in built["catalog"]["entries"]:
        assert entry["target_allowed"] is None
        assert entry["training_allowed"] is None
    assert set(built["report"]["not_derivable_from_the_bucket"]) == {
        "target_allowed", "training_allowed"}


def test_the_control_scroll_is_excluded_by_a_policy_that_names_itself(
    recording: dict
) -> None:
    """PHerc0139 qualifies on the bucket's evidence and is still not a target.

    It is the development-only public positive control, and `physical_scale.py`
    resolves its scale from the frozen control manifest precisely because the
    catalogue does not carry it. The exclusion is read from the control policy
    rather than written into the generator, so there is one definition of it.
    """
    excluded = builder.default_exclusions(ROOT)
    assert "PHerc0139" in excluded

    with_control = builder.build(RecordedBucket(recording), excluded={}, workers=4)
    without = builder.build(RecordedBucket(recording), excluded=excluded, workers=4)
    assert "PHerc139" in samples(with_control)
    assert "PHerc139" not in samples(without)
    row = skips(without)["PHerc0139"]
    assert row["code"] == "EXCLUDED_BY_POLICY"
    assert "PHerc0139-w025-public-positive-v1" in row["detail"]


# --------------------------------------------------------------------------
# Compatibility with what the catalogue already is.
# --------------------------------------------------------------------------

def test_the_hand_maintained_rows_are_reproduced_field_for_field(built: dict) -> None:
    """Derivation is only trustworthy if it agrees with what was verified by hand.

    Every one of the 13 rows frozen from the official prizes page is rebuilt from
    the bucket alone: the same eligible scan, the same URIs, the same Zarr shape,
    the same voxel size and energy, the same threshold, and PHerc1203's forbidden
    2.403 um sibling. Only `reason` (prose about the derivation) and the two
    rights fields differ, and those differ on purpose.
    """
    frozen = {row["sample_id"]: row
              for row in json.loads(CATALOG.read_text(encoding="utf-8"))["entries"]}
    rebuilt = {entry["sample_id"]: entry for entry in built["catalog"]["entries"]}
    assert set(frozen) <= set(rebuilt), (
        f"the bucket no longer supports {sorted(set(frozen) - set(rebuilt))}; a row that "
        "stops being derivable is a review, not a silent drop")
    for sample, row in frozen.items():
        for field, value in row.items():
            if field in {"reason", "target_allowed", "training_allowed"}:
                continue
            assert rebuilt[sample][field] == value, f"{sample}.{field}"


def test_the_bucket_spelling_becomes_the_catalogue_spelling(built: dict) -> None:
    """PHerc0125 in the bucket is PHerc125 in every frozen plan that hashes it.

    Only the zeros right after the prefix go: PHerc0009B is PHerc9B, and
    PHerc1203, PHercMAN5 and PHercParis4 are left exactly as the bucket spells
    them, because there is nothing there to strip.
    """
    assert builder.catalog_sample_id("PHerc0125") == "PHerc125"
    assert builder.catalog_sample_id("PHerc0009B") == "PHerc9B"
    assert builder.catalog_sample_id("PHerc1203") == "PHerc1203"
    assert builder.catalog_sample_id("PHercMAN5") == "PHercMAN5"
    assert builder.catalog_sample_id("PHercParis2Fr47") == "PHercParis2Fr47"
    assert "PHerc0125" not in samples(built) and "PHerc125" in samples(built)


def test_the_catalogue_keeps_the_order_a_reviewer_already_reads(built: dict) -> None:
    """Ordered by the bucket's zero-padded name, so PHerc800 precedes PHerc1203.

    Sorting on the catalogue spelling instead would put PHerc1203 second and turn
    the first regeneration into a diff about line order, which is the fastest way
    to get a reviewer to skim the file they were asked to read.
    """
    order = [entry["sample_id"] for entry in built["catalog"]["entries"]]
    assert order.index("PHerc800") < order.index("PHerc1203")
    assert order[:3] == ["PHerc125", "PHerc139", "PHerc175A"]


# --------------------------------------------------------------------------
# Reproducibility and honest failure.
# --------------------------------------------------------------------------

def test_two_runs_over_the_same_bucket_state_are_byte_identical(recording: dict) -> None:
    """So a regeneration diff is about the bucket and nothing else.

    Including the timestamp: `frozen_at_utc` is the newest `Last-Modified` of the
    objects actually read, not the wall clock, because a generation stamp makes
    every run a diff and trains the reviewer to ignore it.
    """
    first = builder.build(RecordedBucket(recording), excluded={}, workers=4)["catalog"]
    second = builder.build(RecordedBucket(recording), excluded={}, workers=1)["catalog"]
    dumped = json.dumps(first, indent=2, sort_keys=True)
    assert dumped == json.dumps(second, indent=2, sort_keys=True)
    assert first["frozen_at_utc"] == "2026-07-02T11:40:57Z"


def test_an_unreadable_scroll_aborts_rather_than_shrinking_the_catalogue(
    recording: dict
) -> None:
    """A 503 on one scroll must not read as "that scroll is not eligible".

    This is the same failure as the original bug wearing a different hat: an
    output that looks complete and is missing a scroll for a reason nobody
    recorded. So the build raises and the caller decides, instead of returning
    25 entries that a reviewer would have no way to question.
    """
    outage = {"PHerc0800/volumes/",
              "PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr/0/.zarray"}
    with pytest.raises(builder.SourceUnreachable) as raised:
        builder.build(RecordedBucket(recording, unreadable=outage), excluded={}, workers=4)
    assert "refusing to write a catalogue that would look complete" in str(raised.value)


def test_a_source_that_lists_nothing_is_an_error_and_not_an_empty_catalogue(
    recording: dict
) -> None:
    """An empty result from a wrong or unreachable URL is not "no scrolls exist"."""
    empty = copy.deepcopy(recording)
    empty["listings"][""] = []
    with pytest.raises(builder.SourceUnreachable):
        builder.build(RecordedBucket(empty), excluded={}, workers=4)


def test_a_truncated_listing_page_does_not_lose_a_scroll() -> None:
    """45 scrolls fit in one page today; the code must not depend on that.

    A second page dropped on the floor is a scroll that vanishes from the
    catalogue with no skip line and no error -- exactly the silent gap this file
    exists to make impossible, arriving from the transport instead of the logic.
    """
    pages = [
        ("<?xml version='1.0'?><ListBucketResult xmlns='http://s3.amazonaws.com/doc/2006-03-01/'>"
         "<CommonPrefixes><Prefix>PHerc0125/</Prefix></CommonPrefixes>"
         "<IsTruncated>true</IsTruncated><NextContinuationToken>page2</NextContinuationToken>"
         "</ListBucketResult>"),
        ("<?xml version='1.0'?><ListBucketResult xmlns='http://s3.amazonaws.com/doc/2006-03-01/'>"
         "<CommonPrefixes><Prefix>PHerc0800/</Prefix></CommonPrefixes>"
         "<IsTruncated>false</IsTruncated></ListBucketResult>"),
    ]
    asked: list[str] = []

    class _Response:
        def __init__(self, body: str):
            self.body = body.encode()
            self.headers = {}

        def read(self) -> bytes:
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    client = builder.BucketClient("https://example.invalid")
    client._open = lambda url: (asked.append(url), _Response(pages[len(asked) - 1]))[1]
    assert client.list_prefixes("") == ["PHerc0125", "PHerc0800"]
    assert "continuation-token=page2" in asked[1]


def test_the_comparison_says_what_regenerating_would_change(built: dict) -> None:
    """The point of the script is a diff somebody reads before applying it."""
    against = builder.compare(json.loads(CATALOG.read_text(encoding="utf-8")),
                              built["catalog"])
    assert not against["removed"], (
        f"regenerating would drop {against['removed']} from the catalogue")
    assert against["added"], "the bucket holds scrolls the frozen catalogue does not"
    for sample, fields in against["changed"].items():
        assert set(fields) <= {"reason", "target_allowed", "training_allowed"}, (
            f"{sample} would change a derived field: {fields}")

import ast
import inspect
import textwrap


# --- the deployment keeps its own catalogue current -------------------------

def test_reading_the_catalogue_never_touches_the_network():
    """Reading and refreshing are split, and this is the half that matters.

    The first version refreshed lazily, inside the reader. Every phase reads
    this -- P0 to freeze a scale, P1 to find the m7 volume -- so a walk of
    forty-five bucket prefixes landed in the path of a page load, and the test
    suite started depending on somebody else's S3: three tests began failing
    because the live bucket answered with scales the fixtures did not have.
    """
    import inspect
    import panel.app as panel_app

    # The code, not the prose: the docstring names the refresher to say it is
    # somewhere else, and a check that reads it catches the explanation.
    tree = ast.parse(textwrap.dedent(inspect.getsource(
        panel_app.eligible_catalogue_path)))
    called = {
        node.func.id if isinstance(node.func, ast.Name) else
        getattr(node.func, "attr", "")
        for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    for reaching_out in ("BucketClient", "urlopen", "refresh_eligible_catalogue",
                         "build"):
        assert reaching_out not in called, (
            f"eligible_catalogue_path() calls {reaching_out}; a read that can "
            "block on a bucket is a read that hangs a page")


def test_an_unusable_cache_falls_back_rather_than_answering_none(tmp_path, monkeypatch):
    """A cache that exists and holds nothing is worse than no cache.

    Measured: a truncated 37 KB cache with an empty `entries` list made the
    panel report that PHerc0826 -- which the checked-in catalogue carries --
    resolved to no volume this deployment could read.
    """
    import panel.app as panel_app

    broken = tmp_path / "eligible_volumes.json"
    broken.write_text('{"entries": []}')
    monkeypatch.setattr(panel_app, "ELIGIBLE_CACHE", broken)
    assert panel_app.eligible_catalogue_path() == panel_app.CATALOG

    broken.write_text("{ not json")
    assert panel_app.eligible_catalogue_path() == panel_app.CATALOG

    good = tmp_path / "good.json"
    good.write_text('{"entries": [{"sample_id": "PHerc826"}]}')
    monkeypatch.setattr(panel_app, "ELIGIBLE_CACHE", good)
    assert panel_app.eligible_catalogue_path() == good


def test_the_refresh_can_be_turned_off():
    """A suite that walks a bucket it did not ask about is a suite that fails
    when the network does."""
    import panel.app as panel_app

    source = inspect_source(panel_app)
    assert "CX_CATALOG_REFRESH" in source


def inspect_source(module) -> str:
    import inspect
    return inspect.getsource(module._keep_the_catalogue_current)
