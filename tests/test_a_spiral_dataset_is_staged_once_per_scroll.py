"""Three inputs, three natures, and one of them has no URL at all.

A P1 spiral job could be queued and refused on every machine, because nothing
put a dataset where the fit reads one. The only way to get one was for somebody
to copy files onto each host by hand.

    tracks     one file of 5.42 GB      bandwidth-bound, and truncation is
                                        invisible until the fit fails strangely
    lasagna    ~152,000 small objects   latency-bound, and a chunk that failed
                                        to transfer reads as air exactly like a
                                        chunk that was never published
    umbilicus  a few kB of JSON         annotated by a person; in no bucket

The URL shape and the acquisition id below were read from dl.ash2txt.org rather
than assumed: PHerc0826 publishes one scan, 20250821151701, and its tracks
answer `content-length: 5422211072` with `accept-ranges: bytes`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation/scripts"))

import stage_spiral_dataset as staging  # noqa: E402

LISTING = ('<a href="../">../</a>\n'
           '<a href="20250821151701/">20250821151701/</a>\n')


# -- resolving the source --------------------------------------------------

def test_the_tracks_url_is_the_one_the_source_publishes():
    """Checked against dl.ash2txt.org, not constructed from the tutorial."""
    assert staging.tracks_url("0826", "20250821151701") == (
        "https://dl.ash2txt.org/datasets/spiral_datasets/PHerc0826/"
        "20250821151701/tracks/PHerc0826_20250821151701_surface_m7_L0_th0.2.dbm")


def test_the_scan_is_discovered_rather_than_guessed():
    assert staging.discover_scan("0826", opener=lambda url: LISTING) == "20250821151701"


def test_a_scroll_with_several_scans_refuses_instead_of_picking_one():
    """Which acquisition the tracks came from is part of what a fit means."""
    listing = LISTING + '<a href="20260101120000/">20260101120000/</a>\n'
    with pytest.raises(staging.StagingRefused, match="name one with --scan"):
        staging.discover_scan("0826", opener=lambda url: listing)


def test_a_scroll_with_no_dataset_says_that_rather_than_404ing_later():
    with pytest.raises(staging.StagingRefused, match="no scan directory"):
        staging.discover_scan("9999", opener=lambda url: '<a href="../">../</a>')


# -- the large file --------------------------------------------------------

def test_a_whole_local_copy_is_not_fetched_again(tmp_path):
    """5.42 GB is fifty seconds of a good link and nine minutes of a bad one."""
    target = tmp_path / "t.dbm"
    target.write_bytes(b"x" * 100)
    fetched = []

    outcome = staging.fetch_tracks(
        "https://example/t.dbm", target,
        length=lambda url: 100, run=lambda argv: fetched.append(argv) or 0)

    assert outcome["fetched"] is False and fetched == []


def test_a_short_file_is_fetched_and_then_checked_again(tmp_path):
    target = tmp_path / "t.dbm"
    target.write_bytes(b"x" * 40)

    def run(argv):
        target.write_bytes(b"x" * 100)
        return 0

    outcome = staging.fetch_tracks("https://example/t.dbm", target,
                                   length=lambda url: 100, run=run)
    assert outcome["fetched"] is True and outcome["size_bytes"] == 100


def test_a_download_that_stops_short_is_refused_by_the_second_check(tmp_path):
    """The failure this exists for. A truncated .dbm opens, reads, and yields
    fewer tracks; nothing downstream can tell."""
    target = tmp_path / "t.dbm"

    def run(argv):
        target.write_bytes(b"x" * 60)      # the link dropped
        return 0

    with pytest.raises(staging.StagingRefused, match="incomplete: 60 != 100"):
        staging.fetch_tracks("https://example/t.dbm", target,
                             length=lambda url: 100, run=run)


def test_the_fetch_resumes_rather_than_restarting(tmp_path):
    target = tmp_path / "t.dbm"
    target.write_bytes(b"x" * 40)
    seen = []

    def run(argv):
        seen.append(argv)
        target.write_bytes(b"x" * 100)
        return 0

    staging.fetch_tracks("https://example/t.dbm", target,
                         length=lambda url: 100, run=run)
    assert "-C" in seen[0] and "-" in seen[0]


# -- the many small objects ------------------------------------------------

class Bucket:
    """An S3 that can be told which keys are absent and which fail."""

    def __init__(self, keys, absent=(), failing=()):
        self.keys = list(keys)
        self.absent = set(absent)
        self.failing = set(failing)
        self.attempts: dict[str, int] = {}

    def ls(self, path):
        return [f"{path}/lasagna_v1"]

    def find(self, source, detail=True):
        return {f"{source}/{key}": {"type": "file", "size": 3}
                for key in self.keys}

    def get_file(self, key, path):
        self.attempts[key] = self.attempts.get(key, 0) + 1
        if any(key.endswith(name) for name in self.absent):
            raise FileNotFoundError(key)
        if any(key.endswith(name) for name in self.failing):
            raise OSError("connection reset")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"abc")


def test_one_level_is_staged_and_counted(tmp_path):
    bucket = Bucket(["0.0.0", "0.0.1"])

    summary = staging.fetch_lasagna("0826", "4", tmp_path, filesystem=bucket,
                                    workers=2)

    assert summary["level"] == "4"
    assert summary["arrays"]["nx"]["objects"] == 2
    assert summary["absent_404"] == 0 and summary["transfer_failures"] == 0


def test_an_absent_chunk_is_not_a_transfer_failure(tmp_path):
    """In a masked volume an absence is normal and means air. Counting it as a
    failure stops a staging that is fine."""
    bucket = Bucket(["0.0.0", "0.0.1"], absent=["0.0.1"])

    summary = staging.fetch_lasagna("0826", "4", tmp_path, filesystem=bucket,
                                    workers=2)

    assert summary["absent_404"] == 3      # one per array
    assert summary["transfer_failures"] == 0


def test_an_absent_chunk_is_not_retried(tmp_path):
    """Four retries with backoff is fifteen seconds spent proving what the
    first answer already said, and on a masked volume it is most of them."""
    bucket = Bucket(["0.0.0"], absent=["0.0.0"])

    staging.fetch_lasagna("0826", "4", tmp_path, filesystem=bucket, workers=1)

    assert all(count == 1 for count in bucket.attempts.values())


def test_a_transfer_failure_stops_the_staging_and_says_which(tmp_path):
    """The two look identical in the array and mean the opposite: one is air,
    the other is a hole in the evidence."""
    bucket = Bucket(["0.0.0", "0.0.1"], failing=["0.0.1"])

    with pytest.raises(staging.StagingRefused, match="failed to transfer"):
        staging.fetch_lasagna("0826", "4", tmp_path, filesystem=bucket, workers=2)
    # And it did retry that one, unlike an absence.
    assert max(bucket.attempts.values()) == 4


def test_a_level_that_is_not_published_is_not_an_empty_one(tmp_path):
    with pytest.raises(staging.StagingRefused, match="not published"):
        staging.fetch_lasagna("0826", "9", tmp_path,
                              filesystem=Bucket([]), workers=1)


# -- the one with no URL ---------------------------------------------------

def umbilicus(points: int = 40) -> str:
    return json.dumps({"control_points":
                       [{"z": z, "y": 1.0, "x": 2.0} for z in range(points)]})


def test_the_umbilicus_is_supplied_because_no_bucket_has_it(tmp_path):
    supplied = tmp_path / "PHerc0826_umbilicus.json"
    supplied.write_text(umbilicus())
    dataset = tmp_path / "ds"

    outcome = staging.require_umbilicus(dataset, supplied)

    assert outcome["control_points"] == 40
    assert "not published" in outcome["source"]
    assert (dataset / "umbilicus.json").is_file()


def test_a_missing_umbilicus_says_there_is_nowhere_to_fetch_one(tmp_path):
    """Inventing a URL for it would put a download in a receipt that never
    happened. It is annotated by a person."""
    with pytest.raises(staging.StagingRefused, match="no bucket to"):
        staging.require_umbilicus(tmp_path / "ds", None)


def test_a_truncated_umbilicus_is_refused(tmp_path):
    supplied = tmp_path / "u.json"
    supplied.write_text(umbilicus(points=6))

    with pytest.raises(staging.StagingRefused, match="control points"):
        staging.require_umbilicus(tmp_path / "ds", supplied)


# -- the manifest ----------------------------------------------------------

def staged(tmp_path) -> Path:
    dataset = tmp_path / "PHerc0826"
    (dataset / "tracks").mkdir(parents=True)
    (dataset / "tracks/t.dbm").write_bytes(b"x" * 500)
    (dataset / "umbilicus.json").write_text(umbilicus())
    (dataset / "lasagna_inputs/PHerc0826_nx.ome.zarr/4").mkdir(parents=True)
    (dataset / "lasagna_inputs/PHerc0826_nx.ome.zarr/4/0.0.0").write_bytes(b"abc")
    manifest = staging.build_manifest(
        dataset, scroll="0826", scan="20250821151701", level="4",
        volume_name="PHerc{scroll}_{array}.ome.zarr",
        tracks={"path": "t.dbm", "size_bytes": 500, "source": "https://example"},
        umbilicus={"path": "umbilicus.json"}, lasagna={"level": "4"})
    (dataset / staging.MANIFEST).write_text(json.dumps(manifest, indent=2))
    return dataset


def test_the_manifest_records_the_layout_the_fit_has_to_be_rebound_to(tmp_path):
    """The staging chose the level; the fit has to read that level, and the
    scale it checks shapes against is 2 to that level."""
    dataset = staged(tmp_path)
    manifest = json.loads((dataset / staging.MANIFEST).read_text())

    # Resolved, not templated: the stager knows the scroll and a reader would
    # have to guess it. Only {array} survives, because the three volumes are one
    # setting.
    assert manifest["layout"] == {
        "lasagna_volume_name": "PHerc0826_{array}.ome.zarr",
        "normal_zarr_group": "4", "lasagna_scale": 16, "tracks_file": "t.dbm"}


def test_a_volume_name_with_a_field_nobody_fills_is_refused():
    """A `{scroll}` that survives into the binding reaches the fit as a
    directory that does not exist."""
    sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation/backends/spiral"))
    import adapter

    with pytest.raises(adapter.ScrollSpecRefused, match="still carries"):
        adapter.validate_layout(
            {"lasagna_volume_name": "PHerc{scroll}_{array}.ome.zarr"})


def test_a_whole_dataset_verifies_by_size_without_reading_every_byte(tmp_path):
    report = staging.verify(staged(tmp_path))

    assert report["whole"] is True and report["checked_digests"] is False


def test_a_truncated_file_is_caught_by_the_cheap_check(tmp_path):
    dataset = staged(tmp_path)
    (dataset / "tracks/t.dbm").write_bytes(b"x" * 100)

    report = staging.verify(dataset)

    assert report["whole"] is False
    assert report["wrong"][0]["problem"] == "size"


def test_bytes_that_changed_without_changing_length_need_the_digests(tmp_path):
    dataset = staged(tmp_path)
    (dataset / "umbilicus.json").write_text(umbilicus().replace('"z": 0', '"z": 9'))

    assert staging.verify(dataset)["whole"] is True
    assert staging.verify(dataset, hashes=True)["whole"] is False


def test_a_dataset_nobody_staged_is_not_a_dataset_that_verifies(tmp_path):
    (tmp_path / "byhand").mkdir()

    with pytest.raises(staging.StagingRefused, match="assembled by hand"):
        staging.verify(tmp_path / "byhand")
