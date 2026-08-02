"""fetch_fixtures.py tests. Offline: the one code path that needs a
socket is exercised with a mocked urllib.request.urlopen; everything else
(slicing, hashing, content-pin verification, shipped-manifest invariants)
is real logic on synthetic data.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import fetch_fixtures


class TestClassify(unittest.TestCase):
    def test_not_fetchable_now(self):
        can, reason = fetch_fixtures.classify({"fetchable_now": False, "provenance": {"detail": "internal only, no public URL"}})
        self.assertFalse(can)
        self.assertIn("internal", reason)

    def test_fetchable_flag_set_but_no_url_provenance(self):
        can, _ = fetch_fixtures.classify({"fetchable_now": True, "provenance": {"type": "local_provenance"}})
        self.assertFalse(can)

    def test_fetchable_with_url(self):
        can, reason = fetch_fixtures.classify(
            {"fetchable_now": True, "provenance": {"type": "url", "url": "https://example.invalid/x.png"}}
        )
        self.assertTrue(can)
        self.assertEqual(reason, "ok")


class TestShippedManifestInvariants(unittest.TestCase):
    """Ties this script's behavior to the actual shipped manifest:
    exactly the positive entry is publicly fetchable; the three blank
    entries are honestly marked NOT_PUBLICLY_FETCHABLE (locally-rendered
    content, no public raster) and carry a content_pin so their exact
    reviewed bytes remain verifiable. See README "Fixture manifest".
    """

    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parent.parent
        cls.manifest = json.loads((repo_root / "fixtures" / "manifest.json").read_text())
        cls.fixtures = cls.manifest["fixtures"]

    def test_exactly_the_positive_entry_is_fetchable(self):
        fetchable = [e["id"] for e in self.fixtures if fetch_fixtures.classify(e)[0]]
        self.assertEqual(fetchable, ["gate0_s1_known_text"])

    def test_blank_entries_are_pinned_and_marked_not_publicly_fetchable(self):
        blanks = [e for e in self.fixtures if e["kind"] == "blank"]
        self.assertEqual(len(blanks), 3)
        for entry in blanks:
            self.assertEqual(entry["status"], "NOT_PUBLICLY_FETCHABLE", entry["id"])
            self.assertFalse(entry["fetchable_now"], entry["id"])
            pin = entry.get("content_pin") or {}
            self.assertRegex(pin.get("array_sha256", ""), r"^[0-9a-f]{64}$", entry["id"])
            # window shape must agree with the region slice it pins
            rows, cols = entry["region"]["rows"], entry["region"]["cols"]
            self.assertEqual(pin["shape"], [rows[1] - rows[0], cols[1] - cols[0]], entry["id"])
            # coordinate translation to the public segment frame is recorded
            frame = entry["region"]["coordinate_frame"]
            self.assertIn("dl.ash2txt.org", frame["ppm_url"], entry["id"])
            dims = frame["ppm_header_dims"]
            self.assertLessEqual(rows[1], dims["height"], entry["id"])
            self.assertLessEqual(cols[1], dims["width"], entry["id"])

    def test_positive_entry_has_public_url_and_verified_identity(self):
        entry = next(e for e in self.fixtures if e["id"] == "gate0_s1_known_text")
        self.assertEqual(entry["provenance"]["type"], "url")
        self.assertTrue(entry["provenance"]["url"].startswith("https://"))
        self.assertEqual(entry["provenance"]["expected_size_bytes"], 72508353)
        ranges = entry["provenance"]["identity_verification"]["ranges_checked"]
        self.assertGreaterEqual(len(ranges), 3)
        for r in ranges:
            self.assertRegex(r["sha256"], r"^[0-9a-f]{64}$")

    def test_no_sha256_without_fetch(self):
        """sha256 is only ever set by an actual fetch run (status
        FETCHED_VERIFIED); unfetched entries must carry null."""
        for entry in self.fixtures:
            if entry["status"] != "FETCHED_VERIFIED":
                self.assertIsNone(entry["sha256"], entry["id"])
            else:
                self.assertRegex(entry["sha256"] or "", r"^[0-9a-f]{64}$", entry["id"])


class TestDryRun(unittest.TestCase):
    def test_dry_run_does_not_touch_network_or_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            manifest = {
                "schema_version": "v0",
                "fixtures": [{
                    "id": "hypothetical", "kind": "blank", "status": "PROVISIONAL_UNFETCHED", "sha256": None,
                    "fetchable_now": True,
                    "provenance": {"type": "url", "url": "https://example.invalid/x.png", "detail": "hypothetical"},
                }],
            }
            path = d / "manifest.json"
            path.write_text(json.dumps(manifest))
            original_text = path.read_text()
            with mock.patch("fetch_fixtures.urllib.request.urlopen") as mocked:
                summary = fetch_fixtures.run(path, d / "cache", dry_run=True)
                mocked.assert_not_called()
            self.assertEqual(summary["fetched"], [])
            self.assertEqual(path.read_text(), original_text)

    def test_dry_run_on_shipped_manifest_is_network_free(self):
        repo_root = Path(__file__).resolve().parent.parent
        real_manifest = repo_root / "fixtures" / "manifest.json"
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            copy_path = d / "manifest.json"
            copy_path.write_text(real_manifest.read_text())  # never touch the real file
            with mock.patch("fetch_fixtures.urllib.request.urlopen") as mocked:
                summary = fetch_fixtures.run(copy_path, d / "cache", dry_run=True)
                mocked.assert_not_called()
            self.assertEqual(summary["fetched"], [])
            self.assertEqual(copy_path.read_text(), real_manifest.read_text())


class TestSliceRegion(unittest.TestCase):
    def test_slices_rows_cols(self):
        arr = np.arange(100, dtype=np.uint8).reshape(10, 10)
        entry = {"id": "t", "region": {"rows": [2, 5], "cols": [3, 7]}}
        win = fetch_fixtures.slice_region(arr, entry)
        self.assertEqual(win.shape, (3, 4))
        self.assertEqual(int(win[0, 0]), 23)

    def test_out_of_bounds_raises(self):
        arr = np.zeros((10, 10), dtype=np.uint8)
        entry = {"id": "t", "region": {"rows": [2, 11], "cols": [0, 5]}}
        with self.assertRaises(fetch_fixtures.FetchFixturesError):
            fetch_fixtures.slice_region(arr, entry)


class TestFetchOneWithMockedNetwork(unittest.TestCase):
    """fetch_one() is real download+slice+hash logic; we mock only the
    socket (the 'downloaded' bytes are a real .npy raster)."""

    def _fake_response(self, payload: bytes):
        chunks = [payload, b""]

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, n=-1):
                return chunks.pop(0) if chunks else b""

        return FakeResponse()

    def test_fetch_one_slices_hashes_and_stamps(self):
        raster = np.arange(400, dtype=np.uint8).reshape(20, 20)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "raster.npy"
            np.save(src, raster)
            payload = src.read_bytes()
            entry = {
                "id": "fake_fixture", "status": "PROVISIONAL_UNFETCHED", "sha256": None,
                "region": {"rows": [5, 15], "cols": [2, 12]},
                "provenance": {"type": "url", "url": "https://example.invalid/a.npy",
                               "expected_size_bytes": len(payload)},
                "fetchable_now": True,
            }
            with mock.patch("fetch_fixtures.urllib.request.urlopen", return_value=self._fake_response(payload)):
                updated = fetch_fixtures.fetch_one(entry, d / "cache")
            self.assertEqual(updated["status"], "FETCHED_VERIFIED")
            self.assertEqual(updated["source_file"]["sha256"], hashlib.sha256(payload).hexdigest())
            fixture = np.load(updated["fixture"]["path"])
            np.testing.assert_array_equal(fixture, raster[5:15, 2:12])
            self.assertEqual(
                updated["fixture"]["array_sha256"],
                hashlib.sha256(raster[5:15, 2:12].tobytes()).hexdigest(),
            )
            self.assertEqual(updated["sha256"], updated["fixture"]["npy_file_sha256"])

    def test_size_mismatch_refuses_to_stamp(self):
        payload = b"short"
        with tempfile.TemporaryDirectory() as d:
            entry = {
                "id": "fake_fixture",
                "region": {"rows": [0, 2], "cols": [0, 2]},
                "provenance": {"type": "url", "url": "https://example.invalid/a.npy",
                               "expected_size_bytes": 999999},
                "fetchable_now": True,
            }
            with mock.patch("fetch_fixtures.urllib.request.urlopen", return_value=self._fake_response(payload)):
                with self.assertRaises(fetch_fixtures.FetchFixturesError):
                    fetch_fixtures.fetch_one(entry, Path(d) / "cache")


class TestVerifyLocalOne(unittest.TestCase):
    def _entry(self, pin_sha):
        return {
            "id": "blank_x", "kind": "blank", "status": "NOT_PUBLICLY_FETCHABLE", "sha256": None,
            "region": {"rows": [1, 4], "cols": [2, 6]},
            "provenance": {"type": "local_render_from_public_inputs", "local_source": "mosaic.npy"},
            "content_pin": {"array_sha256": pin_sha, "shape": [3, 4], "dtype": "float16"},
            "fetchable_now": False,
        }

    def test_matching_pin_writes_fixture_and_keeps_status(self):
        rng = np.random.default_rng(0)
        mosaic = rng.random((8, 8)).astype(np.float16)
        window = mosaic[1:4, 2:6]
        pin = hashlib.sha256(np.ascontiguousarray(window).tobytes()).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            np.save(d / "mosaic.npy", mosaic)
            updated = fetch_fixtures.verify_local_one(self._entry(pin), d / "cache", d)
            self.assertEqual(updated["status"], "NOT_PUBLICLY_FETCHABLE")  # unchanged
            self.assertIsNone(updated["sha256"])  # still no fetched hash
            self.assertTrue(updated["local_verification"]["matches_content_pin"])
            np.testing.assert_array_equal(np.load(updated["local_verification"]["fixture"]["path"]), window)

    def test_changed_source_refuses_to_write(self):
        mosaic = np.zeros((8, 8), dtype=np.float16)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            np.save(d / "mosaic.npy", mosaic)
            entry = self._entry("0" * 64)  # pin that cannot match
            with self.assertRaises(fetch_fixtures.FetchFixturesError):
                fetch_fixtures.verify_local_one(entry, d / "cache", d)
            self.assertFalse((d / "cache" / "blank_x.npy").exists())


class TestCLI(unittest.TestCase):
    def test_missing_manifest_reports_error(self):
        rc = fetch_fixtures.main(["--manifest", "/no/such/manifest.json"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
