"""Discovery of scrolls in a bucket, and the guard on where we will look.

Two pure functions, both of which fail quietly if they regress.

The scan-name parser reads physical scale off a directory name. The distance
field is optional -- PHerc0172 has none, PHercParis4 does -- so parsing by
position gets the wrong token and nothing complains: a µm figure that is
actually a beam energy still renders as a number.

The source guard decides what URL the server will fetch on a caller's behalf.
The panel binds 0.0.0.0 with no authentication, so this is the only thing
between a text box and a read of anything the host can reach.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import HTTPException  # noqa: E402

from panel.app import check_source_is_fetchable, parse_scan_name  # noqa: E402


@pytest.mark.parametrize(
    ("scan", "micron", "energy"),
    [
        # with the distance field
        ("20250728140407-9.362um-1.2m-113keV-masked.zarr", "9.362", 113.0),
        ("20260310170716-45.532um-11.0m-74keV-masked.zarr", "45.532", 74.0),
        # without it -- the reason this parses by token and not by position
        ("20241024131838-7.910um-53keV-masked.zarr", "7.910", 53.0),
        # sub-micron, and the suffix stripped either way
        ("20250101000000-0.500um-0.2m-65keV.zarr", "0.500", 65.0),
    ],
)
def test_scan_name_yields_scale_and_energy(scan, micron, energy):
    parsed = parse_scan_name(scan)
    assert parsed["pixel_um"] == micron
    assert parsed["energy_kev"] == energy
    assert not parsed["scan_id"].endswith(".zarr")


def test_a_name_in_another_layout_yields_nothing_rather_than_a_wrong_number():
    parsed = parse_scan_name("some-other-bucket-directory.zarr")
    assert parsed["pixel_um"] == ""
    assert parsed["energy_kev"] is None


@pytest.mark.parametrize(
    "source",
    [
        "http://example.com/",             # plaintext
        "ftp://example.com/",              # not http at all
        "https://127.0.0.1/",              # loopback
        "https://localhost/",              # loopback by name
        "https://169.254.169.254/",        # link-local: the metadata endpoint
        "https://10.0.0.1/",               # private
        "https://192.168.1.1/",            # private
        "https://[::1]/",                  # loopback, v6
        "https://",                        # no host
    ],
)
def test_the_panel_refuses_to_fetch_from_its_own_network(source):
    with pytest.raises(HTTPException) as raised:
        check_source_is_fetchable(source)
    assert raised.value.status_code == 400


def test_a_public_https_source_is_allowed():
    check_source_is_fetchable("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
