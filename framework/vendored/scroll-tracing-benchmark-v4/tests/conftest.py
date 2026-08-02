"""Shared pytest configuration for the stb test suite."""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: exercises the full PHerc0332 band (~10-15 min); not part of "
        "the default fast loop but must be run at least once per change.",
    )


# Session-scoped fixtures shared by Agent B's tests (test_arms.py,
# test_contract.py, test_strips.py): loading the band + computing its
# normals is cheap (~0.5s total) but every test module that needs a
# stb.core.Reference at a specific window otherwise re-pays it; scoping
# to "session" means the ~217-candidate slow selection test
# (test_regression_332.py, which builds its own band/cfg directly and
# does not use these fixtures) is unaffected either way.

@pytest.fixture(scope="session")
def cfg_332():
    from stb import config as stb_config

    return stb_config.load_config(REPO_ROOT / "configs" / "pherc0332.json")


@pytest.fixture(scope="session")
def band_332(cfg_332):
    from stb import band as stb_band

    xyz, valid, row0 = stb_band.load_band(cfg_332.band_path)
    return xyz, valid, row0


@pytest.fixture(scope="session")
def resolved_cfg_332(cfg_332, band_332):
    from stb import config as stb_config

    xyz, valid, _row0 = band_332
    return stb_config.resolve(cfg_332, xyz, valid)


@pytest.fixture(scope="session")
def normals_332(band_332):
    from stb import normals as stb_normals

    xyz, valid, _row0 = band_332
    normals_band, n_ok = stb_normals.band_normals(xyz, valid)
    return normals_band, n_ok
