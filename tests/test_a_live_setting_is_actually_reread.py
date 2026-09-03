"""A setting reported "live" has to be one `_apply_settings` actually rebinds.

Every knob declared with `_setting(...)` is read once, at import, into a
module global the request handlers close over. `_apply_settings`'s own
docstring states the rule: "Only the ones that can change meaning mid-process
are rebound; anything marked requires_restart is written and reported, not
pretended to be live." Four settings broke that rule silently --
CX_ARTIFACTS, CX_REQUIRE_LOGIN, CX_TLS_CERT and CX_TLS_KEY were neither
rebound there nor marked `requires_restart`, so `PUT /api/config/env/{name}`
answered `"live": true` while the global everything else read stayed exactly
what it was at boot. CX_REQUIRE_LOGIN is the dangerous one: turning the login
gate on through the panel reported success and left every request checking
the old value.

This reads `_apply_settings`'s own source for the globals it reassigns and
checks every other setting against that list, so a sixth setting added later
without a rebind fails here instead of shipping a second silent one.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="panel dependencies are in panel/.venv")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rebound_setting_names(source: str) -> set[str]:
    """The CX_ names `_apply_settings` reads through `value(...)`.

    Reading the function's own body rather than hand-maintaining a second
    list here is the point: the two can only agree by construction.
    """
    tree = ast.parse(source)
    function = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_settings")
    names = set()
    for node in ast.walk(function):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "value" and node.args
                and isinstance(node.args[0], ast.Constant)):
            names.add(node.args[0].value)
    return names


def test_every_setting_not_rebound_is_marked_requires_restart():
    source = (ROOT / "panel/app.py").read_text()
    rebound = _rebound_setting_names(source)
    assert "CX_RUNS" in rebound, "the parser above found nothing; _apply_settings moved"

    import panel.app as module

    live_but_stale = [
        s["name"] for s in module.SETTINGS
        if s["name"] not in rebound and not s["requires_restart"]
    ]
    assert live_but_stale == [], (
        f"{live_but_stale} report live but _apply_settings never rebinds "
        "their global -- add restart=True or rebind them")


def test_the_four_that_broke_this_are_now_marked():
    import panel.app as module
    current = {s["name"]: s["requires_restart"] for s in module.SETTINGS}
    for name in ("CX_ARTIFACTS", "CX_REQUIRE_LOGIN", "CX_TLS_CERT", "CX_TLS_KEY"):
        assert current[name] is True, f"{name} regressed to reporting live again"
