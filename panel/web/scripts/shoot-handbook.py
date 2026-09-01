#!/usr/bin/env python3
"""Take the handbook's figures, repeatably.

Screenshots in documentation rot faster than the prose around them, and the
usual reason is that nobody can remember how the last one was taken. This is
how: a list of shots, a viewport, and a panel to point at.

    python3 panel/web/scripts/shoot-handbook.py --panel http://127.0.0.1:5175

Against a deployment with real work in it, point `--panel` at that panel and
pass `--insecure` if it serves a self-signed certificate. What lands in
`src/assets/handbook/` is what the pages reference by name, so re-running this
updates every figure at once.

The shots are declared rather than driven: each one is a URL, an optional
element to crop to, and a caption that lives in the Markdown. A shot that
cannot find its element fails loudly rather than saving a picture of the whole
page under a name that promises otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "src" / "assets" / "handbook"

# name -> (path, selector or None, what it is for)
SHOTS: dict[str, tuple[str, str | None, str]] = {
    "handbook-navigation.png": (
        "/documentation#/docs/start/pipeline", ".hb",
        "the handbook's own sidebar, for the page that explains how to read it"),
    "pipeline-rail.png": (
        "/", ".rail",
        "the phase rail: every phase and what is queued against it"),
    "mission.png": (
        "/", "body",
        "a mission with its scrolls and their declared scale"),
    "command-form.png": (
        "/command", "body",
        "the queue form, with only the fields the phase accepts"),
    "phase-anatomy.png": (
        "/phase/P4", "body",
        "one phase's page: what it has run, and the form that queues more"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="http://127.0.0.1:5175",
                        help="the panel to photograph")
    parser.add_argument("--insecure", action="store_true",
                        help="accept a self-signed certificate")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--only", action="append",
                        help="take only these shots, by name")
    # So the same script can run inside a throwaway container on the deployment
    # being photographed, writing to a mounted directory. A panel with real work
    # in it is the only place most of these figures mean anything, and it is not
    # reachable from a laptop.
    parser.add_argument("--out", type=Path, default=None,
                        help="where the figures land (default: the assets dir)")
    # A deployed panel wants a session, and every figure worth taking is behind
    # it. The cookie is passed in rather than a username and a password: this
    # script should never be somewhere a credential is typed, and a session that
    # can be revoked is a smaller thing to hand a screenshot job.
    #
    # Get it from a browser already signed in -- the panel's own cookie -- and
    # keep it out of your shell history:
    #     read -rs HELENA_COOKIE && export HELENA_COOKIE
    parser.add_argument("--cookie", default=None,
                        help="the panel session cookie; $HELENA_COOKIE if unset")
    arguments = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed: pip install playwright "
              "&& python3 -m playwright install chromium", file=sys.stderr)
        return 2

    assets = arguments.out or ASSETS
    assets.mkdir(parents=True, exist_ok=True)
    wanted = {name: shot for name, shot in SHOTS.items()
              if not arguments.only or name in arguments.only}
    failures: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(
            viewport={"width": arguments.width, "height": arguments.height},
            device_scale_factor=2,          # legible when the page is scaled down
            ignore_https_errors=arguments.insecure,
            color_scheme="dark",            # the panel's own default
        )
        cookie = arguments.cookie or os.environ.get("HELENA_COOKIE")
        if cookie:
            host = urlparse(arguments.panel).hostname or "127.0.0.1"
            context.add_cookies([{ "name": "helena_session", "value": cookie,
                                   "domain": host, "path": "/" }])
        page = context.new_page()
        for name, (path, selector, _why) in wanted.items():
            url = arguments.panel.rstrip("/") + path
            try:
                page.goto(url, wait_until="networkidle", timeout=30_000)
            except Exception as failure:  # noqa: BLE001 - reported per shot
                failures.append(f"{name}: {url} did not load ({failure})")
                continue
            target = None
            if selector:
                for one in selector.split(","):
                    found = page.query_selector(one.strip())
                    if found:
                        target = found
                        break
                if target is None:
                    failures.append(
                        f"{name}: none of {selector!r} is on {url} -- refusing "
                        "to save the whole page under a name that promises a "
                        "part of it")
                    continue
            if page.query_selector("input[type=password]"):
                failures.append(
                    f"{name}: this panel wants a session, and what would have "
                    "been saved is a picture of the sign-in box. Pass --cookie "
                    "or set HELENA_COOKIE.")
                continue
            (target or page).screenshot(path=str(assets / name))
            print(f"  {name}")
        browser.close()

    for failure in failures:
        print(f"missing: {failure}", file=sys.stderr)
    print(f"{len(wanted) - len(failures)}/{len(wanted)} shots -> {assets}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
