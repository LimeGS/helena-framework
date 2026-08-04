#!/usr/bin/env python3
"""Helena Framework control panel -- JSON API and SPA host.

Reads what the pipeline already writes -- receipts on disk, ink lane profiles
under git, the fleet tables in PostgreSQL, nvidia-smi -- and never invents
state. See the panel's developer reference.

The map endpoint serves *luminance*, not colour. The client uploads it once as
a texture and applies the threshold and the colour ramp in a shader, so moving
the threshold slider costs no request at all. And it serves the visible window
resampled to screen resolution rather than the whole array, which is what keeps
a 4096x4096 float32 map (67 MiB) off the wire.

Run:
    uvicorn panel.app:app --host 0.0.0.0 --port 8800
"""

from __future__ import annotations

import functools
import hashlib
import io
import ipaddress
import json
import os
import re
import secrets
import socket
import shutil
import subprocess
import tempfile
import sys
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlparse

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from framework.contracts import config_version  # noqa: E402
from framework.contracts import artifact as artifact_contract  # noqa: E402
from framework.contracts import auth as auth_contract  # noqa: E402
from framework.contracts import images as image_contract  # noqa: E402
from framework.contracts import mission as mission_contract  # noqa: E402

HERE = Path(__file__).resolve().parent


SETTINGS: list[dict] = []

# Runtime overrides live in one JSON file so a value changed from the panel
# survives a restart without anybody editing a unit file. Precedence is
# override, then environment, then the built-in default -- and every setting
# reports which of the three it came from, because a value whose origin is
# unclear is worse than no value.
OVERRIDES_PATH = Path(os.environ.get("CX_OVERRIDES", str(HERE / "settings.local.json")))
# Every edit writes a whole new configuration version here, with its own id and
# a sha256 over the entire settings map.
VERSIONS_ROOT = Path(os.environ.get("CX_CONFIG_VERSIONS", str(HERE / "config-versions")))


def _overrides() -> dict[str, str]:
    try:
        return json.loads(OVERRIDES_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _setting(name: str, default: str, doc: str, *, kind: str = "path",
             allowed: list[str] | None = None, example: str = "",
             restart: bool = False, secret: bool = False) -> str:
    """Read one knob and remember where its value came from and what it accepts."""
    override = _overrides().get(name)
    raw = os.environ.get(name)
    if override is not None:
        value, source = override, "override"
    elif raw is not None:
        value, source = raw, "environment"
    else:
        value, source = default, "default"
    SETTINGS.append({
        "name": name, "value": value, "default": default, "source": source,
        "doc": doc, "kind": kind, "allowed": allowed, "example": example,
        "requires_restart": restart, "secret": secret,
    })
    return value

REPO = Path(_setting(
    "CX_REPO", str(HERE.parent),
    "Repository root the panel reads contracts, profiles and registries from. "
    "Everything relative resolves against it.",
    example="/srv/helena/framework", restart=True))
RUNS = Path(_setting(
    "CX_RUNS", "/srv/helena/runs",
    "Parent directory of the missions. Each mission is a subdirectory with a "
    "MISSION.json; runs directly inside are reported as the implicit "
    "'unfiled' mission.",
    example="/srv/helena/runs"))
ARTIFACTS = Path(_setting(
    "CX_ARTIFACTS", "/artifacts",
    "Where surfaces published over the network are kept. Object storage is "
    "optional: with no s3:// prefix configured, workers publish here and this "
    "volume is the deployment's copy of record.",
    example="/artifacts"))
DSN = _setting(
    "CX_DB", "",
    "PostgreSQL connection string for the fleet and the job queue. Empty "
    "disables the Fleet page and the command plane, which then explain what is "
    "missing rather than failing.",
    kind="dsn", example="postgresql://user:password@127.0.0.1:5432/campaignx",
    secret=True, restart=True)
CACHE = Path(_setting(
    "CX_CACHE", "/tmp/helena-panel",
    "Writable directory for cached inventories. Losing it costs one refetch.",
    example="/var/cache/campaignx"))
SCROLL_SOURCE = _setting(
    "CX_SCROLL_SOURCE",
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/",
    "Bucket listing that enumerates every scroll. The panel is not limited to "
    "the priority cohort; this is where the full inventory comes from.",
    kind="url", example="https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/")
SCROLL_TTL = int(_setting(
    "CX_SCROLL_TTL", "86400",
    "Seconds before the scroll inventory is refetched. The listing changes on "
    "the scale of months, so a day is generous.",
    kind="int", example="86400"))
CATALOG = Path(_setting(
    "CX_CATALOG", str(HERE.parent / "workspace" / "catalog" / "eligible_volumes.json"),
    "Frozen catalog naming the priority cohort and its physical scales. Every "
    "other scroll stays browsable; this only decides which are flagged.",
    example="workspace/catalog/eligible_volumes.json"))
GEOMETRY_CATALOG = Path(_setting(
    "CX_GEOMETRY_CATALOG",
    str(
        HERE.parent
        / "workspace/catalog/geometry_surface_catalog_v2/"
        "GEOMETRY_SURFACE_CATALOG.sqlite"
    ),
    "Frozen geometry surface catalog imported by the segmentation fleet.",
    example=(
        "workspace/catalog/geometry_surface_catalog_v2/"
        "GEOMETRY_SURFACE_CATALOG.sqlite"
    ),
    restart=True,
))
_SEED_PROBE_BENCHMARK_RECEIPT = _setting(
    "CX_SEED_PROBE_BENCHMARK_RECEIPT",
    "",
    "Local immutable APPROVED_SELECT benchmark decision used to authorize "
    "production seed-probe select. Empty keeps select unavailable.",
    example="/srv/helena/approvals/SEED_PROBE_BENCHMARK_DECISION.json",
    restart=True,
)
SEED_PROBE_BENCHMARK_RECEIPT = (
    Path(_SEED_PROBE_BENCHMARK_RECEIPT)
    if _SEED_PROBE_BENCHMARK_RECEIPT.strip()
    else None
)
SEED_PROBE_REVIEW_OWNER = _setting(
    "CX_SEED_PROBE_REVIEW_OWNER",
    "",
    "Named operator or queue responsible for resolving seed-probe abstentions. "
    "Empty keeps production select unavailable.",
    example="segmentation-review-oncall",
    restart=True,
).strip()
PROFILE_DIR = Path(_setting(
    "CX_PROFILE_DIR", str(HERE.parent / "framework" / "profiles" / "03-ink"),
    "Directory of ink lane profiles the launcher offers. A detector is a "
    "profile in here, not a script.",
    example="framework/profiles/03-ink"))
AUTH_ROOT = Path(_setting(
    "CX_AUTH", str(HERE / "auth"),
    "Where panel accounts and sessions live. Holds password hashes, so it is "
    "written 0600 and belongs on local disk, not a share.",
    example="panel/auth"))
REQUIRE_LOGIN = _setting(
    "CX_REQUIRE_LOGIN", "true",
    "Whether the panel requires an account. Turning this off on a host that is "
    "reachable from anywhere hands the queue to whoever finds it.",
    kind="bool", allowed=("true", "false")).lower() != "false"
# Read here only so the session cookie knows whether it may be Secure. uvicorn
# is what actually serves the certificate, from the same two paths.
TLS_CERT = _setting(
    "CX_TLS_CERT", "",
    "The certificate uvicorn serves. Set by the panel's start script, which "
    "generates a self-signed pair on first boot if none is supplied; point it "
    "at a real certificate to replace it. Empty means plain http.")
TLS_KEY = _setting(
    "CX_TLS_KEY", "",
    "The private key for CX_TLS_CERT.")
STRIPS = Path(_setting(
    "CX_STRIPS", str(HERE.parent / "workspace" / "strips"),
    "Where qualified reference strips live. A strip is one strip-v0 .npz plus "
    "its qualification report; scoring reads from here.",
    example="workspace/strips"))
REFERENCE_STRIPS = Path(_setting(
    "CX_REFERENCE_STRIPS", str(HERE.parent / "framework" / "vendored" / "reference-strips"),
    "The vendored reference-strips checkout that qualifies and scores.",
    example="framework/vendored/reference-strips"))
REGISTRY = Path(_setting(
    "CX_REGISTRY", str(HERE.parent / "framework" / "registries" / "method-capabilities-0.1.0.json"),
    "Method capability registry. It is what disqualifies a method: a profile "
    "whose method is DISQUALIFIED here is refused before any GPU time.",
    example="framework/registries/method-capabilities-0.1.0.json"))
MAP_MAX_SIZE = int(_setting(
    "CX_MAP_MAX_SIZE", "4096",
    "Largest raster edge the map endpoint will render. Raising it lets the "
    "viewer request more texels than most screens can show.",
    kind="int", allowed=["512", "1024", "2048", "4096", "8192"], example="4096"))
DIST = HERE / "web" / "dist"

# Upstream input contract (villa ink-detection/optimized_inference/inference.py):
# the tile is clipped to CFG.max_clip_value and then divided by that same value.
UPSTREAM_CLIP = 200

# FastAPI claims /docs for Swagger by default, which is exactly the path the
# SPA uses for the framework reference. Swagger moves out of the way.
# openapi_url under /api/ because the SPA serves a catch-all at /{full_path},
# and it answered /openapi.json with index.html -- a 200 full of HTML, which is
# the worst shape of wrong for something a client parses as JSON.
app = FastAPI(title="Helena Framework", docs_url="/api/swagger",
              openapi_url="/api/openapi.json", redoc_url=None)

# The framework reference is a ~570 KB JSON of signatures and docstrings. It
# compresses to a fraction of that and is the only large payload the API emits,
# so compression is worth more here than anywhere else in the app.
app.add_middleware(GZipMiddleware, minimum_size=1024)


# The paths that must answer before anybody is logged in. Everything else --
# every API route, every page -- needs a session. Deny by default: a list of
# what is protected goes stale the moment somebody adds a route, and the one
# they forget is the one that queues GPU work.
OPEN_PATHS = frozenset({"/api/session", "/api/session/bootstrap", "/login"})

# The only paths a machine token opens. A tuple because it is matched with
# startswith, and deliberately short: a worker needs to publish and fetch
# artifacts, and nothing else. Widening this is widening what a token copied off
# a worker host can do.
MACHINE_PATHS = ("/api/artifacts/",)


AUDIT_ROOT = Path(_setting(
    "CX_AUDIT", str(Path(_setting("CX_AUTH", "panel/auth",
                                  "Where accounts live.")).parent / "audit"),
    "Where the audit trail is appended. One file per month of JSON lines, under "
    "the panel's state directory so the backup already carries it."))

# Reading changes nothing and a log of reads is a log nobody searches. What is
# recorded is every request that could change something -- including the ones
# that were refused, because an attempt that failed is the half of an audit
# trail people actually go looking for.
AUDITED_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def record_audit(*, user: str | None, method: str, path: str, query: str,
                 status: int, milliseconds: int, client: str | None) -> None:
    """Append one line to the trail. Never raises: an audit write that can fail
    the request it describes would be a reason to turn auditing off.

    Bodies are deliberately not captured. One of these routes sets S3
    credentials and another sets a password; a trail that recorded what was sent
    would be the most sensitive file on the machine, and "who changed the
    secrets, and when" is the question an audit answers.
    """
    try:
        AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
        moment = datetime.now(timezone.utc)
        line = json.dumps({
            "id": uuid.uuid4().hex[:16],
            "at": moment.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "user": user or "anonymous",
            "action": f"{method} {path}",
            "method": method, "path": path,
            "query": query or None,
            "status": status, "ms": milliseconds, "client": client,
        }, separators=(",", ":"))
        with (AUDIT_ROOT / f"{moment:%Y-%m}.jsonl").open("a", encoding="utf-8") as trail:
            trail.write(line + "\n")
    except Exception as failure:  # noqa: BLE001
        print(f"audit not recorded: {type(failure).__name__}: {failure}",
              file=sys.stderr, flush=True)


def read_audit(*, limit: int = 200, user: str | None = None,
               contains: str | None = None) -> list[dict]:
    """The trail, newest first. Reads whole months rather than seeking: a month
    of mutations on this platform is thousands of lines, not millions."""
    entries: list[dict] = []
    if not AUDIT_ROOT.is_dir():
        return entries
    for month in sorted(AUDIT_ROOT.glob("*.jsonl"), reverse=True):
        for raw in reversed(month.read_text(encoding="utf-8").splitlines()):
            try:
                entry = json.loads(raw)
            except ValueError:
                continue
            if user and entry.get("user") != user:
                continue
            if contains and contains.lower() not in entry.get("action", "").lower():
                continue
            entries.append(entry)
            if len(entries) >= limit:
                return entries
    return entries


# Deliberately generous. What this has to stop is somebody working through a
# password list; what it must never do is get in the way of a harness that signs
# in on every run, or of a team behind one address. So: only failures count,
# only the credential endpoint is guarded, and a success clears the record.
#
# ponytail: in-process dict, which is per-worker. The panel runs one uvicorn
# worker; if that ever changes this moves to the control plane.
LOGIN_WINDOW_SECONDS = 300
LOGIN_FAILURES_PER_ACCOUNT = 10
LOGIN_FAILURES_PER_CLIENT = 40
_login_failures: dict[str, list[float]] = {}


def _recent_failures(key: str, now: float) -> list[float]:
    kept = [t for t in _login_failures.get(key, []) if now - t < LOGIN_WINDOW_SECONDS]
    if kept:
        _login_failures[key] = kept
    else:
        _login_failures.pop(key, None)
    return kept


def login_refusal(account: str, client: str | None) -> tuple[str, int] | None:
    """Why this attempt should not be tried, and for how many seconds.

    Keyed on the account first: one person fumbling their password must not
    lock out a colleague on the same address, and an attacker who cannot see
    which usernames exist gains nothing from spreading the guesses. The
    per-client ceiling is four times as loose and catches the spray.
    """
    now = time.time()
    for key, ceiling, what in ((f"account:{account}", LOGIN_FAILURES_PER_ACCOUNT,
                                "this account"),
                               (f"client:{client or 'unknown'}",
                                LOGIN_FAILURES_PER_CLIENT, "this address")):
        attempts = _recent_failures(key, now)
        if len(attempts) >= ceiling:
            wait = int(LOGIN_WINDOW_SECONDS - (now - attempts[0])) + 1
            return (f"{ceiling} failed sign-ins for {what} in the last "
                    f"{LOGIN_WINDOW_SECONDS // 60} minutes", wait)
    return None


def record_login_failure(account: str, client: str | None) -> None:
    now = time.time()
    for key in (f"account:{account}", f"client:{client or 'unknown'}"):
        _login_failures.setdefault(key, []).append(now)


def clear_login_failures(account: str) -> None:
    """A correct password ends the matter. Someone who has just proved who they
    are should not be serving out a lockout earned by their own typing."""
    _login_failures.pop(f"account:{account}", None)


# ---------------------------------------------------------------------------
# Modules: one vocabulary for every way a phase can be done
#
# The platform grew three separate extension mechanisms -- lanes for most
# phases, profiles routing to adapters for ink, seeders and backends for
# segmentation -- and each is right for what it does. What was missing is a
# single answer to "what can this phase run, and is it turned on".
#
# A module is that answer. It is not a fourth mechanism: it is the existing
# ones, reported in one shape and switchable in one place.
MODULES_PATH = Path(_setting(
    "CX_MODULES", str(Path(_setting("CX_AUTH", "panel/auth",
                                    "Where accounts live.")).parent / "modules.json"),
    "Which modules are switched off, and any module added through the panel. "
    "Under the state directory, so the backup carries it."))


def _module_state() -> dict:
    try:
        return json.loads(MODULES_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {"disabled": [], "added": []}


def _write_module_state(state: dict) -> None:
    MODULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODULES_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def module_disabled(phase: str, module_id: str) -> bool:
    return f"{phase}:{module_id}" in set(_module_state().get("disabled", []))


def phase_modules(phase: str) -> list[dict]:
    """Everything this phase can be done with, whatever mechanism provides it.

    `kind` says which mechanism, because they are not interchangeable and
    somebody integrating needs to know which contract they are writing to:

        lane     a program the queue starts, declared in PHASE_LANES
        profile  a model with its weights and scale, routed to an adapter
        backend  what grows a surface, for segmentation
        seeder   what chooses the point to grow from
        source   where the frozen catalog is read from
    """
    off = set(_module_state().get("disabled", []))
    out: list[dict] = []

    def add(module_id: str, kind: str, name: str, note: str | None = None,
            removable: bool = False, detail: dict | None = None) -> None:
        out.append({
            "id": module_id, "phase": phase, "kind": kind, "name": name,
            "note": note, "enabled": f"{phase}:{module_id}" not in off,
            "removable": removable, **(detail or {}),
        })

    if phase == "P0":
        add(SCROLL_SOURCE or "default", "source", "Catalog source",
            "Where the frozen scroll inventory is read from. Set with "
            "CX_SCROLL_SOURCE.")
        return out

    if phase == "P1":
        for backend in SEGMENTATION_BACKENDS:
            add(backend["id"], "backend", backend["name"], backend.get("note"),
                detail={"adoptable": backend.get("adoptable", False)})
        for seeder in SEGMENTATION_SEEDERS:
            add(seeder["id"], "seeder", seeder["name"], seeder.get("note"),
                detail={"repeatable": seeder.get("repeatable", False)})
        return out

    try:
        import job_store  # type: ignore

        lanes = job_store.PHASE_LANES.get(phase) or {}
    except ImportError:
        # The queue module ships with the worker runtime. Without it the panel
        # can still list the phases it knows about, which is what a deployment
        # that has not built a worker image yet looks like.
        lanes = {}
    for lane_id, lane in lanes.items():
        add(lane_id, "lane", lane.get("name", lane_id), lane.get("note"),
            detail={"runner": lane.get("runner")})

    if phase == "P5":
        for profile in ink_profiles():
            add(profile["profile_id"], "profile", profile["profile_id"],
                profile.get("known_limits"),
                removable=bool(profile.get("added_through_panel")),
                detail={"adapter": profile.get("adapter"),
                        "source": profile.get("source")})
    return out


def is_loopback(request: Request) -> bool:
    """Whether this came from the host itself.

    Only used to gate the very first account. X-Forwarded-For is deliberately
    not consulted: a proxy header is set by whoever is talking to us.
    """
    client = request.client.host if request.client else ""
    return client in ("127.0.0.1", "::1", "localhost")


@app.middleware("http")
async def require_session(request: Request, call_next):
    """The gate, and the trail. Wrapping rather than sprinkling: a route added
    next month is audited because it is a route, not because somebody
    remembered to decorate it."""
    started = time.monotonic()
    response = await gate(request, call_next)
    if request.method in AUDITED_METHODS and request.url.path.startswith("/api/"):
        record_audit(user=getattr(request.state, "username", None),
                     method=request.method, path=request.url.path,
                     query=str(request.url.query), status=response.status_code,
                     milliseconds=int((time.monotonic() - started) * 1000),
                     client=request.client.host if request.client else None)
    return response


async def gate(request: Request, call_next):
    """No session, no panel.

    Static assets are open because the login page is one of them, and an
    unauthenticated request for a hashed JS chunk reveals nothing that the
    login page does not. Everything that reads or changes anything is closed.
    """
    path = request.url.path
    if (not REQUIRE_LOGIN
            or path in OPEN_PATHS
            or path.startswith("/assets/")
            # The identity: an icon set and a manifest, which the browser asks
            # for before anybody has signed in. A login page with a broken
            # favicon is a login page that looks like the wrong site.
            or path in ("/favicon.ico", "/favicon.svg", "/robots.txt",
                        "/site.webmanifest", "/apple-touch-icon.png")
            or (path.startswith("/favicon-") and path.endswith(".png"))):
        return await call_next(request)

    who = auth_contract.whoami(AUTH_ROOT, request.cookies.get(auth_contract.COOKIE))
    # A worker is not a person and has no browser to hold a cookie. It presents
    # a named, revocable machine token instead, and is recorded under that name
    # -- so the audit log says "gpu-1-segment" rather than whichever human's
    # password would otherwise have been copied onto a worker host.
    #
    # Machine tokens reach the artifact endpoints and nothing else. They exist
    # so a remote worker can publish a surface; they are not a second way to
    # drive the panel, and a leaked one should not be able to queue GPU work or
    # read somebody's missions.
    if who is None and path.startswith(MACHINE_PATHS):
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            machine = auth_contract.whoami_machine(AUTH_ROOT, header[7:].strip())
            if machine is not None:
                request.state.username = f"machine:{machine}"
                return await call_next(request)
    request.state.username = who
    if who is None:
        if path.startswith("/api/"):
            return JSONResponse(
                {"detail": "not signed in", "bootstrap_available":
                    not auth_contract.any_users(AUTH_ROOT)},
                status_code=401)
        # A page request: serve the app, which shows the login screen. A
        # redirect would lose the address somebody typed.
        return FileResponse(DIST / "index.html")
    request.state.username = who
    return await call_next(request)


# --------------------------------------------------------------------------
# Receipt index
#
# ponytail: the design calls for a PostgreSQL receipts table. There are six run
# directories. A filesystem scan behind an mtime cache is the same answer for
# three orders of magnitude less machinery -- add the table when a scan stops
# being instant, not before.
# --------------------------------------------------------------------------

_cache: dict[str, Any] = {"stamp": None, "runs": []}


@dataclass
class Run:
    run_id: str
    path: Path
    receipt_path: Path
    receipt: dict
    schema: str
    sample_id: str
    lane_id: str
    checkpoint_sha: str
    generated_at: str
    stats: dict[str, float]
    clip_value: int | None
    divisor: int | None
    normalization: str
    maps: list[str] = field(default_factory=list)
    mission_id: str = "unfiled"

    @property
    def contract_ok(self) -> bool:
        if self.clip_value is None or self.divisor is None:
            return True
        return self.clip_value == self.divisor

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "schema": self.schema,
            "sample_id": self.sample_id,
            "lane_id": self.lane_id,
            "checkpoint_sha": self.checkpoint_sha,
            "generated_at": self.generated_at,
            "stats": self.stats,
            "clip_value": self.clip_value,
            "divisor": self.divisor,
            "normalization": self.normalization,
            "maps": self.maps,
            "contract_ok": self.contract_ok,
            "liveness": self.receipt.get("liveness"),
            "receipt_path": str(self.receipt_path),
            "mission_id": self.mission_id,
        }


def _dig(d: dict, *paths, default=None):
    for path in paths:
        node: Any = d
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if node is not None:
            return node
    return default


def _parse_normalization(text: str) -> tuple[int | None, int | None]:
    """Pull (clip, divisor) out of prose like 'clamp(0,200)/255'."""
    import re

    m = re.search(r"cl(?:amp|ip)\(\s*0\s*,\s*(\d+)\s*\)\s*(?:then divide by\s*|/)\s*(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"cl(?:amp|ip)\(0,(\d+)\)", text)
    if m:
        return int(m.group(1)), None
    return None, None


def _load_receipt(path: Path) -> Run | None:
    try:
        receipt = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    stats = _dig(receipt, "statistics", default={}) or {}
    normalization = str(
        _dig(receipt, "input.normalization", "physical_normalization.model_input_normalization",
             "model_input_normalization", default="")
    )
    clip, divisor = _parse_normalization(normalization)
    if clip is None:
        clip = _dig(receipt, "input.max_clip_value", "lane.max_clip_value")

    return Run(
        run_id=path.parent.name,
        path=path.parent,
        receipt_path=path,
        receipt=receipt,
        # The timesformer lane predates the `schema` convention and names its
        # receipt type `kind`. Both are the same field to a reader.
        schema=str(receipt.get("schema") or receipt.get("kind") or "?"),
        sample_id=str(receipt.get("sample_id", "?")),
        lane_id=str(_dig(receipt, "lane.profile_id", "lane.method_id", "method_id",
                         "checkpoint.method_identity.method_id", default="?")),
        checkpoint_sha=str(_dig(receipt, "lane.checkpoint_sha256", "checkpoint_sha256",
                                "checkpoint.sha256", default="")),
        generated_at=str(receipt.get("generated_at_utc", "")),
        stats={k: float(v) for k, v in stats.items() if isinstance(v, (int, float))},
        clip_value=clip,
        divisor=divisor,
        normalization=normalization,
        maps=sorted(p.name for p in path.parent.glob("*.npy")),
    )


def _scan(directory: Path, mission_id: str) -> list[Run]:
    runs = []
    for entry in sorted(directory.iterdir()):
        if not entry.is_dir():
            continue
        for receipt in sorted(entry.glob("*RECEIPT*.json")):
            run = _load_receipt(receipt)
            if run:
                run.mission_id = mission_id
                runs.append(run)
                break
    return runs


def index_runs(force: bool = False, mission_id: str | None = None) -> list[Run]:
    """Every run, or every run of one mission.

    Runs live at CX_RUNS/<mission>/<run>/. Runs directly under CX_RUNS predate
    missions and are attributed to the implicit "unfiled" mission rather than
    being hidden.
    """
    if not RUNS.exists():
        return []
    stamp = tuple(sorted(
        (p.name, p.stat().st_mtime) for p in RUNS.rglob("*") if p.is_dir()
    ))
    if not force and _cache["stamp"] == stamp:
        runs = _cache["runs"]
    else:
        runs = _scan(RUNS, "unfiled")
        for entry in sorted(RUNS.iterdir()):
            if entry.is_dir() and (entry / mission_contract.MANIFEST_NAME).exists():
                runs += _scan(entry, entry.name)
        runs.sort(key=lambda r: r.generated_at, reverse=True)
        _cache.update(stamp=stamp, runs=runs)
    if mission_id:
        return [r for r in runs if r.mission_id == mission_id]
    return runs


def get_run(run_id: str) -> Run:
    for run in index_runs():
        if run.run_id == run_id:
            return run
    raise HTTPException(404, f"no run {run_id!r}")


# --------------------------------------------------------------------------
# Profiles, registry, catalog
# --------------------------------------------------------------------------

def ink_profiles() -> list[dict]:
    out = []
    for path in sorted(PROFILE_DIR.glob("*.json")):
        try:
            profile = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if profile.get("schema") == "campaignx.ink_lane_profile.v1":
            profile["_path"] = str(path.relative_to(REPO))
            out.append(profile)
    return out


def registry_entries() -> dict[str, dict]:
    if not REGISTRY.exists():
        return {}
    data = json.loads(REGISTRY.read_text())
    entries = data.get("ink") or data.get("entries") or []
    return {e["method_id"]: e for e in entries if "method_id" in e}


def catalog_metadata() -> dict[str, dict]:
    """Physical scale per scroll, from the frozen catalog.

    This used to define a priority cohort. Missions do that now -- a mission
    names the scrolls it attempts and freezes that selection -- so the catalog
    is left doing the one thing only it can: telling you the voxel size and
    beam energy of a scan, which every micron figure downstream depends on.
    """
    if not CATALOG.exists():
        return {}
    out = {}
    for entry in json.loads(CATALOG.read_text()).get("entries", []):
        uri = entry.get("ct_uri", "")
        sample = uri.split("/")[3] if uri.count("/") > 3 else None
        if not sample:
            continue
        scan = entry.get("eligible_scan_id", "")
        out[sample] = {
            "pixel_um": next((p.replace("um", "") for p in scan.split("-") if p.endswith("um")), ""),
            "energy_kev": entry.get("energy_kev"),
            "scan_id": scan,
            "ct_uri": uri,
            "higher_res": bool(entry.get("higher_resolution_sibling_uri")),
        }
    return out


# The bucket is laid out by convention, and discovery depends on it holding:
#
#   <sample_id>/volumes/<timestamp>-<pixel>um[-<distance>m]-<energy>keV[-masked].zarr/
#
# A top-level prefix is a scroll only if it has scans under volumes/. That test
# is what separates a scroll from the bucket's other top-level entries -- the
# live bucket carries a `_thumbnails/` prefix that was being listed as a scroll
# because the old code trusted the top level alone. It also makes the source
# swappable: any bucket in this layout enumerates correctly, and one that is not
# in this layout reports zero scrolls rather than a list of its directories.
#
# The distance field is optional (PHerc0172 has none), so the scan name is
# parsed by looking for the token that ends in `um` and the one that ends in
# `keV`, never by position.
SCAN_SUFFIX = ".zarr"


class NoRedirect(__import__("urllib.request", fromlist=["HTTPRedirectHandler"]).HTTPRedirectHandler):
    """A listing does not redirect; a redirect here is a way around the host check."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, msg, headers, newurl
        raise OSError(f"the source redirected ({code}); listings are read from the URL given")


def scan_date(scan_id: str) -> str | None:
    """The day a scan was taken, from the timestamp its name starts with.

    Derived at read time rather than trusted from the parsed record, because a
    cached inventory written before this existed has the scan ids and not the
    field -- and the answer is in the id either way. The listing carries no other
    date, so this is the only source for "since when do we have this scroll".
    """
    head = scan_id.split("-")[0] if scan_id else ""
    if len(head) != 14 or not head.isdigit():
        return None
    return f"{head[0:4]}-{head[4:6]}-{head[6:8]}"


def parse_scan_name(scan: str) -> dict:
    """Physical scale out of a scan directory name, by token not by position."""
    tokens = scan.removesuffix(SCAN_SUFFIX).split("-")
    micron = next((t.removesuffix("um") for t in tokens if t.endswith("um")), "")
    kev = next((t.removesuffix("keV") for t in tokens if t.endswith("keV")), "")
    try:
        energy = float(kev) if kev else None
    except ValueError:
        energy = None
    # When the scan was made, from the timestamp the name starts with. The
    # inventory is an S3 listing and carries no other date, and "when did this
    # scroll become available to us" is a question the page could not answer at
    # all -- so it comes from the one place the answer already exists.
    scan_id = scan.removesuffix(SCAN_SUFFIX)
    return {"scan_id": scan_id, "pixel_um": micron, "energy_kev": energy,
            "scanned_on": scan_date(scan_id)}


def list_prefixes(source: str, prefix: str = "", timeout: int = 25) -> list[str]:
    """One ListObjectsV2 page of common prefixes, deepest component only."""
    import urllib.request
    import xml.etree.ElementTree as ET

    query = f"?list-type=2&delimiter=/&max-keys=1000&prefix={quote(prefix)}"
    # Redirects are not followed: the address check happens before the fetch, and
    # a 302 to a link-local address would step around it entirely.
    opener = urllib.request.build_opener(NoRedirect)
    with opener.open(source.rstrip("/") + "/" + query, timeout=timeout) as response:
        root = ET.fromstring(response.read())
    depth = prefix.count("/")
    return [
        element.text.rstrip("/").split("/")[-1]
        for element in root.iter()
        if element.tag.endswith("Prefix")
        and element.text
        and element.text.rstrip("/").count("/") == depth
    ]


def check_source_is_fetchable(source: str) -> None:
    """Refuse a source that would turn the panel into a proxy for its own network.

    The listing is fetched by the server, so a caller-supplied URL is a
    server-side request to wherever they name. This panel binds 0.0.0.0 with no
    authentication, which makes that a way to read anything it can reach and
    nobody else can -- link-local metadata endpoints and internal services
    first among them. https only, public addresses only, every resolved address
    checked rather than just the first.
    """
    parsed = urlparse(source)
    if parsed.scheme != "https":
        raise HTTPException(400, "the source must be an https:// URL")
    if not parsed.hostname:
        raise HTTPException(400, "the source has no host")
    try:
        resolved = socket.getaddrinfo(parsed.hostname, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise HTTPException(400, f"the source host does not resolve: {exc}") from exc
    for family, _type, _proto, _canon, address in resolved:
        del family, _type, _proto, _canon
        ip = ipaddress.ip_address(address[0])
        if not ip.is_global:
            raise HTTPException(
                400,
                f"{parsed.hostname} resolves to {ip}, which is not a public address; "
                "the panel will not fetch a listing from its own network",
            )


def scroll_inventory(refresh: bool = False, source: str | None = None) -> dict:
    """Every scroll a bucket exposes, discovered by the volumes/ convention.

    Cached on disk because the listing changes on the scale of months and the
    panel must render with no network at all. Falls back, in order, to the
    cache and then to the priority cohort, and says which of those it used.

    A custom source is never cached to the default file and never falls back to
    the priority cohort: those scrolls belong to the open-data bucket, and
    showing them as the contents of somebody else's bucket would be a lie.
    """
    import urllib.error
    import xml.etree.ElementTree as ET

    custom = source is not None and source.rstrip("/") != SCROLL_SOURCE.rstrip("/")
    source = source or SCROLL_SOURCE
    cache_file = CACHE / ("scroll_inventory.json" if not custom else
                          f"scroll_inventory-{hashlib.sha256(source.encode()).hexdigest()[:16]}.json")
    if not refresh and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if time.time() - cached.get("fetched_at", 0) < SCROLL_TTL:
                return cached
        except (json.JSONDecodeError, OSError):
            pass

    try:
        candidates = [p for p in list_prefixes(source) if p]
        # 45 scrolls means 45 probes; serially that is a minute of latency for
        # something the user pressed a button for. They are independent reads.
        scans: dict[str, list[dict]] = {}
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = {
                pool.submit(list_prefixes, source, f"{name}/volumes/", 20): name
                for name in candidates
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    found = future.result()
                except (urllib.error.URLError, ET.ParseError, OSError, ValueError):
                    continue
                found = [s for s in found if s.endswith(SCAN_SUFFIX)]
                if found:
                    scans[name] = [parse_scan_name(s) for s in sorted(found)]

        payload = {
            "scrolls": sorted(scans),
            "scans": scans,
            "skipped": sorted(set(candidates) - set(scans)),
            "fetched_at": time.time(),
            "source": source,
            "origin": "Custom S3 bucket" if custom else "Official Open Data S3 Bucket",
        }
        try:
            CACHE.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(payload))
        except OSError:
            payload["origin"] += " (not cached: cache directory not writable)"
        return payload
    except (urllib.error.URLError, ET.ParseError, OSError, ValueError) as exc:
        if cache_file.exists():
            try:
                stale = json.loads(cache_file.read_text())
                stale["origin"] = f"Stale cache — the source did not answer ({type(exc).__name__})"
                return stale
            except (json.JSONDecodeError, OSError):
                pass
        if custom:
            return {"scrolls": [], "scans": {}, "skipped": [], "fetched_at": 0.0,
                    "source": source,
                    "origin": f"Unreachable ({type(exc).__name__}: {exc})"}
        return {"scrolls": sorted(catalog_metadata()), "scans": {}, "skipped": [],
                "fetched_at": 0.0, "source": source,
                "origin": f"Priority cohort only ({type(exc).__name__}: {exc})"}


def scrolls(refresh: bool = False, source: str | None = None) -> dict:
    """Every known scroll, joined with the cohort flag and whatever has been run."""
    inventory = scroll_inventory(refresh=refresh, source=source)
    catalog = catalog_metadata()
    runs = index_runs()
    by_sample: dict[str, list[Run]] = {}
    for run in runs:
        by_sample.setdefault(run.sample_id, []).append(run)

    # The default view unions the bucket with the catalog and with whatever has
    # been run, so a scroll with local work never disappears from the list. A
    # custom source shows only what that bucket holds: the catalog describes the
    # open data, and listing its scrolls under someone else's URL would claim
    # they are there.
    if source is None:
        names = sorted(set(inventory["scrolls"]) | set(catalog) | set(by_sample))
    else:
        names = sorted(inventory["scrolls"])
    rows = []
    for sample in names:
        mine = by_sample.get(sample, [])
        best = max(mine, key=lambda r: r.stats.get("p90", 0.0), default=None)
        meta = catalog.get(sample, {})
        # The catalog is frozen and hash-locked, so it wins where it speaks. It
        # describes 13 of 45 scrolls; the rest get their scale from the scan
        # directory name, which the listing already returned.
        scans = inventory.get("scans", {}).get(sample, [])
        finest = min((s for s in scans if s["pixel_um"]),
                     key=lambda s: float(s["pixel_um"]), default=None)
        rows.append({
            "sample_id": sample,
            "pixel_um": meta.get("pixel_um") or (finest or {}).get("pixel_um", ""),
            "energy_kev": meta.get("energy_kev") or (finest or {}).get("energy_kev"),
            "scans": len(scans),
            # When this scroll first showed up, from the earliest scan's own
            # timestamp. The inventory is an S3 listing with no date of its own,
            # and the page could not answer "since when do we have this" at all.
            "scanned_on": min((d for d in (scan_date(s.get("scan_id", "")) for s in scans)
                               if d), default=None),
            "scale_from": "catalog" if meta.get("pixel_um") else ("scan name" if finest else ""),
            "higher_res": meta.get("higher_res", False),
            "runs": len(mine),
            "lane": best.lane_id if best else None,
            "p90": best.stats.get("p90") if best else None,
            "run_id": best.run_id if best else None,
            "verdict": "SCREENED" if mine else "NOT_SCREENED",
        })
    return {
        "scrolls": rows,
        "total": len(rows),
        "with_scale": sum(1 for r in rows if r["pixel_um"]),
        "screened_count": sum(1 for r in rows if r["runs"]),
        "inventory_origin": inventory.get("origin", "?"),
        # Top-level prefixes with no volumes/ -- the open bucket has six that
        # are photographs only. Reported rather than dropped quietly: a count
        # that shrinks with no explanation is its own kind of bug.
        "skipped": inventory.get("skipped", []),
        "fetched_at": inventory.get("fetched_at", 0.0),
    }


# Copied from fleet/store.py rather than imported: the stage lives under
# "01-segmentation", which is not an importable module name. test_terminal_states
# asserts the two agree, because a copy that drifts is how the "162 waiting" tile
# came to count every finished task as waiting.
SEGMENT_TERMINAL_STATES = (
    "ARCHIVED", "FIXTURE_ONLY", "QC_PENDING", "NO_SEED", "GROW_FAILED",
    "DUPLICATE_SURFACE", "FINALIZATION_FAILED", "POLICY_REJECTED",
    "BLOCKED_SOURCE_UNAVAILABLE", "PROBE_REVIEW_PENDING",
    "PROBE_REJECTED_ALL",
)


def stored_scroll(name: str | None) -> str | None:
    """A scroll name in the spelling the control plane stores.

    Everything that reaches this panel from outside names a scroll the way the
    bucket does -- PHerc0826 -- because that is what P0 read it as. Every row in
    the control plane names it the way the frozen catalog does -- PHerc826 --
    because that is what the fleet was bootstrapped from. Nothing reconciled the
    two, so every page that filtered by scroll matched no row and reported zero
    while the fleet held fifty-two surfaces.

    Applied at each point a request's scroll becomes a query, rather than at each
    endpoint: there are eight endpoints and four of them are one line long, and I
    fixed this three times in three of them before doing it here.
    """
    return catalog_sample_id(name, strict=False) if name else name


def mission_scrolls(mission_id: str | None) -> set[str]:
    """The scrolls a mission attempts, in the spelling the control plane stores.

    One place, because every page that scopes to a mission needs it and the two
    that already did had their own copy.

    Translated on the way out. A mission names scrolls the way the bucket does --
    PHerc0826 -- and the control plane stores them the way the frozen catalog
    does -- PHerc826. Comparing the two raw made every mission-scoped page read
    zero while the fleet held forty-two surfaces, which looks like an empty
    campaign and is a spelling.
    """
    if not mission_id:
        return set()
    try:
        _, manifest = mission_contract.resolve(RUNS, mission_id)
    except mission_contract.MissionError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {catalog_sample_id(scroll, strict=False)
            for scroll in (manifest.get("scrolls") or [])}


def read_scope(mission_id: str | None = None,
               sample: str | None = None) -> set[str] | None:
    """Resolve a read to the samples it is allowed to see.

    ``None`` means an explicitly global read.  An empty set means a scoped read
    whose mission contains no scrolls (or whose requested sample is outside the
    mission).  Keeping those values distinct is the isolation boundary: using
    ``scope or None`` turns a brand-new mission into a query over the whole
    control plane.

    ``unfiled`` is historical rather than an explicit P0 selection, but it is
    still a scope: its scrolls come from the receipts discovered by the mission
    contract.  It must never mean the whole control plane.
    """
    normalized = stored_scroll(sample)
    if mission_id:
        allowed = mission_scrolls(mission_id)
        if normalized:
            return {normalized} if normalized in allowed else set()
        return allowed
    if normalized:
        return {normalized}
    return None


def require_write_sample(mission_id: str | None, sample: str | None,
                         operation: str) -> str | None:
    """Bind a mutation to one scroll in its explicit mission.

    Reads can aggregate a mission.  A worker command cannot safely interpret an
    empty or multi-scroll scope when its CLI accepts one optional ``--sample``:
    omitting that flag means the whole fleet.  Resolve a single-scroll mission,
    require an explicit choice for a larger one, and reject a sample outside the
    mission before any job or subprocess is created.
    """
    normalized = stored_scroll(sample)
    if not mission_id:
        return normalized
    if mission_id == "unfiled":
        raise HTTPException(
            409, f"{operation} cannot run in unfiled: it is a read-only view of "
                 "historical receipts, not a declared P0 selection.")
    allowed = mission_scrolls(mission_id)
    if not allowed:
        raise HTTPException(
            409, f"{operation} cannot run: mission {mission_id} has no scrolls. "
                 "Select and freeze a scroll in P0 first.")
    if normalized:
        if normalized not in allowed:
            raise HTTPException(
                409, f"{operation} cannot run on {sample}: it is not selected "
                     f"in mission {mission_id}.")
        return normalized
    if len(allowed) == 1:
        return next(iter(allowed))
    raise HTTPException(
        409, f"{operation} needs one scroll; mission {mission_id} contains "
             f"{len(allowed)}. Choose a scroll first.")


def targets(mission_id: str | None = None) -> list[dict]:
    """The scrolls a mission attempts, in the shape the mission table wants.

    With no mission selected there is no cohort to fall back on -- scope is a
    mission's job now -- so this reports the scrolls that have runs, which is
    the only defensible answer to "what is being worked on".
    """
    rows = {r["sample_id"]: r for r in scrolls()["scrolls"]}
    if mission_id:
        try:
            _, manifest = mission_contract.resolve(RUNS, mission_id)
        except mission_contract.MissionError:
            return []
        return [rows.get(s, {"sample_id": s, "pixel_um": "", "energy_kev": None,
                             "runs": 0, "lane": None, "p90": None, "run_id": None,
                             "higher_res": False, "verdict": "NOT_SCREENED"})
                for s in manifest.get("scrolls", [])]
    return [r for r in rows.values() if r["runs"]]


# --------------------------------------------------------------------------
# Live host + fleet state
# --------------------------------------------------------------------------

def fleet_status(samples: set[str] | None = None) -> dict:
    """The control plane, optionally narrowed to one mission's scrolls.

    It counted the whole fleet unconditionally, and the mission dashboard drew
    its tiles from it -- so a mission holding one scroll reported 43 surfaces
    from 162 tasks belonging to a different one, and named that scroll in
    "surfaces by scroll" underneath. Numbers that large are read as progress,
    which is the worst way for a scoping bug to be wrong.

    ``samples is None`` means the whole fleet, which is what the fleet page
    wants.  An empty set means no rows.  That distinction is deliberate: an
    empty new mission must not fall through to the whole fleet.

    sample_id lives on the snapshot, so tasks, attempts and events reach it
    through a join. Events with no task_id cannot be attributed to a scroll at
    all and are dropped when scoping rather than counted into whichever mission
    happens to be open.
    """
    if not DSN:
        return {"available": False, "reason": "CX_DB is not set"}
    try:
        import psycopg
    except ImportError:
        return {"available": False, "reason": "psycopg is not installed"}
    scope = None if samples is None else sorted(samples)
    args = {"samples": scope, "terminal": list(SEGMENT_TERMINAL_STATES)}
    # One predicate, spelled once, appended to whichever table is being counted.
    task_where = "WHERE TRUE" if scope is None else """WHERE t.source_snapshot_id IN (
                     SELECT source_snapshot_id FROM segment_source_snapshots
                      WHERE sample_id = ANY(%(samples)s))"""
    surface_where = "WHERE TRUE" if scope is None else "WHERE sample_id = ANY(%(samples)s)"
    # An event reaches a scroll only through its task, so scoping adds the join.
    event_from = ("FROM segment_events e" if scope is None else
                  "FROM segment_events e JOIN segment_tasks t ON t.task_id = e.task_id")
    event_where = "WHERE TRUE" if scope is None else task_where
    try:
        with psycopg.connect(DSN, connect_timeout=5) as conn, conn.cursor() as cur:
            def scalar(sql: str) -> int:
                cur.execute(sql, args)
                row = cur.fetchone()
                return int(row[0]) if row else 0

            cur.execute(f"SELECT t.state, count(*) FROM segment_tasks t {task_where} "
                        "GROUP BY t.state ORDER BY 2 DESC", args)
            task_states = [{"state": s, "count": c} for s, c in cur.fetchall()]
            cur.execute(
                f"""SELECT e.event_type, count(*) {event_from} {event_where}
                    GROUP BY e.event_type ORDER BY 2 DESC LIMIT 10""", args)
            events = [{"type": t, "count": c} for t, c in cur.fetchall()]
            cur.execute(
                f"""SELECT a.worker_id, count(*) FROM segment_attempts a
                     JOIN segment_tasks t ON t.task_id = a.task_id
                     {task_where} AND a.worker_id IS NOT NULL
                    GROUP BY a.worker_id ORDER BY 2 DESC LIMIT 12""", args)
            workers = [{"worker_id": w, "attempts": c} for w, c in cur.fetchall()]
            # Split by origin, because this is what the "surfaces we grew" tile
            # reads when a scroll is selected -- which is most of the time -- and
            # it was counting imported catalogue surfaces as fleet output. An
            # attempt means the fleet grew it; no attempt means it arrived from a
            # catalogue. Same rule /api/segmentation/segments uses.
            cur.execute(f"""SELECT sample_id,
                              count(*) FILTER (WHERE payload ->> 'attempt_id' IS NOT NULL),
                              coalesce(sum(area_cm2) FILTER (
                                WHERE payload ->> 'attempt_id' IS NOT NULL), 0),
                              count(*) FILTER (WHERE payload ->> 'attempt_id' IS NULL),
                              coalesce(sum(area_cm2) FILTER (
                                WHERE payload ->> 'attempt_id' IS NULL), 0)
                            FROM segment_surfaces {surface_where}
                            GROUP BY sample_id ORDER BY 2 DESC LIMIT 20""", args)
            surfaces = [{"sample_id": s, "count": c, "area_cm2": float(a),
                         "imported": i, "imported_area_cm2": float(ia)}
                        for s, c, a, i, ia in cur.fetchall()]
            return {
                "available": True,
                "scoped_to": scope,
                "tasks": scalar(f"SELECT count(*) FROM segment_tasks t {task_where}"),
                "attempts": scalar(
                    f"""SELECT count(*) FROM segment_attempts a
                          JOIN segment_tasks t ON t.task_id = a.task_id {task_where}"""),
                # Grown, not every row: the same correction as above, for the
                # figure the page shows when no scroll is chosen.
                "surfaces": scalar(
                    f"""SELECT count(*) FROM segment_surfaces {surface_where}
                          AND payload ->> 'attempt_id' IS NOT NULL"""),
                "imported": scalar(
                    f"""SELECT count(*) FROM segment_surfaces {surface_where}
                          AND payload ->> 'attempt_id' IS NULL"""),
                "events": scalar(f"SELECT count(*) {event_from} {event_where}"),
                "leased": scalar(
                    f"""SELECT count(*) FROM segment_tasks t {task_where}
                          AND t.lease_expires_at > now()"""),
                # The terminal list comes from the constant now. It was three
                # states this fleet never uses, so every finished task counted
                # as a stale lease waiting to be recovered.
                "stale_leases": scalar(
                    f"""SELECT count(*) FROM segment_tasks t {task_where}
                          AND t.lease_expires_at IS NOT NULL
                          AND t.lease_expires_at <= now()
                          AND t.state <> ALL(%(terminal)s::text[])"""),
                "task_states": task_states,
                # QC jobs a worker refused because something is configured
                # wrong: a profile hash that does not match what the deployment
                # pins, a checkpoint that is not the one the profile names.
                #
                # Counted here because the alternative was what happened on
                # 2026-07-29: two GPUs failed every claim for two days and the
                # only place it showed was as jobs sitting in PENDING, which is
                # also what a healthy queue with no free worker looks like.
                # These stay blocked until a person changes a setting, so a
                # number above zero is always something to act on.
                "qc_blocked_on_configuration": scalar(
                    "SELECT count(*) FROM segment_qc_jobs q "
                    + ("WHERE q.state = 'BLOCKED_CONFIGURATION'" if scope is None else
                       "JOIN segment_surfaces f ON f.surface_id = q.surface_id "
                       "WHERE q.state = 'BLOCKED_CONFIGURATION' "
                       "AND f.sample_id = ANY(%(samples)s)")),
                "events_by_type": events,
                "workers": workers,
                "surfaces_by_sample": surfaces,
            }
    except Exception as exc:  # noqa: BLE001 -- the panel must render without the DB
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------
# P1 segmentation
#
# Two populations of surface, and the difference is the point of the phase.
# *Public* segments are what the community already traced and published beside
# the scan; *private* ones are what this fleet grew. A number that mixes them
# says nothing about whether anything was accomplished here.
#
# Backends are declared with why they may or may not be adopted, because three
# exist and only one of them is a producer.
# --------------------------------------------------------------------------

SEGMENTATION_BACKENDS = [
    {
        "id": "vc3d",
        "name": "VC3D seeded grow",
        "method_id": "vc3d-m7-seed-grow@1.0.0",
        "binary": "vc_grow_seg_from_seed",
        "profile_schema": "campaignx.segment_fleet_profile.v1",
        "adoptable": True,
        "note": "The production seeded segmenter. Surfaces it grows are catalogued.",
    },
    {
        "id": "scrollfiesta",
        "name": "ScrollFiesta mesh",
        "method_id": "scrollfiesta-m7-mesh@0.1.1",
        "binary": "grid_weld",
        "profile_schema": "campaignx.scientific_profile.v1",
        "adoptable": False,
        "note": "FAILED_REFERENCE_CONTROL. A comparative backend and geometric "
                "auditor; its surfaces are not catalogued until the seam, "
                "reference-agreement and flattening gates pass.",
    },
    {
        "id": "thaumato",
        "name": "ThaumatoAnakalyptor",
        "method_id": "thaumato-anakalyptor@1.0.0",
        "binary": "ThaumatoAnakalyptor.py",
        "profile_schema": None,
        "adoptable": False,
        "note": "Upstream moved this to deprecated/. Runnable for comparison; "
                "nothing it produces is catalogued without its own control.",
    },
]

# The six the fleet CLI accepts. Kept here rather than derived because the panel
# must render the choice even when the CLI is not importable.
# How the seed gets chosen. This is what P1 decides -- the volume comes from
# P0, the seed does not -- so the six are described by what each one does
# differently, not by their version numbers.
# How the seed gets chosen. This is what P1 decides -- the volume comes from P0,
# the seed does not.
#
# One agent harness, not three. OpenCode already takes --model in provider/model
# form, so Anthropic, OpenAI and OpenRouter are a parameter rather than three
# integrations to build and keep working. There is no separate Claude Code or
# Codex seeder and there does not need to be.
#
# Model ids are suggestions, not a catalogue: they move faster than this file,
# and a closed list would refuse a model that works. What is fixed is the shape,
# provider/model, because that is what OpenCode parses.
# Five, in the order somebody would try them: repeatable and free first, then
# the ones that spend tokens.
#
# The ids are unchanged, because a receipt names the planner it ran and renaming
# one would orphan every attempt that cites it. What changed is what the page
# calls them, and the numbers are gone: "v1" above "v2" reads as simple above
# advanced, when the real difference is whether the planner looks at what already
# failed in that region. So the one that looks is "Deterministic", and the one
# that does not says so.
#
# `opencode` -- the v1 packet schema -- is no longer offered. It exists only to
# reproduce runs that used it, which is a CLI errand, and it was the sixth entry
# nobody starting a run should have to rule out.
# Planners the fleet still runs but the form no longer offers. Kept so a receipt
# that names one stays reproducible, and so removing an entry from the list above
# is a change to what is recommended rather than to what is possible.
# Full entries, the same shape as an offered one. A partial dict here meant the
# validation after it read a key the retired path did not carry, and a request
# naming `opencode` answered 500 instead of queueing -- the failure mode of
# describing a thing two different ways.
RETIRED_SEEDERS = [
    {"id": "opencode", "name": "LLM agent (OpenCode, v1 packets)", "kind": "agent",
     "repeatable": False,
     "configures": [
         {"field": "provider", "label": "provider", "type": "select",
          "options": ["anthropic", "openai", "openrouter"], "note": ""},
         {"field": "model", "label": "model", "type": "text",
          "suggestions": ["claude-opus-5", "gpt-5.6-sol-pro"], "note": ""},
         {"field": "reconsider_covered", "label": "reconsider covered cells",
          "type": "text",
          "note": "Offer cells clearance would skip because a surface already "
                  "grew there. Those are the cells with alternatives to grow, "
                  "so a rank above 1 needs this to reach anything."},
     ],
     "note": "The same harness against the older packet schema. Not offered; it "
             "exists to reproduce runs that used it."},
]

SEGMENTATION_SEEDERS = [
    {"id": "deterministic-v2", "name": "Deterministic", "kind": "deterministic",
     "repeatable": True, "configures": [
         {"field": "candidate_rank", "label": "candidate rank", "type": "number",
          "note": "Which rung of m7's frozen ordering to grow. 1 is the best "
                  "candidate and the default; a higher rank grows one of the "
                  "alternatives this planner would otherwise only record as "
                  "rejected. The rank is stamped on the proposal, because a "
                  "surface from the third choice is not the evidence a first "
                  "choice is."},         {"field": "reconsider_covered", "label": "reconsider covered cells",
          "type": "text",
          "note": "Offer cells clearance would skip because a surface already "
                  "grew there. Those are the cells with alternatives to grow, "
                  "so a rank above 1 needs this to reach anything."},

     ],
     "note": "Scores candidates by cell volume and clearance, takes the best, and "
             "skips any recipe that already failed in this region. Same queue and "
             "same history, same seed, every time."},
    {"id": "deterministic", "name": "Deterministic (history blind)",
     "kind": "deterministic", "repeatable": True, "configures": [
         {"field": "candidate_rank", "label": "candidate rank", "type": "number",
          "note": "Which rung of m7's frozen ordering to grow. 1 is the best "
                  "candidate and the default; a higher rank grows one of the "
                  "alternatives this planner would otherwise only record as "
                  "rejected. The rank is stamped on the proposal, because a "
                  "surface from the third choice is not the evidence a first "
                  "choice is."},
         {"field": "reconsider_covered", "label": "reconsider covered cells",
          "type": "text",
          "note": "Offer cells clearance would skip because a surface already "
                  "grew there. Those are the cells with alternatives to grow, "
                  "so a rank above 1 needs this to reach anything."},
     ],
     "note": "The same scoring with the history ignored, so it will pick a recipe "
             "that already failed here. That is the point of it: reproducing an "
             "older run exactly, and standing as the control arm that shows "
             "whether reading history changes anything."},
    {"id": "cost-aware-v2",
     "name": "Cost-aware router (deterministic first, LLM if necessary)",
     # Declared rather than described. The note has said "the fleet's default"
     # all along and nothing read it, so "default P1" depended on how you came
     # in: the stage contract's active_runtime_profile said cost-aware v2, this
     # API defaulted to the history-blind deterministic planner, the form took
     # whichever entry happened to be first in this list, and the CLI deferred
     # to whatever the worker was started with. Four answers to one question.
     "default": True,
     "kind": "router", "repeatable": False, "configures": [],
     "note": "The fleet's default. Tries the cheapest thing that can answer: "
             "deterministic when there is no history, then a validated cache, "
             "then one model, then a panel -- and falls back to deterministic if "
             "no provider answers, so it always decides."},
    {"id": "opencode-v2", "name": "LLM agent (OpenCode)", "kind": "agent",
     "repeatable": False,
     "configures": [
         {"field": "provider", "label": "provider", "type": "select",
          "options": ["anthropic", "openai", "openrouter"],
          "note": "Your Claude or OpenAI subscription, or OpenRouter for anything else."},
         {"field": "model", "label": "model", "type": "text",
          "suggestions": ["claude-opus-5", "claude-fable-5", "gpt-5.6-sol-pro"],
          "note": "Sent as provider/model, which is what OpenCode parses."},
     ],
     "note": "One model decides, in a disposable project directory so it cannot "
             "reach this checkout. One harness for every provider."},
    {"id": "fusion-v2",
     "name": "Panel of LLM experts (OpenCode + OpenRouter Fusion API)",
     "kind": "panel", "repeatable": False,
     "configures": [
         {"field": "panel", "label": "panel", "type": "list",
          "suggestions": ["anthropic/claude-opus-5", "openai/gpt-5.6-sol-pro",
                          "anthropic/claude-fable-5", "openai/gpt-5.6-sol"],
          "note": "One to eight models, each proposing a seed."},
         {"field": "judge", "label": "synthesiser", "type": "text",
          "suggestions": ["openai/gpt-5.6-sol-pro", "anthropic/claude-opus-5"],
          "note": "Decides between the proposals."},
         {"field": "effort", "label": "reasoning", "type": "select",
          "options": ["high", "max"],
          "note": "max is OpenRouter's strongest and the most expensive."},
     ],
     "note": "Several models propose and one judges. The most expensive lane and "
             "the one with the most reasoning behind each choice."},
]
SEGMENTATION_PLANNERS = SEGMENTATION_SEEDERS   # the CLI still calls them planners

# One answer to "what runs when nobody chose".
#
# Every entry point read its own: this API defaulted to `deterministic`, the
# history-blind v1 planner, while the stage contract's active_runtime_profile
# named cost-aware v2 and the form took whichever seeder came first in the list
# above. A default that depends on the door you came through is not a scientific
# policy, and the one it landed on was the planner that will re-pick a recipe
# that already failed on this cell.
#
# Asserted rather than assumed: a second entry marked default, or none, is a
# configuration error worth failing at import rather than at the first run.
DEFAULT_SEEDER = next(s["id"] for s in SEGMENTATION_SEEDERS if s.get("default"))
assert sum(1 for s in SEGMENTATION_SEEDERS if s.get("default")) == 1, (
    "exactly one seeder is the fleet's default")


# The seed-probe controller is an additive migration. Panel and fleet releases
# are deliberately allowed to roll independently, so every read checks for all
# seven relations before referring to one. An older control plane therefore
# reports "not installed" instead of taking the whole Runs page down with an
# UndefinedTable error.
PROBE_TABLES = (
    "segment_probe_runs",
    "segment_probe_trials",
    "segment_probe_attempts",
    "segment_probe_artifact_sets",
    "segment_probe_evaluations",
    "segment_probe_decisions",
    "segment_probe_promotions",
)


def _probe_tables(cur) -> tuple[bool, list[str]]:
    """Whether the closed-loop probe ledger is installed on this database."""
    cur.execute(
        "SELECT " + ",".join("to_regclass(%s)" for _ in PROBE_TABLES),
        PROBE_TABLES,
    )
    found = cur.fetchone() or tuple(None for _ in PROBE_TABLES)
    missing = [name for name, relation in zip(PROBE_TABLES, found) if relation is None]
    return not missing, missing


def _probe_status_query(cur, samples: list[str] | None,
                        mission_id: str | None = None) -> dict:
    """Read aggregate probe activity without depending on optional columns.

    Mode is kept separate from action so a shadow CONTINUE_WINNER is never
    presented as an actual steering event.
    """
    installed, missing = _probe_tables(cur)
    if not installed:
        return {
            "available": False,
            "reason": "the seed-probe migration is not installed",
            "missing_tables": missing,
            "counts": {
                "runs": 0, "trials": 0, "decisions": 0, "promotions": 0,
            },
            "by_state": {
                "runs": {}, "trials": {}, "decisions": {}, "promotions": {},
            },
            "by_action": {},
            "by_mode": {},
            "by_mode_action": {},
        }

    cur.execute(
        """
        WITH eligible_runs AS (
          SELECT r.*
            FROM segment_probe_runs r
            JOIN segment_tasks t ON t.task_id=r.task_id
            JOIN segment_source_snapshots s
              ON s.source_snapshot_id=t.source_snapshot_id
           WHERE (%(samples)s::text[] IS NULL
              OR s.sample_id=ANY(%(samples)s))
             AND (%(mission)s::text IS NULL OR t.mission_id=%(mission)s)
        ),
        eligible_trials AS (
          SELECT t.* FROM segment_probe_trials t
          JOIN eligible_runs r ON r.probe_run_id=t.probe_run_id
        ),
        eligible_decisions AS (
          SELECT d.* FROM segment_probe_decisions d
          JOIN eligible_runs r ON r.probe_run_id=d.probe_run_id
        ),
        eligible_promotions AS (
          SELECT p.* FROM segment_probe_promotions p
          JOIN eligible_decisions d ON d.decision_id=p.decision_id
        ),
        states AS (
          SELECT 'runs'::text AS kind,
                 coalesce(state, 'UNKNOWN') AS label,
                 count(*)::bigint AS count
            FROM eligible_runs GROUP BY 2
          UNION ALL
          SELECT 'trials',
                 coalesce(state, 'UNKNOWN'),
                 count(*)::bigint
            FROM eligible_trials GROUP BY 2
          UNION ALL
          SELECT 'decisions',
                 coalesce(action, 'RECORDED'),
                 count(*)::bigint
            FROM eligible_decisions GROUP BY 2
          UNION ALL
          SELECT 'promotions',
                 coalesce(state, 'UNKNOWN'),
                 count(*)::bigint
            FROM eligible_promotions GROUP BY 2
        )
        SELECT kind, label, count FROM states
        ORDER BY kind, count DESC, label
        """,
        {"samples": samples, "mission": mission_id},
    )
    by_state: dict[str, dict[str, int]] = {
        "runs": {}, "trials": {}, "decisions": {}, "promotions": {},
    }
    for kind, label, count in cur.fetchall():
        by_state[str(kind)][str(label)] = int(count)

    cur.execute(
        """
        WITH eligible_runs AS (
          SELECT r.*
            FROM segment_probe_runs r
            JOIN segment_tasks t ON t.task_id=r.task_id
            JOIN segment_source_snapshots s
              ON s.source_snapshot_id=t.source_snapshot_id
           WHERE (%(samples)s::text[] IS NULL
              OR s.sample_id=ANY(%(samples)s))
             AND (%(mission)s::text IS NULL OR t.mission_id=%(mission)s)
        ),
        eligible_decisions AS (
          SELECT d.*,r.policy->>'mode' AS mode
            FROM segment_probe_decisions d
            JOIN eligible_runs r ON r.probe_run_id=d.probe_run_id
        )
        SELECT coalesce(mode,'unknown') AS mode,
               coalesce(action,'RECORDED') AS action,
               count(*)::bigint
          FROM eligible_decisions
         GROUP BY 1,2 ORDER BY 3 DESC, 1,2
        """,
        {"samples": samples, "mission": mission_id},
    )
    action_rows = [
        (str(mode), str(action), int(count))
        for mode, action, count in cur.fetchall()
    ]
    actions: dict[str, int] = {}
    modes: dict[str, int] = {}
    mode_actions: dict[str, dict[str, int]] = {}
    for mode, action, count in action_rows:
        actions[action] = actions.get(action, 0) + count
        modes[mode] = modes.get(mode, 0) + count
        mode_actions.setdefault(mode, {})[action] = count
    return {
        "available": True,
        "counts": {kind: sum(states.values()) for kind, states in by_state.items()},
        "by_state": by_state,
        "by_action": actions,
        "by_mode": modes,
        "by_mode_action": mode_actions,
        "note": (
            "Probe decisions compare bounded micro-growth. They do not establish "
            "that a patch follows the correct lamina; geometry and human review "
            "remain separate evidence."
        ),
    }


def probe_status(sample: str | None = None,
                 samples: set[str] | None = None,
                 mission_id: str | None = None) -> dict:
    """Migration-safe, read-only seed-probe status for the Runs view."""
    empty = {
        "counts": {
            "runs": 0, "trials": 0, "decisions": 0, "promotions": 0,
        },
        "by_state": {
            "runs": {}, "trials": {}, "decisions": {}, "promotions": {},
        },
        "by_action": {},
        "by_mode": {},
        "by_mode_action": {},
    }
    if not DSN:
        return {"available": False, "reason": "CX_DB is not set", **empty}
    try:
        import psycopg
    except ImportError:
        return {"available": False, "reason": "psycopg is not installed", **empty}
    normalized = stored_scroll(sample)
    if samples is None:
        scope = [normalized] if normalized else None
    elif normalized:
        scope = [normalized] if normalized in samples else []
    else:
        scope = sorted(samples)
    try:
        with psycopg.connect(DSN, connect_timeout=5) as conn, conn.cursor() as cur:
            return _probe_status_query(cur, scope, mission_id)
    except Exception as exc:  # noqa: BLE001 -- additive status never hides Runs
        return {
            "available": False, "reason": f"{type(exc).__name__}: {exc}", **empty,
        }


def _probe_lineage_query(cur, attempt_id: str) -> dict:
    """Every probe ledger row belonging to one canonical attempt."""
    installed, missing = _probe_tables(cur)
    if not installed:
        return {
            "available": False,
            "reason": "the seed-probe migration is not installed",
            "missing_tables": missing,
            "runs": [], "trials": [], "attempts": [], "artifacts": [],
            "evaluations": [], "decisions": [], "promotions": [],
        }

    cur.execute(
        "SELECT task_id::text FROM segment_attempts WHERE attempt_id=%s",
        (attempt_id,),
    )
    task_row = cur.fetchone()
    task_id = str(task_row[0]) if task_row else None
    cur.execute(
        """
        SELECT to_jsonb(r) FROM segment_probe_runs r
         WHERE r.created_by_attempt_id=%(attempt)s
            OR (%(task)s::text IS NOT NULL AND r.task_id=%(task)s)
         ORDER BY r.updated_at DESC
         LIMIT 100
        """,
        {"attempt": attempt_id, "task": task_id},
    )
    runs = [row[0] for row in cur.fetchall()]
    run_ids = [
        str(doc.get("probe_run_id") or doc.get("run_id") or doc.get("id"))
        for doc in runs
        if doc.get("probe_run_id") or doc.get("run_id") or doc.get("id")
    ]

    def children(table: str, column: str, identifiers: list[str]) -> list[dict]:
        if table not in PROBE_TABLES:
            raise ValueError(f"not a probe table: {table}")
        allowed_columns = {
            "probe_run_id", "probe_trial_id", "decision_id",
        }
        if column not in allowed_columns:
            raise ValueError(f"not a probe lineage column: {column}")
        if not identifiers:
            return []
        cur.execute(
            f"""
            SELECT to_jsonb(p) FROM {table} p
             WHERE p.{column}::text=ANY(%s::text[])
             ORDER BY coalesce(
                 to_jsonb(p)->>'updated_at',
                 to_jsonb(p)->>'created_at','') DESC
             LIMIT 500
            """,
            (identifiers,),
        )
        return [row[0] for row in cur.fetchall()]

    trials = children("segment_probe_trials", "probe_run_id", run_ids)
    trial_ids = [str(row["probe_trial_id"]) for row in trials]
    decisions = children("segment_probe_decisions", "probe_run_id", run_ids)
    decision_ids = [str(row["decision_id"]) for row in decisions]
    return {
        "available": True,
        "runs": runs,
        "trials": trials,
        "attempts": children(
            "segment_probe_attempts", "probe_trial_id", trial_ids
        ),
        "artifacts": children(
            "segment_probe_artifact_sets", "probe_trial_id", trial_ids
        ),
        "evaluations": children(
            "segment_probe_evaluations", "probe_trial_id", trial_ids
        ),
        "decisions": decisions,
        "promotions": children(
            "segment_probe_promotions", "decision_id", decision_ids
        ),
        "note": (
            "This is probe lineage, not lamina certification. Rejected "
            "micro-patches are non-canonical and do not enter the surface catalog."
        ),
    }


def public_segments(refresh: bool = False) -> dict:
    """Published segments per scroll, from the same bucket layout P0 reads.

    `<scroll>/segments/<segment_id>/` is a flat list, so this is one listing per
    scroll. Cached like the scroll inventory: it changes on the scale of weeks
    and the panel has to render with no network at all.
    """
    import urllib.error
    import xml.etree.ElementTree as ET

    cache_file = CACHE / "public_segments.json"
    if not refresh and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if time.time() - cached.get("fetched_at", 0) < SCROLL_TTL:
                return cached
        except (json.JSONDecodeError, OSError):
            pass

    inventory = scroll_inventory()
    counts: dict[str, int] = {}
    try:
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = {
                pool.submit(list_prefixes, SCROLL_SOURCE, f"{name}/segments/", 20): name
                for name in inventory.get("scrolls", [])
            }
            for future in as_completed(futures):
                try:
                    found = future.result()
                except (urllib.error.URLError, ET.ParseError, OSError, ValueError):
                    continue
                if found:
                    counts[futures[future]] = len(found)
    except (OSError, ValueError) as exc:
        return {"by_sample": {}, "total": 0, "fetched_at": 0.0,
                "origin": f"unreachable ({type(exc).__name__}: {exc})"}

    payload = {"by_sample": counts, "total": sum(counts.values()),
               "fetched_at": time.time(), "origin": "Official Open Data S3 Bucket"}
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload))
    except OSError:
        payload["origin"] += " (not cached)"
    return payload


def segmentation_runs(limit: int = 200, sample: str | None = None,
                      mission: str | None = None) -> dict:
    """Attempts and what came of them, newest first, for one scroll or one mission."""
    resolved = read_scope(mission, sample)
    scope = None if resolved is None else sorted(resolved)
    if not DSN:
        return {"available": False, "reason": "CX_DB is not set", "runs": []}
    try:
        import psycopg
    except ImportError:
        return {"available": False, "reason": "psycopg is not installed", "runs": []}
    try:
        with psycopg.connect(DSN, connect_timeout=5) as conn, conn.cursor() as cur:
            # Left joins throughout: an attempt that never produced a surface is
            # exactly the row worth seeing, and an inner join would hide it.
            # An attempt reaches its surface through the artifact set it
            # produced: segment_surfaces has no attempt_id, and the two are
            # matched on the artefact hash within one snapshot. The manifest
            # carries that hash, so the join reads it out rather than assuming a
            # column that does not exist.
            cur.execute(
                """SELECT a.attempt_id, a.task_id, a.state, a.worker_id,
                          a.created_at, a.updated_at, a.result,
                          t.cell_id, t.priority, s.sample_id,
                          f.surface_id, f.area_cm2, f.physical_qc_state,
                          f.artifact_uri, f.artifact_sha256,
                          -- geometry_qc_state arrived in a later migration and
                          -- a deployed database may predate it. Reading it out
                          -- of the row as JSON yields NULL when the column is
                          -- absent instead of failing the whole query, so the
                          -- page renders on an older control plane.
                          to_jsonb(f) ->> 'geometry_qc_state'
                     FROM segment_attempts a
                     LEFT JOIN segment_tasks t ON t.task_id = a.task_id
                     LEFT JOIN segment_source_snapshots s
                            ON s.source_snapshot_id = t.source_snapshot_id
                     LEFT JOIN segment_artifact_sets art
                            ON art.attempt_id = a.attempt_id
                     LEFT JOIN segment_surfaces f
                            ON f.source_snapshot_id = t.source_snapshot_id
                           AND f.artifact_sha256 = art.manifest->>'artifact_sha256'
                    WHERE (%s::text[] IS NULL OR s.sample_id = ANY(%s))
                      AND (%s::text IS NULL OR t.mission_id = %s)
                    ORDER BY a.created_at DESC NULLS LAST
                    LIMIT %s""",
                (scope, scope, mission, mission, limit),
            )
            runs = []
            for row in cur.fetchall():
                result = row[6] if isinstance(row[6], dict) else {}
                runs.append({
                    "attempt_id": row[0], "task_id": row[1], "state": row[2],
                    "worker_id": row[3],
                    "created_at": row[4].isoformat() if row[4] else None,
                    "updated_at": row[5].isoformat() if row[5] else None,
                    "cell_id": row[7], "priority": float(row[8]) if row[8] is not None else None,
                    "sample_id": row[9],
                    "surface_id": row[10],
                    "area_cm2": float(row[11]) if row[11] is not None else None,
                    "qc_state": row[12],
                    "output_path": row[13],
                    "artifact_sha256": row[14],
                    "geometry_state": row[15],
                    "executor": result.get("executor"),
                    "exit_code": result.get("exit_code"),
                    "error": result.get("error") or result.get("non_claim"),
                })
            return {"available": True, "runs": runs}
    except Exception as exc:  # noqa: BLE001 -- the panel must render without the DB
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}", "runs": []}


def scoped_queue(samples: set[str] | None,
                 mission_id: str | None = None) -> dict | None:
    """Queue counts for a set of scrolls, or None if the database is not there.

    Scroll selection constrains the input; mission identity constrains ownership.
    Both are required because two missions may legitimately work on the same
    scroll without inheriting each other's queue, attempts, or surfaces.
    """
    if samples is not None and not samples:
        return {
            "tasks": 0, "leased": 0, "stale_leases": 0, "attempts": 0,
            "by_state": {}, "surfaces": 0, "area_cm2": 0.0,
            "imported": 0, "imported_area_cm2": 0.0,
            "certified": 0, "certified_area_cm2": 0.0,
            "ct_supported": 0, "ct_supported_area_cm2": 0.0,
        }
    if not DSN or samples is None:
        return None
    try:
        import psycopg
    except ImportError:
        return None
    try:
        with psycopg.connect(DSN, connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute(
                # TERMINAL_STATES from fleet/store.py, not a guess. The earlier
                # list named three states this fleet never uses, so every
                # finished task counted as waiting and a queue with nothing
                # claimable in it reported 162 waiting.
                """SELECT count(*) FILTER (WHERE t.state NOT IN
                            (SELECT unnest(%(terminal)s::text[]))),
                          count(*) FILTER (WHERE t.lease_expires_at > now()),
                          count(*) FILTER (WHERE t.lease_expires_at IS NOT NULL
                            AND t.lease_expires_at <= now()
                            AND t.state NOT IN
                              (SELECT unnest(%(terminal)s::text[])))
                     FROM segment_tasks t
                     JOIN segment_source_snapshots s
                       ON s.source_snapshot_id = t.source_snapshot_id
                    WHERE s.sample_id = ANY(%(samples)s)
                      AND (%(mission)s::text IS NULL
                           OR t.mission_id = %(mission)s)""",
                {"samples": sorted(samples),
                 "mission": mission_id,
                 "terminal": list(SEGMENT_TERMINAL_STATES)})
            row = cur.fetchone() or (0, 0, 0)
            cur.execute(
                """SELECT a.state, count(*) FROM segment_attempts a
                     JOIN segment_tasks t ON t.task_id = a.task_id
                     JOIN segment_source_snapshots s
                       ON s.source_snapshot_id = t.source_snapshot_id
                    WHERE s.sample_id = ANY(%(samples)s)
                      AND (%(mission)s::text IS NULL
                           OR t.mission_id = %(mission)s)
                    GROUP BY a.state""",
                {"samples": sorted(samples), "mission": mission_id})
            by_state = {state: int(count) for state, count in cur.fetchall()}
            # Surfaces and their area together: a count without area says nothing
            # about whether anything worth reading came out.
            #
            # Split, because this was one sum over every row in the table and the
            # page called it "surfaces we grew". Imported catalogue surfaces live
            # in that same table -- ten of them on this control plane have no
            # attempt anywhere -- so the fleet was being credited with work it did
            # not do. The origin rule is the one /api/segmentation/segments
            # already uses two hundred lines away: an attempt means we grew it.
            #
            # And certified is reported apart from grown, because gross area is an
            # upper bound and not yield. The scientific handoff is explicit that
            # naive VC3D area double-counts identity and can include radial
            # spokes; only the geometry-certified and CT-supported figures are
            # claims about usable sheet.
            surface_scope = """
                SELECT DISTINCT f.*
                  FROM segment_surfaces f
                  JOIN segment_artifact_sets art
                    ON art.manifest->>'artifact_sha256'=f.artifact_sha256
                  JOIN segment_attempts a ON a.attempt_id=art.attempt_id
                  JOIN segment_tasks t ON t.task_id=a.task_id
                 WHERE f.sample_id = ANY(%(samples)s)
                   AND t.mission_id = %(mission)s
            """ if mission_id else """
                SELECT f.* FROM segment_surfaces f
                 WHERE f.sample_id = ANY(%(samples)s)
            """
            cur.execute(
                """WITH eligible_surfaces AS (""" + surface_scope + """)
                   SELECT
                     count(*) FILTER (WHERE payload ->> 'attempt_id' IS NOT NULL),
                     coalesce(sum(area_cm2) FILTER (
                       WHERE payload ->> 'attempt_id' IS NOT NULL), 0),
                     count(*) FILTER (WHERE payload ->> 'attempt_id' IS NULL),
                     coalesce(sum(area_cm2) FILTER (
                       WHERE payload ->> 'attempt_id' IS NULL), 0),
                     count(*) FILTER (
                       WHERE to_jsonb(f) ->> 'geometry_qc_state' = 'GEOMETRY_CERTIFIED'),
                     coalesce(sum(area_cm2) FILTER (
                       WHERE to_jsonb(f) ->> 'geometry_qc_state' = 'GEOMETRY_CERTIFIED'), 0),
                     count(*) FILTER (
                       WHERE physical_qc_state IN ('CT_SUPPORTED', 'CT_SUPPORTED_REVIEW')),
                     coalesce(sum(area_cm2) FILTER (
                       WHERE physical_qc_state IN ('CT_SUPPORTED', 'CT_SUPPORTED_REVIEW')), 0)
                   FROM eligible_surfaces f""",
                {"samples": sorted(samples), "mission": mission_id})
            counts = cur.fetchone() or (0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0)
            grown, grown_area, imported, imported_area, \
                certified, certified_area, supported, supported_area = counts
            return {"tasks": int(row[0]), "leased": int(row[1]),
                    "stale_leases": int(row[2]),
                    "attempts": sum(by_state.values()),
                    "by_state": by_state,
                    # `surfaces` and `area_cm2` keep their names and stop
                    # including what the fleet did not grow.
                    "surfaces": int(grown), "area_cm2": float(grown_area),
                    "imported": int(imported), "imported_area_cm2": float(imported_area),
                    "certified": int(certified),
                    "certified_area_cm2": float(certified_area),
                    "ct_supported": int(supported),
                    "ct_supported_area_cm2": float(supported_area)}
    except Exception:  # noqa: BLE001 -- the page renders without the database
        return None


def segmentation_state(sample: str | None = None,
                       samples: set[str] | None = None,
                       mission_id: str | None = None) -> dict:
    """Everything P1 needs to render, in one call.

    `samples` is the mission's selection. The queue is reported for those
    scrolls only; a mission's page showing the whole fleet's backlog says work
    is waiting that has nothing to do with it.
    """
    sample = stored_scroll(sample)
    effective = ({sample} if sample else None) if samples is None else (
        ({sample} if sample in samples else set()) if sample else set(samples))
    fleet = fleet_status(effective)
    scope = scoped_queue(effective, mission_id)
    public = public_segments()
    public_by_sample = {
        stored_scroll(name) or name: count
        for name, count in public.get("by_sample", {}).items()
    }
    if effective is not None:
        public_by_sample = {name: count for name, count in public_by_sample.items()
                            if stored_scroll(name) in effective}
    private_rows = ([] if mission_id else fleet.get("surfaces_by_sample", []))
    private = {row["sample_id"]: row for row in private_rows}
    in_scope = sample is None or effective is None or sample in effective
    return {
        "public": {
            "total": (public.get("total", 0) if effective is None
                      else sum(public_by_sample.values())),
            "by_sample": public_by_sample,
            "origin": public.get("origin", "?"),
            "for_sample": (public_by_sample.get(sample)
                           if sample and in_scope else None),
        },
        "private": {
            "total": (scope.get("surfaces", 0) if scope is not None
                      else fleet.get("surfaces", 0)),
            "area_cm2": (scope.get("area_cm2") if scope is not None else None),
            # Computed and then not sent, which is its own kind of missing: the
            # split query returns imported, certified and CT-supported and only
            # the grown pair reached the page. Gross area is an upper bound; these
            # are the figures that mean anything downstream.
            "imported": (scope.get("imported", 0) if scope is not None
                         else fleet.get("imported", 0)),
            "certified": scope.get("certified") if scope is not None else None,
            "certified_area_cm2": (scope.get("certified_area_cm2")
                                    if scope is not None else None),
            "ct_supported": scope.get("ct_supported") if scope is not None else None,
            "ct_supported_area_cm2": (scope.get("ct_supported_area_cm2")
                                       if scope is not None else None),
            "by_sample": private_rows,
            "for_sample": (private.get(sample) if sample and in_scope else None),
        },
        "queue": {
            **({"tasks": fleet.get("tasks", 0),
                "attempts": fleet.get("attempts", 0),
                "leased": fleet.get("leased", 0),
                "stale_leases": fleet.get("stale_leases", 0)}
               if scope is None else scope),
            "states": ([] if mission_id else fleet.get("task_states", [])),
            "by_state": scope.get("by_state", {}) if scope is not None else {},
            "scope": ("the whole fleet" if effective is None
                      else f"{len(effective)} scroll(s) in this mission"),
        },
        "workers": fleet.get("workers", []),
        "backends": SEGMENTATION_BACKENDS,
        "planners": SEGMENTATION_PLANNERS,
        "available": fleet.get("available", False),
        "reason": fleet.get("reason"),
    }


# --------------------------------------------------------------------------
# Reference strips: scoring a surface against something outside itself.
#
# A strip is a small patch of scroll where consecutive papyrus wraps are
# recorded as separate labelled point sets. It is not ground truth and does not
# claim to be: it is derived from a segmentation and validated by internal
# consistency, so the one thing it cannot do is judge the segment it came from.
#
# What it does judge is the failure that actually matters. For a mesher that is
# non-manifold edges, open boundaries, and wraps fused into one surface -- which
# is a stronger statement than a backend's own test suite, because the strip was
# not written by the same people.
# --------------------------------------------------------------------------

STRIP_SCHEMA = "strip-v0"
# 512 MB. UC-01 is 66 MB and a strip is a handful of point sets; anything an
# order of magnitude past that is not a strip and should not be written to disk
# before we find out.
MAX_STRIP_BYTES = 512 * 1024 * 1024


def strip_tools() -> Path:
    if not (REFERENCE_STRIPS / "strip_format.py").exists():
        raise HTTPException(
            409, f"reference-strips is not vendored at {REFERENCE_STRIPS}")
    return REFERENCE_STRIPS


def read_strip(path: Path) -> dict:
    """Identity and shape of one strip, read through the vendored loader.

    Deliberately the vendored `load_strip` rather than a bare np.load: it passes
    allow_pickle=False, and an .npz that arrived over HTTP is exactly the file
    where pickle would be arbitrary code execution.
    """
    if str(strip_tools()) not in sys.path:
        sys.path.insert(0, str(strip_tools()))
    import strip_format  # noqa: PLC0415

    strip = strip_format.load_strip(path)
    problems = strip_format.validate_strip(strip)
    meta = dict(strip.meta or {})
    # load_strip hands back {"median","p10","p90"}, not the array it saved.
    # NaN means the pitch is unknown, and JSON has no NaN, so it goes out null.
    pitch = {k: (None if v != v else float(v))
             for k, v in (strip.pitch_um or {}).items()}
    return {
        "strip_id": path.stem,
        "path": str(path),
        "bytes": path.stat().st_size,
        "schema_version": meta.get("schema_version"),
        "scroll": meta.get("scroll"),
        "segment_id": meta.get("segment_id"),
        "window": meta.get("window"),
        "voxel_size_um": meta.get("voxel_size_um"),
        "tier": meta.get("tier"),
        "wraps": len(strip.wrap_indices) if strip.wrap_indices is not None else 0,
        "points": int(sum(len(w) for w in strip.wraps.values())) if strip.wraps else 0,
        "pitch_um": pitch,
        "problems": problems,
        "qualified": strip_format.is_qualified(path),
    }


def strip_qualification(path: Path) -> dict | None:
    report = path.with_suffix(".qualification.json")
    if not report.exists():
        return None
    try:
        return json.loads(report.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def strips() -> list[dict]:
    """Every strip on disk, newest first, with its qualification verdict."""
    if not STRIPS.exists():
        return []
    out = []
    for path in sorted(STRIPS.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            entry = read_strip(path)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 -- one bad file must not empty the list
            out.append({"strip_id": path.stem, "path": str(path),
                        "error": f"{type(exc).__name__}: {exc}", "qualified": False})
            continue
        report = strip_qualification(path)
        # The report calls it "pass", not "passed", and a check that did not
        # run reports pass=None rather than False -- the CT cross-check is off
        # by default, and "not run" is not "failed".
        entry["qualification"] = {
            "overall_pass": report.get("overall_pass"),
            "checks": {k: v.get("pass") for k, v in report.get("checks", {}).items()
                       if isinstance(v, dict)},
        } if report else None
        out.append(entry)
    return out


def integrity(mission_id: str | None = None) -> list[dict]:
    """Receipts whose declared preprocessing contradicts the upstream contract.

    Scoped to a mission's runs. Unscoped it reported findings from every run on
    the host, so a mission with nothing wrong in it still showed a red tile.
    """
    findings = []
    registry = registry_entries()
    for run in index_runs(mission_id=mission_id):
        if run.clip_value is not None:
            if run.divisor is not None and run.divisor != run.clip_value:
                findings.append({
                    "run_id": run.run_id, "sample_id": run.sample_id, "lane": run.lane_id,
                    "kind": "normalization",
                    "detail": f"clips at {run.clip_value} and divides by {run.divisor}",
                    "severity": "critical"})
            elif run.clip_value != UPSTREAM_CLIP:
                findings.append({
                    "run_id": run.run_id, "sample_id": run.sample_id, "lane": run.lane_id,
                    "kind": "clip",
                    "detail": f"max_clip_value {run.clip_value}, upstream declara {UPSTREAM_CLIP}",
                    "severity": "warning"})
        liveness = run.receipt.get("liveness") or {}
        if liveness.get("verdict") not in (None, "ALIVE"):
            findings.append({
                "run_id": run.run_id, "sample_id": run.sample_id, "lane": run.lane_id,
                "kind": "liveness",
                "detail": f"mapa {liveness['verdict']}: {liveness.get('reason', '')}",
                "severity": "critical"})
    return findings


# --------------------------------------------------------------------------
# Probability maps
#
# The array is cached as uint8 (a 4096x4096 map costs 16 MiB instead of 67) and
# every request slices the window the client can actually see. The client gets
# luminance and does its own colouring, so the threshold never round-trips.
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=6)
def _quantised(run_id: str, name: str, mtime: float) -> tuple[np.ndarray, np.ndarray, dict]:
    run = get_run(run_id)
    path = run.path / name
    if not path.exists():
        raise HTTPException(404, f"no map {name!r}")
    a = np.load(path).astype(np.float32)
    valid = a > 0
    v = a[valid]
    if v.size:
        s = np.sort(v[:: max(1, v.size // 400_000)])
        stats = {
            "valid_pixels": int(v.size),
            "p50": float(np.percentile(s, 50)),
            "p90": float(np.percentile(s, 90)),
            "p99": float(np.percentile(s, 99)),
            "min": float(v.min()),
            "max": float(v.max()),
            "fraction_above_0_5": float((v > 0.5).mean()),
        }
    else:
        stats = {"valid_pixels": 0}
    stats["height"], stats["width"] = int(a.shape[0]), int(a.shape[1])
    return np.clip(a * 255.0, 0, 255).astype(np.uint8), valid, stats


def quantised(run_id: str, name: str):
    run = get_run(run_id)
    path = run.path / name
    if not path.exists():
        raise HTTPException(404, f"no map {name!r}")
    return _quantised(run_id, name, path.stat().st_mtime)


@functools.lru_cache(maxsize=64)
def _raster(run_id: str, name: str, mtime: float, box: tuple[int, int, int, int], size: int) -> bytes:
    grey, valid, stats = _quantised(run_id, name, mtime)
    x, y, w, h = box
    x = max(0, min(x, stats["width"] - 1))
    y = max(0, min(y, stats["height"] - 1))
    w = max(1, min(w, stats["width"] - x))
    h = max(1, min(h, stats["height"] - y))

    window = grey[y:y + h, x:x + w]
    mask = valid[y:y + h, x:x + w]
    # Alpha carries validity, so the client can leave holes transparent without
    # a second request or a magic sentinel value inside the luminance channel.
    rgba = np.dstack([window, window, window, np.where(mask, 255, 0).astype(np.uint8)])
    image = Image.fromarray(rgba, "RGBA")
    if max(w, h) > size:
        scale = size / max(w, h)
        image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)

    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=False, compress_level=1)
    return buf.getvalue()


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

@app.get("/api/build")
def api_build():
    """The entry chunk the server is currently serving.

    Superseded chunks are kept on purpose, so a tab that loaded an older
    index.html goes on fetching names that still resolve. lazyRoute only
    recovers from a *missing* chunk, which means a tab open across a deploy runs
    the old build indefinitely and looks like the deploy did nothing -- or worse,
    hits a bug that was fixed hours ago. There were 22 retained chunks for one
    route when this was written.

    The client compares this against the script tag it was loaded from.
    """
    index = REPO / "panel" / "web" / "dist" / "index.html"
    try:
        html = index.read_text(encoding="utf-8")
    except OSError as exc:
        return JSONResponse({"entry": None, "reason": str(exc)})
    match = re.search(r"assets/(index-[A-Za-z0-9_-]+\.js)", html)
    # The footer's link rides along here rather than getting its own request:
    # this one is already polled to notice a deploy, and the answer changes about
    # as often.
    return JSONResponse({"entry": match.group(1) if match else None,
                         "source_url": SOURCE_URL})


@app.get("/api/state")
def api_state(mission: str | None = None):
    """Volatile: the client polls this one."""
    return JSONResponse({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # Every one of these takes the mission now. The dashboard drew tiles from
        # a helper that counted the whole host, so a mission holding one scroll
        # reported another scroll's 43 surfaces and 162 tasks as its own.
        "fleet": fleet_status(None if mission is None else mission_scrolls(mission)),
        "integrity": integrity(mission),
        "targets": targets(mission),
        "scroll_totals": {k: v for k, v in scrolls().items() if k != "scrolls"},
        "run_count": len(index_runs(mission_id=mission)),
        # Profiles are reference material rather than a mission's property: the
        # same lane profile is available to every mission, so this stays global.
        "lane_count": len(ink_profiles()),
        "mission": mission,
    })


@app.get("/api/runs")
def api_runs(mission: str | None = None):
    return JSONResponse({"runs": [r.as_dict() for r in index_runs(mission_id=mission)]})


@app.get("/api/run/{run_id}")
def api_run(run_id: str):
    run = get_run(run_id)
    profile = next((p for p in ink_profiles()
                    if run.lane_id in (p["profile_id"], p["method_id"])), None)
    return JSONResponse({**run.as_dict(), "receipt": run.receipt, "profile": profile})


@app.get("/api/run/{run_id}/map/{name}")
def api_map_meta(run_id: str, name: str):
    _, _, stats = quantised(run_id, name)
    return JSONResponse(stats)


@app.get("/api/run/{run_id}/map/{name}/raster")
def api_map_raster(
    run_id: str,
    name: str,
    x: int = Query(0, ge=0),
    y: int = Query(0, ge=0),
    w: int = Query(0, ge=0),
    h: int = Query(0, ge=0),
    size: int = Query(1024, ge=64, le=MAP_MAX_SIZE),
):
    """Luminance + validity alpha for one window, resampled to `size`."""
    run = get_run(run_id)
    path = run.path / name
    if not path.exists():
        raise HTTPException(404, f"no map {name!r}")
    mtime = path.stat().st_mtime
    _, _, stats = _quantised(run_id, name, mtime)
    if w == 0 or h == 0:
        x, y, w, h = 0, 0, stats["width"], stats["height"]
    data = _raster(run_id, name, mtime, (x, y, w, h), size)
    return Response(data, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400, immutable"})


@app.get("/api/run/{run_id}/map/{name}/screen")
def api_screen(
    run_id: str,
    name: str,
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    min_area: int = Query(12, ge=1, le=100_000),
    max_area: int = Query(4000, ge=1, le=10_000_000),
    row_gap: int = Query(60, ge=1, le=10_000),
    row_min_shapes: int = Query(4, ge=1, le=100),
    min_candidates: int = Query(10, ge=1, le=10_000),
    min_rows: int = Query(2, ge=1, le=100),
):
    """The strict text screen: threshold, shapes, rows, verdict.

    Every parameter of the screen is on the query string rather than compiled
    in, because the screen's thresholds are exactly the thing an operator needs
    to move while watching what moves with them. The defaults are the frozen
    ones.
    """
    from scipy import ndimage

    grey, valid, meta = quantised(run_id, name)
    probability = grey.astype(np.float32) / 255.0
    mask = (probability >= threshold) & valid
    labels, count = ndimage.label(mask)
    if count == 0:
        shapes: list[dict] = []
    else:
        areas = np.bincount(labels.ravel())[1:]
        keep = np.flatnonzero((areas >= min_area) & (areas <= max_area)) + 1
        centres = ndimage.center_of_mass(mask, labels, keep) if keep.size else []
        boxes = ndimage.find_objects(labels)
        shapes = [
            {"label": int(k), "area": int(areas[k - 1]),
             "y": float(c[0]), "x": float(c[1]),
             "bbox": [int(boxes[k - 1][1].start), int(boxes[k - 1][0].start),
                      int(boxes[k - 1][1].stop), int(boxes[k - 1][0].stop)]}
            for k, c in zip(keep.tolist(), centres)
        ]

    # Rows: sort by centroid y and cut wherever the gap exceeds row_gap.
    rows: list[dict] = []
    for shape in sorted(shapes, key=lambda s: s["y"]):
        if rows and shape["y"] - rows[-1]["last_y"] <= row_gap:
            rows[-1]["shapes"] += 1
            rows[-1]["last_y"] = shape["y"]
        else:
            rows.append({"first_y": shape["y"], "last_y": shape["y"], "shapes": 1})
    qualifying = [r for r in rows if r["shapes"] >= row_min_shapes]

    passes = len(shapes) >= min_candidates and len(qualifying) >= min_rows
    return JSONResponse({
        "run_id": run_id, "map": name,
        "parameters": {"threshold": threshold, "min_area": min_area, "max_area": max_area,
                       "row_gap": row_gap, "row_min_shapes": row_min_shapes,
                       "min_candidates": min_candidates, "min_rows": min_rows},
        "shape_count": len(shapes), "row_count": len(rows),
        "qualifying_row_count": len(qualifying),
        "shapes": shapes[:500],
        "rows": [{"y": round(r["first_y"], 1), "shapes": r["shapes"],
                  "qualifies": r["shapes"] >= row_min_shapes} for r in rows],
        "verdict": "PASSES_STRICT_SCREEN" if passes else "DOES_NOT_PASS",
        "percentiles": {k: meta.get(k) for k in ("p50", "p90", "p99", "max")},
        "shape_y_x": [meta["height"], meta["width"]],
        "non_claims": [
            "Passing the strict screen is not a reading and does not accept ink, text or letters.",
            "It says bounded shapes arranged in rows survived the threshold, nothing more.",
            "Run the vetting-card battery before treating any window as a candidate.",
        ],
    })


@app.get("/api/lanes")
def api_lanes():
    registry = registry_entries()
    profiles = []
    for p in ink_profiles():
        entry = registry.get(p["method_id"], {})
        profiles.append({
            **p,
            "registry_status": entry.get("validation_status"),
            "registry_policy": entry.get("recommended_policy"),
            "disqualified": "DISQUALIFIED" in str(entry.get("validation_status", "")),
        })
    return JSONResponse({"profiles": profiles, "upstream_clip": UPSTREAM_CLIP})


@app.get("/api/fleet")
def api_fleet():
    return JSONResponse(fleet_status())


# --------------------------------------------------------------------------
# Missions
#
# A mission is a directory under CX_RUNS with a MISSION.json naming the scrolls
# it attempts. The manifest is authoritative; this API reads and writes it, and
# the database only indexes which jobs belong where.
# --------------------------------------------------------------------------


class MissionRequest(BaseModel):
    mission_id: str = Field(min_length=3, max_length=64)
    name: str = Field(min_length=1, max_length=120)
    # A mission may be created empty: choosing scrolls is P0's job.
    scrolls: list[str] = Field(default_factory=list)
    description: str = Field(default="", max_length=2000)


class AmendRequest(BaseModel):
    add: list[str] = Field(min_length=1)
    # Required only once the mission has produced work. The contract decides
    # that, not the schema: a draft selection is edited freely, and a validator
    # that demanded a reason here would reimpose the ceremony one layer up.
    reason: str = Field("", max_length=500)


def _refuse_implicit(mission_id: str) -> None:
    """The unfiled mission has no manifest to amend.

    It is a view of runs that predate missions, assembled from their receipts,
    not a scope anybody declared. Editing its selection would mean editing a
    description of what already happened.
    """
    if mission_id == "unfiled":
        raise HTTPException(
            409,
            "'unfiled' is not a real mission: it is a view of runs that predate missions, "
            "read from their receipts. Its scroll list describes what exists rather than "
            "what was chosen, so there is nothing to amend. Create a mission and select "
            "the scrolls you mean to attempt.")


@app.get("/api/missions")
def api_missions():
    missions = mission_contract.discover(RUNS)
    counts: dict[str, int] = {}
    if DSN:
        try:
            for job in job_store().jobs(limit=500):
                key = job.get("mission_id") or "unfiled"
                counts[key] = counts.get(key, 0) + 1
        except Exception:  # noqa: BLE001 -- the queue is optional
            counts = {}
    for m in missions:
        m["job_count"] = counts.get(m["mission_id"], 0)
    return JSONResponse({"missions": missions, "runs_root": str(RUNS)})


@app.post("/api/missions")
def api_create_mission(request: MissionRequest):
    try:
        manifest = mission_contract.create(
            RUNS, mission_id=request.mission_id, name=request.name,
            scrolls=request.scrolls, description=request.description)
    except mission_contract.MissionError as exc:
        raise HTTPException(400, str(exc)) from exc
    index_runs(force=True)
    return JSONResponse(manifest, status_code=201)


@app.get("/api/missions/{mission_id}")
def api_mission(mission_id: str):
    try:
        directory, manifest = mission_contract.resolve(RUNS, mission_id)
    except mission_contract.MissionError as exc:
        raise HTTPException(404, str(exc)) from exc
    runs = index_runs(mission_id=mission_id)
    manifest["path"] = str(directory)
    manifest["runs"] = [r.as_dict() for r in runs]
    manifest["run_count"] = len(runs)
    manifest["scrolls_with_runs"] = sorted({r.sample_id for r in runs})
    manifest["scrolls_without_runs"] = sorted(
        set(manifest.get("scrolls", [])) - {r.sample_id for r in runs})
    return JSONResponse(manifest)


def freeze_p0_artifacts(mission_id: str, scrolls_selected: list[str],
                        *, by: str, reason: str = "") -> list[dict]:
    """Write and register what P0 produced: one frozen source per scroll.

    P0's output is not a file anyone was writing. The phase decides which volume
    each scroll is read from and at what physical scale, and that decision was
    living only in the mission manifest -- so "What P0 produced" was empty while
    the empty state claimed every Apply recorded a version. A promise the code
    did not keep is worse than a blank table.

    One artifact per scroll rather than one per mission, because that is the unit
    a later phase consumes: P1 reads one scroll's prediction, and a mission-wide
    blob would make every scroll's run depend on the selection of all the others.

    Content-addressed, so re-applying an unchanged selection returns the same
    artifact instead of a new version. What creates a version is a changed
    decision -- a different scan for a scroll, a corrected scale -- which is the
    only thing a version should mean.
    """
    directory = mission_directory(mission_id)
    outputs = directory / "artifacts" / "P0"
    outputs.mkdir(parents=True, exist_ok=True)
    metadata = catalog_metadata()
    registered: list[dict] = []
    for sample in sorted(scrolls_selected):
        entry = metadata.get(sample, {})
        frozen = {
            "schema": "campaignx.p0_frozen_source.v1",
            "sample_id": sample,
            "scan_id": entry.get("scan_id") or None,
            "ct_uri": entry.get("ct_uri") or None,
            "pixel_um": entry.get("pixel_um") or None,
            "energy_kev": entry.get("energy_kev"),
            "higher_resolution_sibling_exists": bool(entry.get("higher_res")),
            # Absent rather than guessed. A scroll the frozen catalogue does not
            # carry has no declared scale here, and every micron figure
            # downstream depends on that number being measured.
            "scale_known": bool(entry.get("pixel_um")),
            "non_claim": "This freezes which volume is read and at what scale. It "
                         "is not a claim that anything was found in it.",
        }
        path = outputs / f"{_safe_id(sample)}.json"
        path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        try:
            registered.append(artifact_contract.register(
                directory, phase="P0", sample_id=sample, kind="frozen-source",
                path=path, note=reason[:200], by=by))
        except artifact_contract.ArtifactError:
            # A registry that cannot record must not undo a selection the user
            # already made and the manifest already holds.
            continue
    return registered


@app.post("/api/missions/{mission_id}/artifacts/freeze-p0")
def api_freeze_p0(mission_id: str, http: Request):
    """Register what P0 produced for the selection as it stands.

    Amending requires a scroll to add, so a mission whose selection was made
    before this register existed had no way to record it -- and the P0 table said
    "nothing registered yet" about a phase that had, in fact, decided something.

    Idempotent by content: pressing it twice returns the same artifacts rather
    than a second version. A version means the decision changed.
    """
    _refuse_implicit(mission_id)
    try:
        _, manifest = mission_contract.resolve(RUNS, mission_id)
    except mission_contract.MissionError as exc:
        raise HTTPException(404, str(exc)) from exc
    scrolls_selected = list(manifest.get("scrolls") or [])
    if not scrolls_selected:
        raise HTTPException(409, "this mission has no scrolls selected, so P0 has "
                                 "not decided anything to record")
    registered = freeze_p0_artifacts(
        mission_id, scrolls_selected,
        by=str(getattr(http.state, "username", "") or "panel"),
        reason="frozen from the selection as it stands")
    return JSONResponse({"scrolls": scrolls_selected, "artifacts": registered})


@app.post("/api/missions/{mission_id}/amend")
def api_amend_mission(mission_id: str, request: AmendRequest, http: Request):
    """Widen a frozen selection, on the record, and record what it produced."""
    _refuse_implicit(mission_id)
    try:
        directory, _ = mission_contract.resolve(RUNS, mission_id)
        manifest = mission_contract.amend_scrolls(
            directory, add=request.add, reason=request.reason)
    except mission_contract.MissionError as exc:
        raise HTTPException(400, str(exc)) from exc
    # After the manifest, never instead of it: the selection is the decision and
    # the artifact is the record of it, so a registry failure must not be able to
    # lose a choice the user already made.
    registered = freeze_p0_artifacts(
        mission_id, list(manifest.get("scrolls") or []),
        by=str(getattr(http.state, "username", "") or "panel"),
        reason=request.reason or "")
    return JSONResponse({**manifest, "artifacts": registered})


@app.post("/api/missions/{mission_id}/remove")
def api_remove_scrolls(mission_id: str, request: AmendRequest):
    """Remove scrolls. Free while the mission is a draft; scrolls with work are refused."""
    _refuse_implicit(mission_id)
    try:
        directory, _ = mission_contract.resolve(RUNS, mission_id)
    except mission_contract.MissionError as exc:
        raise HTTPException(404, str(exc)) from exc
    # What has produced something here cannot be disowned.
    protected = {r.sample_id for r in index_runs(mission_id=mission_id)}
    if DSN:
        try:
            protected |= {j["sample_id"] for j in job_store().jobs(limit=500,
                                                                  mission_id=mission_id)}
        except Exception as unreachable:  # noqa: BLE001
            # Not optional here. This is the query that decides which scrolls
            # may not be removed, and a queue that cannot be read yields a
            # smaller protected set rather than an error -- so an amendment
            # would disown a scroll with work queued against it, which the line
            # above says cannot happen.
            raise HTTPException(503, {
                "detail": "the queue cannot be read, so what is protected from "
                          "removal cannot be established",
                "error": f"{type(unreachable).__name__}: {unreachable}",
            }) from unreachable
    try:
        manifest = mission_contract.remove_scrolls(
            directory, remove=request.add, reason=request.reason, protected=protected)
    except mission_contract.MissionError as exc:
        raise HTTPException(409, str(exc)) from exc
    index_runs(force=True)
    return JSONResponse(manifest)


@app.post("/api/missions/{mission_id}/state")
def api_mission_state(mission_id: str, state: str = Query(...)):
    try:
        directory, _ = mission_contract.resolve(RUNS, mission_id)
        manifest = mission_contract.set_state(directory, state)
    except mission_contract.MissionError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(manifest)


@app.get("/api/scrolls")
def api_scrolls(refresh: bool = Query(False), source: str | None = Query(None)):
    """Every scroll a bucket exposes, discovered by the volumes/ convention.

    `source` browses a different bucket without changing the configured one --
    the setting is what persists, this is what looks. Any bucket in the same
    layout enumerates; one that is not reports zero scrolls rather than
    guessing at its directories.
    """
    if source:
        check_source_is_fetchable(source)
    return JSONResponse(scrolls(refresh=refresh, source=source))


# --------------------------------------------------------------------------
# Command plane
#
# The panel enqueues; workers on any host claim and execute. It does not run
# anything itself and it does not accept a command -- a request names a profile
# and parameters that must validate against it, and the worker builds the
# command line from the profile. Reaching this API is not the same as being
# able to run arbitrary code on a GPU host.
#
# It is still an unauthenticated write surface. Bind the service to localhost,
# or put something in front of it, before exposing it on a network you do not
# control.
# --------------------------------------------------------------------------

sys.path.insert(0, str(REPO / "framework" / "stages" / "03-ink" / "fleet"))


def job_store():
    if not DSN:
        raise HTTPException(503, "CX_DB is not set: the command plane needs the fleet database")
    try:
        from job_store import InkJobStore  # type: ignore
    except ImportError as exc:
        raise HTTPException(503, f"ink job store unavailable: {exc}") from exc
    return InkJobStore(DSN)


class JobRequest(BaseModel):
    sample_id: str = Field(min_length=1, max_length=64)
    phase: str = Field(default="P5", pattern=r"^P[0-9]$")
    profile_id: str | None = Field(default=None, max_length=128)
    component: str | None = Field(default=None, max_length=128)
    parameters: dict[str, Any]
    priority: int = Field(default=0, ge=-100, le=100)
    requested_host: str | None = Field(default=None, max_length=64)
    mission_id: str | None = Field(default=None, max_length=64)
    max_attempts: int = Field(default=3, ge=1, le=10)


@app.get("/api/jobs")
def api_jobs(limit: int = Query(100, ge=1, le=500), state: str | None = None,
             mission: str | None = None):
    store = job_store()
    states = tuple(s.strip() for s in state.split(",")) if state else None
    return JSONResponse({"jobs": store.jobs(limit=limit, states=states, mission_id=mission)})


def who_asked(http: Request) -> str:
    """The signed-in user, for the rows a request creates. Falls back to the
    process name only where login is switched off in development."""
    return getattr(http, "state", None) and getattr(http.state, "username", None) \
        or "panel"


@app.post("/api/jobs")
def api_enqueue(request: JobRequest, http: Request):
    """Queue one run. Refused if the profile is unknown or disqualified."""
    job_sample = require_write_sample(
        request.mission_id, request.sample_id, f"{request.phase} job")
    # A switch that does not stop the work is decoration. Checked here rather
    # than in the worker so the refusal reaches the person who asked, at the
    # moment they ask, instead of arriving as a failed job an hour later.
    for module_id in (request.profile_id,
                      (request.parameters or {}).get("lane")):
        if module_id and module_disabled(request.phase, str(module_id)):
            raise HTTPException(409, {
                "detail": f"{module_id} is switched off for {request.phase}",
                "why": "Switch it back on under Configuration -> Modules, or "
                       "choose another one.",
            })
    if request.phase == "P5":
        profiles = {p["profile_id"]: p for p in ink_profiles()}
        profile = profiles.get(request.profile_id or "")
        if profile is None:
            raise HTTPException(400, f"unknown profile {request.profile_id!r}")
        entry = registry_entries().get(profile["method_id"], {})
        if "DISQUALIFIED" in str(entry.get("validation_status", "")):
            raise HTTPException(
                409,
                f"{profile['method_id']} is {entry['validation_status']} and must not be routed. "
                f"{entry.get('recommended_policy', '')}",
            )
    elif request.phase not in QUEUEABLE_PHASES:
        raise HTTPException(400, f"phase {request.phase} has no runner registered")
    parameters = dict(request.parameters)
    if request.phase == "P4":
        # Filled here rather than asked of every caller: where a render is
        # published is a property of the deployment, not of the request, and a
        # form that can forget it strands the output on the worker's disk.  The
        # client cannot override it either: the form hides this field, but the
        # HTTP boundary still accepts an arbitrary parameters object.
        if not RENDER_STORE and not parameters.get("allow_local_layers"):
            raise HTTPException(409,
                "CX_RENDER_STORE is not set. A layer stack written only to the "
                "worker's disk is lost with the worker; set it in Settings, or "
                "pass allow_local_layers for a deliberate single-machine run.")
        if RENDER_STORE:
            parameters["artifact_store"] = RENDER_STORE
        else:
            parameters.pop("artifact_store", None)
    if request.phase == "P5":
        # P7 may be claimed by a different worker, so the probability map must
        # outlive the container that computed it.  Its destination is deployment
        # policy for the same reason as P4/P8 and cannot be redirected by a
        # caller through the generic parameters object.
        if not INK_STORE:
            raise HTTPException(
                409,
                "CX_INK_STORE is not set. A probability map written only to the "
                "worker cannot be consumed reliably by P7.",
            )
        parameters["artifact_store"] = INK_STORE
    if request.phase == "P8" and parameters.get("lane") == "vc3d-tifxyz-merge":
        # The immutable evidence destination is deployment policy, not job
        # input.  Always overwrite a client value so direct API callers cannot
        # divert a merge to a worker-local or attacker-chosen path.
        if not RECONSTRUCTION_STORE:
            raise HTTPException(
                409,
                "CX_RECONSTRUCTION_STORE is not set. A merged TIFXYZ and its "
                "lineage must be published to durable storage.",
            )
        parameters["artifact_store"] = RECONSTRUCTION_STORE
    store = job_store()
    try:
        job_id = store.enqueue(
            # The catalog's spelling, like every other row in the control plane.
            # Stored verbatim, the queue held one scroll under two names --
            # PHerc0826 from the bucket beside PHerc826 from the catalog -- and
            # a mission-scoped phase counted only half its own renders.
            sample_id=job_sample or request.sample_id,
            phase=request.phase,
            mission_id=request.mission_id,
            profile_id=request.profile_id, component=request.component,
            parameters=parameters, priority=request.priority,
            requested_host=request.requested_host, max_attempts=request.max_attempts,
            created_by=who_asked(http),
        )
    except Exception as exc:  # JobRejected and friends
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse({"job_id": job_id}, status_code=201)


@app.get("/api/jobs/{job_id}/events")
def api_job_events(job_id: str):
    return JSONResponse({"events": job_store().events(job_id)})


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel(job_id: str):
    cancelled = job_store().cancel(job_id)
    if not cancelled:
        raise HTTPException(409, "only a job that has not started can be cancelled")
    return JSONResponse({"job_id": job_id, "state": "cancelled"})


class HostRequest(BaseModel):
    host_id: str = Field(min_length=1, max_length=64)
    ssh_target: str = Field(min_length=1, max_length=200)
    roles: list[str] = Field(default_factory=list)
    notes: str | None = Field(default=None, max_length=500)
    # Default true: a host added without the worker on it is a row that looks
    # like a machine. Opt out for a host that is already running one.
    provision: bool = True


class SegmentationRunRequest(BaseModel):
    """A request to queue segmentation work for one scroll.

    Deliberately narrow. The panel does not send a command and it does not send
    coordinates: it names a scroll, a backend, a planner and a task budget, and
    the fleet's own bootstrap decides which cells are uncovered and where the
    seeds may be. Anything wider would let a request from a browser choose where
    to look, which is the one decision the whole planner contract exists to keep
    inside versioned policy.
    """

    sample_id: str = Field(min_length=1, max_length=64)
    backend: str = Field("vc3d")
    # Every bootstrap knob, optional. None means "do not pass the flag", so the
    # queue applies its own default rather than a copy this model remembers.
    options: dict[str, str | int | float | None] = Field(default_factory=dict)
    # The fleet's default, not this endpoint's own opinion of one. It said
    # "deterministic" -- the history-blind v1 planner -- so a direct API call
    # got a different policy from the same button in the browser.
    planner: str = Field(default_factory=lambda: DEFAULT_SEEDER)
    max_tasks: int = Field(4, ge=1, le=48)
    grid_step: int = Field(2048, ge=256, le=8192)
    # Not a planner knob: the planners are right that they configure nothing with
    # it. It is provenance stamped on the task, and it is what makes asking a
    # covered cell a second question rather than a duplicate -- the planner
    # refuses a cell that already has a task under the same grid and policy, and
    # that refusal is correct. Growing m7's second candidate over ground its
    # first already covered needs a name for the new question.
    policy_version: str | None = Field(default=None, max_length=64)
    seed_config: dict[str, str] = Field(default_factory=dict)
    reason: str = Field("", max_length=500)
    # A deterministic, bounded experiment inside the selected planner lane.
    # ``shadow`` records the micro-growth comparison and leaves the canonical
    # seed alone. ``select`` may change it, but only in the two deterministic
    # lanes below and only when the evidence separates a winner; otherwise the
    # attempt stops for a person instead of guessing.
    seed_probe_mode: Literal["off", "shadow", "select"] = "off"
    # Raised from 3. Three is a tie-break between m7's best candidates; the
    # question is whether m7's ordering holds further down, and that cannot be
    # asked at 3. Measured at top_k=3 on PHerc826: 8 of 8 candidates ELIGIBLE,
    # a 0% rejection rate against a 34% break-even -- so probing paid for
    # nothing. Either the tail is as good, and m7 orders well, or it degrades
    # and the rejection that makes a cheap filter worth running appears there.
    seed_probe_top_k: int = Field(2, ge=1, le=20)
    seed_probe_generations: int = Field(12, ge=10, le=20)
    # Which mission's P0 selection this run reads. The launcher records a
    # selection and then queued a run that had no idea which one was current, so
    # "what did this task read" was answerable only by the wall clock: whatever
    # the mission happened to point at when somebody looked.
    mission_id: str | None = Field(None, max_length=64)


@app.get("/api/segmentation")
def api_segmentation(sample: str | None = Query(None),
                     mission: str | None = Query(None)):
    """Public segments, private surfaces, the queue, and what can run."""
    selection = None if mission is None else mission_scrolls(mission)
    state = segmentation_state(sample, samples=selection, mission_id=mission)
    # What a new run would read. Shown where the run is started rather than in
    # a register of its own: choosing an input is part of launching, not a
    # separate errand.
    # Every upstream artifact in the mission, not only the selected scroll's.
    #
    # This was scoped to the subject, which made a dropdown that looked like a
    # chooser and offered one item -- and left the run's scroll coming from a
    # selector two panels away. Starting a run is where you decide what to run on,
    # so the input names the scroll: pick PHerc0841's frozen source and the run
    # covers PHerc0841. That is also the lineage story, rather than a hidden
    # dependency on what the sidebar happens to be showing.
    state["reads"] = readable_inputs(mission, "P1", None) if mission else []
    return JSONResponse(state)


def readable_inputs(mission_id: str | None, phase: str, sample: str | None) -> list[dict]:
    """The upstream artifacts a run of this phase could read, newest first."""
    if not mission_id or mission_id == "unfiled":
        return []
    try:
        directory, _ = mission_contract.resolve(RUNS, mission_id)
    except mission_contract.MissionError:
        return []
    chosen = (artifact_contract.current_selection(directory) or {}).get("choices", {})
    out = []
    for upstream in phase_inputs(phase):
        for record in artifact_contract.artifacts(directory, phase=upstream,
                                                  sample_id=sample):
            key = artifact_contract.selection_key(upstream, record["sample_id"])
            out.append({
                "artifact_id": record["artifact_id"],
                "phase": record["phase"],
                "sample_id": record["sample_id"],
                "kind": record["kind"],
                "note": record.get("note", ""),
                "registered_at_utc": record["registered_at_utc"],
                "selected": chosen.get(key) == record["artifact_id"],
            })
    return out


@app.get("/api/segmentation/runs")
def api_segmentation_runs(sample: str | None = Query(None),
                          mission: str | None = Query(None),
                          limit: int = Query(200, ge=1, le=1000)):
    return JSONResponse(segmentation_runs(limit=limit, sample=sample, mission=mission))


@app.get("/api/segmentation/probes")
def api_segmentation_probes(sample: str | None = Query(None),
                            mission: str | None = Query(None)):
    """Counts and decision actions from the optional seed-probe ledger.

    This endpoint is intentionally read-only. Before the additive probe
    migration is installed it answers ``available: false`` with the missing
    table names, so panel and fleet images can be rolled in either order.
    """
    selection = None if mission is None else mission_scrolls(mission)
    return JSONResponse(probe_status(sample=sample, samples=selection,
                                     mission_id=mission))


# Every knob the fleet bootstrap accepts, in one place, because it is used three
# times: to validate a request, to build the command, and to draw the form. A
# hand-typed copy in the front end would be a fourth truth that drifts.
#
# `default` is deliberately absent. The bootstrap owns its defaults; repeating
# them here would freeze a copy that stops matching the moment one changes, and
# an omitted flag means "whatever the queue thinks" rather than "whatever the
# panel remembers". The form shows the bootstrap's own value by leaving the field
# empty and saying so.
#
# Bounds come from the growth profile where the profile has them, because the
# planner contract rejects a value outside its declared minimum/maximum as
# POLICY_REJECTED -- so a form that let you type one would be offering a run
# that cannot survive validation.
SEGMENTATION_OPTIONS = [
    {"flag": "--grid-step", "field": "grid_step", "kind": "int",
     "label": "Grid step", "group": "where to look — cells, chosen when queueing",
     "note": "Edge of a grid cell in CT-L0 voxels. Larger means fewer, bigger cells."},
    {"flag": "--query-radius", "field": "query_radius", "kind": "int",
     "label": "Query radius", "group": "where to look — cells, chosen when queueing",
     "note": "How far around a cell centre the seed search reads the prediction."},
    {"flag": "--volume-edge-margin", "field": "volume_edge_margin", "kind": "int",
     "label": "Volume edge margin", "group": "where to look — cells, chosen when queueing",
     "note": "Voxels of the scan edge left alone; a surface grown into the "
             "boundary is an artefact of the crop."},
    {"flag": "--clearance", "field": "clearance", "kind": "float",
     "label": "Cell clearance", "group": "where to look — cells, chosen when queueing",
     "note": "How far a cell must be from ground already segmented to be queued "
             "at all. Measured at the cell centre minus the query radius, so "
             "every point in the probe cube clears it and not just the middle."},
    {"flag": "--candidate-interior-clearance", "field": "candidate_interior_clearance",
     "kind": "int", "label": "Interior clearance", "group": "which point in the cell — chosen at run time",
     "note": "Distance a candidate must keep from the cell interior wall."},
    {"flag": "--selection-strategy", "field": "selection_strategy", "kind": "choice",
     "label": "Cell ranking", "group": "where to look — cells, chosen when queueing",
     "choices": ["max-clearance-v1", "stratified-clearance-v1"],
     "note": "Whether the best-cleared cells always win, or the volume is sampled "
             "across clearance bands so the sweep spreads out instead of "
             "crowding the emptiest region."},
    {"flag": "--candidate-selection-policy", "field": "candidate_selection_policy",
     "kind": "choice", "label": "Candidate policy", "group": "which point in the cell — chosen at run time",
     "choices": ["score-cell-volume-clearance-v1", "adaptive-geometry-history-v2"],
     "note": "The frozen rule a planner must obey. adaptive-geometry-history-v2 "
             "is the only one that lets a v2 seeder pick a non-top candidate."},
    {"flag": "--seed-region-policy", "field": "seed_region_policy", "kind": "choice",
     "label": "Seed region policy", "group": "which point in the cell — chosen at run time",
     "choices": ["fixed-v1", "m7-recenter-z-v1", "m7-recenter-xyz-v1"],
     "note": "Whether the probe box sits where the grid says, or is recentred on "
             "the prediction first."},
    {"flag": "--recenter-probe-max-candidates", "field": "recenter_probe_max_candidates",
     "kind": "int", "label": "Recentre probe cap", "group": "recentring",
     "note": "Only read when a recentring policy is chosen."},
    {"flag": "--recenter-radius-x", "field": "recenter_radius_x", "kind": "int",
     "label": "Recentre radius X", "group": "recentring", "note": "Voxels."},
    {"flag": "--recenter-radius-y", "field": "recenter_radius_y", "kind": "int",
     "label": "Recentre radius Y", "group": "recentring", "note": "Voxels."},
    {"flag": "--recenter-radius-z", "field": "recenter_radius_z", "kind": "int",
     "label": "Recentre radius Z", "group": "recentring", "note": "Voxels."},
    {"flag": "--ct-material-support-gate", "field": "ct_material_support_gate",
     "off_flag": "--no-ct-material-support-gate", "kind": "toggle",
     "label": "Ask the CT", "group": "does the CT agree",
     "note": "Reject a seed the raw scan has no material at. On, because growing "
             "from a point with nothing there costs hours and cannot succeed. It "
             "was off for the first 142 tasks this fleet queued, which trusted "
             "the prediction alone. The three settings below apply when it is on."},
    {"flag": "--ct-support-level", "field": "ct_support_level", "kind": "int",
     "label": "CT support level", "group": "does the CT agree",
     "note": "Resolution level of the CT the gate reads."},
    {"flag": "--ct-support-radius-l0", "field": "ct_support_radius_l0", "kind": "int",
     "label": "CT support radius", "group": "does the CT agree",
     "note": "Voxels around the candidate the gate samples."},
    {"flag": "--ct-support-minimum-nonzero-voxels",
     "field": "ct_support_minimum_nonzero_voxels", "kind": "int",
     "label": "Minimum non-zero voxels", "group": "does the CT agree",
     "note": "Below this the CT is treated as empty there and the candidate goes."},
    {"flag": "--grid-version", "field": "grid_version", "kind": "text",
     "label": "Grid version", "group": "provenance",
     "note": "Stamped into every task. Changing it makes a new coverage universe "
             "rather than adding to the existing one."},
    {"flag": "--policy-version", "field": "policy_version", "kind": "text",
     "label": "Policy version", "group": "provenance",
     "note": "Recorded on each task so a run is attributable to the rules it ran "
             "under."},
]


def seed_probe_select_readiness(sample: str | None = None) -> dict:
    """Explain every independent gate on steering with probe evidence."""

    def immutable_source_lock(entry: dict[str, Any]) -> bool:
        lock = entry.get("source_content_lock")
        expected = {
            "schema",
            "status",
            "verification_method",
            "verified_at_utc",
            "ct_uri",
            "ct_sha256",
            "ct_version_id",
            "m7_uri",
            "m7_sha256",
            "m7_version_id",
        }
        if not isinstance(lock, dict) or set(lock) != expected:
            return False
        ct_uri = str(entry.get("ct_uri") or "")
        m7_uri = str(entry.get("surface_prediction_uri") or "")
        ct_sha = str(entry.get("ct_sha256") or "")
        m7_sha = str(entry.get("surface_prediction_sha256") or "")
        try:
            verified_time = datetime.fromisoformat(
                str(lock.get("verified_at_utc") or "").replace("Z", "+00:00")
            )
        except ValueError:
            verified_time = None
        return bool(
            lock["schema"] == "campaignx.source_content_lock.v1"
            and lock["status"] == "VERIFIED_IMMUTABLE"
            and lock["verification_method"]
            == "immutable-uri-manifest-sha256-v1"
            and isinstance(lock["verified_at_utc"], str)
            and lock["verified_at_utc"].endswith("Z")
            and verified_time is not None
            and verified_time.utcoffset()
            == timezone.utc.utcoffset(verified_time)
            and re.fullmatch(r"[0-9a-f]{64}", ct_sha)
            and re.fullmatch(r"[0-9a-f]{64}", m7_sha)
            and lock["ct_uri"] == ct_uri
            and lock["m7_uri"] == m7_uri
            and lock["ct_sha256"] == ct_sha
            and lock["m7_sha256"] == m7_sha
            and isinstance(lock["ct_version_id"], str)
            and len(lock["ct_version_id"]) >= 8
            and lock["ct_version_id"] in ct_uri
            and isinstance(lock["m7_version_id"], str)
            and len(lock["m7_version_id"]) >= 8
            and lock["m7_version_id"] in m7_uri
        )

    rollout_enabled = os.environ.get(
        "HELENA_ENABLE_SEED_PROBE_SELECT", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    review_owner_declared = bool(SEED_PROBE_REVIEW_OWNER)
    benchmark_approved = False
    authorized_samples: set[str] = set()
    benchmark_evidence: dict[str, Any] = {}
    if SEED_PROBE_BENCHMARK_RECEIPT is not None:
        try:
            stage_root = REPO / "framework/stages/01-segmentation"
            if str(stage_root) not in sys.path:
                sys.path.insert(0, str(stage_root))
            from fleet.seed_probe import (  # noqa: PLC0415
                load_seed_probe_benchmark_receipt,
            )

            authorization = load_seed_probe_benchmark_receipt(
                SEED_PROBE_BENCHMARK_RECEIPT
            )
            benchmark_approved = True
            authorized_samples = set(authorization["authorized_sample_ids"])
            benchmark_evidence = {
                "benchmark_id": authorization["benchmark_id"],
                "decision_receipt_sha256": authorization[
                    "decision_receipt_sha256"
                ],
                "paired_cell_count": authorization["paired_cell_count"],
                "scroll_count": authorization["scroll_count"],
                "authorized_sample_count": len(authorized_samples),
            }
        except Exception as exc:  # noqa: BLE001 - rollout gate must fail closed
            benchmark_evidence = {
                "error": (
                    "configured benchmark receipt failed validation "
                    f"({type(exc).__name__})"
                )
            }
    locked_samples: set[str] = set()
    try:
        document = json.loads(CATALOG.read_text(encoding="utf-8"))
        entries = document.get("entries", []) if isinstance(document, dict) else document
        locked_samples = {
            str(entry["sample_id"])
            for entry in entries
            if isinstance(entry, dict)
            and entry.get("sample_id")
            and immutable_source_lock(entry)
        }
    except (OSError, TypeError, ValueError):
        locked_samples = set()
    canonical_sample = catalog_sample_id(sample, strict=False) if sample else None
    benchmark_scope_allows = bool(
        canonical_sample and canonical_sample in authorized_samples
    )
    source_locked = (
        canonical_sample in locked_samples
        if canonical_sample
        else bool(locked_samples)
    )
    reasons = []
    if not rollout_enabled:
        reasons.append(
            "select rollout flag is disabled"
        )
    if not benchmark_approved:
        reasons.append(
            benchmark_evidence.get("error")
            or "no approved causal benchmark receipt is configured"
        )
    elif not benchmark_scope_allows:
        reasons.append(
            "choose a sample covered by the approved benchmark receipt"
            if canonical_sample is None
            else "the selected sample is outside the approved benchmark scope"
        )
    if not source_locked:
        reasons.append(
            "the selected source has no verified immutable CT and m7 content lock"
            if canonical_sample
            else "the catalog has no verified immutable CT and m7 source"
        )
    if not review_owner_declared:
        reasons.append("no seed-probe review-resolution owner is configured")
    return {
        "available": (
            rollout_enabled
            and benchmark_approved
            and benchmark_scope_allows
            and source_locked
            and review_owner_declared
        ),
        "rollout_enabled": rollout_enabled,
        "benchmark_approved": benchmark_approved,
        "benchmark_scope_allows": benchmark_scope_allows,
        **benchmark_evidence,
        "source_locked": source_locked,
        "review_owner_declared": review_owner_declared,
        "reason": "; ".join(reasons) or None,
    }


def seed_probe_worker_readiness() -> dict:
    """Whether a recently seen worker can claim probe-required tasks."""

    if not DSN:
        return {"available": False, "reason": "CX_DB is not set"}
    try:
        import psycopg

        with psycopg.connect(DSN, connect_timeout=5) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT count(*)::bigint
                         FROM segment_worker_capabilities
                        WHERE lower(coalesce(
                                capabilities->>'seed_probe_v1','false'))
                              IN ('true','1','yes','on')
                          AND updated_at >= now() - interval '20 minutes'"""
                )
                count = int(cursor.fetchone()[0])
    except Exception as exc:  # noqa: BLE001 - readiness must fail closed
        return {
            "available": False,
            "reason": f"cannot verify probe-capable workers: {type(exc).__name__}: {exc}",
        }
    return {
        "available": count > 0,
        "worker_count": count,
        "reason": (
            None
            if count
            else "no probe-capable worker has checked in during the last 20 minutes"
        ),
    }


@app.get("/api/segmentation/options")
def api_segmentation_options(sample: str | None = Query(None)):
    """What a run can be configured with, and what the runtime pins.

    Served rather than hard-coded in the page so the form and the command are
    built from one list.
    """
    return JSONResponse({
        "options": SEGMENTATION_OPTIONS,
        "probe": {
            "modes": [
                {
                    "id": "off",
                    "name": "Off",
                    "note": "Grow the canonical seed directly.",
                },
                {
                    "id": "shadow",
                    "name": "Shadow",
                    "note": (
                        "Compare bounded deterministic micro-growths and record "
                        "the decision, without changing the canonical grow."
                    ),
                },
                {
                    "id": "select",
                    "name": "Select",
                    "note": (
                        "Let the deterministic comparison choose beneath "
                        "Cost-aware or Deterministic v2. If it cannot separate "
                        "a winner, stop for human review instead of guessing."
                    ),
                },
            ],
            "default_mode": "off",
            "select_readiness": seed_probe_select_readiness(sample),
            "top_k": {"minimum": 1, "maximum": 20, "default": 2},
            "generations": {"minimum": 10, "maximum": 20, "default": 12},
            "note": (
                "seed-probe-v1 is a closed-loop deterministic micro-growth layer "
                "beneath the Cost-aware router, not a replacement for it or an "
                "LLM panel. In select mode an unambiguous winner deliberately "
                "takes Cost-aware's zero-provider deterministic lane, so Fusion "
                "reasoning is bypassed for that attempt."
            ),
            "caveat": (
                "A probe winner is not proof of the correct lamina. Geometry and "
                "human review remain separate evidence."
            ),
        },
        # The VC3D growth parameters are a different layer and the page should say
        # so: they are written per attempt by the executor, and the two the profile
        # freezes cannot be chosen at all.
        "growth": {
            "note": "VC3D's own parameters live in the growth profile and are "
                    "written per attempt. They are shown for reading; the panel "
                    "does not yet carry them with a task.",
            "parameters": [
                {"name": "generations", "range": "20-45", "default": 35},
                {"name": "step_size", "range": "12-24", "default": 20},
                {"name": "inpaint", "range": "true/false", "default": False},
                {"name": "skip_overlap_check", "range": "true/false", "default": False},
                {"name": "min_area_cm", "range": "pinned at 0.0", "default": 0.0},
                {"name": "use_cuda", "range": "pinned false — segmentation is CPU",
                 "default": False},
            ],
        },
    })


# What bootstrap-manual will accept out of the form's option list. Derived from
# the command rather than assumed: it takes the grid and CT-support knobs and none
# of the cell-ranking ones, because a manual seed is told where to look.
MANUAL_OPTION_FLAGS = {
    "grid_step": "--grid-step",
    "query_radius": "--query-radius",
    "volume_edge_margin": "--volume-edge-margin",
    "ct_support_level": "--ct-support-level",
    "ct_support_radius_l0": "--ct-support-radius-l0",
    "ct_support_minimum_nonzero_voxels": "--ct-support-minimum-nonzero-voxels",
}


def _manual_option_flags(options: dict) -> list[str]:
    """The subset of the form's options a manual seed can carry."""
    argv: list[str] = []
    for field, flag in MANUAL_OPTION_FLAGS.items():
        value = options.get(field)
        if value in (None, ""):
            continue
        try:
            argv += [flag, str(int(float(value)))]
        except (TypeError, ValueError):
            raise HTTPException(400, f"{field} must be a number") from None
    # Refused, not quietly dropped -- including the two cell-ranking policies.
    # Excluding them from this check was the first thing I wrote here, which would
    # have accepted them and sent them nowhere: the same pattern this endpoint is
    # being fixed for.
    unknown = sorted(field for field in options
                     if field not in MANUAL_OPTION_FLAGS
                     and options[field] not in (None, ""))
    if unknown:
        raise HTTPException(
            400, f"a manual seed cannot carry {unknown}: bootstrap-manual is told "
                 "where to look, so it takes the grid and CT-support settings and "
                 "none of the ones that rank cells. Leave them empty here, or use "
                 "a discovered run.")
    return argv


class ManualSeedRequest(BaseModel):
    """Points a person wants grown, in CT-L0 voxels."""

    sample_id: str = Field(min_length=1, max_length=64)
    points: list[dict[str, float | str | None]] = Field(min_length=1, max_length=200)
    note: str | None = Field(default=None, max_length=500)
    # A task is identified by volume, grid version, cell and policy version, so
    # the same point under the same policy is one task forever. Growing it again
    # -- after a fix, or with a different screen -- needs a new policy version,
    # and without a way to name one the endpoint could only answer "nothing was
    # queued" and not offer any way past it.
    policy_version: str | None = Field(default=None, max_length=64)
    grid_version: str | None = Field(default=None, max_length=64)
    # Everything below reached nothing before: the browser sent a scroll, points
    # and a note, and the form's planner, options and mission sat beside a button
    # that ignored them. bootstrap-manual takes all of these -- they were simply
    # never passed.
    #
    # No backend field. The fleet grows on VC3D only, which the run endpoint
    # already refuses to pretend otherwise about, so offering one here would be
    # the same lie in a second place.
    planner: str | None = Field(default=None, max_length=64)
    seed_config: dict[str, str] = Field(default_factory=dict)
    options: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    mission_id: str | None = Field(default=None, max_length=64)


@app.post("/api/segmentation/manual-seeds")
def api_manual_seeds(body: ManualSeedRequest, request: Request):
    """Queue growth from points somebody chose, instead of points a model proposed.

    This is the other kind of ground truth. Reference strips judge a surface that
    already exists; these produce one. They share a phrase and nothing else, so
    they are separate endpoints -- folding them together because both are called
    ground truth is the vocabulary mistake that made segments and runs one table.

    The author is taken from the session rather than the body. A human seed whose
    author is self-reported is not auditable, and every surface grown from one
    carries seed_origin human precisely so "the fleet found this" stays a claim
    somebody can check.

    Screening is unchanged and that is the point: a manual seed skips the m7
    prediction, so the CT-material gate asking the raw scan whether there is
    anything at that coordinate is the only check left between a guess and hours
    of growing. It still runs.
    """
    sample = require_write_sample(body.mission_id, body.sample_id, "manual seed run")
    if not DSN:
        raise HTTPException(409, "CX_DB is not set; there is no control plane to queue into")
    script = (REPO / "framework/stages/01-segmentation/scripts"
              / "helena_segment_search_fleet.py")
    if not script.exists():
        raise HTTPException(500, f"the fleet CLI is missing at {script}")

    # From the session the middleware already resolved, not from the body: a
    # human seed whose author is self-reported is not auditable.
    author = str(getattr(request.state, "username", "") or "").strip()
    if not author:
        raise HTTPException(403, "a manual seed needs a signed-in author")

    # Written where the CLI can read it and nowhere durable: the points are in
    # the tasks once this returns, and a second copy on disk is a second thing
    # that can disagree with them.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump({"points": body.points}, handle)
        points_path = Path(handle.name)
    receipt_path = points_path.with_suffix(".receipt.json")
    argv = [
        sys.executable, str(script), "bootstrap-manual",
        "--db", DSN,
        "--catalog", str(GEOMETRY_CATALOG),
        "--sample", sample or catalog_sample_id(body.sample_id),
        "--points", str(points_path),
        "--submitted-by", author,
        "--receipt", str(receipt_path),
        *(["--mission-id", body.mission_id] if body.mission_id else []),
        *(["--policy-version", body.policy_version] if body.policy_version else []),
        *(["--grid-version", body.grid_version] if body.grid_version else []),
        # The planner the form chose, and its model. Without these a manual seed
        # grew under whatever planner the host was started with, which is the same
        # defect the run endpoint had.
        *(["--planner", body.planner] if body.planner else []),
        *(["--planner-model", body.seed_config["model"]]
          if body.seed_config.get("model") else []),
        # Only the options this command takes. bootstrap-manual has no
        # --candidate-selection-policy or --seed-region-policy: it is told where to
        # look, so the knobs that rank cells do not apply. Sending one would be a
        # command line the fleet rejects, and offering it in the form would be the
        # accepted-and-discarded pattern again.
        *_manual_option_flags(body.options),
    ]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed script, validated arguments
            argv, capture_output=True, text=True, timeout=300,
            env={**os.environ, "PYTHONPATH": str(REPO)})
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(504, "queueing the manual seeds did not finish in 300s") from exc
    finally:
        points_path.unlink(missing_ok=True)

    if completed.returncode != 0:
        receipt_path.unlink(missing_ok=True)
        raise HTTPException(400, {
            "detail": "the fleet refused these points",
            "exit_code": completed.returncode,
            # The DSN is in argv and must not come back in an error.
            "stderr_tail": (completed.stderr or "")[-2000:],
        })
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        receipt = {}
    finally:
        receipt_path.unlink(missing_ok=True)
    # The same refusal the run endpoint makes, for the same reason: a 200 on a
    # request that queued nothing tells somebody their points are growing when
    # the queue was untouched.
    queued = (receipt.get("tasks") or {}).get(
        sample or catalog_sample_id(body.sample_id), {})
    inserted = int(queued.get("inserted", 0))
    if inserted == 0:
        raise HTTPException(409, {
            "detail": (f"nothing was queued: all {queued.get('generated', 0)} of "
                       "these points already have a task under policy "
                       f"{body.policy_version or 'ink-blind-v1'}"),
            "why": ("A task is identified by volume, grid version, cell and "
                    "policy version, so the same point under the same policy is "
                    "one task forever -- including one that already finished."),
            "how": ("Name a new policy_version to grow these points again, which "
                    "is what makes the second run a separate experiment rather "
                    "than a silent no-op."),
            "generated": queued.get("generated", 0),
            "inserted": 0,
        })
    return JSONResponse({"queued_for": body.sample_id,
                         "submitted_by": author,
                         "points": len(body.points),
                         "inserted": inserted,
                         "receipt": receipt,
                         "note": body.note})


@app.get("/api/segmentation/segments")
def api_segments(sample: str | None = Query(None),
                 mission: str | None = Query(None),
                 limit: int = Query(300, ge=1, le=2000)):
    """The surfaces that exist, whatever made them.

    Separate from /runs on purpose, and not a second vocabulary for the same
    rows: a run carries its own output, but a surface can outlive the run that
    made it and some never had one. Ten of the surfaces on this control plane
    were imported from a catalogue and have no attempt anywhere -- a runs table
    cannot show those, so a page built only from runs quietly under-reports what
    the fleet actually holds.

    `origin` is the distinction: GROWN HERE for a surface an attempt in this
    control plane produced, IMPORTED for one that arrived some other way.
    """
    resolved = read_scope(mission, sample)
    scope = None if resolved is None else sorted(resolved)
    if not DSN:
        return JSONResponse({"available": False, "reason": "CX_DB is not set",
                             "segments": []})
    try:
        import psycopg
    except ImportError:
        return JSONResponse({"available": False, "reason": "psycopg is not installed",
                             "segments": []})
    try:
        with psycopg.connect(DSN, connect_timeout=5) as conn, conn.cursor() as cur:
            mission_join = """
                  JOIN segment_artifact_sets art
                    ON art.manifest->>'artifact_sha256'=f.artifact_sha256
                  JOIN segment_attempts a ON a.attempt_id=art.attempt_id
                  JOIN segment_tasks t ON t.task_id=a.task_id
            """ if mission else ""
            mission_where = " AND t.mission_id = %s" if mission else ""
            cur.execute(
                """SELECT DISTINCT f.surface_id, f.sample_id, f.area_cm2, f.state,
                          f.physical_qc_state,
                          to_jsonb(f) ->> 'geometry_qc_state',
                          f.artifact_uri, f.artifact_sha256, f.created_at,
                          f.payload ->> 'attempt_id',
                          f.payload ->> 'source_catalog',
                          f.owner,
                          f.payload -> 'human_review'
                     FROM segment_surfaces f
                """ + mission_join + """
                    WHERE (%s::text[] IS NULL OR f.sample_id = ANY(%s))
                """ + mission_where + """
                    ORDER BY f.created_at DESC NULLS LAST
                    LIMIT %s""",
                ((scope, scope, mission, limit) if mission
                 else (scope, scope, limit)))
            rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"available": False,
                             "reason": f"{type(exc).__name__}: {exc}",
                             "segments": []})

    segments = [{
        "surface_id": surface_id, "sample_id": sample_id,
        "area_cm2": float(area) if area is not None else None,
        "state": state, "physical_qc_state": physical,
        "geometry_qc_state": geometry or "GEOMETRY_UNMEASURED",
        "artifact_uri": uri, "artifact_sha256": sha,
        "created_at": created.isoformat() if created else None,
        "attempt_id": attempt,
        "origin": "GROWN_HERE" if attempt else "IMPORTED",
        "source_catalog": catalog, "owner": owner,
        "human_review": review,
    } for (surface_id, sample_id, area, state, physical, geometry, uri, sha,
           created, attempt, catalog, owner, review) in rows]

    grown = sum(1 for s in segments if s["origin"] == "GROWN_HERE")
    return JSONResponse({
        "available": True, "segments": segments, "count": len(segments),
        "grown_here": grown, "imported": len(segments) - grown,
        "total_area_cm2": round(
            sum(s["area_cm2"] or 0.0 for s in segments), 2),
    })


class ReplanRequest(BaseModel):
    grid_version: str = Field(min_length=1, max_length=64)
    policy_version: str = Field(min_length=1, max_length=64)
    planner: str | None = Field(default=None, max_length=64)
    planner_model: str | None = Field(default=None, max_length=128)
    sample_id: str | None = Field(default=None, max_length=64)
    mission_id: str | None = Field(default=None, max_length=64)
    causes: list[str] = Field(default_factory=list)
    limit: int = Field(default=50, ge=1, le=500)
    dry_run: bool = True


@app.post("/api/segmentation/replan")
def api_replan(request: ReplanRequest):
    """Ask the cells that gave no seed again, under a different policy.

    169 of this control plane's 241 tasks ended NO_SEED, the worker recorded for
    each one how many candidates the provider offered and which screen removed
    them, and nothing ever read it back: the cell was terminal, the next
    bootstrap picked cells by distance from known surfaces, and the fleet
    explored without learning.

    Shelled out to the fleet's own command, like the bootstrap is, so the rule
    about what may be re-asked lives in one place. That command refuses a policy
    version that is already on the task -- task identity would make it a silent
    no-op -- so this cannot queue nothing and call it a run.
    """
    sample = require_write_sample(request.mission_id, request.sample_id, "P1 replan")
    if not DSN:
        raise HTTPException(409, "CX_DB is not set; there is no control plane")
    script = REPO / "framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py"
    if not script.exists():
        raise HTTPException(409, f"the fleet command is not at {script}")
    argv = [
        sys.executable, str(script), "replan",
        "--db", DSN,
        "--grid-version", request.grid_version,
        "--policy-version", request.policy_version,
        "--limit", str(request.limit),
        *(["--planner", request.planner] if request.planner else []),
        *(["--planner-model", request.planner_model] if request.planner_model else []),
        *(["--sample", sample] if sample else []),
        *(["--dry-run"] if request.dry_run else []),
    ]
    for cause in request.causes:
        argv += ["--cause", cause]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed script, validated arguments
            argv, capture_output=True, text=True, timeout=600,
            env={**os.environ, "PYTHONPATH": str(REPO)}, check=False)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(504, "the replan did not finish in 600s") from exc
    if completed.returncode != 0:
        # The DSN is in argv and must not come back in an error message.
        raise HTTPException(400, {
            "detail": "the fleet refused this replan",
            "exit_code": completed.returncode,
            "stderr_tail": (completed.stderr or "")[-2000:],
        })
    try:
        receipt = json.loads(completed.stdout)
    except ValueError:
        raise HTTPException(500, "the replan produced no receipt") from None
    return JSONResponse(receipt, status_code=201 if not request.dry_run else 200)


HUMAN_REVIEW_VERDICTS = ("APPROVED", "DEFECTIVE", "REVIEWED", "INSPECT")


class ReviewRequest(BaseModel):
    """A person's opinion of a surface, which is not P2's verdict."""

    verdict: str = Field(min_length=1, max_length=16)
    note: str = Field("", max_length=500)


@app.post("/api/segmentation/surface/{surface_id}/review")
def api_review(surface_id: str, request: ReviewRequest, http: Request):
    """Approved, Defective, Reviewed, Inspect -- from a person, kept apart from QC.

    Stored under `human_review` in the surface's own payload: no migration, and
    no chance of it being read as a geometry or CT verdict. Those two live in
    their own columns and are written by the fleet; this one is written by
    whoever is logged in and says so. A human calling a surface good is not
    certification, and P3/P4 admission still asks the columns.

    ponytail: payload jsonb, not a table. One row per surface already exists.
    """
    if request.verdict not in HUMAN_REVIEW_VERDICTS:
        raise HTTPException(400, f"verdict must be one of {list(HUMAN_REVIEW_VERDICTS)}")
    if not DSN:
        raise HTTPException(409, "CX_DB is not set; there is no control plane")
    import psycopg
    review = {"verdict": request.verdict,
              "note": request.note.strip() or None,
              "by": getattr(http.state, "username", None) or "anonymous",
              "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        with psycopg.connect(DSN, connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute(
                """UPDATE segment_surfaces
                      SET payload = coalesce(payload, '{}'::jsonb)
                                    || jsonb_build_object('human_review', %s::jsonb)
                    WHERE surface_id = %s""",
                (json.dumps(review), surface_id))
            if cur.rowcount == 0:
                raise HTTPException(404, f"no surface {surface_id}")
            conn.commit()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"{type(exc).__name__}: {exc}") from None
    return JSONResponse(review, status_code=201)


@functools.lru_cache(maxsize=8)
def read_tifxyz(uri: str):
    """The three coordinate planes of a TIFXYZ, as float arrays.

    A TIFXYZ is a directory of x.tif, y.tif and z.tif holding the CT coordinate
    of every pixel of the sheet -- so it *is* the surface, where sample_points is
    a 16x16 sample of it. Read the way stage 4 already reads one, with tifffile.

    Cached because one slice request opens the same three objects for each axis,
    and they are private S3 objects a few megabytes each. Eight surfaces is enough
    for somebody clicking through a table and small enough not to matter; the
    cache is per process and dies with it, which is the right lifetime for a
    preview.
    """
    import numpy as np
    import tifffile

    planes = []
    for axis in "xyz":
        target = f"{uri.rstrip('/')}/{axis}.tif"
        if target.startswith("s3://"):
            import fsspec
            with fsspec.open(target, "rb") as handle:
                planes.append(np.asarray(tifffile.imread(handle), dtype="float32"))
        else:
            planes.append(np.asarray(tifffile.imread(target), dtype="float32"))
    if len({plane.shape for plane in planes}) != 1:
        raise ValueError(f"x, y and z disagree on shape: "
                         f"{[plane.shape for plane in planes]}")
    return tuple(planes)


class ModelDownloadRequest(BaseModel):
    """One file from one Hugging Face repository."""

    # Both halves must start with an alphanumeric. Without that, `..` is a legal
    # half -- `a/..` made the destination directory the models root's parent, and
    # `../etc` put two dots in a URL path.
    repo: str = Field(min_length=3, max_length=120,
                      pattern=r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
    file: str = Field(min_length=1, max_length=200)
    revision: str = Field("main", max_length=64)
    # Where it goes under the models root. Defaults to the repository's own name.
    name: str | None = Field(None, max_length=64, pattern=r"^[A-Za-z0-9][\w.-]*$")
    # What a profile says this file has to hash to. Optional, because a model can
    # arrive before a profile exists for it; enforced when given.
    expect_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")


@app.get("/api/models")
def api_models(resolve: bool = False):
    """What the platform's profiles need, and which of them are on disk.

    Not a catalogue of models somebody thinks are good: a list derived from the
    frozen profiles, each naming a checkpoint by SHA-256. That is the only
    preselection that cannot be wrong, because a hardcoded set of repository names
    is a guess about somebody else's naming and ships as a button that 404s.
    """
    rows = declared_checkpoints()
    if resolve:
        # Only for what is missing: a checkpoint already on disk needs no source.
        with ThreadPoolExecutor(max_workers=4) as pool:
            missing = [r for r in rows if not r["installed"]]
            for row, found in zip(missing,
                                  pool.map(resolve_on_hugging_face, missing)):
                row["hugging_face"] = found
    return JSONResponse({
        "root": str(MODELS_ROOT),
        "writable": os.access(MODELS_ROOT, os.W_OK) if MODELS_ROOT.exists() else False,
        "checkpoints": rows,
        "note": "A profile declares checkpoint_sha256 and treats the path as runtime "
                "input, so a checkpoint counts as installed when a file under the "
                "models root hashes to what the profile named.",
    })


@app.post("/api/models/download")
def api_model_download(request: ModelDownloadRequest):
    """Fetch one checkpoint from Hugging Face into the models volume.

    Three refusals, and each is the point rather than paperwork:

    `.safetensors` only. A `.bin` or `.pt` checkpoint is a pickle that executes
    arbitrary code the moment something loads it, and this platform loads
    checkpoints on GPU workers. Safetensors cannot carry code.

    The hash is recorded on arrival and compared against every profile. A file
    that matches one is reported as satisfying it; a file that matches none is
    stored and reported as unrecognised, which is a fact rather than a failure --
    it is how a new model arrives before a profile exists for it.

    Nothing is loaded, imported or executed here. The panel writes bytes to a
    volume and hashes them.
    """
    if not request.file.endswith(".safetensors"):
        raise HTTPException(
            400,
            f"{request.file} is not a .safetensors file. A .bin or .pt checkpoint is "
            "a pickle that runs code when it is loaded, and these are loaded on GPU "
            "workers, so only safetensors is accepted.")
    if "/" in request.file or ".." in request.file:
        raise HTTPException(400, "the file is a name inside the repository, not a path")

    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    # Resolve the revision first, so what lands on disk is recorded against a
    # commit rather than against a branch that moves.
    try:
        with urllib.request.urlopen(
                f"https://huggingface.co/api/models/{request.repo}"
                f"/revision/{request.revision}", timeout=30) as answer:
            meta = json.load(answer)
    except urllib.error.HTTPError as exc:
        raise HTTPException(
            exc.code if exc.code in (401, 403, 404) else 502,
            f"Hugging Face answered {exc.code} for {request.repo} at "
            f"{request.revision}. A 401 or 403 usually means the repository is "
            "gated and needs a token this panel does not hold.") from None
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"could not reach Hugging Face: "
                                 f"{type(exc).__name__}: {exc}") from None

    commit = str(meta.get("sha") or request.revision)
    offered = {f.get("rfilename") for f in meta.get("siblings", [])}
    if request.file not in offered:
        raise HTTPException(
            404, {"detail": f"{request.repo} at {commit[:12]} has no {request.file}",
                  "safetensors_it_does_have":
                      sorted(f for f in offered if str(f).endswith(".safetensors"))[:20]})

    destination = MODELS_ROOT / (request.name or request.repo.split("/")[-1])
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(
            507, f"cannot write to {MODELS_ROOT}: {exc}. The models volume has to be "
                 "mounted read-write on the panel.") from None

    target = destination / request.file
    url = (f"https://huggingface.co/{request.repo}/resolve/{commit}/{request.file}")
    digest = hashlib.sha256()
    written = 0
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=600) as source, partial.open("wb") as sink:
            while chunk := source.read(1024 * 1024):
                sink.write(chunk)
                digest.update(chunk)
                written += len(chunk)
        partial.replace(target)
    except Exception as exc:  # noqa: BLE001
        partial.unlink(missing_ok=True)
        raise HTTPException(502, f"the download failed after {written} bytes: "
                                 f"{type(exc).__name__}: {exc}") from None

    checkpoint_sha256 = digest.hexdigest()
    if request.expect_sha256 and checkpoint_sha256 != request.expect_sha256:
        target.unlink(missing_ok=True)
        raise HTTPException(409, {
            "detail": "the file downloaded is not the one that was asked for",
            "expected": request.expect_sha256,
            "received": checkpoint_sha256,
            "why": "a repository re-uploaded since the profile was frozen serves "
                   "different weights under the same name. It has been deleted "
                   "rather than installed under a hash it does not have.",
        })
    satisfies = [row for row in declared_checkpoints()
                 if row["checkpoint_sha256"] == checkpoint_sha256]
    return JSONResponse({
        "repo": request.repo,
        "revision": commit,
        "file": request.file,
        "path": str(target),
        "bytes": written,
        "checkpoint_sha256": checkpoint_sha256,
        "satisfies": [row["declared_by"] for row in satisfies],
        "recognised": bool(satisfies),
        "note": ("this file is the checkpoint those profiles declare"
                 if satisfies else
                 "no profile declares this hash, so nothing will use it until one "
                 "does. That is expected for a model arriving before its profile."),
    }, status_code=201)


@app.get("/api/segmentation/surface/{surface_id}/slice.png")
def api_slice(surface_id: str,
              axis: str = Query("z", pattern="^[xyz]$"),
              at: float = Query(0.5, ge=0.0, le=1.0),
              window: int = Query(512, ge=64, le=2048),
              threshold: int = Query(0, ge=0, le=255),
              overlay: bool = Query(True),
              slab: int = Query(32, ge=1, le=256)):
    """One orthogonal CT slice through this surface's own bounding box, as a PNG.

    The panel could say a surface exists, its area and its QC state, and could
    not show it -- so the question that decides whether a surface is any good
    ("does this follow one sheet, or did it jump laminae?") could only be asked
    in VC3D. Worse, the human review verdicts this panel now collects were being
    asked of people who had nothing to look at.

    Bounded by the surface, not by the volume. A CT here is 20840x8387x8387
    voxels; a full slice would be hundreds of megabytes and useless. `at` is a
    fraction through the bbox along `axis`, and `window` caps the square read
    around its centre, so one request is one patch-sized read whatever the scroll.

    ponytail: reuses the fleet's open_prediction() rather than a second OME-Zarr
    reader, and PIL for the PNG -- both already in the image. No tile server, no
    caching layer: these are a few hundred kilobytes and the browser caches them.
    """
    if not DSN:
        raise HTTPException(409, "CX_DB is not set; there is no control plane")
    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(DSN, connect_timeout=5) as conn, \
            conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT f.bbox_xyz, f.sample_points, f.artifact_uri, s.ct_uri
                 FROM segment_surfaces f
                 LEFT JOIN segment_source_snapshots s
                        ON s.source_snapshot_id = f.source_snapshot_id
                WHERE f.surface_id = %s""",
            (surface_id,))
        row = cur.fetchone()
    if row is None:
        raise HTTPException(404, f"no surface {surface_id}")
    if not row["ct_uri"]:
        raise HTTPException(409, "this surface's snapshot names no CT volume")
    bbox = row["bbox_xyz"]
    if not (isinstance(bbox, list) and len(bbox) == 2):
        raise HTTPException(409, "this surface has no bounding box to slice through")

    sys.path.insert(0, str(REPO / "framework/stages/01-segmentation"))
    try:
        from mcp.server import open_prediction  # type: ignore
    except ImportError as exc:
        raise HTTPException(503, f"the volume reader is not available: {exc}") from None

    try:
        volume = open_prediction(str(row["ct_uri"]), None, level=0)
    except Exception as exc:  # noqa: BLE001 -- network, auth, missing metadata
        raise HTTPException(502, f"could not open the CT: {type(exc).__name__}: {exc}") from None

    import numpy as np
    # The array is z, y, x; the bbox is x, y, z.
    low = [int(float(bbox[0][k])) for k in range(3)]
    high = [int(float(bbox[1][k])) for k in range(3)]
    depth = "xyz".index(axis)
    plane = int(low[depth] + (high[depth] - low[depth]) * at)

    def span(index: int) -> tuple[int, int]:
        centre = (low[index] + high[index]) // 2
        half = window // 2
        return max(centre - half, 0), max(centre - half, 0) + window

    try:
        if axis == "z":
            (x0, x1), (y0, y1) = span(0), span(1)
            data = volume[plane, y0:y1, x0:x1]
        elif axis == "y":
            (x0, x1), (z0, z1) = span(0), span(2)
            data = volume[z0:z1, plane, x0:x1]
        else:
            (y0, y1), (z0, z1) = span(1), span(2)
            data = volume[z0:z1, y0:y1, plane]
        data = np.asarray(data)
    except Exception as exc:  # noqa: BLE001 -- out of bounds, chunk fetch failures
        raise HTTPException(502, f"could not read that slice: {type(exc).__name__}: {exc}") from None
    if data.ndim != 2 or data.size == 0:
        raise HTTPException(409, f"that slice is empty at {axis}={plane}")

    # Per-slice normalisation. A fixed window would render most of a scroll black:
    # these volumes are 8- or 16-bit with the papyrus occupying a narrow band.
    finite = data[np.isfinite(data)] if data.dtype.kind == "f" else data
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 1.0
    scaled = np.zeros(data.shape, dtype="uint8") if hi <= lo else \
        (((data - lo) / (hi - lo)) * 255).clip(0, 255).astype("uint8")
    if threshold:
        scaled = np.where(scaled >= threshold, scaled, 0).astype("uint8")

    from PIL import Image, ImageDraw
    import io
    image = Image.fromarray(scaled, mode="L")

    # Where the surface sits, from the sample points it already stores.
    #
    # These are a coarse grid -- 256 points over a patch about 1330 voxels
    # across, so roughly 83 voxels apart -- and this is emphatically not a traced
    # outline of the sheet. It answers "is this patch where I think it is", not
    # "does it follow one lamina". The page says so, because a dotted line that
    # looked like a traced surface would be a worse lie than no overlay.
    #
    # The dense version means reading the TIFXYZ, which is the surface: that
    # needs tifffile, an S3 client and the fleet's credentials in this image, and
    # is a separate piece of work rather than a wider slab.
    #
    # `slab` at 32 by default because 8 drew nothing on real data: a sheet meets
    # one plane along a curve, and on an 83-voxel grid almost no sampled point
    # lands within 8 voxels of it.
    drawn = 0
    kind = "none"
    if overlay:
        image = image.convert("RGB")
        pen = ImageDraw.Draw(image)

    # The surface itself, from its TIFXYZ, when that can be read.
    #
    # Every pixel of the sheet carries its CT coordinate, so the pixels whose
    # coordinate along this axis lands near the plane are exactly where the sheet
    # crosses it -- a curve, not a scatter. The sampled grid below is the fallback
    # and stays for surfaces whose artifact cannot be opened: 256 points at 83
    # voxels apart drew 0, 2 and 0 dots on real data, which is a locator and not
    # an outline.
    if overlay and row["artifact_uri"]:
        try:
            px, py, pz = read_tifxyz(str(row["artifact_uri"]))
        except Exception as failure:  # noqa: BLE001 -- private object, absent, bad tiff
            px = None
            fallback_reason = f"{type(failure).__name__}: {failure}"
        else:
            fallback_reason = None
            along = (px, py, pz)[depth]
            # VC3D writes -1 where a pixel of the grid has no surface, and the
            # arrays here run from -1: drawing those would put the sheet at the
            # volume's corner. Every axis has to be valid, not just the one being
            # compared to the plane, or a point lands at a plausible wrong place.
            valid = (np.isfinite(px) & np.isfinite(py) & np.isfinite(pz)
                     & (px > 0) & (py > 0) & (pz > 0))
            near = valid & (np.abs(along - plane) <= slab)
            if near.any():
                # Indexed by depth: dropping x leaves (y, z), dropping y leaves
                # (x, z), dropping z leaves (x, y) -- and these must line up with
                # the u/v offsets below or the curve lands somewhere plausible and
                # wrong, which is the worst kind of wrong for an overlay.
                across = [(py, pz), (px, pz), (px, py)][depth]
                us = across[0][near] - (x0 if axis in ("z", "y") else y0)
                vs = across[1][near] - (y0 if axis == "z" else z0)
                inside = (us >= 0) & (us < image.width) & (vs >= 0) & (vs < image.height)
                # Radius scaled to the slice, for the same reason the sampled
                # grid needed it: single pixels on a 1024-wide image are not
                # visible, and an overlay nobody can see is not an overlay.
                dot = max(window // 512, 1)
                for u, v in zip(us[inside].astype(int), vs[inside].astype(int), strict=True):
                    pen.ellipse([int(u) - dot, int(v) - dot,
                                 int(u) + dot, int(v) + dot], fill=(232, 106, 51))
                drawn = int(inside.sum())
                kind = "tifxyz"
    else:
        fallback_reason = "no artifact"

    if overlay and drawn == 0 and row["sample_points"]:
        flat = [(int(a), int(b), int(c)) for a, b, c in row["sample_points"]]
        for x, y, z in flat:
            here = (x, y, z)[depth]
            if abs(here - plane) > slab:
                continue
            if axis == "z":
                u, v = x - x0, y - y0
            elif axis == "y":
                u, v = x - x0, z - z0
            else:
                u, v = y - y0, z - z0
            if 0 <= u < image.width and 0 <= v < image.height:
                # Radius scaled to the window: fifteen one-pixel dots on a
                # 2048-wide slice are invisible, which made a working overlay
                # look like a broken one.
                dot = max(window // 256, 2)
                pen.ellipse([u - dot, v - dot, u + dot, v + dot], fill=(232, 106, 51))
                drawn += 1
        kind = "sampled-grid" if drawn else "none"

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return Response(
        content=buffer.getvalue(), media_type="image/png",
        headers={"Cache-Control": "private, max-age=300",
                 "X-Slice-Plane": f"{axis}={plane}",
                 "X-Slice-Range": f"{lo:.4g}..{hi:.4g}",
                 # So the page can say "no points near this plane" rather than
                 # leaving somebody to wonder whether the overlay is broken.
                 "X-Overlay-Points": str(drawn),
                 "X-Overlay-Slab": str(slab),
                 "X-Overlay-Kind": kind,
                 **({"X-Overlay-Fallback": fallback_reason[:120]}
                    if kind != "tifxyz" and fallback_reason else {}),
                 "X-Overlay-Available": str(bool(row["sample_points"])).lower()})


class CorrectionPoint(BaseModel):
    x: float
    y: float
    z: float


class CorrectionRequest(BaseModel):
    """One broken area, as VC3D describes it: a chain of points for one problem.

    The first point is special -- it goes on the correct surface near the fault,
    and the rest trace where the sheet should have gone. VC3D's own guidance, kept
    here so the shape of the request matches the shape of the tool.
    """

    points: list[CorrectionPoint] = Field(min_length=2, max_length=500)
    note: str = Field("", max_length=500)
    resume_generations: int | None = Field(None, ge=1, le=40)
    rewind_gen: int | None = Field(None, ge=0, le=1000)


@app.post("/api/segmentation/surface/{surface_id}/corrections")
def api_corrections(surface_id: str, request: CorrectionRequest, http: Request):
    """Store a correction collection and queue a resumed grow from it.

    A drifted surface used to be a dead end: the attempt succeeded, the geometry
    was wrong, and the cell was done. Everything needed to fix it already
    existed -- optional_grow_flags has read resume_from and corrections off the
    locked plan since it was written, and nothing ever wrote them, so --resume
    and --correct were reachable only by hand on a worker.

    The corrected surface is a new artifact with its own identity, never an edit
    of the old one: task identity includes the policy version, so this queues
    under a resume-specific one and the original stays exactly as measured.

    ponytail: the points go to a file beside the manual seeds, in the format
    vc_grow_seg_from_seed already reads, rather than a table of their own.
    """
    if not DSN:
        raise HTTPException(409, "CX_DB is not set; there is no control plane")
    author = getattr(http.state, "username", None) or "anonymous"

    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(DSN, connect_timeout=5) as conn, \
            conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT f.surface_id, f.sample_id, f.artifact_uri, f.bbox_xyz,
                      f.source_snapshot_id
                 FROM segment_surfaces f WHERE f.surface_id = %s""",
            (surface_id,))
        surface = cur.fetchone()
    if surface is None:
        raise HTTPException(404, f"no surface {surface_id}")
    if not surface["artifact_uri"]:
        raise HTTPException(409, "this surface has no artifact to resume from")

    # VC3D's point-collection format, which is also what --correct consumes and
    # what the export endpoint hands back.
    collection = {
        "schema": "campaignx.vc3d_point_collection.v1",
        "surface_id": surface_id,
        "sample_id": surface["sample_id"],
        "coordinate_frame": "ct_l0_xyz",
        "submitted_by": author,
        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": request.note.strip() or None,
        "points": [[p.x, p.y, p.z] for p in request.points],
    }
    digest = hashlib.sha256(
        json.dumps(collection, sort_keys=True).encode()).hexdigest()
    collection["content_sha256"] = digest

    directory = CACHE / "corrections"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{surface_id}-{digest[:12]}.json"
        path.write_text(json.dumps(collection, indent=1), encoding="utf-8")
    except OSError as exc:
        raise HTTPException(500, f"could not store the corrections: {exc}") from None

    script = REPO / "framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py"
    if not script.exists():
        raise HTTPException(409, f"the fleet command is not at {script}")
    argv = [
        sys.executable, str(script), "bootstrap-resume",
        "--db", DSN,
        "--surface", surface_id,
        "--corrections", str(path),
        "--submitted-by", author,
        *(["--resume-generations", str(request.resume_generations)]
          if request.resume_generations else []),
        *(["--rewind-gen", str(request.rewind_gen)]
          if request.rewind_gen is not None else []),
    ]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed script, validated arguments
            argv, capture_output=True, text=True, timeout=300,
            env={**os.environ, "PYTHONPATH": str(REPO)}, check=False)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(504, "the resume did not queue in 300s") from exc
    if completed.returncode != 0:
        raise HTTPException(400, {
            "detail": "the fleet refused this resume",
            "exit_code": completed.returncode,
            "stderr_tail": (completed.stderr or "")[-2000:],
        })
    try:
        receipt = json.loads(completed.stdout)
    except ValueError:
        receipt = {"queued": True, "stdout_tail": (completed.stdout or "")[-2000:]}
    return JSONResponse({"collection": collection, "receipt": receipt},
                        status_code=201)


class MaskRequest(BaseModel):
    """A mask drawn in VC3D, kept for VC3D.

    Deliberately not a grow parameter. vc_grow_seg_from_seed takes exactly
    correct, format, inpaint, mode, resume, resume-generations, resume-opt,
    rewind-gen, segment-name, skip-overlap-check and voxelsize -- there is no
    --mask, and the only --mask in the whole of VC3D belongs to vc_rgb2tifxyz.
    Passing one to the grower would be a control that looks wired and changes
    nothing, which is the defect this panel has spent a while removing.

    So a mask is stored, versioned and handed back: it travels the round trip for
    VC3D's own tools and its GUI overlays, and never pretends to steer a grow.
    """

    png_base64: str = Field(min_length=32, max_length=8_000_000)
    plane: str = Field("z", pattern="^[xyz]$")
    at: int = Field(..., ge=0)
    note: str = Field("", max_length=500)


@app.post("/api/segmentation/surface/{surface_id}/mask")
def api_mask(surface_id: str, request: MaskRequest, http: Request):
    """Store one mask against a surface and a plane through it."""
    import base64
    import io

    try:
        raw = base64.b64decode(request.png_base64, validate=True)
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "png_base64 is not valid base64") from None
    if len(raw) > 6_000_000:
        raise HTTPException(413, f"that mask is {len(raw)} bytes; 6 MB is the cap")

    # Decoded before it is stored, so what comes back out is an image rather than
    # whatever somebody posted.
    from PIL import Image
    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "that is not a readable PNG") from None
    if (image.format or "").upper() != "PNG":
        raise HTTPException(400, f"masks are PNG; that is {image.format}")

    author = getattr(http.state, "username", None) or "anonymous"
    digest = hashlib.sha256(raw).hexdigest()
    directory = CACHE / "masks"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"{surface_id}-{request.plane}{request.at}-{digest[:12]}"
        (directory / f"{stem}.png").write_bytes(raw)
        record = {
            "schema": "campaignx.vc3d_mask.v1",
            "surface_id": surface_id,
            "plane": request.plane,
            "at": request.at,
            "size": list(image.size),
            "content_sha256": digest,
            "bytes": len(raw),
            "submitted_by": author,
            "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": request.note.strip() or None,
            "consumed_by": "vc3d tools and overlays; not vc_grow_seg_from_seed, "
                           "which has no mask option",
        }
        (directory / f"{stem}.json").write_text(json.dumps(record, indent=1),
                                                encoding="utf-8")
    except OSError as exc:
        raise HTTPException(500, f"could not store the mask: {exc}") from None
    return JSONResponse(record, status_code=201)


@app.get("/api/segmentation/surface/{surface_id}/masks")
def api_masks(surface_id: str):
    """The masks stored for this surface, newest first, with their download paths."""
    directory = CACHE / "masks"
    found = []
    if directory.exists():
        for path in sorted(directory.glob(f"{surface_id}-*.json"), reverse=True):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            record["png"] = (f"/api/segmentation/surface/"
                             f"{surface_id}/mask/{record['content_sha256'][:12]}.png")
            found.append(record)
    return JSONResponse({"surface_id": surface_id, "masks": found,
                         "count": len(found)})


@app.get("/api/segmentation/surface/{surface_id}/mask/{digest}.png")
def api_mask_png(surface_id: str, digest: str):
    """One stored mask, back out as the PNG that went in."""
    if not re.fullmatch(r"[0-9a-f]{6,64}", digest):
        raise HTTPException(400, "that is not a hash")
    directory = CACHE / "masks"
    # Globbed on the surface and the digest, so the path cannot be steered by
    # either argument.
    matches = sorted(directory.glob(f"{surface_id}-*-{digest}.png")) if directory.exists() else []
    if not matches:
        raise HTTPException(404, f"no mask {digest} for {surface_id}")
    return Response(content=matches[0].read_bytes(), media_type="image/png",
                    headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/segmentation/surface/{surface_id}/corrections")
def api_corrections_export(surface_id: str):
    """The correction collections for this surface, in VC3D's own format.

    The export half of the round trip: the bundle tells VC3D where to look, this
    hands back the points somebody already placed so they can be reopened,
    amended, or applied on another machine.
    """
    directory = CACHE / "corrections"
    found = []
    if directory.exists():
        for path in sorted(directory.glob(f"{surface_id}-*.json")):
            try:
                found.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
    return JSONResponse({"surface_id": surface_id, "collections": found,
                         "count": len(found)})


class ImportRequest(BaseModel):
    """A patch grown somewhere else, entering the catalogue with its authorship."""

    sample_id: str = Field(min_length=1, max_length=64)
    surfaces: list[dict] = Field(min_length=1, max_length=200)
    owner: str = Field("imported", max_length=64)


@app.post("/api/segmentation/import")
def api_import(request: ImportRequest, http: Request):
    """Register VC3D patch folders as imported surfaces.

    The inbound half. Somebody with a folder of patches grown in VC3D on their
    own machine had no way in: it would have been a manual S3 copy with no hash,
    no owner and no receipt -- a surface the catalogue cannot defend.

    Imported on purpose, never GROWN_HERE: the origin split this panel now draws
    everywhere depends on that, and a patch this fleet did not grow must not be
    counted as its output. Each one needs a bbox and a hash, which is what makes
    it a record rather than a claim.
    """
    if not DSN:
        raise HTTPException(409, "CX_DB is not set; there is no control plane")
    author = getattr(http.state, "username", None) or "anonymous"
    sample = catalog_sample_id(request.sample_id)

    rows = []
    for index, entry in enumerate(request.surfaces):
        uri = str(entry.get("artifact_uri") or entry.get("path") or "").strip()
        digest = str(entry.get("artifact_sha256") or entry.get("sha256") or "").strip()
        bbox = entry.get("bbox_xyz") or entry.get("bbox_l0_xyz")
        if not uri:
            raise HTTPException(400, f"surface {index} has no artifact_uri")
        if not (isinstance(bbox, list) and len(bbox) == 2):
            raise HTTPException(400, f"surface {index} has no bbox_xyz [[x,y,z],[x,y,z]]")
        if len(digest) != 64:
            raise HTTPException(
                400, f"surface {index} needs a 64-character artifact_sha256; "
                     "an unhashed patch is a claim, not a record")
        rows.append((uri, digest, bbox, entry.get("area_cm2")))

    import psycopg
    inserted = 0
    try:
        with psycopg.connect(DSN, connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT source_snapshot_id FROM segment_source_snapshots WHERE sample_id=%s",
                (sample,))
            snapshot = cur.fetchone()
            if snapshot is None:
                raise HTTPException(
                    409, f"{sample} has no source snapshot; bootstrap it before "
                         "importing surfaces against it")
            for uri, digest, bbox, area in rows:
                cur.execute(
                    """INSERT INTO segment_surfaces
                         (surface_id, source_snapshot_id, sample_id, owner,
                          artifact_sha256, artifact_uri, bbox_xyz, area_cm2,
                          state, physical_qc_state, payload, created_at)
                       VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s,
                               'IMPORTED_COVERAGE','UNVALIDATED',%s::jsonb,now())
                       ON CONFLICT (surface_id) DO NOTHING""",
                    (f"import:{digest[:32]}", snapshot[0], sample, request.owner,
                     digest, uri, json.dumps(bbox), area,
                     json.dumps({"source_catalog": "vc3d-import",
                                 "imported_by": author,
                                 "imported_at": datetime.now(timezone.utc)
                                                .isoformat(timespec="seconds")})))
                inserted += cur.rowcount or 0
            conn.commit()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"{type(exc).__name__}: {exc}") from None
    return JSONResponse({"sample_id": sample, "offered": len(rows),
                         "inserted": inserted, "owner": request.owner,
                         "origin": "IMPORTED"}, status_code=201)


@app.get("/api/segmentation/surface/{surface_id}/vc3d")
def api_vc3d_bundle(surface_id: str):
    """Everything VC3D needs to open this surface, and the command that does it.

    Pointers, not copies. vc_grow_seg_from_seed takes --volume, --resume and
    --correct, and the volume here is 20840x8387x8387 voxels behind an HTTP URL --
    a bundle that tried to contain it would be a bundle nobody can download. So
    this names the exact volume, the exact m7 prediction, the frame and scale they
    are in, the surface to resume from with its hash, and prints the command.

    ponytail: JSON, no zip. Everything in it is already a URI the operator's VC3D
    can open, and a zip would be a second thing to keep in step with the
    catalogue.
    """
    if not DSN:
        raise HTTPException(409, "CX_DB is not set; there is no control plane")
    import psycopg
    from psycopg.rows import dict_row
    try:
        with psycopg.connect(DSN, connect_timeout=5) as conn, \
                conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT f.surface_id, f.sample_id, f.area_cm2, f.artifact_uri,
                          f.artifact_sha256, f.bbox_xyz, f.physical_qc_state,
                          to_jsonb(f) ->> 'geometry_qc_state' AS geometry_qc_state,
                          f.payload -> 'human_review' AS human_review,
                          s.ct_uri, s.ct_sha256, s.m7_uri, s.m7_sha256,
                          s.coordinate_frame, s.voxel_size_um, s.shape_xyz,
                          s.payload -> 'eligible_manifest_sha256' AS catalog_sha256,
                          s.payload -> 'source_status' AS source_status
                     FROM segment_surfaces f
                     LEFT JOIN segment_source_snapshots s
                            ON s.source_snapshot_id = f.source_snapshot_id
                    WHERE f.surface_id = %s""",
                (surface_id,))
            row = cur.fetchone()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"{type(exc).__name__}: {exc}") from None
    if row is None:
        raise HTTPException(404, f"no surface {surface_id}")

    row["schema"] = "campaignx.vc3d_open_bundle.v1"
    row["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # The command, with the two things VC3D needs and a placeholder for the third.
    # Correction points are made in VC3D and passed back with --correct; there is
    # nothing to fill in here until somebody draws them.
    row["vc3d_command"] = (
        f"vc_grow_seg_from_seed --volume {row['ct_uri'] or '<volume>'} "
        f"--resume {row['artifact_uri'] or '<surface>'} "
        "--correct <corrections.json>")
    row["corrections"] = f"/api/segmentation/surface/{surface_id}/corrections"
    row["masks"] = f"/api/segmentation/surface/{surface_id}/masks"
    row["notes"] = [
        "Pointers, not copies: --volume takes the CT this surface was grown from.",
        f"Coordinates are {row['coordinate_frame'] or 'unknown'} at "
        f"{row['voxel_size_um'] or '?'} um per voxel.",
        "Draw corrections in VC3D, save the annotation, pass it with --correct.",
        "Masks are stored and handed back for VC3D's own tools. The grower has no "
        "mask option, so one is never passed to a grow.",
    ]
    if row.get("source_status") and "HASH_UNAVAILABLE" in str(row["source_status"]):
        row["notes"].append(
            "The volume is URI-locked, not hash-locked: its bytes were never "
            "hashed, so this bundle identifies it by address and not by content.")
    return JSONResponse(row)


@app.get("/api/segmentation/attempt/{attempt_id}")
def api_attempt(attempt_id: str):
    """Everything recorded about one attempt: its task, its result, its manifest.

    ponytail: no viewer. The browser renders JSON, RunDetail already links to raw
    JSON for P5 receipts, and a bespoke tree widget for a document somebody reads
    twice a month is a widget to maintain. The manifest carries the artifact hash,
    the locked plan's hash, the bbox and the file list, which is the provenance
    the audit asked to be reachable.
    """
    if not DSN:
        raise HTTPException(409, "CX_DB is not set; there is no control plane")
    import psycopg
    from psycopg.rows import dict_row
    try:
        with psycopg.connect(DSN, connect_timeout=5) as conn, \
                conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT a.attempt_id, a.task_id, a.state, a.worker_id,
                          a.created_at, a.updated_at, a.result,
                          -- The plan the worker actually locked, and the proposal
                          -- it locked from. Both are columns on segment_attempts
                          -- and this endpoint promised them in its own docstring
                          -- while selecting neither, so the one place a run's
                          -- decisions could be read showed everything except the
                          -- decisions.
                          a.proposal, a.proposal_sha256,
                          a.locked_plan, a.locked_plan_sha256,
                          t.cell_id, t.policy_version, t.grid_version,
                          t.parameter_envelope, t.catalog_snapshot_sha256, t.payload,
                          art.artifact_set_id, art.manifest, art.manifest_sha256,
                          art.staging_uri
                     FROM segment_attempts a
                     LEFT JOIN segment_tasks t ON t.task_id = a.task_id
                     LEFT JOIN segment_artifact_sets art ON art.attempt_id = a.attempt_id
                    WHERE a.attempt_id = %s""",
                (attempt_id,))
            row = cur.fetchone()
            probe_lineage = {
                "available": False,
                "reason": "probe lineage was not read",
                "runs": [], "trials": [], "attempts": [], "artifacts": [],
                "evaluations": [], "decisions": [], "promotions": [],
            }
            if row is not None:
                # Optional and isolated. A panel image may lead or trail the
                # additive migration, and an unexpected probe-ledger shape must
                # not hide the attempt record that predates it.
                with conn.cursor() as probe_cur:
                    probe_cur.execute("SAVEPOINT panel_probe_lineage")
                    try:
                        probe_lineage = _probe_lineage_query(probe_cur, attempt_id)
                    except Exception as exc:  # noqa: BLE001
                        probe_cur.execute("ROLLBACK TO SAVEPOINT panel_probe_lineage")
                        probe_lineage = {
                            "available": False,
                            "reason": f"{type(exc).__name__}: {exc}",
                            "runs": [], "trials": [], "attempts": [],
                            "artifacts": [], "evaluations": [],
                            "decisions": [], "promotions": [],
                        }
                    finally:
                        probe_cur.execute("RELEASE SAVEPOINT panel_probe_lineage")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"{type(exc).__name__}: {exc}") from None
    if row is None:
        raise HTTPException(404, f"no attempt {attempt_id}")
    for key in ("created_at", "updated_at"):
        if row.get(key):
            row[key] = row[key].isoformat()
    row["probe_lineage"] = probe_lineage
    return JSONResponse(row)


class MaintenanceRequest(BaseModel):
    """One of the fleet's own maintenance commands, by name."""

    action: str = Field(min_length=1, max_length=32)
    sample_id: str | None = Field(None, max_length=64)
    mission_id: str | None = Field(None, max_length=64)
    limit: int = Field(100, ge=1, le=1000)


# ponytail: one endpoint and an allowlist, not three near-identical ones. The
# allowlist is what keeps `action` from being a command line -- it selects a
# subcommand name, it never supplies one. Each entry says which of the request's
# fields that subcommand takes, because coverage does not take a limit and
# republish needs the artifact store.
MAINTENANCE = {
    "republish": ("sample", "limit", "artifact-store"),
    "coverage": ("sample",),
    # `certify` is what the broken "qc" button should have been: it gives a
    # geometry verdict to surfaces that have none, which is exactly what an
    # imported surface needs to leave WAITING_GEOMETRY. Runs as a one-shot with
    # --db and --sample, and is safe to repeat.
    "certify": ("sample", "limit"),
    # No "qc". It was here and it could not work: `qc` is a parser with two
    # subcommands, so `qc --db …` exits 2 on argparse before anything runs. And
    # neither subcommand belongs on a button -- `qc run` is a worker loop wanting
    # --worker-id and --run-root, and `qc enqueue-backfill` wants a manifest and a
    # receipt path the panel does not have. An audit found the dead button; the
    # test below now runs each action's argv through --help so the next one cannot
    # ship.
}


@app.post("/api/segmentation/maintenance")
def api_maintenance(request: MaintenanceRequest):
    """Republish, coverage and QC, which existed only as a terminal command.

    Same shape as the replan endpoint above: shell out to the fleet's own
    command so the rule about what may be re-done lives in one place, and give
    back its receipt. Recovery you can only reach by ssh is recovery that does
    not happen at 2am.
    """
    takes = MAINTENANCE.get(request.action)
    if takes is None:
        raise HTTPException(400, f"not a maintenance action: {request.action!r}; "
                                 f"available: {sorted(MAINTENANCE)}")
    sample = require_write_sample(
        request.mission_id, request.sample_id, f"P1 {request.action}")
    if not DSN:
        raise HTTPException(409, "CX_DB is not set; there is no control plane")
    script = REPO / "framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py"
    if not script.exists():
        raise HTTPException(409, f"the fleet command is not at {script}")

    argv = [sys.executable, str(script), request.action, "--db", DSN]
    if "sample" in takes and sample:
        argv += ["--sample", sample]
    if "limit" in takes:
        argv += ["--limit", str(request.limit)]
    if "artifact-store" in takes:
        # The worker's own variable, so republish writes where the fleet
        # publishes. A second setting here would be a second thing to keep in
        # step with it.
        store = os.environ.get("ARTIFACT_ROOT") or os.environ.get("HELENA_SEGMENT_ARTIFACTS")
        if not store:
            raise HTTPException(409, "neither ARTIFACT_ROOT nor "
                                     "HELENA_SEGMENT_ARTIFACTS is set, and "
                                     "republish moves surfaces into it")
        argv += ["--artifact-store", store]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed script, allowlisted action
            argv, capture_output=True, text=True, timeout=900,
            env={**os.environ, "PYTHONPATH": str(REPO)}, check=False)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(504, f"{request.action} did not finish in 900s") from exc
    if completed.returncode != 0:
        raise HTTPException(400, {
            "detail": f"the fleet refused this {request.action}",
            "exit_code": completed.returncode,
            "stderr_tail": (completed.stderr or "")[-2000:],
        })
    try:
        return JSONResponse(json.loads(completed.stdout), status_code=201)
    except ValueError:
        # coverage prints a report rather than a receipt.
        return JSONResponse({"action": request.action,
                             "stdout_tail": (completed.stdout or "")[-4000:]})


@app.get("/api/segmentation/no-seed")
def api_no_seed(sample: str | None = Query(None),
                mission: str | None = Query(None)):
    """Why segmentation found nothing to grow from.

    NO_SEED is not a cause, it is the label for "none survived". The worker
    writes which of the three screens removed the proposals, and those are
    different problems: an empty prediction means the model says there is no
    sheet there; a CT-support rejection means the model proposed and the raw
    scan disagreed; a clearance rejection means the candidates were good and
    too close to ground already covered -- which is the fleet re-treading, and
    is good news wearing a failure's label.
    """
    resolved = read_scope(mission, sample)
    scope = None if resolved is None else sorted(resolved)
    if not DSN:
        return JSONResponse({"available": False, "reason": "CX_DB is not set"})
    try:
        import psycopg
    except ImportError:
        return JSONResponse({"available": False, "reason": "psycopg is not installed"})
    try:
        with psycopg.connect(DSN, connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT a.result, s.sample_id
                     FROM segment_attempts a
                     LEFT JOIN segment_tasks t ON t.task_id = a.task_id
                     LEFT JOIN segment_source_snapshots s
                            ON s.source_snapshot_id = t.source_snapshot_id
                    WHERE a.state = 'NO_SEED'
                      AND (%s::text[] IS NULL OR s.sample_id = ANY(%s))
                      AND (%s::text IS NULL OR t.mission_id = %s)""",
                (scope, scope, mission, mission))
            rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"available": False, "reason": f"{type(exc).__name__}: {exc}"})

    # These are the keys the worker actually stores in the database, which are
    # not the keys of the fuller diagnosis it writes to
    # NO_SEED_CAUSAL_DIAGNOSIS.json beside the attempt. Reading the source of
    # that file and assuming the row matched was the same mistake this whole
    # audit has been about.
    causes: dict[str, int] = {}
    # Only the two counts the worker writes. There is no post-CT count in the
    # row: reporting a third stage that is always zero invented a screen that
    # rejected everything.
    stages = {"raw_candidate_count": 0, "usable_candidate_count": 0}
    undiagnosed = 0
    diagnosed = 0
    by_sample: dict[str, int] = {}
    reasons: dict[str, int] = {}
    never_proposed = 0
    for result, sample_id in rows:
        by_sample[sample_id or "?"] = by_sample.get(sample_id or "?", 0) + 1
        payload = result if isinstance(result, dict) else {}
        for key in stages:
            stages[key] += int(payload.get(key) or 0)
        if not int(payload.get("raw_candidate_count") or 0):
            never_proposed += 1
        # Tallied before the cause-count check, not after. The reason string is
        # the field these rows actually carry, and skipping to the next row on
        # a missing no_seed_cause_counts threw away the only thing there was to
        # read -- which is why this reported 116 attempts and nothing about them.
        if payload.get("reason"):
            reasons[payload["reason"]] = reasons.get(payload["reason"], 0) + 1
        counts = payload.get("no_seed_cause_counts")
        if not isinstance(counts, dict):
            undiagnosed += 1
            continue
        # An attempt can fail two screens, and by_cause counts occurrences so
        # that filtering on either finds it. Its sum is therefore not a count of
        # attempts, and reading it as one made the arithmetic miss by exactly
        # the number of double-diagnosed rows: 163 causes over 161 attempts.
        #
        # diagnosed is the attempt-level partner: one per row that carries any
        # cause at all, so diagnosed + undiagnosed == attempts holds however
        # many screens a single attempt trips.
        named = False
        for cause, count in counts.items():
            if int(count or 0) > 0:
                causes[cause] = causes.get(cause, 0) + 1
                named = True
        if named:
            diagnosed += 1
        else:
            undiagnosed += 1

    # An attempt whose reason blames a screen while its raw count is zero did
    # not fail that screen: nothing ever reached it. That combination means the
    # seed service returned no candidates, which is a broken search rather than
    # an empty scroll, and the two lead opposite ways -- one is a bug to fix,
    # the other is ground to stop probing.
    contradictory = never_proposed and any(
        "clearance" in reason or "policy" in reason for reason in reasons)
    return JSONResponse({
        "available": True, "attempts": len(rows), "by_cause": causes,
        "candidates_surviving_each_screen": stages,
        "diagnosed": diagnosed,
        "undiagnosed": undiagnosed, "by_sample": by_sample,
        "reasons": reasons,
        "no_candidate_ever_proposed": never_proposed,
        "seed_service_suspected": bool(contradictory),
        "note": "NO_SEED names the screen that removed the proposals. It does "
                "not establish that no physical surface is there."
                + (f" {never_proposed} of {len(rows)} attempts had zero raw "
                   "candidates, so no screen rejected anything -- the seed "
                   "search returned nothing to screen." if contradictory else ""),
    })


GEOMETRY_MEANING = {
    "GEOMETRY_CERTIFIED": "passed all eight requirements of the CT/fibre gate",
    "GEOMETRY_REJECTED_BRIDGE": "two laminae welded across a gap — the join is not papyrus",
    "GEOMETRY_REJECTED_LAMINA_SWITCH": "the sheet jumps to a neighbouring wrap partway",
    "GEOMETRY_REJECTED_DISTORTION": "locally stretched past what a lamina can be",
    "GEOMETRY_REJECTED_COVERAGE": "too little of the grid carries real coordinates",
    "GEOMETRY_UNMEASURED": "the gate did not run — not a verdict, an absence of one",
}


class CertifyRequest(BaseModel):
    sample_id: str | None = Field(default=None, max_length=64)
    mission_id: str | None = Field(default=None, max_length=64)
    limit: int = Field(default=25, ge=1, le=200)
    dry_run: bool = False


@app.post("/api/geometry/certify")
def api_certify(body: CertifyRequest, http: Request):
    """Run P2 over surfaces that carry no verdict.

    P2 and P3 were reachable only from a shell on the host: the phases worked and
    nobody but their author could run them, which is the same as not having them.
    """
    sample = require_write_sample(body.mission_id, body.sample_id, "P2 certification")
    return JSONResponse(enqueue_fleet_phase("P2", {
        "limit": body.limit,
        **({"dry_run": True} if body.dry_run else {}),
        **({"sample": sample} if sample else {}),
    }, sample, body.mission_id, who_asked(http)), status_code=201)


class FlattenRequest(BaseModel):
    sample_id: str | None = Field(default=None, max_length=64)
    mission_id: str | None = Field(default=None, max_length=64)
    surface_id: str | None = Field(default=None, max_length=128)
    limit: int = Field(default=5, ge=1, le=100)
    dry_run: bool = False
    # The override, exposed so the page that shows what the policy holds back
    # can also be the page that decides to proceed anyway.
    allow_unvalidated: bool = False


@app.post("/api/flattening/run")
def api_flatten(body: FlattenRequest, http: Request):
    """Run P3 over certified surfaces this profile has not unrolled yet."""
    sample = require_write_sample(body.mission_id, body.sample_id, "P3 flattening")
    if not FLATTEN_STORE:
        raise HTTPException(409,
            "CX_FLATTEN_STORE is not set. A flattened sheet published nowhere is "
            "one P4 cannot read, the same way a surface on a worker's disk was.")
    return JSONResponse(enqueue_fleet_phase("P3", {
        "limit": body.limit,
        "artifact_store": FLATTEN_STORE,
        **({"dry_run": True} if body.dry_run else {}),
        **({"allow_unvalidated": True} if body.allow_unvalidated else {}),
        **({"sample": sample} if sample else {}),
        **({"surface_id": body.surface_id} if body.surface_id else {}),
    }, sample, body.mission_id, who_asked(http)), status_code=201)


def fleet_store():
    """The segmentation control plane, for the few things the panel writes there."""
    if not DSN:
        raise HTTPException(409, "CX_DB is not set; there is no control plane")
    sys.path.insert(0, str(REPO / "framework/stages/01-segmentation"))
    from fleet.store_factory import open_fleet_store  # noqa: PLC0415

    store = open_fleet_store(DSN)
    store.initialize()
    return store


class SecretEdit(BaseModel):
    value: str = Field(min_length=1, max_length=4096)


@app.get("/api/secrets")
def api_secrets():
    """Which credentials the fleet holds. Never what they are.

    They lived in a file on one host's tmpfs: lost on every reboot, absent on
    every other machine, and placed by hand each time -- which is why surface QC
    and the ink worker both refuse to start after a restart until somebody
    remembers. A worker is ephemeral and has to be able to start from nothing
    but a database URL.
    """
    return JSONResponse({
        "secrets": fleet_store().secret_status(),
        "note": ("Stored in the control plane, where the workers already look. "
                 "Anything that can read that database can read these, which is "
                 "already true of the database password every worker carries -- "
                 "this is not encryption at rest and does not pretend to be."),
    })


@app.put("/api/secrets/{name}")
def api_set_secret(name: str, edit: SecretEdit, request: Request):
    """Set one credential. The value is never read back through this API."""
    author = str(getattr(request.state, "username", "") or "").strip()
    if not author:
        raise HTTPException(403, "setting a credential needs a signed-in author")
    try:
        recorded = fleet_store().set_secret(name, edit.value, author)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse({**recorded, "note": "workers pick it up when they next start"})


@app.delete("/api/secrets/{name}")
def api_forget_secret(name: str):
    if not fleet_store().forget_secret(name):
        raise HTTPException(404, f"{name} is not set")
    return JSONResponse({"name": name, "forgotten": True})


def admissible_physical_states() -> list[str]:
    """The downstream admissibility policy, from the module that owns it.

    Imported rather than repeated: a copy of this tuple in the panel is a second
    policy that agrees with the first until someone edits one of them.
    """
    sys.path.insert(0, str(REPO / "framework/stages/01-segmentation"))
    from fleet.store import ADMISSIBLE_PHYSICAL_QC_STATES  # noqa: PLC0415

    return list(ADMISSIBLE_PHYSICAL_QC_STATES)


@app.get("/api/phases/{phase}/parameters")
def api_phase_parameters(phase: str):
    """Every job parameter this phase accepts, as a form can draw it.

    The panel used to carry its own copy of this list, so a parameter added to
    the queue was invisible in the browser until somebody remembered to add it
    twice -- which is how the direction along the normal, the depth window and
    the render-to-detector chain all reached the API without reaching anyone
    who was not typing curl.
    """
    sys.path.insert(0, str(REPO / "framework/stages/03-ink/fleet"))
    try:
        from job_store import phase_parameter_schema  # noqa: PLC0415

        schema = phase_parameter_schema(phase.upper())
    except Exception as failure:  # noqa: BLE001
        return JSONResponse({"available": False,
                             "reason": f"{type(failure).__name__}: {failure}"})
    if not schema["fields"]:
        return JSONResponse({"available": False,
                             "reason": f"{phase.upper()} takes no queued parameters"})
    return JSONResponse({"available": True, **schema})


@app.get("/api/ink/lanes")
def api_ink_lanes():
    """Every ink model this deployment knows, and whether P5 can run it.

    Three populations nobody had listed together: the lane profiles, the
    adapters that can execute one, and the method registry's record of what each
    checkpoint is worth. A method with no lane profile cannot be queued however
    good it is, and a lane whose adapter the queue has no command for used to be
    routed to whichever runner came first -- which is how every TimeSformer lane
    was handed the ResNet runner's flags.
    """
    sys.path.insert(0, str(REPO / "framework/stages/03-ink/fleet"))
    try:
        from job_store import ink_lane_inventory  # noqa: PLC0415

        lanes = ink_lane_inventory()
    except Exception as failure:  # noqa: BLE001
        return JSONResponse({"available": False,
                             "reason": f"{type(failure).__name__}: {failure}"})
    return JSONResponse({
        "available": True,
        "lanes": lanes,
        "routable": sum(1 for lane in lanes if lane["routable"]),
        "note": ("Routable means the queue can build this lane's command from "
                 "its own adapter. It says nothing about whether the model is "
                 "any good on this scroll: that is validation_status, and "
                 "several routable lanes are CONTROL_SUPPORTED_TARGET_"
                 "UNCALIBRATED or worse."),
    })


@app.get("/api/coverage")
def api_coverage(sample: str | None = Query(None), mission: str | None = Query(None)):
    """How much of a scroll has been explored, per grid version.

    The framework is named for exploration and could not answer this. Coverage
    existed only as a ranking input inside the bootstrap -- how far a candidate
    cell is from the surfaces already grown -- and was never reported, so
    progress was read off a surface count, which rises whether the fleet is
    finding new ground or re-treading old.

    The interesting column is the hit rate. On this control plane one grid found
    a lamina in 30 of 30 cells and another in 1 of 128, which is the difference
    between a policy that works and one that does not, and nothing said so.
    """
    resolved = read_scope(mission, sample)
    if resolved is not None and not resolved:
        return JSONResponse({
            "available": True,
            "schema": "campaignx.segment_coverage.v1",
            "grids": [], "volumes": [], "non_claims": [],
        })
    try:
        store = fleet_store()
        if resolved is None:
            coverage = (store.coverage(None, mission_id=mission)
                        if mission else store.coverage(None))
        else:
            parts = [(store.coverage(scroll, mission_id=mission)
                      if mission else store.coverage(scroll))
                     for scroll in sorted(resolved)]
            coverage = {
                "schema": "campaignx.segment_coverage.v1",
                "grids": [row for part in parts for row in part.get("grids", [])],
                "volumes": [row for part in parts for row in part.get("volumes", [])],
                "non_claims": (parts[0].get("non_claims", []) if parts else []),
            }
        return JSONResponse({"available": True, **coverage})
    except HTTPException as refusal:
        # Degraded like every other read on this panel rather than as a status
        # code: the pages all know the {available, reason} shape, and a 409 here
        # reached the browser as "coverage could not be read" with the reason
        # thrown away.
        return JSONResponse({"available": False, "reason": str(refusal.detail)})
    except Exception as failure:  # noqa: BLE001
        # Said out loud. A coverage page reading "nothing attempted" because the
        # query failed is indistinguishable from an unexplored scroll, which is
        # the exact silence this phase exists to break.
        return JSONResponse({"available": False,
                             "reason": f"{type(failure).__name__}: {failure}"})


@app.get("/api/flattening")
def api_flattening(sample: str | None = Query(None),
                   mission: str | None = Query(None)):
    """P3: which certified surfaces have been unrolled, and how much they moved.

    Two populations, and the gap between them is the phase's backlog: surfaces
    P2 certified, and surfaces P3 has flattened. The area ratio is here rather
    than buried in a receipt because it is the one number that says whether the
    flattening did something reasonable -- a sheet that lost a third of its area
    is not a sheet anyone should measure ink on.
    """
    if not DSN:
        return JSONResponse({"available": False, "reason": "CX_DB is not set"})
    try:
        import psycopg
    except ImportError:
        return JSONResponse({"available": False, "reason": "psycopg is not installed"})
    # A scroll narrows to one; a mission narrows to its own. Same defect as the
    # geometry page had: unscoped, this answered for the whole fleet on a
    # mission's page.
    # stored_scroll, not the strict form: reading is not queueing. A scroll the
    # frozen catalog does not register -- PHerc0139 has published segments and no
    # catalogue entry -- made this page 400, and a phase page that refuses to
    # open is worse than one that reports nothing.
    resolved = read_scope(mission, sample)
    scope = None if resolved is None else sorted(resolved)
    try:
        with psycopg.connect(DSN, connect_timeout=5) as conn, conn.cursor() as cur:
            surface_filters = []
            arguments = []
            if scope is not None:
                surface_filters.append("s.sample_id = ANY(%s)")
                arguments.append(scope)
            if mission:
                # A mission owns both surfaces grown by its P1 tasks and new
                # N->1 surfaces derived by one of its queued jobs.  Joining only
                # through segment_attempts made a successful P8 child disappear
                # from P3 precisely when the lineage changed shape.
                surface_filters.append("""(
                  EXISTS (
                    SELECT 1 FROM segment_artifact_sets art
                    JOIN segment_attempts a ON a.attempt_id=art.attempt_id
                    JOIN segment_tasks t ON t.task_id=a.task_id
                    WHERE art.manifest->>'artifact_sha256'=s.artifact_sha256
                      AND t.mission_id=%s)
                  OR EXISTS (
                    SELECT 1 FROM surface_derivations d
                    JOIN ink_jobs j ON j.job_id=d.job_id
                    WHERE d.child_surface_id=s.surface_id
                      AND j.mission_id=%s)
                )""")
                arguments.extend([mission, mission])
            eligible = """WITH eligible_surfaces AS (
                SELECT DISTINCT s.* FROM segment_surfaces s
            """ + (
                " WHERE " + " AND ".join(surface_filters)
                if surface_filters else ""
            ) + ") "
            cur.execute(
                eligible + "SELECT count(*) FROM eligible_surfaces s "
                "WHERE s.geometry_qc_state = 'GEOMETRY_CERTIFIED'",
                arguments)
            certified = int(cur.fetchone()[0])
            # Certified and waiting on the other axis. P3 consumes a surface the
            # CT supports; one whose CT support was never measured is waiting for
            # a QC job, not failing it, and a backlog that stops moving for that
            # reason should say so here rather than look like a stalled phase.
            cur.execute(
                eligible + "SELECT count(*) FROM eligible_surfaces s "
                "WHERE s.geometry_qc_state = 'GEOMETRY_CERTIFIED' "
                "AND s.physical_qc_state <> ALL(%s)",
                [*arguments, admissible_physical_states()])
            awaiting_physical = int(cur.fetchone()[0])
            cur.execute(
                eligible + "SELECT f.state, count(*) FROM surface_flattenings f "
                "JOIN eligible_surfaces s ON s.surface_id = f.surface_id "
                "GROUP BY 1", arguments)
            by_state = {row[0]: int(row[1]) for row in cur.fetchall()}
            cur.execute(
                eligible + "SELECT f.surface_id, s.sample_id, f.profile_id, f.state, "
                "       f.area_ratio, f.artifact_uri, f.created_at "
                "FROM surface_flattenings f "
                "JOIN eligible_surfaces s ON s.surface_id = f.surface_id "
                "ORDER BY f.created_at DESC LIMIT 60", arguments)
            rows = [{"surface_id": r[0], "sample_id": r[1], "profile_id": r[2],
                     "state": r[3], "area_ratio": float(r[4]) if r[4] is not None else None,
                     "artifact_uri": r[5],
                     "created_at": r[6].isoformat() if r[6] else None}
                    for r in cur.fetchall()]
    except Exception as failure:  # noqa: BLE001
        return JSONResponse({"available": False, "reason": f"{type(failure).__name__}: {failure}"})
    flattened = int(by_state.get("FLATTENED", 0))
    return JSONResponse({
        "available": True,
        "certified": certified,
        "flattened": flattened,
        "awaiting": max(certified - flattened - awaiting_physical, 0),
        "awaiting_physical_qc": awaiting_physical,
        "by_state": by_state,
        "rows": rows,
        "note": ("P3 consumes a certified surface and produces a flat sheet that "
                 "is still a TIFXYZ, so every flattened pixel still names the CT "
                 "voxel it came from. The area ratio is flattened area over "
                 "surface area: 1.0 would be a perfectly developable sheet, and "
                 "no real lamina is one. A sheet below the floor recorded in its "
                 "receipt is kept as a measurement and not published, so P4 "
                 "cannot read it."),
    })


@app.get("/api/geometry")
def api_geometry(mission: str | None = Query(None),
                 sample: str | None = Query(None)):
    """P2's verdict on every surface, and how many carry no verdict at all.

    The count that matters here is GEOMETRY_UNMEASURED. Certification is
    fail-soft by design -- a gate that cannot load records "unmeasured" rather
    than losing the segmentation -- which means the phase reports success by
    saying nothing, and a gate that silently never runs looks exactly like a
    gate that ran and found nothing wrong. Only counting the silence separates
    them.
    """
    if not DSN:
        return JSONResponse({"available": False, "reason": "CX_DB is not set"})
    try:
        import psycopg
    except ImportError:
        return JSONResponse({"available": False, "reason": "psycopg is not installed"})
    try:
        with psycopg.connect(DSN, connect_timeout=5) as conn, conn.cursor() as cur:
            # to_jsonb, not the bare column: a control plane that predates the
            # geometry migration has no such column, and reading it out of the
            # row as JSON yields NULL there instead of failing the whole page.
            # Scoped like every other page. Unscoped, this reported the whole
            # fleet's verdicts on a mission's own page -- forty-two certified
            # beside a segmentation tab reading zero, which is not two views of
            # one campaign but two different questions answered as if they were.
            resolved = read_scope(mission, sample)
            scope = None if resolved is None else sorted(resolved)
            if mission:
                cur.execute(
                    """SELECT to_jsonb(f) ->> 'geometry_qc_state', f.physical_qc_state,
                              f.sample_id, count(DISTINCT f.surface_id)
                         FROM segment_surfaces f
                         JOIN segment_artifact_sets art
                           ON art.manifest->>'artifact_sha256'=f.artifact_sha256
                         JOIN segment_attempts a ON a.attempt_id=art.attempt_id
                         JOIN segment_tasks t ON t.task_id=a.task_id
                        WHERE (%s::text[] IS NULL OR f.sample_id = ANY(%s))
                          AND t.mission_id=%s
                        GROUP BY 1, 2, 3""", (scope, scope, mission))
            elif scope is not None:
                cur.execute(
                    """SELECT to_jsonb(f) ->> 'geometry_qc_state', f.physical_qc_state,
                              f.sample_id, count(*)
                         FROM segment_surfaces f
                        WHERE f.sample_id = ANY(%s)
                        GROUP BY 1, 2, 3""", (scope,))
            else:
                cur.execute(
                    """SELECT to_jsonb(f) ->> 'geometry_qc_state', f.physical_qc_state,
                              f.sample_id, count(*)
                         FROM segment_surfaces f
                        GROUP BY 1, 2, 3""")
            rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"available": False, "reason": f"{type(exc).__name__}: {exc}"})

    geometry: dict[str, int] = {}
    physical: dict[str, int] = {}
    by_sample: dict[str, int] = {}
    total = 0
    for geometry_state, physical_state, sample, count in rows:
        state = geometry_state or "GEOMETRY_UNMEASURED"
        geometry[state] = geometry.get(state, 0) + count
        physical[physical_state or "UNVALIDATED"] = \
            physical.get(physical_state or "UNVALIDATED", 0) + count
        by_sample[sample or "?"] = by_sample.get(sample or "?", 0) + count
        total += count

    unmeasured = geometry.get("GEOMETRY_UNMEASURED", 0)
    return JSONResponse({
        "available": True, "surfaces": total,
        "by_geometry_state": geometry, "by_physical_state": physical,
        "by_sample": by_sample, "unmeasured": unmeasured,
        "meaning": GEOMETRY_MEANING,
        # Said plainly rather than left for the reader to infer from a zero:
        # every surface predating the gate is unmeasured, and that is a
        # different sentence from "every surface was checked and passed".
        "note": ("no surface here carries a geometry verdict: all of them were "
                 "finalized before the gate existed" if unmeasured == total and total
                 else f"{unmeasured} of {total} surfaces were never measured"),
    })


@app.get("/api/segmentation/public")
def api_public_segments(refresh: bool = Query(False)):
    return JSONResponse(public_segments(refresh=refresh))


def catalog_sample_id(name: str, *, strict: bool = True) -> str:
    """The frozen catalog's spelling of a scroll the rest of the panel calls `name`.

    P0 takes scroll names from the bucket, where the folder is PHerc0826. The
    catalog registers the same scroll as PHerc826, and the catalog name is the
    one hashed into every frozen plan -- so renaming it invalidates the cohort
    and fails closeout, which is what happened when it was tried.

    Translating at this boundary instead. Anything queued for a scroll the
    catalog spells differently used to reach the fleet unchanged and die there
    with `ValueError: unknown samples requested`, surfacing in the form as a
    Python traceback, which made New Run unusable for that scroll.
    """
    import re

    def spelling(value: str) -> str:
        # Leading zeros inside the number are the whole difference; case and
        # separators are folded because a third spelling would cost nothing.
        return re.sub(r"0+(\d)", r"\1",
                      value.replace("-", "").replace("_", "").lower())

    try:
        rows = json.loads(CATALOG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return name
    # The same rule bootstrap_sources uses, so the panel and the fleet cannot
    # disagree about which scrolls the catalog registers.
    if isinstance(rows, dict):
        rows = rows.get("entries") or []
    known = {spelling(str(r.get("sample_id", ""))): str(r.get("sample_id", ""))
             for r in rows if isinstance(r, dict)}
    if spelling(name) not in known and not strict:
        # Reading is not queueing. A mission may name a scroll the frozen cohort
        # does not carry -- PHerc0841 and PHerc0846A are in mission `test` and
        # in no catalog -- and a page that 400s because of one is a page nobody
        # can open. Queueing still refuses, because there the name has to resolve
        # to a volume.
        return name
    if spelling(name) not in known:
        raise HTTPException(400, {
            "detail": f"{name} is not a scroll the frozen catalog registers",
            "why": ("The fleet resolves a scroll through the catalog, so a name "
                    "only P0 knows reaches it and fails there."),
            "known": sorted(known.values()),
        })
    return known[spelling(name)]


# Where the platform puts what it makes.
#
# Both of these default into /artifacts, which the compose file backs with a
# named Docker volume. A deployment therefore stores everything it produces on
# its own, with no bucket, no credentials and nothing to configure -- and the
# volume outlives the containers.
#
# Object storage is the alternative, not the foundation: give either setting an
# s3://bucket/prefix and open_artifact_store sends that artifact class there
# instead. It is what you reach for when work runs on more than one machine,
# because a Docker volume lives on one host and a second worker cannot see it.
MODELS_ROOT = Path(_setting(
    "CX_MODELS", "/models",
    "Where model checkpoints live. A Docker volume, mounted read-write here and "
    "read-only by the workers: the panel is the only thing that writes one, and a "
    "worker that could overwrite a checkpoint could change what a frozen profile "
    "means.",
    example="/models"))

# What the platform's own profiles say they need, which is the only preselected
# list worth offering. A hardcoded catalogue of repository names would be a guess
# about somebody else's naming, and a wrong one ships as a button that 404s.
_HASHED: dict[tuple, str] = {}

# Every model_family a profile names is the repository name Scroll Prize
# publishes it under, which is why the preselected list is derived rather than
# typed: a hardcoded catalogue is a guess about somebody else's naming that
# ships as a button returning 404.
HF_PUBLISHER = "scrollprize"

# What a checkpoint can arrive as. Only .safetensors is downloadable -- the rest
# execute code when they load -- but all of them count as installed.
CHECKPOINT_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".pth", ".bin")


def hugging_face_model(repo: str, revision: str = "main", timeout: int = 30) -> dict:
    """Ask Hugging Face what is in a repository. Metadata only, no download."""
    import urllib.request  # noqa: PLC0415

    url = (f"https://huggingface.co/api/models/{repo}/revision/{revision}"
           "?blobs=true")
    with urllib.request.urlopen(url, timeout=timeout) as answer:
        return json.load(answer)


def resolve_on_hugging_face(row: dict) -> dict:
    """Find the file whose hash is the one the profile declared.

    The listing carries `lfs.sha256` for every large file, so a checkpoint can be
    matched against a frozen profile before a byte is transferred. A repository
    that has been re-uploaded stops matching and says so, instead of installing
    something else under the right name.
    """
    family = row.get("model_family")
    if not family:
        return {"state": "no_family",
                "why": "the profile declares a hash and no model family, so there "
                       "is no repository name to derive"}
    repo = f"{HF_PUBLISHER}/{family}"
    import urllib.error  # noqa: PLC0415

    try:
        meta = hugging_face_model(repo)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"repo": repo, "state": "not_published",
                    "why": "no repository of that name. The model family is an "
                           "internal recipe name rather than something published, "
                           "so this checkpoint has to be installed by hand"}
        if exc.code in (401, 403):
            return {"repo": repo, "state": "gated",
                    "why": "Hugging Face answers 401 both for a gated repository "
                           "and for one that does not exist, so as not to leak "
                           "which. Either way the panel cannot fetch it"}
        return {"repo": repo, "state": "unreachable",
                "why": f"Hugging Face answered {exc.code}"}
    except Exception as exc:  # noqa: BLE001
        return {"repo": repo, "state": "unreachable",
                "why": f"{type(exc).__name__}: {exc}"}

    files = meta.get("siblings", [])
    common = {"repo": repo, "revision": str(meta.get("sha") or "main"),
              "gated": bool(meta.get("gated"))}

    def hashes_to_ours(f: dict) -> bool:
        return (f.get("lfs") or {}).get("sha256") == row["checkpoint_sha256"]

    # The hash has to match *and* the file has to be one the panel will fetch.
    # A .ckpt that hashes correctly is still a pickle, and reporting it as
    # available would put a button on the page that the download endpoint
    # refuses -- which reads as a broken panel rather than as a deliberate rule.
    match = next((f for f in files if hashes_to_ours(f)
                  and str(f.get("rfilename", "")).endswith(".safetensors")), None)
    if match:
        return {**common, "state": "exact", "file": match["rfilename"],
                "bytes": (match.get("lfs") or {}).get("size"),
                "why": "the file in this repository hashes to what the profile "
                       "declares, so downloading it satisfies the profile"}

    pickled = next((f for f in files if hashes_to_ours(f)), None)
    if pickled:
        return {**common, "state": "pickle_only", "file": pickled["rfilename"],
                "bytes": (pickled.get("lfs") or {}).get("size"),
                "why": f"the checkpoint this profile names is published only as "
                       f"{pickled['rfilename']}, which is a pickle: loading it "
                       "executes whatever was serialised into it, on a GPU worker. "
                       "Convert it to safetensors and install it by hand"}

    safetensors = [f["rfilename"] for f in files
                   if str(f.get("rfilename", "")).endswith(".safetensors")]
    if safetensors:
        return {**common, "state": "mismatch", "safetensors": safetensors,
                "why": "the repository exists and none of its files hash to what "
                       "the profile declares, so it has been re-uploaded since the "
                       "profile was frozen"}
    return {**common, "state": "no_safetensors",
            "files": [f.get("rfilename") for f in files][:20],
            "why": "this repository publishes no safetensors, and the remaining "
                   "formats are pickles the panel will not fetch"}


def declared_checkpoints() -> list[dict]:
    """Every checkpoint a frozen profile names, and whether it is installed.

    A profile is authoritative about identity -- it declares checkpoint_sha256 and
    says the path is runtime input only -- so "do we have this model" is a question
    about hashes on disk and not about filenames.
    """
    import re as _re

    wanted: dict[str, dict] = {}
    for path in sorted((REPO / "framework/profiles").rglob("*.json")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for match in _re.finditer(r'"checkpoint_sha256":\s*"([0-9a-f]{64})"', text):
            digest = match.group(1)
            family = _re.search(r'"model_family":\s*"([^"]+)"', text)
            wanted.setdefault(digest, {
                "checkpoint_sha256": digest,
                "model_family": family.group(1) if family else None,
                "declared_by": [],
            })
            wanted[digest]["declared_by"].append(path.name)

    installed = {}
    if MODELS_ROOT.exists():
        # Every checkpoint format, not only the one the panel will fetch.
        #
        # Refusing to download a pickle is a policy about what this panel will
        # pull off the internet. Refusing to *see* one is just being wrong: the
        # canonical ink checkpoint on gpu-1 is an r152.ckpt that a frozen
        # profile names by hash, it is installed, and the page called it
        # missing.
        for candidate in sorted(
                path for suffix in CHECKPOINT_SUFFIXES
                for path in MODELS_ROOT.rglob(f"*{suffix}")):
            try:
                stat = candidate.stat()
            except OSError:
                continue
            # A checkpoint is hundreds of megabytes and this runs on a page load,
            # so the hash is remembered against size and mtime. A file that
            # changed changes one of them.
            key = (str(candidate), stat.st_size, stat.st_mtime_ns)
            digest = _HASHED.get(key)
            if digest is None:
                try:
                    digest = artifact_contract.file_sha256(candidate)
                except OSError:
                    continue
                _HASHED[key] = digest
            installed[digest] = str(candidate)
    for digest, row in wanted.items():
        row["installed_at"] = installed.get(digest)
        row["installed"] = digest in installed
    return sorted(wanted.values(), key=lambda r: (not r["installed"],
                                                  r["model_family"] or ""))


RENDER_STORE = _setting(
    "CX_RENDER_STORE", "/artifacts/layer-stacks-v1",
    "Where P4 publishes a layer stack. A local path -- the default is inside "
    "the platform's own durable volume -- or s3://bucket/prefix to publish to "
    "object storage instead, which is what a fleet spanning hosts needs. "
    "Everything else in the pipeline publishes, and the stack the detector eats "
    "was once the one artifact left on the worker's own disk.",
    example="s3://bucket/layer-stacks-v1")
INK_STORE = _setting(
    "CX_INK_STORE", "/artifacts/ink-maps-v1",
    "Where P5 publishes probability maps so P7 can consume them from another "
    "worker. A durable local path is used by default; use an s3:// destination "
    "for a fleet spanning hosts.",
    example="s3://bucket/ink-maps-v1")
SOURCE_URL = _setting(
    "CX_SOURCE_URL", "",
    "Where this platform's source lives, linked in the panel's footer. Empty "
    "hides the link: there is no repository URL anywhere in this checkout, and a "
    "guessed one in the footer of a control panel is worse than no link at all.",
    example="https://github.com/your-org/helena")
FLATTEN_STORE = _setting(
    "CX_FLATTEN_STORE", "/artifacts/flattened-v1",
    "Where flattened sheets are published: a local path, by default inside the "
    "platform's own durable volume, or s3://bucket/prefix for object storage. "
    "Empty keeps them nowhere, which is only useful for a trial run.",
    example="s3://bucket/flattened-v1")
RECONSTRUCTION_STORE = _setting(
    "CX_RECONSTRUCTION_STORE", "/artifacts/reconstruction-v1",
    "Where P8 publishes merged TIFXYZ surfaces and their immutable lineage: "
    "a durable local path or s3://bucket/prefix for a multi-host fleet.",
    example="s3://bucket/reconstruction-v1")


def enqueue_fleet_phase(phase: str, parameters: dict, sample_id: str | None,
                        mission_id: str | None = None,
                        created_by: str = "panel") -> dict:
    """Queue a phase. The panel does not run anything.

    P2 and P3 used to be a `docker run` from this process, which meant
    containerising the panel would have required handing it the Docker socket --
    host control given to a web process to save writing a row. They are jobs
    now, claimed by a worker that already carries the runtime, like every other
    phase.
    """
    store = job_store()
    try:
        job_id = store.enqueue(
            sample_id=catalog_sample_id(sample_id) if sample_id else "fleet",
            phase=phase, parameters=parameters,
            # Attributable. Without the mission these rows belong to no campaign
            # and a mission's own P2 and P3 work is invisible on its page.
            mission_id=mission_id, profile_id=None, component=None,
            priority=0, requested_host=None, max_attempts=1,
            created_by=created_by)
    except Exception as exc:  # JobRejected and friends
        raise HTTPException(400, str(exc)) from exc
    return {"job_id": job_id, "phase": phase, "parameters": parameters,
            "note": ("queued; a worker claims it. Watch it on the Fleet page or "
                     "at /api/jobs.")}


def bootstrap_counts(stdout: str) -> tuple[int, int]:
    """How many cells the bootstrap considered, and how many became tasks.

    Task identity is (volume, grid version, cell, policy version) with an
    ON CONFLICT DO NOTHING behind it, so a second run over the same ground is
    a no-op by design. It used to return 201 anyway: the form said a run had
    started, the queue was untouched, and the seeder chosen was never tried.
    """
    try:
        queued = json.loads(stdout).get("queued") or {}
    except (ValueError, AttributeError):
        return (0, 0)
    generated = sum(int(v.get("generated", 0)) for v in queued.values())
    inserted = sum(int(v.get("inserted", 0)) for v in queued.values())
    return (generated, inserted)



@app.post("/api/segmentation/runs")
def api_queue_segmentation(request: SegmentationRunRequest, http: Request):
    """Queue segmentation work by running the fleet's own bootstrap.

    The panel shells out to `helena_segment_search_fleet.py bootstrap` rather
    than inserting into segment_tasks itself. That command is what decides which
    grid cells are uncovered, applies the seed-region and candidate policies, and
    stamps the catalogue snapshot hash every task carries -- reproducing any of
    that here would mean two implementations of the same policy, and the one in
    the panel would be the one nobody tested.
    """
    backend = next((b for b in SEGMENTATION_BACKENDS if b["id"] == request.backend), None)
    if backend is None:
        raise HTTPException(400, f"unknown backend {request.backend!r}; "
                                f"available: {[b['id'] for b in SEGMENTATION_BACKENDS]}")
    # Validate the P0 boundary before reading an artifact or creating tasks.  A
    # registry row is evidence, not authority: older versions of the artifact
    # endpoint allowed a row for any sample to be placed inside any mission.
    # The manifest's frozen scroll selection remains the authorization source.
    fleet_sample = require_write_sample(
        request.mission_id, request.sample_id, "P1 segmentation")
    # Not gated on adoptable, deliberately. The form disables the two that do not
    # work so nobody picks one by accident, but the endpoint stays open: the alpha
    # keeps experimenting with them, and a browser affordance is the wrong place to
    # decide what a researcher may try. The choice is recorded either way, so a run
    # made with one is identifiable afterwards rather than indistinguishable.
    # `opencode` is not in the offered list any more -- it reproduces runs written
    # against the v1 packet schema, which is a CLI errand -- but it is still a
    # planner the fleet can run, so a request naming it is honoured. Same reasoning
    # as the backends the form disables: the page decides what is worth offering,
    # not what a researcher may try.
    if module_disabled("P1", request.planner):
        raise HTTPException(409, {
            "detail": f"the {request.planner} seeder is switched off",
            "why": "Switch it back on under Configuration -> Modules, or choose "
                   "another one.",
        })
    if getattr(request, "backend", None) and module_disabled("P1", request.backend):
        raise HTTPException(409, {
            "detail": f"the {request.backend} backend is switched off",
            "why": "Switch it back on under Configuration -> Modules.",
        })
    seeder = next((p for p in SEGMENTATION_SEEDERS if p["id"] == request.planner), None)
    if seeder is None:
        seeder = next((p for p in RETIRED_SEEDERS if p["id"] == request.planner), None)
    if seeder is None:
        raise HTTPException(400, f"unknown seeder {request.planner!r}; "
                                f"offered: {[p['id'] for p in SEGMENTATION_SEEDERS]}; "
                                f"still runnable: {[p['id'] for p in RETIRED_SEEDERS]}")
    if (request.seed_probe_mode == "select"
            and request.planner not in {"cost-aware-v2", "deterministic-v2"}):
        raise HTTPException(
            400,
            "seed-probe select mode may steer only cost-aware-v2 or "
            "deterministic-v2. Shadow mode is available with every lane because "
            "it records the comparison without changing the canonical grow.",
        )
    if request.seed_probe_mode == "select":
        if request.seed_probe_top_k < 2:
            raise HTTPException(
                400,
                "seed-probe select needs at least two candidates; top_k=1 is "
                "a geometry preflight, not a comparison",
            )
        readiness = seed_probe_select_readiness(fleet_sample)
        if not readiness["available"]:
            raise HTTPException(
                409,
                {
                    "detail": "seed-probe select is not authorized for this run",
                    **readiness,
                    "how": (
                        "Run shadow mode first. Enable select only after the "
                        "matched benchmark is approved and the source carries "
                        "content-locked CT and m7 digests."
                    ),
                },
            )
    # A v2 policy needs a v2 seeder.
    #
    # The generator sets planner_contract_version from the policy --
    # adaptive-geometry-history-v2 makes it a v2 task -- and the worker then
    # refuses a v2 task whose planner cannot read a v2 packet, with
    # "planner v2 task has an unsupported candidate selection policy". Refusing
    # is right; letting the form build it and finding out one lease later is not,
    # because the operator learns from a failed attempt rather than from the
    # control that offered it.
    if (request.options.get("candidate_selection_policy") == "adaptive-geometry-history-v2"
            and not request.planner.endswith("-v2")):
        raise HTTPException(
            400,
            f"adaptive-geometry-history-v2 is a v2 policy and {request.planner!r} is a "
            "v1 seeder, so the worker would refuse every task this queues. Choose a v2 "
            "seeder, or leave the policy at score-cell-volume-clearance-v1.",
        )

    # Declared, offered -- and able to travel. Those are three things.
    #
    # The task carries `planner` and `planner_model`, and the worker's
    # planner_factory takes exactly those two, so a seeder field that is not the
    # model reaches no host. Fusion declares panel, judge and effort: they were
    # validated here, echoed back, and dropped, which means an operator could
    # assemble a panel of four models, be told it was accepted, and get the
    # planner's defaults -- at four models' cost, attributed to a configuration
    # that never ran.
    #
    # Refused rather than silently ignored, the same as an unrunnable backend.
    # Wiring them means a per-task planner config through the factory the worker
    # calls; until that exists, saying so is the honest answer.
    # candidate_rank is deliberately absent: the worker reads it off the task and
    # sets it on the planner it builds, so it reaches the host that grows the
    # seed. The three below still do not.
    UNTRAVELLED = {"panel", "judge", "effort"}
    stranded = sorted(UNTRAVELLED & {f for f, v in request.seed_config.items()
                                     if str(v).strip()})
    if stranded:
        raise HTTPException(
            501,
            f"{seeder['name']} cannot carry {stranded} onto a task yet: the worker "
            "builds a planner from the task's planner and planner_model only, so "
            "these would be accepted here and never reach the host that grows the "
            "seed. Leave them empty to run the planner's own defaults.",
        )

    declared = {c["field"] for c in seeder["configures"]}
    extra = sorted(set(request.seed_config) - declared)
    if extra:
        raise HTTPException(400, f"{seeder['name']} takes no {extra}; it configures "
                                f"{sorted(declared) or 'nothing'}")
    # OpenCode is handed provider/model as one string, which is what it parses.
    if seeder["kind"] == "agent" and request.seed_config.get("model"):
        provider = request.seed_config.get("provider", "anthropic")
        model = request.seed_config["model"]
        request.seed_config["model"] = (model if "/" in model
                                        else f"{provider}/{model}")
    if not backend["adoptable"] and not request.reason.strip():
        raise HTTPException(
            400,
            f"{backend['name']} is not an adoptable backend: {backend['note']} "
            "Queueing it anyway is a comparison, so it needs a reason on the record.",
        )
    # Refused rather than run as something else.
    #
    # This parameter was validated, echoed back in the response, and reached
    # nothing: the bootstrap script has no --backend argument, a task carries no
    # backend field, and _worker in the fleet CLI instantiates VC3DGrowExecutor
    # unconditionally. So a request naming ScrollFiesta ran VC3D and the reply
    # said ScrollFiesta -- which is not a missing feature, it is an attribution
    # the record would carry and nobody could detect afterwards.
    #
    # Not implemented instead of dispatched because there is nothing to dispatch
    # to: no ScrollFiesta or Thaumato executor exists in the fleet. Wiring the
    # backend onto the task is worth doing when one does, and until then saying
    # so is the only honest answer.
    if backend["id"] != "vc3d":
        raise HTTPException(
            501,
            f"the queue runs VC3D only. {backend['name']} is a comparison backend "
            "with no executor in the fleet, so a task naming it would grow on VC3D "
            "and be recorded under the wrong method. Run it outside the queue and "
            "register what it produces, or add its executor first.",
        )
    if not DSN:
        raise HTTPException(409, "CX_DB is not set; there is no control plane to queue into")
    if request.seed_probe_mode != "off":
        worker_readiness = seed_probe_worker_readiness()
        if not worker_readiness["available"]:
            raise HTTPException(
                409,
                {
                    "detail": (
                        "no recently checked-in worker can execute seed-probe-v1; "
                        "nothing was queued"
                    ),
                    **worker_readiness,
                    "how": (
                        "Start a worker with HELENA_SEED_PROBE_SUPPORT=1 and "
                        "--seed-probe-support, then wait for its capability check-in."
                    ),
                },
            )
    if not CATALOG.exists():
        raise HTTPException(409, f"the frozen catalog is not at {CATALOG}")

    script = REPO / "framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py"
    if not script.exists():
        raise HTTPException(409, f"the fleet bootstrap is not at {script}")

    # One owner for the grid step.
    #
    # It was declared twice: a top-level request field with a default of 2048,
    # and an entry in SEGMENTATION_OPTIONS carrying the same --grid-step flag.
    # The loop below therefore had to skip it to avoid passing the flag twice --
    # and the form sends it inside `options`, so the number an operator typed
    # went into the dict the loop ignores while the command line got the
    # untouched default. Every run was 2048 whatever the page said.
    #
    # Options win when present, because that is where the form puts it, and the
    # bounds from the field are re-checked here: validation lived on the
    # top-level field and nothing was validating the path actually in use.
    # What this run will read, resolved from the control plane rather than taken
    # from the browser's word for it: a request naming its own selection would be
    # a per-run override nothing else knows about.
    #
    # artifact_contract.resolve is the function the rest of the platform uses for
    # "which artifact should this phase read", and it distinguishes a selection
    # from "the newest one, because nothing was selected". Those are different
    # claims and the task records which it was.
    p0_selection = None
    p0_artifact = None
    p0_record = None
    if request.mission_id:
        # No longer fails open. It used to swallow everything and queue a run with
        # blank provenance, which is the failure mode that makes a record
        # untrustworthy: the run looks provenanced and is not. A mission that
        # cannot be read is a reason to stop, because the operator asked for a run
        # against a selection and we cannot say which one that is.
        try:
            directory = mission_directory(request.mission_id)
            p0_record = artifact_contract.resolve(directory, "P0", request.sample_id)
            p0_selection = artifact_contract.current_selection(directory)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                409, f"cannot read mission {request.mission_id}'s P0 selection "
                     f"({type(exc).__name__}: {exc}), so this run cannot say what "
                     "it read. Fix the mission or queue without a mission.") from None
        if p0_record is None:
            raise HTTPException(
                409, f"no P0 artifact is registered for {request.sample_id} in "
                     f"mission {request.mission_id}. P1 grows from what P0 froze, "
                     "so there is nothing to grow from yet.")
        p0_artifact = p0_record.get("artifact_id")

    # The silent no-op, refused.
    #
    # Task identity is (source_snapshot, grid_version, cell, policy_version) with
    # ON CONFLICT DO NOTHING, and the P0 artifact is not part of it. So reselecting
    # and requeueing inserted nothing while the reply said queued, and the tasks
    # that already existed kept the old artifact id -- a control plane that reads
    # one thing and, to anyone asking the task, says it reads another.
    #
    # Adding the artifact to the unique constraint is the other fix and is not
    # this one: on a live table whose existing rows have no artifact id, and where
    # PostgreSQL treats NULLs as distinct, it would weaken the constraint for
    # every task queued without a mission. Refusing the collision gives the same
    # guarantee -- no task can carry a selection it was not created under -- and
    # tells the operator which two selections are in play.
    if p0_artifact and DSN:
        try:
            import psycopg
            with psycopg.connect(DSN, connect_timeout=5) as conn, conn.cursor() as cur:
                cur.execute(
                    """SELECT DISTINCT t.payload ->> 'p0_artifact_id'
                         FROM segment_tasks t
                         JOIN segment_source_snapshots s
                           ON s.source_snapshot_id = t.source_snapshot_id
                        WHERE s.sample_id = %s
                          AND t.payload ->> 'p0_artifact_id' IS NOT NULL
                          AND t.payload ->> 'p0_artifact_id' <> %s""",
                    (fleet_sample, p0_artifact))
                conflicting = [row[0] for row in cur.fetchall()]
        except HTTPException:
            raise
        except Exception:  # noqa: BLE001 -- the check is not the run
            conflicting = []
        if conflicting:
            raise HTTPException(
                409,
                {"detail": f"{request.sample_id} already has tasks queued against a "
                           "different P0 artifact, and task identity does not "
                           "include the artifact -- so these would silently insert "
                           "nothing and keep the older provenance.",
                 "selected_now": p0_artifact,
                 "already_queued_against": conflicting,
                 "remedy": "restore that selection, or queue under a new "
                           "policy_version so the new tasks have an identity of "
                           "their own."})

    grid_step = request.grid_step
    if request.options.get("grid_step") not in (None, ""):
        try:
            grid_step = int(request.options["grid_step"])
        except (TypeError, ValueError):
            raise HTTPException(400, "grid_step must be a whole number") from None
        if not 256 <= grid_step <= 8192:
            raise HTTPException(400, f"grid_step must be between 256 and 8192, not {grid_step}")

    argv = [
        sys.executable, str(script), "bootstrap",
        "--db", DSN,
        "--eligible", str(CATALOG),
        "--catalog", str(GEOMETRY_CATALOG),
        "--sample", fleet_sample,
        "--max-tasks-per-sample", str(request.max_tasks),
        "--grid-step", str(grid_step),
        # Written on every task, so the seeder chosen here is the one that grows
        # them. It used to be sent nowhere: whatever the form said, the planner
        # was whichever one the host happened to be started with.
        "--planner", request.planner,
        # Likewise the model: choosing one and having the host's default run is
        # the same kind of nothing as choosing a planner that never travels.
        *(["--planner-model", request.seed_config["model"]]
          if request.seed_config.get("model") else []),
        # The name of the question this run is asking. Without it a covered cell
        # is a duplicate; with it, the same ground under a new policy.
        *(["--policy-version", request.policy_version] if request.policy_version else []),
        # The rung of m7's ordering to grow. Travels to the task, and from there
        # the worker sets it on the planner it builds.
        *(["--candidate-rank", str(int(request.seed_config["candidate_rank"]))]
          if request.seed_config.get("candidate_rank") else []),
        # Revisiting ground that already produced a surface, which is where
        # m7's alternatives live.
        *(["--reconsider-covered"]
          if str(request.seed_config.get("reconsider_covered", "")).lower()
          in {"1", "true", "yes"} else []),
        # A bounded deterministic layer inside the chosen lane. Pass every
        # value, including ``off``, so the task records what the operator asked
        # for rather than inheriting whichever default a later worker image has.
        "--seed-probe-mode", request.seed_probe_mode,
        "--seed-probe-top-k", str(request.seed_probe_top_k),
        "--seed-probe-generations", str(request.seed_probe_generations),
        *(
            [
                "--seed-probe-benchmark-receipt",
                str(SEED_PROBE_BENCHMARK_RECEIPT),
            ]
            if request.seed_probe_mode == "select"
            and SEED_PROBE_BENCHMARK_RECEIPT is not None
            else []
        ),
        *(
            [
                "--seed-probe-review-owner",
                SEED_PROBE_REVIEW_OWNER,
            ]
            if request.seed_probe_mode == "select"
            and SEED_PROBE_REVIEW_OWNER
            else []
        ),
        # The form says this is kept with the run, so it is: onto every task the
        # bootstrap creates. It used to come back in the HTTP reply and go
        # nowhere -- and the audit trail is not the place for it, because that
        # trail holds no request bodies by design.
        *(["--reason", request.reason.strip()] if request.reason.strip() else []),
        # Who asked. The audit trail already records the caller, but the trail
        # is not the queue: a task row that cannot name its owner cannot be
        # shown to that person as theirs, and cannot be weighed against anybody
        # else's share of the machines.
        "--queued-by", who_asked(http),
        # A scroll is the scientific input; the mission owns this run. The same
        # scroll can be explored by several missions without sharing attempts.
        *(["--mission-id", request.mission_id] if request.mission_id else []),
        # The P0 selection this run reads, by version and by artifact.
        #
        # The catalogue already gives a task its CT and m7 URIs, the coordinate
        # frame, the voxel size and the manifest hash, through the source
        # snapshot it points at. What it could not say was which P0 selection
        # was current when somebody pressed the button -- so a mission that
        # reselected between two runs produced two tasks that look identical and
        # read different inputs.
        #
        # Byte hashes are a separate matter and are not coming: a CT volume here
        # is 20840x8387x8387 voxels behind an HTTP URL, so the snapshot marks
        # itself URI_LOCKED_HASH_UNAVAILABLE and that is the honest ceiling
        # rather than a gap to close.
        *(["--p0-selection-version", p0_selection["version_id"]]
          if p0_selection else []),
        *(["--p0-artifact-id", p0_artifact] if p0_artifact else []),
        # The digest and how it was resolved. An id alone cannot show that the
        # artifact changed underneath, and "selected" is a different claim from
        # "the only one registered".
        *(["--p0-artifact-sha256", str(p0_record["content_sha256"])]
          if p0_record and p0_record.get("content_sha256") else []),
        *(["--p0-resolved-by", str(p0_record["resolved_by"])]
          if p0_record and p0_record.get("resolved_by") else []),
    ]
    # Built from SEGMENTATION_OPTIONS, so a knob cannot reach the command line
    # without being declared, described and offered by the form -- which is what
    # stops this becoming another selector that is accepted and never sent.
    known = {o["field"]: o for o in SEGMENTATION_OPTIONS}
    unknown = sorted(set(request.options) - set(known))
    if unknown:
        raise HTTPException(400, f"not configurable: {unknown}; "
                                f"available: {sorted(known)}")
    for field, value in request.options.items():
        if value is None or value == "" or field == "grid_step":
            continue
        option = known[field]
        # A boolean flag takes no value. Sending one made argparse read it as a
        # positional and refuse the whole command -- "unrecognized arguments:
        # on" -- so asking the CT to check was the one option that could not be
        # set from here.
        if option["kind"] == "toggle":
            on = str(value).lower() in ("1", "true", "on", "yes")
            flag = option["flag"] if on else option.get("off_flag")
            if flag:
                argv += [flag]
            continue
        if option["kind"] == "choice" and str(value) not in option["choices"]:
            raise HTTPException(400, f"{field} must be one of {option['choices']}")
        if option["kind"] in ("int", "float"):
            try:
                float(value)
            except (TypeError, ValueError):
                raise HTTPException(400, f"{field} must be a number") from None
        argv += [option["flag"], str(value)]
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=600,
            env={**os.environ, "PYTHONPATH": str(REPO)})
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(504, "the fleet bootstrap did not finish in 600s") from exc

    # The DSN is in argv and must not come back in an error message.
    redacted = [a if a != DSN else "postgresql://redacted" for a in argv]
    if completed.returncode != 0:
        raise HTTPException(400, {
            "detail": "the fleet bootstrap refused this request",
            "exit_code": completed.returncode,
            "stderr_tail": (completed.stderr or "")[-2000:],
            "command": redacted,
        })
    generated, inserted = bootstrap_counts(completed.stdout)
    if inserted == 0:
        raise HTTPException(409, {
            "detail": (
                f"nothing was queued: all {generated} cells this run covers "
                f"already have a task under grid {request.options.get('grid_version') or 'ct-l0-v1'} "
                f"and policy {request.options.get('policy_version') or 'ink-blind-v1'}."),
            "why": ("A task is identified by volume, grid version, cell and policy "
                    "version, so re-running the same scroll with a different seeder "
                    "writes into the identity that already exists and inserts nothing."),
            "how": ("Name a new policy version to make this a new experiment over "
                    "the same ground, or a smaller grid step to cover ground the "
                    "current tiling skipped."),
            "generated": generated,
            "inserted": 0,
        })
    return JSONResponse({
        "queued_for": request.sample_id,
        "generated": generated,
        "inserted": inserted,
        "backend": backend["id"],
        "planner": request.planner,
        "seed_probe": {
            "mode": request.seed_probe_mode,
            "top_k": request.seed_probe_top_k,
            "generations": request.seed_probe_generations,
        },
        "reason": request.reason.strip() or None,
        "stdout_tail": (completed.stdout or "")[-2000:],
        "command": redacted,
        # The planner is chosen by the worker, not by the queue. Saying so here
        # stops the UI from implying the choice was bound to these tasks.
        "seed_config": request.seed_config or None,
        "planner_note": (f"These tasks carry --planner {request.planner}; a "
                         "worker that claims one uses that planner, or puts the "
                         "task back if it is not equipped to run it"
                         + (f". Configured with {request.seed_config}"
                            if request.seed_config else "") + "."),
    }, status_code=201)


# --------------------------------------------------------------------------
# Artifacts and the selection between them.
#
# An artifact is immutable and identified by its content, so correcting an
# earlier phase adds a version rather than replacing one. Which version a phase
# uses is a separate, versioned choice -- the same discipline configuration
# gets, for the same reason: the question afterwards is "what was selected when
# that run happened".
# --------------------------------------------------------------------------

def mission_directory(mission_id: str) -> Path:
    _refuse_implicit(mission_id)
    try:
        directory, _ = mission_contract.resolve(RUNS, mission_id)
    except mission_contract.MissionError as exc:
        raise HTTPException(404, str(exc)) from exc
    return directory


def phase_inputs(phase_id: str) -> list[str]:
    """Which phases produce what this one needs, from the pipeline contract.

    Read rather than restated: pipeline_phases.json already declares it, and
    two copies of a dependency graph is one copy too many.
    """
    if not PHASES.exists():
        return []
    for entry in json.loads(PHASES.read_text()).get("phases", []):
        if entry.get("id") == phase_id.upper():
            return list((entry.get("prerequisites") or {}).get("produced_by") or [])
    return []


class ArtifactRequest(BaseModel):
    phase: str = Field(min_length=2, max_length=4)
    sample_id: str = Field(min_length=1, max_length=64)
    kind: str = Field(min_length=1, max_length=64)
    path: str = Field(min_length=1, max_length=1024)
    produced_by: str | None = None
    inputs: list[str] = Field(default_factory=list)
    note: str = Field("", max_length=500)


class SelectionRequest(BaseModel):
    """The whole map, never a patch -- see the contract for why."""

    choices: dict[str, str]
    reason: str = Field("", max_length=500)


@app.get("/api/missions/{mission_id}/artifacts")
def api_artifacts(mission_id: str, phase: str | None = Query(None),
                  sample: str | None = Query(None)):
    directory = mission_directory(mission_id)
    rows = artifact_contract.artifacts(directory, phase=phase, sample_id=sample)
    selection = artifact_contract.current_selection(directory)
    chosen = set((selection or {}).get("choices", {}).values())
    for row in rows:
        row["selected"] = row["artifact_id"] in chosen
        row["exists"] = Path(row["path"]).exists()
    return JSONResponse({
        "artifacts": rows,
        "selection": selection,
        "consumes_from": phase_inputs(phase) if phase else [],
    })


@app.post("/api/missions/{mission_id}/artifacts")
def api_register_artifact(mission_id: str, request: ArtifactRequest):
    """Record what a phase produced.

    The path is resolved inside the runs root and refused anywhere else: this
    reads and hashes whatever it is given, so an unconstrained path would let a
    request walk the filesystem this process can see.
    """
    directory = mission_directory(mission_id)
    require_write_sample(mission_id, request.sample_id, "artifact registration")
    root = RUNS.resolve()
    target = Path(request.path)
    target = (root / target).resolve() if not target.is_absolute() else target.resolve()
    if not target.is_relative_to(root):
        raise HTTPException(400, f"an artifact must live under {root}")
    if not target.exists():
        raise HTTPException(404, f"nothing at {target}")
    try:
        record = artifact_contract.register(
            directory, phase=request.phase, sample_id=request.sample_id,
            kind=request.kind, path=target, produced_by=request.produced_by,
            inputs=request.inputs, note=request.note)
    except artifact_contract.ArtifactError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(record, status_code=201)


@app.get("/api/missions/{mission_id}/artifacts/{artifact_id:path}/affects")
def api_artifact_affects(mission_id: str, artifact_id: str):
    """What was computed from this, directly or through a chain.

    The question after correcting an earlier phase. Those results are not
    wrong -- they are answers to a question that was asked with different
    inputs, and this is the only thing that can still say which.
    """
    directory = mission_directory(mission_id)
    try:
        record = artifact_contract.get(directory, artifact_id)
    except artifact_contract.ArtifactError as exc:
        raise HTTPException(404, str(exc)) from exc
    return JSONResponse({
        "artifact": record,
        "affects": artifact_contract.descendants(directory, artifact_id),
    })


def fleet_surfaces_for_backfill() -> list[dict]:
    """Finalised P1 surfaces, with where their bytes are."""
    if not DSN:
        return []
    try:
        import psycopg
    except ImportError:
        return []
    try:
        with psycopg.connect(DSN, connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT surface_id, sample_id, artifact_uri
                     FROM segment_surfaces
                    WHERE artifact_uri IS NOT NULL
                    ORDER BY created_at DESC""")
            return [{"surface_id": s, "sample_id": m, "artifact_uri": u}
                    for s, m, u in cur.fetchall()]
    except Exception:  # noqa: BLE001 -- backfill must not need a healthy database
        return []


@app.post("/api/missions/{mission_id}/artifacts/backfill")
def api_backfill_artifacts(mission_id: str, dry_run: bool = Query(True)):
    """Register what already ran, from the receipts it left.

    Runs predate this register, and a lineage graph that starts empty is a
    lineage graph nobody trusts. A receipt already names its phase, its scroll
    and its output directory, so the register can be reconstructed from what is
    on disk rather than from anybody's memory.

    The inputs are resolved through the current selection, which is honest but
    worth saying plainly: a backfilled edge records what that run *would* read
    today, not what it read then. Runs recorded going forward carry the real
    thing. Backfilled ones are marked so the two are never confused.

    Defaults to a dry run because it writes to an append-only log.
    """
    directory = mission_directory(mission_id)
    pipeline = REPO / "framework" / "contracts" / "pipeline_phases.json"
    planned, done = [], []
    # index_runs() reads INK_PROFILE_RECEIPTs, which are P5 by construction.
    # When another phase starts leaving receipts here, this needs to read the
    # phase off the receipt rather than assume it.
    phase = "P5"
    for run in index_runs():
        output = Path(run.path)
        if not output.exists():
            continue
        entry = {"phase": phase, "sample_id": run.sample_id,
                 "path": str(output), "run_id": run.run_id}
        if dry_run:
            planned.append(entry)
            continue
        record = artifact_contract.record_run(
            directory, pipeline,
            phase=phase, sample_id=run.sample_id, output=output,
            produced_by=f"run:{run.run_id}",
            note="backfilled from the receipt; inputs are today's selection",
            by="backfill")
        if record:
            done.append(record["artifact_id"])

    # P1 surfaces come from the fleet database rather than a receipt on this
    # host. Registering them here rather than in the finalizer is deliberate:
    # segment_tasks carries no mission, and adding a column to a schema with
    # workers holding leases against it is not a thing to do for bookkeeping.
    # The panel already knows which mission is open, so it does the recording.
    for surface in fleet_surfaces_for_backfill():
        entry = {"phase": "P1", "sample_id": surface["sample_id"],
                 "path": surface["artifact_uri"], "run_id": surface["surface_id"]}
        path = Path(surface["artifact_uri"])
        if not path.exists():
            entry["skipped"] = "artifact_uri is not a path on this host"
            planned.append(entry) if dry_run else None
            continue
        if dry_run:
            planned.append(entry)
            continue
        record = artifact_contract.record_run(
            directory, pipeline, phase="P1", sample_id=surface["sample_id"],
            output=path, produced_by=f"surface:{surface['surface_id']}",
            note="backfilled from the fleet database", by="backfill")
        if record:
            done.append(record["artifact_id"])

    return JSONResponse({
        "dry_run": dry_run,
        "would_register" if dry_run else "registered": planned if dry_run else done,
        "count": len(planned if dry_run else done),
        "caveat": "a backfilled input edge is what the run would read today, "
                  "not what it read then",
    })


@app.get("/api/missions/{mission_id}/selection")
def api_selection(mission_id: str):
    directory = mission_directory(mission_id)
    return JSONResponse({
        "current": artifact_contract.current_selection(directory),
        "history": list(reversed(artifact_contract.selections(directory))),
    })


@app.post("/api/missions/{mission_id}/selection")
def api_select(mission_id: str, request: SelectionRequest):
    directory = mission_directory(mission_id)
    try:
        record = artifact_contract.select(
            directory, choices=request.choices, reason=request.reason)
    except artifact_contract.ArtifactError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(record, status_code=201)


@app.post("/api/missions/{mission_id}/selection/{version_id}/restore")
def api_restore_selection(mission_id: str, version_id: str):
    directory = mission_directory(mission_id)
    try:
        record = artifact_contract.restore_selection(directory, version_id)
    except artifact_contract.ArtifactError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(record, status_code=201)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class UserRequest(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=auth_contract.MINIMUM_PASSWORD, max_length=256)


def _session_cookie(response: JSONResponse, token: str) -> JSONResponse:
    """HttpOnly so script cannot read it, Lax so a cross-site form post cannot
    ride it, Secure whenever the panel is actually serving TLS.

    Secure is conditional rather than always-on because a Secure cookie over
    plain http is a cookie the browser silently refuses to send -- the login
    appears to succeed and the next request is a 401 with nothing in the log to
    explain it.
    """
    response.set_cookie(
        auth_contract.COOKIE, token,
        httponly=True, samesite="lax", secure=bool(TLS_CERT),
        max_age=auth_contract.SESSION_DAYS * 24 * 3600, path="/")
    return response


@app.get("/api/session")
def api_session(request: Request):
    """Who is signed in, and whether anybody can be."""
    who = auth_contract.whoami(AUTH_ROOT, request.cookies.get(auth_contract.COOKIE))
    return JSONResponse({
        "username": who,
        "required": REQUIRE_LOGIN,
        "any_users": auth_contract.any_users(AUTH_ROOT),
        # The first account can only be claimed from the host itself.
        "bootstrap_available": (not auth_contract.any_users(AUTH_ROOT)
                                and is_loopback(request)),
        "bootstrap_note": "the first account is created from the host itself; "
                          "a panel already reachable cannot let whoever finds "
                          "it first claim the door",
    })


@app.post("/api/session")
def api_login(request: Request, body: LoginRequest):
    # Named for the trail before the attempt, so a refused sign-in records who
    # was being claimed rather than "anonymous, denied". Whether it worked is
    # the status column's job.
    account = body.username.strip().lower()
    request.state.username = account
    client = request.client.host if request.client else None

    refusal = login_refusal(account, client)
    if refusal:
        why, wait = refusal
        raise HTTPException(429, {"detail": f"too many attempts: {why}",
                                  "retry_after_seconds": wait},
                            headers={"Retry-After": str(wait)})
    try:
        token = auth_contract.login(AUTH_ROOT, body.username, body.password)
    except auth_contract.AuthError as exc:
        record_login_failure(account, client)
        raise HTTPException(401, str(exc)) from exc
    clear_login_failures(account)
    return _session_cookie(JSONResponse({"username": body.username.strip().lower()}), token)


@app.delete("/api/session")
def api_logout(request: Request):
    auth_contract.logout(AUTH_ROOT, request.cookies.get(auth_contract.COOKIE))
    response = JSONResponse({"signed_out": True})
    response.delete_cookie(auth_contract.COOKIE, path="/")
    return response


@app.post("/api/session/bootstrap")
def api_bootstrap(request: Request, body: UserRequest):
    """Create the very first account, from the host only.

    Open so that a panel with no accounts is not a panel nobody can enter, and
    refused from anywhere but loopback so that being first through the door is
    not a way in. Once one account exists this endpoint is closed for good.
    """
    if auth_contract.any_users(AUTH_ROOT):
        raise HTTPException(409, "accounts already exist; sign in and add users there")
    if not is_loopback(request):
        raise HTTPException(
            403,
            "the first account is created from the host itself. On the panel host: "
            "curl -s -X POST localhost:8800/api/session/bootstrap "
            "-H 'Content-Type: application/json' "
            "-d '{\"username\":\"you\",\"password\":\"...\"}'")
    try:
        record = auth_contract.create_user(AUTH_ROOT, body.username, body.password,
                                           by="bootstrap")
    except auth_contract.AuthError as exc:
        raise HTTPException(400, str(exc)) from exc
    token = auth_contract.login(AUTH_ROOT, body.username, body.password)
    return _session_cookie(JSONResponse(record, status_code=201), token)


class ModuleToggle(BaseModel):
    enabled: bool
    reason: str | None = Field(default=None, max_length=200)


class HuggingFaceModule(BaseModel):
    """An ink detector, named by its Hugging Face repository."""

    repo_id: str = Field(min_length=3, max_length=120)
    adapter: str = Field(min_length=3, max_length=120)
    training_pixel_um: float = Field(gt=0, lt=1000)
    frames: int = Field(ge=1, le=512)
    revision: str | None = Field(default=None, max_length=64)
    checkpoint_file: str = Field(default="model.safetensors", max_length=120)
    known_limits: str | None = Field(default=None, max_length=500)


@app.get("/api/modules")
def api_modules(phase: str | None = None):
    """What every phase can be done with, and what is switched on."""
    phases = [phase] if phase else [p["id"] for p in
                                    json.loads(PHASES.read_text())["phases"]]
    return JSONResponse({
        "phases": [{"phase": p, "modules": phase_modules(p)} for p in phases],
        "kinds": {
            "lane": "a program the queue starts",
            "profile": "a model with its weights and physical scale",
            "backend": "what grows a surface",
            "seeder": "what chooses the point to grow from",
            "source": "where the frozen catalog is read from",
        },
    })


@app.post("/api/modules/P5/huggingface")
def api_add_huggingface_module(body: HuggingFaceModule):
    """Add an ink detector by naming its Hugging Face repository.

    A profile is what this platform runs, not a checkpoint: it fixes which
    weights, at what physical scale, through which adapter, and what is not
    claimed. This writes one from the little that cannot be derived, so that
    adding somebody else's model is a form rather than a pull request.

    The weights are not downloaded here. The worker fetches them when the job
    runs and verifies the digest recorded below, which is what makes the run
    reproducible by somebody who was not there.
    """
    try:
        import job_store  # type: ignore
    except ImportError as exc:
        raise HTTPException(503, f"the queue module is unavailable: {exc}") from exc

    adapters = set(job_store.INK_ADAPTERS)
    if body.adapter not in adapters:
        raise HTTPException(400, {
            "detail": f"unknown adapter {body.adapter!r}",
            "why": "An adapter is the command-line contract the model is run "
                   "through. A model that fits none of these needs one written "
                   "for it; the four here are what exist.",
            "adapters": sorted(adapters),
        })
    if "/" not in body.repo_id:
        raise HTTPException(400, "a Hugging Face repository is owner/name")

    # The docstring below promises the worker verifies a digest. It recorded
    # None, so there was nothing to verify and the promise was empty. Hugging
    # Face publishes the SHA-256 of every large file in its listing, so the
    # digest can be recorded at registration without transferring the weights.
    revision = body.revision or "main"
    try:
        meta = hugging_face_model(body.repo_id, revision)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, {
            "detail": f"could not read {body.repo_id} at {revision}: "
                      f"{type(exc).__name__}: {exc}",
            "why": "the repository is checked before a profile is written for it, "
                   "so a typo does not become a profile that fails at run time",
        }) from None
    named = next((f for f in meta.get("siblings", [])
                  if f.get("rfilename") == body.checkpoint_file), None)
    if named is None:
        raise HTTPException(404, {
            "detail": f"{body.repo_id} at {revision} has no {body.checkpoint_file}",
            "files": sorted(f.get("rfilename") for f in meta.get("siblings", []))[:25],
        })
    checkpoint_sha256 = (named.get("lfs") or {}).get("sha256")
    resolved_revision = str(meta.get("sha") or revision)

    slug = body.repo_id.replace("/", "-").lower()
    profile_id = f"{slug}-screening@0.1.0"
    path = PROFILE_DIR / f"{slug}-screening-0.1.0.json"
    if path.exists():
        raise HTTPException(409, f"{profile_id} is already registered")

    profile = {
        "schema": "campaignx.ink_lane_profile.v1",
        "profile_id": profile_id,
        "method_id": f"{slug}@0.1.0",
        "adapter": body.adapter,
        "source": {
            "kind": "huggingface",
            "repo_id": body.repo_id,
            # The commit, not the branch that was named to reach it. A branch
            # moves and the profile is supposed to be frozen.
            "revision": resolved_revision,
            "requested_revision": revision,
            "file": body.checkpoint_file,
        },
        "added_through_panel": True,
        "input_contract": {
            "modality": "registered surface CT stack",
            "frames": body.frames,
            "training_pixel_um": body.training_pixel_um,
            "training_slice_um": body.training_pixel_um,
            "physical_resampling_required": True,
        },
        # Said out loud, because the platform cannot know it and the number a
        # run produces will be read as if somebody did.
        "known_limits": body.known_limits or (
            "Added through the panel from a Hugging Face repository. Nothing "
            "here has been validated against a known positive on this "
            "deployment." + ("" if checkpoint_sha256 else
                             " The repository publishes this file without a "
                             "SHA-256, which happens for small files stored "
                             "outside LFS, so no digest was recorded and the "
                             "worker has nothing to verify against.")),
        "checkpoint_sha256": checkpoint_sha256,
    }
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2) + "\n")

    state = _module_state()
    state.setdefault("added", []).append(
        {"phase": "P5", "id": profile_id, "repo_id": body.repo_id,
         "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    _write_module_state(state)
    # CX_PROFILE_DIR can point outside the checkout -- a deployment that keeps
    # its profiles on a mounted volume is a normal deployment -- so the path is
    # only shortened when it is genuinely inside.
    try:
        shown = str(path.relative_to(REPO))
    except ValueError:
        shown = str(path)
    return JSONResponse({"phase": "P5", "id": profile_id,
                         "path": shown,
                         "checkpoint_sha256": checkpoint_sha256,
                         "note": "It is registered and switched on. The weights "
                                 "are fetched by the worker on first use and "
                                 "checked against the digest above."
                                 if checkpoint_sha256 else
                                 "It is registered and switched on, with no "
                                 "digest: the repository does not publish one "
                                 "for this file."},
                        status_code=201)


@app.post("/api/modules/{phase}/{module_id}")
def api_toggle_module(phase: str, module_id: str, body: ModuleToggle):
    """Switch a module on or off for this deployment.

    Off means the queue refuses it and the forms stop offering it. It does not
    delete anything: a run that used it keeps its receipt, and its record still
    says which module produced it.
    """
    known = {m["id"] for m in phase_modules(phase)}
    if module_id not in known:
        raise HTTPException(404, f"{phase} has no module {module_id!r}; "
                                 f"it offers {sorted(known)}")
    remaining = [m for m in phase_modules(phase)
                 if m["enabled"] and m["id"] != module_id]
    if not body.enabled and not remaining:
        raise HTTPException(409, {
            "detail": f"{module_id} is the last module {phase} has left",
            "why": "A phase with everything switched off cannot run at all, and "
                   "the page would say it is ready.",
        })

    state = _module_state()
    disabled = set(state.get("disabled", []))
    key = f"{phase}:{module_id}"
    disabled.discard(key) if body.enabled else disabled.add(key)
    state["disabled"] = sorted(disabled)
    _write_module_state(state)
    return JSONResponse({"phase": phase, "id": module_id, "enabled": body.enabled})


@app.get("/api/audit")
def api_audit(limit: int = Query(200, ge=1, le=2000),
              user: str | None = None, contains: str | None = None):
    """Every non-read-only request this panel has served, newest first.

    Refusals are included. "Who tried to remove that scroll and was told no" is
    the question this gets asked, and a trail of successes alone cannot answer
    it -- the status column is where that lives.
    """
    entries = read_audit(limit=limit, user=user, contains=contains)
    return JSONResponse({
        "entries": entries, "count": len(entries), "limit": limit,
        "months": sorted((f.stem for f in AUDIT_ROOT.glob("*.jsonl")), reverse=True)
        if AUDIT_ROOT.is_dir() else [],
        "users": sorted({e["user"] for e in entries}),
        "root": str(AUDIT_ROOT),
        # Said out loud rather than discovered: nobody should assume a body was
        # captured, and nobody should worry that a password was.
        "captures": "timestamp, id, user, action, status, duration and client "
                    "address. Request bodies are never recorded.",
    })


@app.get("/api/users")
def api_users():
    return JSONResponse({"users": auth_contract.user_list(AUTH_ROOT),
                         "minimum_password": auth_contract.MINIMUM_PASSWORD})


@app.post("/api/users")
def api_create_user(request: Request, body: UserRequest):
    """No roles: anyone signed in can add anyone. Who did it is recorded."""
    try:
        record = auth_contract.create_user(
            AUTH_ROOT, body.username, body.password,
            by=getattr(request.state, "username", "panel"))
    except auth_contract.AuthError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(record, status_code=201)


@app.post("/api/users/{username}/password")
def api_set_password(username: str, body: UserRequest):
    try:
        auth_contract.set_password(AUTH_ROOT, username.strip().lower(), body.password)
    except auth_contract.AuthError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse({"username": username, "password_changed": True})


@app.delete("/api/users/{username}")
def api_delete_user(username: str):
    try:
        auth_contract.delete_user(AUTH_ROOT, username.strip().lower())
    except auth_contract.AuthError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse({"username": username, "deleted": True})


# --------------------------------------------------------------------------
# Machine tokens
# --------------------------------------------------------------------------
#
# A worker on another host publishes its surfaces here and needs a credential
# to do it. The alternative was copying a person's password into an env file on
# every worker machine, which makes the audit log a lie and turns one leak into
# a password rotation everywhere.

class MachineRequest(BaseModel):
    name: str


@app.get("/api/machines")
def api_machines():
    return JSONResponse({"machines": auth_contract.machine_list(AUTH_ROOT)})


@app.post("/api/machines")
def api_create_machine(request: Request, body: MachineRequest):
    """Mint a token. Returned once, here, and never recoverable afterwards.

    Stored as a hash like everything else that authenticates, so this response
    is the only time the token exists in readable form. Losing it means
    revoking and minting another, which is the correct trade.
    """
    try:
        token = auth_contract.create_machine_token(
            AUTH_ROOT, body.name, by=getattr(request.state, "username", "panel"))
    except auth_contract.AuthError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(
        {"name": body.name.strip().lower(), "token": token,
         "shown_once": True,
         "hint": "put it in the worker's env file as HELENA_PANEL_TOKEN"},
        status_code=201)


@app.delete("/api/machines/{name}")
def api_revoke_machine(name: str):
    revoked = auth_contract.revoke_machine_token(AUTH_ROOT, name.strip().lower())
    if not revoked:
        raise HTTPException(404, f"no machine token named {name!r}")
    return JSONResponse({"name": name, "revoked": True})


@app.get("/api/strips")
def api_strips():
    """Reference strips on disk, with what each one is qualified to judge."""
    return JSONResponse({"strips": strips(), "root": str(STRIPS),
                         "tools": str(REFERENCE_STRIPS)})


@app.post("/api/strips")
async def api_upload_strip(request: Request):
    """Accept one strip-v0 .npz as the raw request body.

    A raw body rather than multipart: this takes exactly one file and multipart
    would mean a dependency to parse a envelope with one part in it.

    No filename from the client is used as a path. The identity comes from the
    strip's own metadata; the body lands in a staging file, is read with
    allow_pickle=False -- an .npz that arrived over HTTP is precisely where
    pickle would be arbitrary code execution -- and is only moved into place
    once it parses and validates as a strip.
    """
    strip_tools()
    STRIPS.mkdir(parents=True, exist_ok=True)

    staged = STRIPS / f".incoming-{os.getpid()}-{int(time.time())}.npz"
    written = 0
    try:
        with staged.open("wb") as handle:
            async for chunk in request.stream():
                written += len(chunk)
                if written > MAX_STRIP_BYTES:
                    raise HTTPException(
                        413, f"a strip over {MAX_STRIP_BYTES // 1024 // 1024} MB is not a strip")
                handle.write(chunk)
        if not written:
            raise HTTPException(400, "the request body was empty")
        try:
            described = read_strip(staged)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"this does not read as a strip: "
                                     f"{type(exc).__name__}: {exc}") from exc
        if described["schema_version"] != STRIP_SCHEMA:
            raise HTTPException(400, f"schema {described['schema_version']!r} is not {STRIP_SCHEMA}")
        if described["problems"]:
            raise HTTPException(400, {"detail": "the strip does not validate",
                                      "problems": described["problems"]})

        # Identity from the metadata, sanitised to one path component. A strip
        # that names another strip's scroll+segment+window *is* that strip's
        # replacement, and overwriting is the honest outcome.
        raw = "-".join(str(described.get(k) or "unknown")
                       for k in ("scroll", "segment_id", "window"))
        strip_id = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-")[:120] or "strip"
        final = STRIPS / f"{strip_id}.npz"
        staged.replace(final)
    finally:
        staged.unlink(missing_ok=True)

    return JSONResponse({**read_strip(final), "qualified": False,
                         "next": "qualify it before anything is scored against it"},
                        status_code=201)


@app.post("/api/strips/{strip_id}/qualify")
def api_qualify_strip(strip_id: str, ct_check: bool = Query(False)):
    """Run the four checks. Without a passing report the scorer says UNQUALIFIED.

    The CT cross-check is off by default because it is the one that reaches the
    network -- and it is also the only check that appeals to anything outside
    the segmentation the strip came from, so a strip qualified without it is
    self-consistent and no more than that.
    """
    path = (STRIPS / f"{_safe_id(strip_id)}.npz")
    if not path.exists():
        raise HTTPException(404, f"no strip {strip_id}")
    argv = [sys.executable, str(strip_tools() / "qualify_strip.py"), str(path)]
    if ct_check:
        argv.append("--ct-check")
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=1800, check=False)
    report = strip_qualification(path)
    # Exit 2 is "the report was written and it did not pass", which is a result.
    if completed.returncode not in (0, 2) or report is None:
        raise HTTPException(400, {"detail": "qualification did not complete",
                                  "exit_code": completed.returncode,
                                  "stderr_tail": (completed.stderr or "")[-2000:]})
    return JSONResponse({"strip_id": strip_id, "overall_pass": report.get("overall_pass"),
                         "checks": report.get("checks", {}), "config": report.get("config", {}),
                         "ct_check": ct_check})


class StripScoreRequest(BaseModel):
    """What to score. A surface directory, or a run that produced one."""

    pred_path: str = Field(min_length=1, max_length=1024)
    mode: str = Field("mesh")


@app.post("/api/strips/{strip_id}/score")
def api_score_against_strip(strip_id: str, request: StripScoreRequest):
    """Score a surface against a strip.

    `pred_path` is resolved inside the runs root and refused anywhere else. The
    scorer is a subprocess that reads whatever path it is handed, so an
    unconstrained one would read any file this process can.
    """
    if request.mode not in ("mesh", "points"):
        raise HTTPException(400, "mode is 'mesh' or 'points'")
    path = (STRIPS / f"{_safe_id(strip_id)}.npz")
    if not path.exists():
        raise HTTPException(404, f"no strip {strip_id}")

    root = RUNS.resolve()
    pred = Path(request.pred_path)
    pred = (root / pred).resolve() if not pred.is_absolute() else pred.resolve()
    if not pred.is_relative_to(root):
        raise HTTPException(400, f"the surface must live under {root}")
    if not pred.exists():
        raise HTTPException(404, f"nothing at {pred}")

    argv = [sys.executable, str(strip_tools() / "score_strip.py"),
            "--strip", str(path), "--mode", request.mode, "--pred", str(pred)]
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=1800, check=False)
    scorecard = pred.with_suffix(".scorecard.json") if pred.is_file() else \
        pred.parent / f"{pred.name}.scorecard.json"
    card = None
    for candidate in (scorecard, pred / "scorecard.json"):
        try:
            if candidate.exists():
                card = json.loads(candidate.read_text())
                break
        except (json.JSONDecodeError, OSError):
            continue
    if completed.returncode != 0 and card is None:
        raise HTTPException(400, {"detail": "scoring did not complete",
                                  "exit_code": completed.returncode,
                                  "stderr_tail": (completed.stderr or "")[-3000:]})
    return JSONResponse({
        "strip_id": strip_id, "mode": request.mode, "pred": str(pred),
        "qualified_reference": read_strip(path)["qualified"],
        "scorecard": card,
        "stdout_tail": (completed.stdout or "")[-4000:],
    })


def _safe_id(value: str) -> str:
    """One path component, or nothing. Never a traversal."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "", value)
    if not cleaned or cleaned in (".", ".."):
        raise HTTPException(400, f"{value!r} is not a strip id")
    return cleaned


def host_images(ssh_target: str) -> dict[str, str] | None:
    """Which image digests a host actually holds.

    Asked over SSH rather than trusted from a record, because the record is
    what somebody meant and this is what is there. A host that cannot be
    reached returns None, which is different from a host with no images.
    """
    try:
        completed = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", ssh_target,
             # The layer chain, not the reported id: the classic and containerd
             # stores give different ids for the same image, which reads as
             # drift that is not there.
             "for i in $(sudo docker images --format '{{.Repository}}' | grep '^helena-' | sort -u); do "
             "printf '%s ' \"$i\"; sudo docker inspect \"$i\" "
             "--format '{{range .RootFS.Layers}}{{slice . 7 19}}+{{end}}' 2>/dev/null; done"],
            capture_output=True, text=True, timeout=25)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    found: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].startswith("helena-"):
            found.setdefault(parts[0], parts[1].rstrip("+"))
    return found


@app.get("/api/hosts/{host_id}/images")
def api_host_images(host_id: str):
    """What this host runs, against what its roles require."""
    row = next((h for h in job_store().hosts() if h["host_id"] == host_id), None)
    if row is None:
        raise HTTPException(404, f"no host {host_id}")
    expected = image_contract.images_for(row.get("roles") or [])
    present = host_images(row["ssh_target"])
    if present is None:
        return JSONResponse({"host_id": host_id, "reachable": False,
                             "expected": expected, "present": {}, "drift": [],
                             "note": "the host did not answer over SSH"})
    return JSONResponse({
        "host_id": host_id, "reachable": True,
        "expected": expected, "present": present,
        "drift": image_contract.drift(expected, present),
        "note": "a role runs an image identified by digest; a tag identifies "
                "nothing, which is how two hosts came to run different bytes "
                "under the same name",
    })


@app.get("/api/hosts")
def api_hosts():
    """Every host, with the panel's own measured live rather than remembered.

    ``last_state`` is written by the ink worker's heartbeat, so a host whose
    worker is stopped keeps showing whatever it last said -- here, hardware from
    eight hours earlier. The panel runs on one of these hosts and can just look,
    so for that row it does, and says which reading it is. Remote hosts still
    report through their worker, because there is nothing else to ask.
    """
    rows = job_store().hosts()
    try:
        # Flat, like every other import of this package in this file, and with no
        # path insert: the module-level one above already put 03-ink/fleet on the
        # path. Importing it as `fleet.ink_worker` meant inserting 03-ink itself,
        # which binds the name `fleet` to that directory for the whole process --
        # and that directory has job_store.py and no store.py. So one call to
        # /api/hosts left `from fleet.store import ...` failing everywhere else,
        # which is P3 reporting ModuleNotFoundError until the panel restarts.
        # ink_worker itself does `from job_store import ...`, so flat is also how
        # it expects to be imported.
        from ink_worker import host_state  # noqa: PLC0415

        live = host_state(RUNS)
    except Exception:  # noqa: BLE001 -- a probe that fails must not empty the page
        live = None
    if live:
        for row in rows:
            if row.get("host_id") != live.get("hostname"):
                continue
            # Merged, not replaced. The panel is on the machine and its reading
            # of cores, RAM and disk is the better one -- but it runs in a
            # container with no card, so its GPU list is empty on a host with
            # two of them. Replacing wholesale is why the Hosts page said "not
            # reported" minutes after that machine rendered a layer stack.
            sys.path.insert(0, str(REPO / "framework/stages/03-ink/fleet"))
            from job_store import merge_gpu_observations  # noqa: PLC0415

            stored = row.get("last_state") or {}
            row["last_state"] = {**stored, **live,
                                 "gpus": merge_gpu_observations(stored, live)}
            row["state_source"] = ("measured now by the panel; cards as the "
                                   "workers on this host last saw them")
            break

    # A segmentation host heartbeats into segment_worker_capabilities, not into
    # ink_hosts.last_state, so a machine running a segmentation worker read as
    # "never seen" while its worker had checked in minutes earlier. Two heartbeats
    # in two namespaces and the page only knew one.
    #
    # Liveness and GPU are taken from it. Cores, RAM and disk are not, because
    # that row is an admission contract -- the fleet filters claims on it -- and
    # widening a frozen schema to fill a table column is the wrong direction.
    for row in rows:
        # Only when nothing better exists. A host running the inventory reporter
        # writes cores, RAM and disk into last_state, and overwriting that with
        # the admission row would trade a full reading for a GPU flag and label
        # it "capabilities only" -- worse than what it replaced.
        if row.get("state_source") or (row.get("last_state") or {}).get("cores"):
            continue
        seen = segmentation_heartbeat(row.get("host_id", ""))
        if not seen:
            continue
        row["last_seen_at"] = seen["updated_at"]
        row["last_state"] = {**(row.get("last_state") or {}), **seen["hardware"]}
        row["state_source"] = (
            f"segmentation worker {seen['worker_id']} — admission capabilities "
            "only, so cores, RAM and disk are not reported")
    return JSONResponse({"hosts": rows})


def segmentation_heartbeat(host_id: str) -> dict | None:
    """The most recent segmentation check-in for a host, if there is one.

    Matched on the worker id being the host id or starting with it: a host runs
    its worker as <host>-<role>, and one machine can run several.
    """
    if not DSN or not host_id:
        return None
    try:
        import psycopg
    except ImportError:
        return None
    try:
        with psycopg.connect(DSN, connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT worker_id, capabilities, updated_at
                     FROM segment_worker_capabilities
                    WHERE worker_id = %(host)s OR worker_id LIKE %(host)s || '-%%'
                    ORDER BY updated_at DESC LIMIT 1""", {"host": host_id})
            row = cur.fetchone()
    except Exception:  # noqa: BLE001 -- the page renders without this
        return None
    if not row:
        return None
    worker_id, capabilities, updated_at = row
    caps = capabilities if isinstance(capabilities, dict) else {}
    hardware: dict = {}
    if caps.get("cuda_available") and caps.get("gpu_model"):
        # Shaped like the ink probe's gpus so the table renders one path.
        hardware["gpus"] = [{
            "index": int(caps.get("cuda_device_index") or 0),
            "name": caps["gpu_model"],
            "used_mb": 0,
            "total_mb": int(float(caps.get("gpu_vram_gb") or 0) * 1024),
            "util_pct": 0,
            "uuid": "",
        }]
    return {"worker_id": worker_id, "hardware": hardware,
            "updated_at": updated_at.isoformat() if updated_at else None}


PROVISION_LOG = CACHE / "provision"


@app.post("/api/hosts")
def api_register_host(request: HostRequest):
    """Register a host, and put the worker on it.

    Registering used to write a row and stop. The host then appeared in the
    table looking ready and claimed nothing, and the only symptom was a queue
    that did not move -- which reads as "no work" rather than "this machine was
    never set up". Everything the worker needs is in the image, so provisioning
    is: install Docker if it is missing, hand over the image, install the two
    units. Nothing is compiled on the host, which is what having an image is
    for.

    Detached rather than awaited: streaming a multi-gigabyte image over SSH
    takes minutes and an HTTP request that holds the connection open that long
    is a request that times out somewhere in the middle, leaving a half-built
    host and no record of why. The log is the record; /api/hosts/{id}/provision
    reads it back.
    """
    job_store().register_host(request.host_id, request.ssh_target, request.roles,
                              request.notes)
    if not request.provision:
        return JSONResponse({"host_id": request.host_id, "provisioning": False},
                            status_code=201)

    script = REPO / "containers" / "provision-host.sh"
    if not script.exists():
        raise HTTPException(500, f"the provisioning script is missing at {script}")
    PROVISION_LOG.mkdir(parents=True, exist_ok=True)
    log = PROVISION_LOG / f"{_safe_id(request.host_id)}.log"
    role = (request.roles or ["worker"])[0]
    with log.open("wb") as sink:
        subprocess.Popen(  # noqa: S603 - fixed script, arguments are validated
            ["/bin/sh", str(script), request.ssh_target, role],
            stdout=sink, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            cwd=str(REPO), start_new_session=True)
    return JSONResponse({"host_id": request.host_id, "provisioning": True,
                         "log": f"/api/hosts/{request.host_id}/provision"},
                        status_code=201)


@app.get("/api/hosts/{host_id}/provision")
def api_provision_log(host_id: str):
    """How provisioning went, or that it never ran."""
    log = PROVISION_LOG / f"{_safe_id(host_id)}.log"
    if not log.exists():
        return JSONResponse({"host_id": host_id, "provisioned": False,
                             "log": "", "reason": "this host was never provisioned "
                                                  "from the panel"})
    text = log.read_text(encoding="utf-8", errors="replace")
    # The script's last line reports the unit states, so "finished" is whether
    # it got there -- not whether the process exited, which a detached child
    # cannot be asked after the fact.
    return JSONResponse({"host_id": host_id, "provisioned": True,
                         "finished": "worker:" in text,
                         "log": text[-8000:]})


# Roles a person assigns. `postgres` is not here and neither is anything else
# that describes where infrastructure happens to live: those are facts about the
# deployment, not choices about what a machine should be asked to do, and a
# checkbox that can silently move the database is a checkbox nobody wants.
#
# `build` has no image requirement -- it means this host compiles images -- so
# it is assignable but contributes nothing to the drift check.
ASSIGNABLE_ROLES = ("segment", "render", "ink", "mesh", "build")
# Real roles that this page cannot assign. Named so a request carrying one is
# understood and ignored rather than refused, while a typo is still an error.
KNOWN_ROLES = ASSIGNABLE_ROLES + ("postgres",)


class HostRolesRequest(BaseModel):
    roles: list[str] = Field(default_factory=list, max_length=8)


@app.post("/api/hosts/{host_id}/roles")
def api_set_host_roles(host_id: str, request: HostRolesRequest):
    """Change what a host is asked to do.

    Roles decide which images the host is expected to hold, and nothing else --
    no claim path filters on them. Said here because the control looks like it
    routes work and does not: a host with no `ink` role still claims ink work if
    a worker is running on it.

    Roles the panel does not offer are preserved rather than dropped. A host
    carrying the control-plane database is tagged `postgres`; a request from a
    page that cannot express that must not be able to remove it, or saving an
    unrelated change would quietly rewrite where the database is meant to be.
    """
    sent = [r.strip().lower() for r in request.roles if r.strip()]
    # A role that exists but is not assignable is dropped from the request rather
    # than refused: it is preserved from the row below regardless, so rejecting
    # it only breaks clients that echo back what they were shown -- removing
    # one assignable role would fail whenever the row carried a role the page
    # cannot show. A string that is no role at all is still an error: silently
    # ignoring a typo would lose the role the caller meant.
    wanted = [r for r in sent if r in ASSIGNABLE_ROLES]
    unknown = [r for r in sent if r not in ASSIGNABLE_ROLES and r not in KNOWN_ROLES]
    if unknown:
        raise HTTPException(400, f"no such role: {unknown}; "
                                 f"assignable: {list(ASSIGNABLE_ROLES)}")

    existing = next((h for h in job_store().hosts()
                     if h.get("host_id") == host_id), None)
    if existing is None:
        raise HTTPException(404, f"no host {host_id!r}")
    kept = [r for r in (existing.get("roles") or []) if r not in ASSIGNABLE_ROLES]
    roles = sorted(set(wanted) | set(kept))
    if job_store().set_host_roles(host_id, roles) == 0:
        raise HTTPException(404, f"no host {host_id!r}")
    return JSONResponse({"host_id": host_id, "roles": roles,
                         "preserved": kept,
                         "note": "roles decide which images this host is expected "
                                 "to hold; they do not gate what it claims"})


@app.post("/api/hosts/{host_id}/enabled")
def api_host_enabled(host_id: str, enabled: bool = Query(...)):
    job_store().set_host_enabled(host_id, enabled)
    return JSONResponse({"host_id": host_id, "enabled": enabled})


@app.post("/api/jobs/init")
def api_jobs_init():
    """Create the ink job tables. Idempotent."""
    job_store().initialize()
    return JSONResponse({"initialized": True})


# --------------------------------------------------------------------------
# Configuration surface
#
# An exploration framework whose knobs are buried in source is an exploration
# framework you cannot explore with. Everything tunable is listed here with its
# current value and where that value came from.
# --------------------------------------------------------------------------

TUNABLE_ROOTS = [
    ("contracts", REPO / "framework" / "contracts", "*.py"),
    ("03-ink", REPO / "framework" / "stages" / "03-ink" / "scripts", "*.py"),
    ("04-validation", REPO / "framework" / "stages" / "04-validation" / "scripts", "*.py"),
    ("06-discovery", REPO / "framework" / "stages" / "06-discovery" / "scripts", "*.py"),
]


@functools.lru_cache(maxsize=1)
def _module_constants(stamp: float) -> list[dict]:
    """Module-level UPPER_CASE literals: the framework's compiled-in knobs."""
    import ast

    del stamp
    out = []
    for group, directory, pattern in TUNABLE_ROOTS:
        if not directory.exists():
            continue
        for path in sorted(directory.glob(pattern)):
            if path.name.startswith("_"):
                continue
            try:
                tree = ast.parse(path.read_text())
            except (SyntaxError, OSError):
                continue
            for node in tree.body:
                targets_ = (
                    [node.target] if isinstance(node, ast.AnnAssign) else
                    list(node.targets) if isinstance(node, ast.Assign) else []
                )
                for target in targets_:
                    if not (isinstance(target, ast.Name) and target.id.isupper()):
                        continue
                    if node.value is None:
                        continue
                    try:
                        value = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        continue
                    if isinstance(value, (str, int, float, bool, list, tuple, dict)):
                        out.append({
                            "group": group,
                            "module": path.stem,
                            "path": str(path.relative_to(REPO)),
                            "name": target.id,
                            "value": list(value) if isinstance(value, tuple) else value,
                            "line": node.lineno,
                        })
    return out


class EnvEdit(BaseModel):
    value: str = Field(max_length=4000)
    reason: str = Field(default="", max_length=300)


class ConstantEdit(BaseModel):
    path: str = Field(max_length=500)
    name: str = Field(max_length=120)
    value: str = Field(max_length=4000)
    reason: str = Field(default="", max_length=500)


def _apply_settings() -> None:
    """Re-resolve the module globals after an override changes.

    The settings are read once at import, so an edit has to push the new value
    back into the globals that the request handlers close over. Only the ones
    that can change meaning mid-process are rebound; anything marked
    requires_restart is written and reported, not pretended to be live.
    """
    global REPO, RUNS, CACHE, SCROLL_SOURCE, SCROLL_TTL, CATALOG
    global PROFILE_DIR, REGISTRY, MAP_MAX_SIZE, PHASES, IMPLEMENTATIONS
    current = {s["name"]: s for s in SETTINGS}

    def value(name: str) -> str:
        overrides = _overrides()
        if name in overrides:
            return overrides[name]
        return os.environ.get(name) or current[name]["default"]

    REPO = Path(value("CX_REPO"))
    RUNS = Path(value("CX_RUNS"))
    CACHE = Path(value("CX_CACHE"))
    SCROLL_SOURCE = value("CX_SCROLL_SOURCE")
    SCROLL_TTL = int(value("CX_SCROLL_TTL"))
    CATALOG = Path(value("CX_CATALOG"))
    PROFILE_DIR = Path(value("CX_PROFILE_DIR"))
    REGISTRY = Path(value("CX_REGISTRY"))
    MAP_MAX_SIZE = int(value("CX_MAP_MAX_SIZE"))
    PHASES = REPO / "framework" / "contracts" / "pipeline_phases.json"
    IMPLEMENTATIONS = REPO / "framework" / "registries" / "phase-implementations-0.1.0.json"

    for setting in SETTINGS:
        overrides = _overrides()
        if setting["name"] in overrides:
            setting["value"], setting["source"] = overrides[setting["name"]], "override"
        elif os.environ.get(setting["name"]) is not None:
            setting["value"] = os.environ[setting["name"]]
            setting["source"] = "environment"
        else:
            setting["value"], setting["source"] = setting["default"], "default"

    index_runs(force=True)
    _quantised.cache_clear()
    _raster.cache_clear()


def _effective() -> dict[str, str]:
    """The settings map as it stands: override, else environment, else default."""
    overrides = _overrides()
    out = {}
    for setting in SETTINGS:
        name = setting["name"]
        out[name] = overrides.get(name) or os.environ.get(name) or setting["default"]
    return out


def _write_overrides(values: dict[str, str]) -> None:
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDES_PATH.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")


@app.put("/api/config/env/{name}")
def api_set_env(name: str, edit: EnvEdit):
    """Change one setting, which produces one new configuration version.

    A value never moves alone: the whole map is snapshotted, hashed and given
    an id, because the question asked afterwards is what the configuration was
    when a run happened, not what one knob was.
    """
    setting = next((s for s in SETTINGS if s["name"] == name), None)
    if setting is None:
        raise HTTPException(404, f"no setting {name}")
    if setting["kind"] == "int":
        try:
            int(edit.value)
        except ValueError as exc:
            raise HTTPException(400, f"{name} must be an integer") from exc
    if setting["allowed"] and edit.value not in setting["allowed"]:
        raise HTTPException(400, f"{name} must be one of {setting['allowed']}")

    _ensure_config_baseline()
    before = _effective()
    overrides = _overrides()
    overrides[name] = edit.value
    _write_overrides(overrides)
    _apply_settings()
    after = _effective()
    try:
        version = config_version.commit(
            VERSIONS_ROOT, after,
            reason=edit.reason or f"set {name}",
            changed=config_version.diff(before, after))
    except config_version.ConfigVersionError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse({
        "version_id": version["version_id"], "content_sha256": version["content_sha256"],
        "changed": version["changed"],
        "live": not setting["requires_restart"],
        "note": None if not setting["requires_restart"] else
                "written, but this setting is read at startup: restart to apply it",
    })


@app.delete("/api/config/env/{name}")
def api_clear_env(name: str):
    """Drop an override, which is also a new version."""
    overrides = _overrides()
    if name not in overrides:
        raise HTTPException(404, f"{name} has no override to clear")
    _ensure_config_baseline()
    before = _effective()
    del overrides[name]
    _write_overrides(overrides)
    _apply_settings()
    after = _effective()
    version = config_version.commit(
        VERSIONS_ROOT, after, reason=f"cleared the override on {name}",
        changed=config_version.diff(before, after))
    return JSONResponse({"version_id": version["version_id"], "changed": version["changed"]})


def _ensure_config_baseline() -> None:
    """Version 0 exists before anything else asks about versions."""
    config_version.ensure_baseline(
        VERSIONS_ROOT, {s["name"]: s["default"] for s in SETTINGS})


@app.get("/api/config/versions")
def api_config_versions():
    _ensure_config_baseline()
    active = config_version.current(VERSIONS_ROOT)
    return JSONResponse({
        "versions": list(reversed(config_version.history(VERSIONS_ROOT))),
        "current_id": active["version_id"] if active else None,
        "root": str(VERSIONS_ROOT),
    })


@app.post("/api/config/versions/{version_id}/restore")
def api_restore_config(version_id: str):
    """Go back. This writes a new version equal to the old one rather than
    rewinding, so the fact that you went back stays in the record."""
    try:
        version = config_version.restore(VERSIONS_ROOT, version_id)
    except config_version.ConfigVersionError as exc:
        raise HTTPException(400, str(exc)) from exc
    defaults = {s["name"]: s["default"] for s in SETTINGS}
    _write_overrides({k: v for k, v in version["settings"].items()
                      if v != defaults.get(k) and v != os.environ.get(k)})
    _apply_settings()
    return JSONResponse({
        "version_id": version["version_id"], "restored_from": version_id,
        "changed": version["changed"],
        "note": "some settings are read at startup; restart to apply those",
    })


@app.put("/api/config/constant")
def api_set_constant(edit: ConstantEdit):
    """Rewrite a module-level constant in the source.

    This is a code change and is treated as one: the file is edited in place,
    the old and new literals are returned, and nothing is committed. A constant
    that decides whether a screen passes should leave a diff behind.
    """
    import ast as ast_module

    target = (REPO / edit.path).resolve()
    if not str(target).startswith(str(REPO.resolve())) or not target.exists():
        raise HTTPException(400, f"{edit.path} is not a file inside the repository")
    if target.suffix != ".py":
        raise HTTPException(400, "only Python constants are editable here")

    source = target.read_text()
    tree = ast_module.parse(source)
    for node in tree.body:
        targets = ([node.target] if isinstance(node, ast_module.AnnAssign)
                   else list(node.targets) if isinstance(node, ast_module.Assign) else [])
        for name_node in targets:
            if not (isinstance(name_node, ast_module.Name) and name_node.id == edit.name):
                continue
            if node.value is None:
                continue
            try:
                previous = ast_module.literal_eval(node.value)
                replacement = ast_module.literal_eval(edit.value)
            except (ValueError, SyntaxError) as exc:
                raise HTTPException(400, f"{edit.value!r} is not a literal") from exc
            if type(previous) is not type(replacement):
                raise HTTPException(
                    400,
                    f"{edit.name} is {type(previous).__name__}; refusing to make it "
                    f"{type(replacement).__name__}")
            lines = source.splitlines(keepends=True)
            start = node.value.lineno - 1
            end = (node.value.end_lineno or node.value.lineno) - 1
            head = lines[start][: node.value.col_offset]
            tail = lines[end][node.value.end_col_offset:]
            lines[start:end + 1] = [head + edit.value + tail]
            target.write_text("".join(lines))
            _module_constants.cache_clear()
            return JSONResponse({
                "path": edit.path, "name": edit.name,
                "previous": repr(previous), "value": edit.value,
                "reason": edit.reason,
                "note": "the source file changed; this is a code edit and is not committed",
            })
    raise HTTPException(404, f"{edit.name} is not a module-level constant in {edit.path}")


@app.get("/api/config")
def api_config():
    stamp = max(
        (p.stat().st_mtime for _, d, pat in TUNABLE_ROOTS if d.exists() for p in d.glob(pat)),
        default=0.0,
    )
    return JSONResponse({
        "environment": SETTINGS,
        "constants": _module_constants(stamp),
        "overrides_path": str(OVERRIDES_PATH),
        "version": (_ensure_config_baseline() or config_version.current(VERSIONS_ROOT)),
        "paths_exist": {
            s["name"]: Path(s["value"]).exists()
            for s in SETTINGS
            if s["name"] not in ("CX_DB", "CX_SCROLL_SOURCE", "CX_SCROLL_TTL", "CX_MAP_MAX_SIZE")
        },
    })


# --------------------------------------------------------------------------
# Documentation
#
# Read out of the source with `ast`, never imported: the docs cannot drift from
# the code, and generating them costs no dependency and runs no framework code.
# --------------------------------------------------------------------------

DOC_ROOTS = [
    ("contracts", REPO / "framework" / "contracts", "*.py"),
    ("01-segmentation", REPO / "framework" / "stages" / "01-segmentation" / "scripts", "*.py"),
    ("02-flattening", REPO / "framework" / "stages" / "02-flattening" / "scripts", "*.py"),
    ("03-ink", REPO / "framework" / "stages" / "03-ink" / "scripts", "*.py"),
    ("04-validation", REPO / "framework" / "stages" / "04-validation" / "scripts", "*.py"),
    ("05-reconstruction", REPO / "framework" / "stages" / "05-reconstruction" / "scripts", "*.py"),
    ("06-discovery", REPO / "framework" / "stages" / "06-discovery" / "scripts", "*.py"),
    ("audits", REPO / "framework" / "audits", "*.py"),
]


def _signature(node) -> str:
    import ast

    args = node.args
    parts: list[str] = []
    positional = list(args.posonlyargs) + list(args.args)
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for arg, default in zip(positional, defaults):
        text = arg.arg
        if arg.annotation is not None:
            text += f": {ast.unparse(arg.annotation)}"
        if default is not None:
            text += f" = {ast.unparse(default)}"
        parts.append(text)
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        text = arg.arg
        if arg.annotation is not None:
            text += f": {ast.unparse(arg.annotation)}"
        if default is not None:
            text += f" = {ast.unparse(default)}"
        parts.append(text)
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{node.name}({', '.join(parts)}){returns}"


def _cli_options(tree) -> list[dict]:
    """Recover an argparse interface without running the parser."""
    import ast

    options = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        flags = [ast.literal_eval(a) for a in node.args
                 if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if not flags:
            continue
        info: dict[str, Any] = {"flags": flags}
        for keyword in node.keywords:
            if keyword.arg in ("required", "default", "help", "choices"):
                try:
                    info[keyword.arg] = ast.literal_eval(keyword.value)
                except (ValueError, SyntaxError):
                    info[keyword.arg] = ast.unparse(keyword.value)
        options.append(info)
    return options


@functools.lru_cache(maxsize=1)
def _framework_docs(stamp: float) -> list[dict]:
    import ast

    del stamp
    groups = []
    for label, directory, pattern in DOC_ROOTS:
        if not directory.exists():
            continue
        modules = []
        for path in sorted(directory.glob(pattern)):
            if path.name.startswith("_"):
                continue
            try:
                tree = ast.parse(path.read_text())
            except (SyntaxError, OSError):
                continue
            functions = [
                {
                    "name": node.name,
                    "signature": _signature(node),
                    "doc": ast.get_docstring(node) or "",
                    "line": node.lineno,
                }
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not node.name.startswith("_")
            ]
            classes = [
                {
                    "name": node.name,
                    "doc": ast.get_docstring(node) or "",
                    "methods": [
                        {"name": m.name, "signature": _signature(m), "doc": ast.get_docstring(m) or ""}
                        for m in node.body
                        if isinstance(m, ast.FunctionDef) and not m.name.startswith("_")
                    ],
                }
                for node in tree.body
                if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
            ]
            cli = _cli_options(tree)
            if not (functions or classes or cli):
                continue
            modules.append({
                "path": str(path.relative_to(REPO)),
                "name": path.stem,
                "doc": ast.get_docstring(tree) or "",
                "functions": functions,
                "classes": classes,
                "cli": cli,
                "is_entrypoint": bool(cli),
            })
        if modules:
            groups.append({"group": label, "modules": modules})
    return groups


PHASES = REPO / "framework" / "contracts" / "pipeline_phases.json"


@app.get("/api/phases")
def api_phases():
    """The phase vocabulary, from its single committed source."""
    if not PHASES.exists():
        raise HTTPException(404, "pipeline_phases.json is missing")
    return JSONResponse(json.loads(PHASES.read_text()))


IMPLEMENTATIONS = REPO / "framework" / "registries" / "phase-implementations-0.1.0.json"


QUEUEABLE_PHASES = ("P4", "P5", "P7", "P8", "P9")

# Which stage directory holds a phase's profiles. The concept is shared -- a
# profile is a frozen declaration of how a stage runs -- but the schema is not:
# an ink lane profile declares frames, tile and training scale, which mean
# nothing outside P5.
PHASE_PROFILE_DIR = {
    "P1": "01-segmentation", "P2": "01-segmentation",
    "P3": "02-flattening", "P4": "02-flattening",
    "P5": "03-ink", "P6": "03-ink",
    "P7": "04-validation",
    "P8": "05-reconstruction",
    "P9": "06-discovery",
}


# A stage can keep runtime envelopes beside the code that consumes them rather
# than in the shared profile tree. P1 does: the six profiles under
# framework/profiles/01-segmentation are scrollfiesta and hybrid recipes, all
# EXPERIMENTAL_LOCKED and one of them backed by a method that failed its
# reference control, while the envelope the fleet actually grows with lives in
# fleet/profiles. Reading only the first directory meant the panel offered the
# backend that does not work and hid the one that does.
EXTRA_PROFILE_DIRS = {
    "P1": ["framework/stages/01-segmentation/fleet/profiles"],
    "P2": ["framework/stages/01-segmentation/fleet/profiles"],
}


def phase_profiles(phase_id: str) -> list[dict]:
    """Every profile declared for this phase's stage, whatever its schema."""
    stage = PHASE_PROFILE_DIR.get(phase_id)
    directories = [REPO / "framework" / "profiles" / stage] if stage else []
    directories += [REPO / extra for extra in EXTRA_PROFILE_DIRS.get(phase_id, [])]
    directories = [d for d in directories if d.exists()]
    if not directories:
        return []
    registry = registry_entries()
    out = []
    for path in sorted(p for d in directories for p in d.glob("*.json")):
        try:
            profile = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        identifier = (profile.get("profile_id") or profile.get("policy_id")
                      or profile.get("id") or path.stem)
        entry = registry.get(profile.get("method_id", ""), {})
        out.append({
            "profile_id": identifier,
            "schema": profile.get("schema", "?"),
            "method_id": profile.get("method_id"),
            "adapter": profile.get("adapter"),
            "path": str(path.relative_to(REPO)),
            "input_contract": profile.get("input_contract", {}),
            "default_execution": profile.get("default_execution", {}),
            "disqualified": "DISQUALIFIED" in str(entry.get("validation_status", "")),
            "registry_status": entry.get("validation_status"),
            "registry_policy": entry.get("recommended_policy"),
            # What the fleet can actually grow with, as opposed to a recipe that
            # describes a backend whose binaries are not here.
            "runnable": profile.get("schema") == "campaignx.segment_fleet_profile.v1",
            "status": profile.get("status"),
            "parameters": profile.get("parameters", {}),
        })
    return out


def scrolls_never_intaken(mission_id: str | None) -> list[str]:
    """Scrolls a mission names that P0 never registered.

    Without this they are silently absent from every count: mission `test` names
    three scrolls, one has a source snapshot, and the other two look like work
    that is finished rather than work that cannot start.
    """
    named = mission_scrolls(mission_id)
    if not named or not DSN:
        return []
    try:
        import psycopg

        with psycopg.connect(DSN, connect_timeout=5) as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT DISTINCT sample_id FROM segment_source_snapshots "
                "WHERE sample_id = ANY(%s)", [sorted(named)])
            registered = {row[0] for row in cursor.fetchall()}
    except Exception:  # noqa: BLE001 -- the control plane is optional
        return []
    return sorted(named - registered)


def phase_state(phase_id: str, mission_id: str | None = None,
                subject: str | None = None) -> dict:
    """What exists at this phase right now, and what can be done to it.

    Every phase answers the same four questions -- state, action, artefacts,
    contract -- so a phase with nothing yet reports an empty action with a
    reason rather than being absent from the interface.
    """
    subject = stored_scroll(subject)
    contract = next(
        (p for p in json.loads(PHASES.read_text())["phases"] if p["id"] == phase_id), None
    ) if PHASES.exists() else None
    if contract is None:
        raise HTTPException(404, f"no phase {phase_id}")

    # Resolve once and reuse for every source below.  In particular, receipts
    # and queue rows are not allowed to resurrect a sample that the mission's
    # P0 manifest does not select, even if an older/corrupt row carries both the
    # mission id and that foreign sample.
    permitted = read_scope(mission_id, subject)

    def scoped_receipts() -> list[Run]:
        rows = index_runs(mission_id=mission_id)
        if permitted is not None:
            rows = [row for row in rows if stored_scroll(row.sample_id) in permitted]
        if subject:
            rows = [row for row in rows if stored_scroll(row.sample_id) == subject]
        return rows

    def scoped_jobs(phase: str, *, limit: int = 500) -> list[dict[str, Any]]:
        if not DSN or (permitted is not None and not permitted):
            return []
        try:
            rows = job_store().jobs(
                limit=limit, mission_id=mission_id, phase=phase,
                sample_id=subject or None)
        except Exception:  # noqa: BLE001 -- the queue is optional
            return []
        if permitted is not None:
            rows = [job for job in rows
                    if stored_scroll(job.get("sample_id")) in permitted]
        return rows

    registry = json.loads(IMPLEMENTATIONS.read_text())["implementations"] \
        if IMPLEMENTATIONS.exists() else []
    components = [i for i in registry if phase_id in i.get("phases", [])]

    state: dict[str, Any] = {}
    artefacts: list[dict] = []

    if phase_id == "P0":
        summary = scrolls()
        state = {"scrolls_available": summary["total"],
                 "with_declared_scale": summary["with_scale"],
                 "with_runs": summary["screened_count"],
                 "inventory_origin": summary["inventory_origin"]}
        if mission_id:
            try:
                _, manifest = mission_contract.resolve(RUNS, mission_id)
                state["in_this_mission"] = len(manifest.get("scrolls", []))
            except mission_contract.MissionError:
                pass
            missing = scrolls_never_intaken(mission_id)
            if missing:
                # Named and never registered. P1 cannot queue a task against a
                # scroll with no source snapshot, so these are not slow -- they
                # have not started and nothing else on the page would say so.
                state["never_intaken"] = len(missing)
                state["never_intaken_scrolls"] = ", ".join(missing)
    elif phase_id in ("P1", "P2", "P3"):
        # A phase page is asked about one scroll, or about a mission's scrolls,
        # or about nothing in particular -- and in the last case it is the fleet
        # page, which is what this branch always was. The first two reported the
        # fleet's numbers regardless, and P1 and P2 shared one branch, so two
        # phases answering two different questions printed one number.
        scope = None if permitted is None else sorted(permitted)
        if phase_id == "P3":
            sheets = json.loads(api_flattening(sample=subject, mission=mission_id).body)
            state = ({"certified": sheets["certified"], "flattened": sheets["flattened"],
                      "awaiting": sheets["awaiting"]} if sheets.get("available")
                     else {"unavailable": sheets.get("reason")})
            artefacts = sheets.get("rows", [])[:50]
        elif scope is None:
            fleet = fleet_status()
            if fleet.get("available"):
                state = {"tasks": fleet["tasks"], "surfaces": fleet["surfaces"],
                         "attempts": fleet["attempts"], "stale_leases": fleet["stale_leases"]}
                artefacts = fleet.get("surfaces_by_sample", [])[:20]
            else:
                state = {"unavailable": fleet.get("reason")}
        elif not scope:
            state = ({"surfaces": 0, "area_cm2": 0.0, "tasks": 0, "attempts": 0}
                     if phase_id == "P1" else
                     {"surfaces": 0, "certified": 0, "unmeasured": 0,
                      "rejected": 0})
        else:
            surfaces = [s for scroll in scope
                        for s in subject_surfaces(scroll, mission_id)]
            artefacts = surfaces[:50]
            if phase_id == "P1":
                # What growing produced. Verdicts are P2's answer, not this one.
                state = {"surfaces": len(surfaces),
                         "area_cm2": round(sum(s["area_cm2"] for s in surfaces), 2)}
                queue = scoped_queue(set(scope), mission_id)
                if queue is not None:
                    state["tasks"] = queue["tasks"]
                    state["attempts"] = queue["attempts"]
                # And how much ground that covers, which a surface count cannot
                # say: it rises whether the fleet is finding new ground or
                # re-treading old.
                #
                # Every scroll in scope, not the first one: a mission's scrolls
                # sort alphabetically and the first of `test` is one nobody ever
                # intook, so this reported nothing for a scroll with 198 cells
                # attempted two names later.
                grids = []
                for scroll in scope:
                    try:
                        grids += fleet_store().coverage(
                            scroll, mission_id=mission_id)["grids"]
                    except Exception:  # noqa: BLE001 -- the control plane is optional
                        continue
                if grids:
                    state["cells_attempted"] = sum(g["cells_attempted"] for g in grids)
                    state["cells_with_surface"] = sum(
                        g["cells_with_surface"] for g in grids)
                    state["grid_versions"] = len(grids)
            else:
                # The verdict lives in geometry_qc_state. This read `state` and
                # `physical_qc_state`, which hold QC_SCREENED and CT_SUPPORTED --
                # neither starts with "certif", so a scroll with forty-two
                # certified surfaces reported zero certified.
                tally: dict[str, int] = {}
                for surface in surfaces:
                    verdict = str(surface.get("geometry_qc_state") or "GEOMETRY_UNMEASURED")
                    tally[verdict] = tally.get(verdict, 0) + 1
                state = {"surfaces": len(surfaces),
                         "certified": tally.get("GEOMETRY_CERTIFIED", 0),
                         "unmeasured": tally.get("GEOMETRY_UNMEASURED", 0),
                         "rejected": sum(n for verdict, n in tally.items()
                                         if verdict.startswith("GEOMETRY_REJECTED"))}
    elif phase_id == "P4":
        runs = scoped_receipts()
        legacy = len({str(r.receipt.get("input", {}).get("tiff_dir")) for r in runs
                      if r.receipt.get("input", {}).get("tiff_dir")})
        state = {"layer_stacks_referenced": legacy} if legacy else {}
        # And what this phase has actually rendered, which the receipt count
        # cannot see: those receipts predate the queue, so a phase that rendered
        # two stacks this afternoon still reported zero.
        state.update(render_status(subject, mission_id))
    elif phase_id == "P5":
        runs = scoped_receipts()
        state = {"receipts": len(runs),
                 "samples": len({r.sample_id for r in runs})}
        # And what the queue did, which the receipt index cannot see: a job
        # queued without a mission files its receipt under `unfiled`, so a
        # screening that ran this afternoon read as zero runs on the mission's
        # page while its probability map sat on disk.
        state.update(render_status(subject, mission_id, phase="P5"))
        artefacts = [r.as_dict() for r in runs[:20]]
    elif phase_id == "P6":
        runs = scoped_receipts()
        # P6 is the liveness decision produced inside P5. Queue-driven P5 runs
        # keep that decision in ink_jobs.result; they are not copied into the
        # legacy CX_RUNS receipt tree. Reading only index_runs therefore left a
        # mission at zero immediately after its successful ALIVE screening.
        queued_p5 = scoped_jobs("P5")
        terminal_p5 = [
            job for job in queued_p5
            if job.get("state") in {"succeeded", "failed"}
        ]
        # If a deployment later indexes queue outputs as legacy receipts, keep
        # the transition additive without double-counting the same directory.
        legacy_paths = {str(run.path) for run in runs}
        terminal_p5 = [
            job for job in terminal_p5
            if not job.get("output_dir")
            or str(Path(str(job["output_dir"]))) not in legacy_paths
        ]
        verdicts: dict[str, int] = {}
        for run in runs:
            verdict = (run.receipt.get("liveness") or {}).get("verdict")
            if verdict:
                verdicts[verdict] = verdicts.get(verdict, 0) + 1
        for job in terminal_p5:
            verdict = ((job.get("result") or {}).get("liveness") or {}).get("verdict")
            if verdict:
                verdicts[str(verdict)] = verdicts.get(str(verdict), 0) + 1
        total = len(runs) + len(terminal_p5)
        state = {"runs_with_liveness": sum(verdicts.values()), "verdicts": verdicts,
                 "runs_without_liveness": total - sum(verdicts.values())}
    elif phase_id == "P7":
        runs = scoped_receipts()
        legacy_paths = {str(run.path) for run in runs}
        alive_p5 = [
            job for job in scoped_jobs("P5")
            if job.get("state") == "succeeded"
            and ((job.get("result") or {}).get("liveness") or {}).get("verdict") == "ALIVE"
            and (not job.get("output_dir")
                 or str(Path(str(job["output_dir"]))) not in legacy_paths)
        ]
        state = {"maps_available": sum(len(r.maps) for r in runs) + len(alive_p5),
                 "runs_screenable": sum(1 for r in runs if r.maps) + len(alive_p5)}
        state.update(render_status(subject, mission_id, phase="P7"))
        artefacts = [{"run_id": r.run_id, "sample_id": r.sample_id, "maps": r.maps}
                     for r in runs if r.maps][:20]
        artefacts += [{"job_id": job.get("job_id"),
                       "sample_id": job.get("sample_id"),
                       "output_dir": job.get("output_dir"),
                       "liveness": "ALIVE"}
                      for job in alive_p5[:max(0, 20 - len(artefacts))]]
    elif phase_id in ("P8", "P9"):
        # The last two phases had no numbers at all: their work is queued like
        # every other phase and nothing counted it, so the pages that produce
        # the relation graph and the plates -- the only artefact a person reads
        # -- looked like phases nobody had ever run.
        state = render_status(subject, mission_id, phase=phase_id)
    else:
        state = {}

    # Is this phase's input actually present? Blocking on "the previous phase
    # has not run" would be wrong -- several phases can be entered from public
    # artefacts, and P5 in particular was entered that way for every result this
    # project has. So the check is on the input, not on the predecessor.
    prerequisites = contract.get("prerequisites", {})
    runs_here = scoped_receipts()
    surfaces_here = (subject_surfaces(subject, mission_id)
                     if subject and (permitted is None or bool(permitted)) else [])
    if not subject and DSN:
        if mission_id and permitted is not None:
            queue = scoped_queue(set(permitted), mission_id)
            surfaces_here = ([{}] * int(queue.get("surfaces", 0) or 0)
                             if queue is not None else [])
        else:
            fleet = fleet_status(permitted)
            surfaces_here = ([{}] * int(fleet.get("surfaces", 0) or 0)
                             if fleet.get("available") else [])

    have: bool
    if phase_id == "P0":
        have = True
    elif phase_id == "P1":
        have = permitted is None or bool(permitted)
    elif phase_id in ("P2", "P3"):
        have = bool(surfaces_here)
    elif phase_id == "P8":
        # P8 may enter from public segment meshes without a P1/P2 surface in
        # this control plane. A successful assembly is definitive evidence
        # that those external inputs existed; do not show the same phase as
        # both completed and blocked merely because its source was external.
        have = bool(surfaces_here) or any(
            job.get("state") == "succeeded" for job in scoped_jobs("P8")
        )
    elif phase_id == "P4":
        have = bool(surfaces_here)
    elif phase_id == "P5":
        have = any(r.receipt.get("input", {}).get("tiff_dir") for r in runs_here) or bool(surfaces_here)
    elif phase_id == "P6":
        have = bool(state.get("runs_with_liveness"))
    elif phase_id == "P7":
        have = bool(state.get("maps_available"))
    elif phase_id == "P9":
        have = any(
            job.get("state") == "succeeded"
            and bool((job.get("parameters") or {}).get("out_path"))
            for job in scoped_jobs("P8")
        )
    else:
        have = False

    blocked = None
    if not have and prerequisites.get("needs"):
        producers = prerequisites.get("produced_by") or []
        # A phase whose input has no producer here is not waiting on an earlier
        # phase -- P8 and P9 read published artefacts and no phase feeds them.
        # Saying "normally from an earlier phase" invented a dependency the
        # contract deliberately says is not there.
        if producers:
            blocked = f"needs {prerequisites['needs']}, from {' or '.join(producers)}"
            if prerequisites.get("external_source"):
                blocked += f"; or from {prerequisites['external_source']}"
        elif prerequisites.get("external_source"):
            blocked = (f"needs {prerequisites['needs']}, which no phase here "
                       f"produces: {prerequisites['external_source']}")
        else:
            blocked = f"needs {prerequisites['needs']}"

    # Filtered in SQL. Fetching the newest fifty rows overall and keeping the
    # ones for this phase returns nothing whenever another phase has produced
    # fifty jobs more recently, so a quiet phase looked idle while it ran.
    queued = scoped_jobs(phase_id, limit=50)

    return {
        "contract": contract,
        "components": components,
        "profiles": phase_profiles(phase_id),
        "profile_stage": PHASE_PROFILE_DIR.get(phase_id),
        "state": state,
        "artefacts": artefacts,
        "subject": subject,
        "prerequisites": prerequisites,
        "input_available": have,
        "blocked": blocked,
        "queueable": phase_id in QUEUEABLE_PHASES,
        "queueable_reason": None if phase_id in QUEUEABLE_PHASES else
            "no runner is registered for this phase yet; it is observed rather than driven",
        "jobs": queued,
    }


def adjudication_outcome_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    """Separate P7 execution success from the claim-check outcome it measured."""
    counts = {"adjudications_passed": 0, "adjudications_refuted": 0,
              "adjudications_without_verdict": 0}
    for result in results:
        verdict = str(((result or {}).get("adjudication") or {}).get("verdict") or "")
        if verdict == "PASS":
            counts["adjudications_passed"] += 1
        elif verdict == "FAIL":
            counts["adjudications_refuted"] += 1
        else:
            counts["adjudications_without_verdict"] += 1
    return counts


def render_status(sample_id: str | None = None,
                  mission_id: str | None = None,
                  phase: str = "P4") -> dict[str, int]:
    """What this phase's queue has produced, by outcome.

    From the job queue rather than from run receipts. A receipt is filed under
    the mission the job carried, and a job queued without one lands in `unfiled`
    -- so a phase that ran this afternoon reported zero runs on the mission's
    own page while its output sat on disk with a receipt beside it. The queue
    knows what was asked for and what came of it; that is the number a phase
    page owes the reader.
    """
    if not DSN:
        return {}
    resolved = read_scope(mission_id, sample_id)
    if resolved is not None and not resolved:
        by_state: dict[str, int] = {}
        noun = {"P4": "renders", "P5": "screenings", "P7": "adjudications",
                "P8": "assemblies", "P9": "plate_runs"}.get(phase, "jobs")
        empty = {f"{noun}_succeeded": 0, f"{noun}_failed": 0,
                 f"{noun}_queued": 0}
        if phase == "P7":
            empty.update(adjudication_outcome_counts([]))
        return empty
    scientific: dict[str, int] = {}
    try:
        import psycopg

        with psycopg.connect(DSN, connect_timeout=5) as connection, connection.cursor() as cursor:
            conditions = ["phase = %s"]
            arguments: list[Any] = [phase]
            if mission_id:
                conditions.append("mission_id = %s")
                arguments.append(mission_id)
            if resolved is not None:
                conditions.append("sample_id = ANY(%s)")
                arguments.append(sorted(resolved))
            if phase == "P7":
                cursor.execute(
                    "SELECT state, result FROM ink_jobs WHERE "
                    + " AND ".join(conditions), arguments)
                rows = cursor.fetchall()
                by_state = {}
                for state, _result in rows:
                    by_state[str(state)] = by_state.get(str(state), 0) + 1
                scientific = adjudication_outcome_counts([
                    dict(result or {}) for state, result in rows
                    if str(state) == "succeeded"
                ])
            else:
                cursor.execute(
                    "SELECT state, count(*) FROM ink_jobs WHERE "
                    + " AND ".join(conditions) + " GROUP BY 1",
                    arguments)
                by_state = {str(row[0]): int(row[1]) for row in cursor.fetchall()}
    except Exception:  # noqa: BLE001
        return {}
    noun = {"P4": "renders", "P5": "screenings", "P7": "adjudications",
            "P8": "assemblies", "P9": "plate_runs"}.get(phase, "jobs")
    status = {f"{noun}_succeeded": by_state.get("succeeded", 0),
              f"{noun}_failed": by_state.get("failed", 0),
              # leased counts as queued: it is claimed and not yet running, and a
              # phase that reports nothing in flight while a worker holds a job
              # reads as idle.
              f"{noun}_queued": sum(by_state.get(s, 0)
                                    for s in ("pending", "leased", "running"))}
    status.update(scientific)
    return status


def subject_surfaces(sample_id: str,
                     mission_id: str | None = None) -> list[dict]:
    """Surfaces produced for one scroll, optionally by one mission only."""
    sample_id = stored_scroll(sample_id) or sample_id
    if not DSN:
        return []
    try:
        import psycopg

        with psycopg.connect(DSN, connect_timeout=5) as connection, connection.cursor() as cursor:
            mission_where = """ AND (
                EXISTS (
                  SELECT 1 FROM segment_artifact_sets art
                  JOIN segment_attempts a ON a.attempt_id=art.attempt_id
                  JOIN segment_tasks t ON t.task_id=a.task_id
                  WHERE art.manifest->>'artifact_sha256'=f.artifact_sha256
                    AND t.mission_id=%s)
                OR EXISTS (
                  SELECT 1 FROM surface_derivations d
                  JOIN ink_jobs j ON j.job_id=d.job_id
                  WHERE d.child_surface_id=f.surface_id
                    AND j.mission_id=%s)
              )""" if mission_id else ""
            cursor.execute(
                """SELECT DISTINCT f.surface_id, f.state, f.physical_qc_state,
                          f.area_cm2, f.created_at,
                          to_jsonb(f) ->> 'geometry_qc_state'
                   FROM segment_surfaces f
                  WHERE f.sample_id=%s
                """ + mission_where + """
                  ORDER BY f.created_at DESC LIMIT 200""",
                ((sample_id, mission_id, mission_id)
                 if mission_id else (sample_id,)))
            return [{"surface_id": s_, "state": st, "physical_qc_state": qc,
                     "geometry_qc_state": geometry,
                     "area_cm2": float(a or 0), "created_at": c.isoformat() if c else None}
                    for s_, st, qc, a, c, geometry in cursor.fetchall()]
    except Exception:  # noqa: BLE001 -- the fleet database is optional
        return []


@app.get("/api/subjects")
def api_subjects(mission: str | None = None):
    """What can be followed through the phases.

    A run is the output of one phase and does not exist in the other nine, so
    it is the wrong thing to select. What crosses phases is the *subject*: a
    scroll, and within it a segment. A segment is born in P1, certified in P2,
    flattened in P3, rendered in P4, screened in P5 and adjudicated in P7 --
    that chain is what an operator actually follows.
    """
    permitted = read_scope(mission)
    runs = index_runs(mission_id=mission)
    if permitted is not None:
        runs = [run for run in runs if stored_scroll(run.sample_id) in permitted]
    by_sample: dict[str, list[Run]] = {}
    for run in runs:
        by_sample.setdefault(run.sample_id, []).append(run)

    names: set[str] = (set(by_sample) if permitted is None else set(permitted))

    catalog = catalog_metadata()
    subjects = []
    for sample in sorted(names):
        surfaces = subject_surfaces(sample, mission)
        mine = by_sample.get(sample, [])
        certified = sum(1 for s in surfaces if str(s["state"]).lower().startswith("certif")
                        or str(s["physical_qc_state"] or "").lower().startswith("certif"))
        # Where did this subject get to? The furthest phase with anything in it.
        reached = "P0"
        if surfaces:
            reached = "P2" if certified else "P1"
        if mine:
            reached = "P5"
        if any(r.maps for r in mine):
            reached = "P7"
        subjects.append({
            "sample_id": sample,
            "pixel_um": catalog.get(sample, {}).get("pixel_um", ""),
            "energy_kev": catalog.get(sample, {}).get("energy_kev"),
            "surfaces": len(surfaces),
            "certified_surfaces": certified,
            "surface_area_cm2": round(sum(s["area_cm2"] for s in surfaces), 2),
            "runs": len(mine),
            "maps": sum(len(r.maps) for r in mine),
            "reached_phase": reached,
            "segments": [s["surface_id"] for s in surfaces[:50]],
        })
    return JSONResponse({"subjects": subjects, "mission": mission})


@app.get("/api/phase/{phase_id}")
def api_phase(phase_id: str, mission: str | None = None, subject: str | None = None):
    return JSONResponse(phase_state(phase_id.upper(), mission_id=mission, subject=subject))


@app.get("/api/phase-summary")
def api_phase_summary(mission: str | None = None, subject: str | None = None):
    """One badge per phase for the navigation rail."""
    progress_keys = {
        "P0": (("in_this_mission",) if mission else ("scrolls_available",)),
        "P1": ("surfaces", "tasks", "attempts"),
        "P2": ("certified", "surfaces"),
        "P3": ("flattened", "certified"),
        "P4": ("renders_succeeded", "renders_queued"),
        "P5": ("screenings_succeeded", "receipts"),
        "P6": ("runs_with_liveness",),
        "P7": ("adjudications_succeeded", "maps_available"),
        "P8": ("assemblies_succeeded",),
        "P9": ("plate_runs_succeeded",),
    }
    out = []
    for contract in json.loads(PHASES.read_text())["phases"]:
        pid = contract["id"]
        detail = phase_state(pid, mission_id=mission, subject=subject)
        counts = detail["state"]
        badge = next((str(counts[key]) for key in progress_keys.get(pid, ())
                      if isinstance(counts.get(key), int) and counts[key] > 0), None)
        work = phase_work(detail["jobs"])
        out.append({"id": pid, "name": contract["name"], "slug": contract["slug"],
                    "maturity": contract["maturity"], "queueable": detail["queueable"],
                    "input_available": detail["input_available"],
                    "blocked": detail["blocked"],
                    "badge": badge, "components": len(detail["components"]),
                    "active_jobs": work["running"] + work["queued"],
                    "running_jobs": work["running"], "queued_jobs": work["queued"],
                    "failed_jobs": work["failed"],
                    "status": phase_status(contract, detail, work),
                    "why": phase_status_reason(contract, detail, work)})
    return JSONResponse({"phases": out})


def phase_work(jobs: list[dict]) -> dict:
    """What this phase's queue is doing now, and how its last attempt ended.

    Two distinctions the rail could not previously make.

    Queued is not running. A job nobody has claimed and a job a worker is
    executing look identical in a count of "active", and the rail pulsed for
    both -- so a phase whose image carries no runner for it pulsed "running now"
    indefinitely while nothing was happening. That is exactly the failure this
    rail exists to surface.

    And the last attempt is not any attempt. Counting every failure in the
    window left a phase red for as long as one old failure stayed in it, however
    many times it had succeeded since.
    """
    running = sum(1 for j in jobs if j["state"] in ("leased", "running"))
    queued = sum(1 for j in jobs if j["state"] == "pending")
    # Newest first, so the first terminal row is the most recent outcome.
    last = next((j for j in jobs
                 if j["state"] in ("succeeded", "failed", "cancelled")), None)
    return {"running": running, "queued": queued,
            "failed": sum(1 for j in jobs if j["state"] == "failed"),
            "last": (last or {}).get("state"),
            "attempts": len(jobs)}


def phase_status(contract: dict, detail: dict, work: dict) -> str:
    """One word for what this phase is doing, derived once and here.

    Ordered by what somebody scanning the rail needs first: work happening now,
    then work waiting for a worker, then how the last attempt ended, then
    whether this phase could start at all.

    A phase with no command is not one that runs somewhere else. A committed
    catalog and a check that lives inside another phase are both "nothing to
    run"; calling either "elsewhere" points at a machine that does not exist.
    """
    if work["running"]:
        return "running"
    if work["queued"]:
        return "queued"
    if work["last"] == "failed":
        return "failed"
    if work["last"] == "cancelled":
        return "stopped"
    if not contract.get("runnable_here", True):
        return "no-run" if not contract.get("runner") else "elsewhere"
    if detail.get("blocked"):
        return "blocked"
    if detail.get("artefacts") or work["last"] == "succeeded":
        return "done"
    if detail.get("input_available"):
        return "ready"
    return "waiting"


def phase_status_reason(contract: dict, detail: dict, work: dict) -> str:
    """The sentence behind the colour. A dot nobody can interrogate is decoration."""
    def plural(n: int) -> str:
        return "s" if n != 1 else ""

    if work["running"]:
        return f"{work['running']} job{plural(work['running'])} running on a worker"
    if work["queued"]:
        return (f"{work['queued']} job{plural(work['queued'])} queued; no worker has "
                "claimed them yet")
    if work["last"] == "failed":
        return "the last attempt failed"
    if work["last"] == "cancelled":
        return "the last attempt was cancelled"
    if not contract.get("runnable_here", True):
        return contract.get("how_to_run", "nothing here runs this phase")
    if detail.get("blocked"):
        return str(detail["blocked"])
    if detail.get("artefacts"):
        return f"{len(detail['artefacts'])} artefact(s) here"
    if work["last"] == "succeeded":
        return "the last attempt succeeded"
    if detail.get("input_available"):
        return "its prerequisites are met; nothing has run yet"
    return "its prerequisites are not met yet"


@app.get("/api/implementations")
def api_implementations():
    """Which component implements each phase, and where it is published."""
    if not IMPLEMENTATIONS.exists():
        raise HTTPException(404, "phase-implementations registry is missing")
    return JSONResponse(json.loads(IMPLEMENTATIONS.read_text()))


@app.get("/api/docs")
def api_docs():
    stamp = max(
        (p.stat().st_mtime for _, d, pat in DOC_ROOTS if d.exists() for p in d.glob(pat)),
        default=0.0,
    )
    groups = _framework_docs(stamp)
    return JSONResponse({
        "groups": groups,
        "module_count": sum(len(g["modules"]) for g in groups),
        "function_count": sum(
            len(m["functions"]) + sum(len(c["methods"]) for c in m["classes"])
            for g in groups for m in g["modules"]
        ),
        "profiles": [
            {"profile_id": p["profile_id"], "method_id": p["method_id"],
             "adapter": p["adapter"], "contract": p["input_contract"]}
            for p in ink_profiles()
        ],
        "paths": {"repo": str(REPO), "runs": str(RUNS)},
    })


@app.post("/api/reindex")
def api_reindex():
    _quantised.cache_clear()
    _raster.cache_clear()
    return JSONResponse({"runs": len(index_runs(force=True))})


# --------------------------------------------------------------------------
# Artifacts over the network
# --------------------------------------------------------------------------
#
# Object storage is optional. Where there is no s3:// prefix, a worker publishes
# its surfaces here and this volume is the deployment's copy of record.
#
# A worker on the panel's own host can share the volume directly, and does. This
# is for the other case -- a worker on a different machine -- which is why it
# exists at all: a surface written to a worker's own disk is invisible to QC and
# to flattening, and is lost with the machine.
#
# A directory at a time, gzipped, because that is the unit the artifact store
# publishes: a surface plus its manifest, and a half-written surface is worse
# than none. Measured on real work: a published surface is about 800 KB, so this
# is one request rather than a chunked protocol nobody needs.

def _artifact_path(key: str) -> Path:
    """Resolve a key under the artifact root, or refuse.

    The check is on the resolved path, not the key: `a/../../etc` contains no
    suspicious component after normalisation, and a symlink inside the root can
    point anywhere at all. Comparing the real path to the real root is the only
    form of this that holds.
    """
    root = ARTIFACTS.resolve()
    candidate = (root / key).resolve()
    if candidate != root and root not in candidate.parents:
        raise HTTPException(400, "that key is outside the artifact root")
    return candidate


@app.put("/api/artifacts/{key:path}")
async def api_artifact_put(key: str, http: Request):
    """Receive one directory, gzipped, and place it atomically.

    Unpacked beside the destination and moved into place, so a transfer that
    dies halfway leaves nothing that looks like a published surface. A reader
    downstream cannot tell "still uploading" from "finished" by looking, so it
    must never see the first state.
    """
    import shutil
    import tarfile

    destination = _artifact_path(key)
    body = await http.body()
    if not body:
        raise HTTPException(400, "empty body: expected a gzipped tar")

    staging = destination.parent / f".incoming-{secrets.token_hex(8)}"
    try:
        staging.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as archive:
            for member in archive.getmembers():
                # A tar can carry absolute paths, `..`, symlinks and device
                # nodes. This one is written by our own worker, which is
                # exactly the assumption not to make: the token that reaches
                # this endpoint lives in a file on a worker host.
                if member.name.startswith("/") or ".." in Path(member.name).parts:
                    raise HTTPException(400, f"unsafe path in archive: {member.name}")
                if not (member.isfile() or member.isdir()):
                    raise HTTPException(400, f"only files and directories: {member.name}")
            archive.extractall(staging)  # noqa: S202 - members checked above
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return JSONResponse({"artifact_uri": str(destination), "bytes": len(body)})


@app.post("/api/artifacts/{key:path}")
async def api_artifact_copy(key: str, http: Request):
    """Copy one artifact to another key, here rather than over the wire.

    Promotion moves a surface from staging to its final name. The bytes are
    already on this host, and S3's backend does the same thing with a
    server-side copy -- sending them up a second time would make publishing
    twice as expensive for no gain.
    """
    import shutil

    body = await http.json()
    source = _artifact_path(str(body.get("copy_from", "")))
    if not source.is_dir():
        raise HTTPException(404, f"nothing to copy at {body.get('copy_from')!r}")
    destination = _artifact_path(key)
    if destination.is_dir():
        # Already there. Publication is retried, and a retry that refuses is a
        # surface that never gets its final name.
        return JSONResponse({"artifact_uri": str(destination), "already_present": True})
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".copying-{secrets.token_hex(8)}"
    try:
        shutil.copytree(source, staging)
        staging.replace(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return JSONResponse({"artifact_uri": str(destination), "already_present": False})


@app.get("/api/artifacts/{key:path}")
def api_artifact_get(key: str):
    """Hand back one directory, gzipped."""
    import tarfile

    source = _artifact_path(key)
    if not source.is_dir():
        raise HTTPException(404, f"no artifact at {key}")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(source, arcname=".")
    return Response(buffer.getvalue(), media_type="application/gzip")


@app.delete("/api/artifacts/{key:path}")
def api_artifact_delete(key: str, http: Request):
    import shutil

    target = _artifact_path(key)
    existed = target.is_dir()
    if existed:
        shutil.rmtree(target)
    return JSONResponse({"deleted": existed})


@app.head("/api/artifacts/{key:path}")
def api_artifact_head(key: str):
    """Whether something is there, without moving it.

    The store asks this before promoting, and a GET of a surface to answer a
    yes-or-no question is the sort of thing that only hurts once it is busy.
    """
    return Response(status_code=200 if _artifact_path(key).is_dir() else 404)


# --------------------------------------------------------------------------
# SPA
# --------------------------------------------------------------------------

if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        candidate = DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(DIST / "index.html")
else:
    @app.get("/")
    def not_built():
        return JSONResponse(
            {"error": "the frontend is not built",
             "fix": "cd panel/web && npm ci && npm run build"},
            status_code=503,
        )
