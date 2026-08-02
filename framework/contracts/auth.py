"""Who may use the panel.

No roles. Everyone who can log in can do everything, which is the honest shape
for a small team all of whom are already trusted with the GPU host -- roles that
nobody enforces are worse than none, because they read as a boundary that is
not there.

What this does guard is the difference between "somebody on the network" and
"somebody we invited". The panel binds 0.0.0.0, accepts file uploads and queues
GPU work, so that difference is the whole point.

Passwords are stored as scrypt hashes with a per-user salt. Session tokens are
stored hashed too, exactly like the fleet's lease tokens: a stolen sessions file
is then a list of what a token would have to hash to, not a list of tokens.

The first account is the one problem this cannot solve on its own. A panel that
is already reachable and unauthenticated cannot safely let the first request
through the door claim it, so the first account may only be created from
loopback -- somebody with a shell on the host, which is a boundary that already
exists.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

USERS_SCHEMA = "campaignx.panel_users.v1"
SESSION_SCHEMA = "campaignx.panel_sessions.v1"

USERS_NAME = "USERS.json"
SESSIONS_NAME = "SESSIONS.json"

COOKIE = "helena_session"
SESSION_DAYS = 30

# scrypt at the parameters Python's own documentation suggests for interactive
# logins. n=2**15 is about 100ms and 32 MB per attempt on this hardware, which
# is the point: it is the cost of guessing, paid once per real login.
SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1
SALT_BYTES = 16
KEY_BYTES = 32
# scrypt needs about 128 * n * r bytes -- 32 MiB at these parameters -- and
# OpenSSL's default ceiling is exactly 32 MiB, so it refuses by a hair with
# "memory limit exceeded". Stated rather than tuned down: the memory hardness
# is the property being bought.
SCRYPT_MAXMEM = 128 * SCRYPT_N * SCRYPT_R * 2

NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")
MINIMUM_PASSWORD = 10


class AuthError(ValueError):
    """The account or session could not be created or read."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(moment: datetime) -> str:
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(SALT_BYTES)
    derived = hashlib.scrypt(password.encode("utf-8"), salt=salt,
                             n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=KEY_BYTES,
                             maxmem=SCRYPT_MAXMEM)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${derived.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, n, r, p, salt, expected = stored.split("$")
        if algorithm != "scrypt":
            return False
        derived = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt),
                                 n=int(n), r=int(r), p=int(p), dklen=KEY_BYTES,
                                 maxmem=128 * int(n) * int(r) * 2)
    except (ValueError, TypeError):
        return False
    # compare_digest, not ==: the obvious comparison leaks how much of the hash
    # matched through how long it took to say no.
    return hmac.compare_digest(derived.hex(), expected)


def token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _read(path: Path, schema: str) -> dict:
    if not path.exists():
        return {"schema": schema, "records": {}}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"schema": schema, "records": {}}
    if document.get("schema") != schema:
        return {"schema": schema, "records": {}}
    return document


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    # 0600 before it is in place: the file holds password hashes, and a window
    # where it is world-readable is a window.
    os.chmod(temporary, 0o600)
    temporary.replace(path)


# ------------------------------------------------------------------ users --

def users(root: Path) -> dict[str, dict]:
    return _read(Path(root) / USERS_NAME, USERS_SCHEMA)["records"]


def user_list(root: Path) -> list[dict]:
    """Everything about each account except anything that could authenticate."""
    return sorted(
        ({"username": name,
          "created_at_utc": record.get("created_at_utc"),
          "created_by": record.get("created_by"),
          "last_login_utc": record.get("last_login_utc")}
         for name, record in users(root).items()),
        key=lambda r: r["username"],
    )


def any_users(root: Path) -> bool:
    return bool(users(root))


def create_user(root: Path, username: str, password: str, *, by: str = "panel") -> dict:
    username = (username or "").strip().lower()
    if not NAME_PATTERN.match(username):
        raise AuthError("a username is 2 to 64 characters of lowercase letters, "
                        "digits, dot, underscore or hyphen")
    if len(password or "") < MINIMUM_PASSWORD:
        raise AuthError(f"a password is at least {MINIMUM_PASSWORD} characters")

    path = Path(root) / USERS_NAME
    document = _read(path, USERS_SCHEMA)
    if username in document["records"]:
        raise AuthError(f"{username} already exists")
    document["records"][username] = {
        "password": hash_password(password),
        "created_at_utc": _stamp(_now()),
        "created_by": by,
        "last_login_utc": None,
    }
    _write(path, document)
    return {"username": username, "created_at_utc": document["records"][username]["created_at_utc"]}


def set_password(root: Path, username: str, password: str) -> None:
    if len(password or "") < MINIMUM_PASSWORD:
        raise AuthError(f"a password is at least {MINIMUM_PASSWORD} characters")
    path = Path(root) / USERS_NAME
    document = _read(path, USERS_SCHEMA)
    if username not in document["records"]:
        raise AuthError(f"no user {username}")
    document["records"][username]["password"] = hash_password(password)
    _write(path, document)


def delete_user(root: Path, username: str) -> None:
    """Refuses the last one. A panel with no accounts is a panel nobody can
    reach, and the way back in is a shell on the host."""
    path = Path(root) / USERS_NAME
    document = _read(path, USERS_SCHEMA)
    if username not in document["records"]:
        raise AuthError(f"no user {username}")
    if len(document["records"]) == 1:
        raise AuthError("that is the only account; deleting it locks everyone out")
    del document["records"][username]
    _write(path, document)
    # Every session that account held stops now, not when it expires.
    sessions_path = Path(root) / SESSIONS_NAME
    sessions = _read(sessions_path, SESSION_SCHEMA)
    sessions["records"] = {k: v for k, v in sessions["records"].items()
                           if v.get("username") != username}
    _write(sessions_path, sessions)


# --------------------------------------------------------------- sessions --

def login(root: Path, username: str, password: str) -> str:
    """A session token, or nothing. The failure never says which half was wrong.

    Distinguishing "no such user" from "wrong password" hands over a list of
    real accounts, and this panel's usernames are people's names.
    """
    username = (username or "").strip().lower()
    path = Path(root) / USERS_NAME
    document = _read(path, USERS_SCHEMA)
    record = document["records"].get(username)

    if record is None:
        # Spend the same time as a real verification would, so the answer does
        # not arrive faster for a username that does not exist.
        hash_password(password or "")
        raise AuthError("that username and password do not match")
    if not verify_password(password or "", record["password"]):
        raise AuthError("that username and password do not match")

    record["last_login_utc"] = _stamp(_now())
    _write(path, document)

    token = secrets.token_urlsafe(32)
    sessions_path = Path(root) / SESSIONS_NAME
    sessions = _read(sessions_path, SESSION_SCHEMA)
    sessions["records"] = {k: v for k, v in sessions["records"].items()
                           if v.get("expires_at_utc", "") > _stamp(_now())}
    sessions["records"][token_fingerprint(token)] = {
        "username": username,
        "created_at_utc": _stamp(_now()),
        "expires_at_utc": _stamp(_now() + timedelta(days=SESSION_DAYS)),
    }
    _write(sessions_path, sessions)
    return token


def whoami(root: Path, token: str | None) -> str | None:
    """The account a token belongs to, if it is live."""
    if not token:
        return None
    sessions = _read(Path(root) / SESSIONS_NAME, SESSION_SCHEMA)
    record = sessions["records"].get(token_fingerprint(token))
    if record is None:
        return None
    if record.get("expires_at_utc", "") <= _stamp(_now()):
        return None
    return record.get("username")


def logout(root: Path, token: str | None) -> None:
    if not token:
        return
    path = Path(root) / SESSIONS_NAME
    sessions = _read(path, SESSION_SCHEMA)
    if sessions["records"].pop(token_fingerprint(token), None) is not None:
        _write(path, sessions)


# --------------------------------------------------------------- machines --
#
# A worker is not a person.
#
# Workers on other hosts publish surfaces to the panel over the network, and
# until now the only way in was a session cookie -- which meant either putting a
# human's password in an env file on every worker host, or leaving the upload
# path unauthenticated. Both are worse than a token that says what it is.
#
# The differences from a session that matter:
#
#   * it is named, so the audit log says "gpu-1-segment" and not a person who
#     was not there;
#   * it does not expire, because a worker restarting at 3am cannot log in
#     again;
#   * it is revocable one at a time, so a compromised host is one revocation
#     rather than a password rotation everywhere.
#
# Stored hashed, exactly like session tokens: the file is a list of what a token
# would have to hash to, not a list of tokens. It is shown once, when created.

MACHINES_SCHEMA = "campaignx.panel_machines.v1"
MACHINES_NAME = "MACHINES.json"

# Long enough that guessing is not a strategy: 32 bytes of urlsafe base64.
MACHINE_TOKEN_BYTES = 32
MACHINE_PREFIX = "helena-machine-"


def machines(root: Path) -> dict[str, dict]:
    return _read(Path(root) / MACHINES_NAME, MACHINES_SCHEMA)["records"]


def machine_list(root: Path) -> list[dict]:
    """Everything about each machine token except the token."""
    return sorted(
        ({"name": record.get("name"),
          "created_at_utc": record.get("created_at_utc"),
          "created_by": record.get("created_by"),
          "last_used_utc": record.get("last_used_utc")}
         for record in machines(root).values()),
        key=lambda r: r["name"] or "",
    )


def create_machine_token(root: Path, name: str, *, by: str = "panel") -> str:
    """Mint a token for a worker. Returned once and never recoverable.

    The name is checked against the same pattern as a username, and against the
    same namespace: a machine called `limegs` reading as a person in the audit
    log is the confusion this is supposed to remove.
    """
    name = (name or "").strip().lower()
    if not NAME_PATTERN.match(name):
        raise AuthError(
            "a machine name is 2 to 64 characters of a-z, 0-9, dot, dash or "
            "underscore, starting with a letter or digit")
    if name in users(root):
        raise AuthError(f"{name!r} is already a person; pick another name")
    path = Path(root) / MACHINES_NAME
    document = _read(path, MACHINES_SCHEMA)
    if any(r.get("name") == name for r in document["records"].values()):
        raise AuthError(f"a machine token named {name!r} already exists")
    token = MACHINE_PREFIX + secrets.token_urlsafe(MACHINE_TOKEN_BYTES)
    document["records"][token_fingerprint(token)] = {
        "name": name,
        "created_at_utc": _stamp(_now()),
        "created_by": by,
        "last_used_utc": None,
    }
    _write(path, document)
    return token


def revoke_machine_token(root: Path, name: str) -> bool:
    """By name, because the token is not recoverable to revoke it by."""
    path = Path(root) / MACHINES_NAME
    document = _read(path, MACHINES_SCHEMA)
    doomed = [k for k, r in document["records"].items() if r.get("name") == name]
    for key in doomed:
        document["records"].pop(key)
    if doomed:
        _write(path, document)
    return bool(doomed)


def whoami_machine(root: Path, token: str | None) -> str | None:
    """The machine a token belongs to, if it is live.

    Records last_used_utc, so a token nobody uses any more is visible as one
    that can be revoked. Written only when the day changes: a worker polling
    every ten seconds would otherwise rewrite this file forever.
    """
    if not token or not token.startswith(MACHINE_PREFIX):
        return None
    path = Path(root) / MACHINES_NAME
    document = _read(path, MACHINES_SCHEMA)
    record = document["records"].get(token_fingerprint(token))
    if record is None:
        return None
    today = _stamp(_now())[:10]
    if (record.get("last_used_utc") or "")[:10] != today:
        record["last_used_utc"] = _stamp(_now())
        _write(path, document)
    return record.get("name")
