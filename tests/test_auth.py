"""Accounts, sessions, and the four things that must not slip.

Passwords never stored recoverably. Session tokens never stored as tokens. A
failed login never says which half was wrong. And the last account cannot be
deleted, because a panel with no accounts is one nobody can reach.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.contracts import auth  # noqa: E402

PASSWORD = "a-long-enough-one"


def test_a_password_is_never_stored_in_a_form_that_can_be_read_back(tmp_path: Path):
    auth.create_user(tmp_path, "ana", PASSWORD)
    raw = (tmp_path / auth.USERS_NAME).read_text(encoding="utf-8")
    assert PASSWORD not in raw
    stored = json.loads(raw)["records"]["ana"]["password"]
    assert stored.startswith("scrypt$")
    # Two accounts with the same password do not look alike: the salt is
    # per-user, so a matching pair in the file reveals nothing.
    auth.create_user(tmp_path, "ben", PASSWORD)
    assert json.loads(raw := (tmp_path / auth.USERS_NAME).read_text())["records"]["ben"]["password"] != stored


def test_the_users_file_is_not_world_readable(tmp_path: Path):
    auth.create_user(tmp_path, "ana", PASSWORD)
    mode = os.stat(tmp_path / auth.USERS_NAME).st_mode & 0o777
    assert mode == 0o600, f"password hashes at mode {mode:o}"


def test_a_session_token_is_stored_only_as_its_fingerprint(tmp_path: Path):
    auth.create_user(tmp_path, "ana", PASSWORD)
    token = auth.login(tmp_path, "ana", PASSWORD)
    raw = (tmp_path / auth.SESSIONS_NAME).read_text(encoding="utf-8")
    assert token not in raw, "the sessions file holds live tokens"
    assert auth.token_fingerprint(token) in raw
    assert auth.whoami(tmp_path, token) == "ana"


def test_a_wrong_password_and_a_missing_user_are_the_same_answer(tmp_path: Path):
    auth.create_user(tmp_path, "ana", PASSWORD)
    messages = set()
    for username, password in (("ana", "wrong-one-entirely"), ("nobody", PASSWORD)):
        with pytest.raises(auth.AuthError) as raised:
            auth.login(tmp_path, username, password)
        messages.add(str(raised.value))
    assert len(messages) == 1, f"the failure distinguishes the two: {messages}"


def test_an_unknown_user_does_not_answer_faster(tmp_path: Path):
    """A quick "no" is a directory of who exists. The usernames here are
    people's names, so that is worth the deliberate cost."""
    auth.create_user(tmp_path, "ana", PASSWORD)

    def attempt(username: str) -> float:
        started = time.perf_counter()
        try:
            auth.login(tmp_path, username, "wrong-one-entirely")
        except auth.AuthError:
            pass
        return time.perf_counter() - started

    real = min(attempt("ana") for _ in range(3))
    missing = min(attempt("nobody") for _ in range(3))
    # Generous: this catches "returns instantly", not a timing side channel.
    assert missing > real * 0.5, f"missing user answered in {missing:.3f}s vs {real:.3f}s"


def test_case_and_whitespace_do_not_make_a_second_account(tmp_path: Path):
    auth.create_user(tmp_path, "ana", PASSWORD)
    with pytest.raises(auth.AuthError, match="already exists"):
        auth.create_user(tmp_path, "  ANA ", PASSWORD)
    assert auth.login(tmp_path, " Ana ", PASSWORD)


@pytest.mark.parametrize("username", ["", "a", "Has Capitals", "with/slash", "-lead", "x" * 65])
def test_usernames_stay_boring(tmp_path: Path, username):
    with pytest.raises(auth.AuthError):
        auth.create_user(tmp_path, username, PASSWORD)


def test_a_short_password_is_refused(tmp_path: Path):
    with pytest.raises(auth.AuthError, match="at least"):
        auth.create_user(tmp_path, "ana", "short")


def test_the_last_account_cannot_be_deleted(tmp_path: Path):
    auth.create_user(tmp_path, "ana", PASSWORD)
    with pytest.raises(auth.AuthError, match="only account"):
        auth.delete_user(tmp_path, "ana")
    auth.create_user(tmp_path, "ben", PASSWORD)
    auth.delete_user(tmp_path, "ana")
    assert [u["username"] for u in auth.user_list(tmp_path)] == ["ben"]


def test_deleting_an_account_ends_its_sessions_now(tmp_path: Path):
    auth.create_user(tmp_path, "ana", PASSWORD)
    auth.create_user(tmp_path, "ben", PASSWORD)
    token = auth.login(tmp_path, "ana", PASSWORD)
    assert auth.whoami(tmp_path, token) == "ana"
    auth.delete_user(tmp_path, "ana")
    assert auth.whoami(tmp_path, token) is None


def test_logout_ends_the_session_and_a_forged_token_is_nobody(tmp_path: Path):
    auth.create_user(tmp_path, "ana", PASSWORD)
    token = auth.login(tmp_path, "ana", PASSWORD)
    auth.logout(tmp_path, token)
    assert auth.whoami(tmp_path, token) is None
    assert auth.whoami(tmp_path, "not-a-token") is None
    assert auth.whoami(tmp_path, None) is None


def test_changing_a_password_invalidates_the_old_one(tmp_path: Path):
    auth.create_user(tmp_path, "ana", PASSWORD)
    auth.set_password(tmp_path, "ana", "a-different-long-one")
    with pytest.raises(auth.AuthError):
        auth.login(tmp_path, "ana", PASSWORD)
    assert auth.login(tmp_path, "ana", "a-different-long-one")


def test_the_listing_carries_nothing_that_could_authenticate(tmp_path: Path):
    auth.create_user(tmp_path, "ana", PASSWORD)
    auth.login(tmp_path, "ana", PASSWORD)
    for record in auth.user_list(tmp_path):
        assert set(record) == {"username", "created_at_utc", "created_by", "last_login_utc"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
