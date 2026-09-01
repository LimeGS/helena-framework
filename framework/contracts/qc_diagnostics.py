from __future__ import annotations

import re
from typing import Any

_ERROR_PREFIX = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]{0,127}):")
_TRACEBACK_FILE = re.compile(r'''^\s*File\s+["'].+["'],\s+line\s+\d+''')

_QUOTED_VALUE = r'''(?:"[^"\r\n]*"|'[^'\r\n]*')'''
_SENSITIVE_KEY = (
    r"(?:aws[_-](?:access[_-]key[_-]id|secret[_-]access[_-]key|"
    r"session[_-]token|security[_-]token)|api[_-]?key|client[_-]?secret|"
    r"private[_-]?key|credential(?:s)?|access[_-]?token|refresh[_-]?token|"
    r"password|passwd|secret|token|access[_-]?key(?:_id)?|"
    r"worker(?:[ _-]?(?:id|identity|name))?)"
)
_HEADER_NAME = r"(?:authorization|proxy-authorization|cookie|set-cookie)"
_HEADER_KEY = rf"(?:\"{_HEADER_NAME}\"|'{_HEADER_NAME}'|{_HEADER_NAME})"
_PATH_START = r"(?:[A-Za-z]:[\\/]|\\\\|/)"
_FIELD_BOUNDARY = (
    rf"(?=$|\s+(?:[\"']?{_SENSITIVE_KEY}[\"']?\s*[:=]|"
    rf"{_HEADER_KEY}(?=\s|[:=]|$)|"
    rf"[A-Za-z][A-Za-z0-9+.-]{{0,31}}://|{_PATH_START}))"
)
_CREDENTIAL_BOUNDARY = (
    rf"(?=$|[,;}}\]]|\s+(?:[\"']?{_SENSITIVE_KEY}[\"']?\s*[:=]|"
    rf"{_HEADER_KEY}(?=\s|[:=]|$)|"
    rf"[A-Za-z][A-Za-z0-9+.-]{{0,31}}://|{_PATH_START}))"
)
_PATH_BOUNDARY = (
    rf"(?=$|[,;}}\]]|\s+(?:<[^>]+>|[\"']?{_SENSITIVE_KEY}[\"']?\s*[:=]|"
    rf"{_HEADER_KEY}(?=\s|[:=]|$)|"
    rf"[A-Za-z][A-Za-z0-9+.-]{{0,31}}://|{_PATH_START}))"
)
_SUBSTITUTIONS = (
    (
        rf"(?i)(?<![A-Za-z0-9_-])(?:\{{\s*)?{_HEADER_KEY}\s*[:=]\s*"
        rf".+?{_FIELD_BOUNDARY}",
        "<redacted>",
    ),
    (
        rf"(?i)(?<![A-Za-z0-9_-])(?:\{{\s*)?{_HEADER_KEY}\s+"
        rf"(?:bearer|basic|token)\s+.+?{_FIELD_BOUNDARY}",
        "<redacted>",
    ),
    (
        rf"(?i)(?<![A-Za-z0-9_-])(?:\{{\s*)?{_HEADER_KEY}\s*[:=]\s*"
        rf".+?{_FIELD_BOUNDARY}",
        "<redacted>",
    ),
    (r"(?i)\b(?:bearer|basic)\s+[^\s,;]+", "<redacted>"),
    (
        rf"(?i)(?<![A-Za-z0-9_-])[\"']?{_SENSITIVE_KEY}[\"']?\s*[:=]\s*"
        rf"{_QUOTED_VALUE}",
        "<redacted>",
    ),
    (
        rf"(?i)(?<![A-Za-z0-9_-])[\"']?{_SENSITIVE_KEY}[\"']?\s*[:=]\s*"
        rf".+?{_CREDENTIAL_BOUNDARY}",
        "<redacted>",
    ),
    (r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b", "<redacted>"),
    (r"(?i)\b(?:postgres(?:ql)?|mysql|mariadb|redis|rediss|mongodb(?:\+srv)?|amqp(?:s)?)://[^\s]+", "<dsn>"),
    (r"(?i)\bhttps?://[^\s]+", "<url>"),
    (r"(?i)\b(?:s3|gs|az|abfs|wasbs)://[^\s]+", "<artifact-uri>"),
    (r"(?i)\b[a-z][a-z0-9+.-]{0,31}://[^\s]+", "<uri>"),
    (r'''(?i)"(?:[A-Za-z]:[\\/]|\\\\|/)[^"]+"''', "<path>"),
    (r"(?i)'(?:[A-Za-z]:[\\/]|\\\\|/)[^']+'", "<path>"),
    (
        rf"(?i)(?<![A-Za-z0-9_.-]){_PATH_START}.+?{_PATH_BOUNDARY}",
        "<path>",
    ),
)


def sanitize_error(raw: object) -> tuple[str | None, str | None]:
    if not isinstance(raw, str):
        return None, None
    message = " ".join(raw.split())
    matched = _ERROR_PREFIX.match(message)
    error_type = matched.group(1) if matched else None
    for pattern, replacement in _SUBSTITUTIONS:
        message = re.sub(pattern, replacement, message)
    if not message:
        return error_type, None
    if len(message) > 500:
        message = message[:499] + "…"
    return error_type, message


def safe_message(raw: object, fallback: str) -> str:
    """Return one bounded safe message, never serializing unsafe objects."""
    _error_type, message = sanitize_error(raw)
    return message or fallback


def receipt_with_safe_error(
    receipt: dict[str, Any], fallback: str,
) -> dict[str, Any]:
    """Clone a caller-owned receipt and replace only its durable error field."""
    safe_receipt = dict(receipt)
    safe_receipt["error"] = safe_message(receipt.get("error"), fallback)
    return safe_receipt


def extract_last_python_exception(output: str) -> tuple[str | None, str | None]:
    lines = output.splitlines()
    traceback_starts = [
        index
        for index, line in enumerate(lines)
        if line.strip() == "Traceback (most recent call last):"
    ]
    if not traceback_starts:
        return None, None
    traceback = lines[traceback_starts[-1] + 1:]
    if not any(_TRACEBACK_FILE.match(line) for line in traceback):
        return None, None
    for line in reversed(traceback):
        error_type, error = sanitize_error(line)
        if (
            error_type
            and error
            and error_type.rsplit(".", 1)[-1].endswith(("Error", "Exception"))
        ):
            return error_type, error
    return None, None
