#!/usr/bin/env python3
"""Structured (JSON-lines) logging with size-based rotation for the campaign
loop. One line per event: {"ts", "level", "msg", ...extra fields}. Rotation
caps disk use on the remote instance over an unattended multi-hour run --
the same disk-full failure mode that hit the local Mac earlier today.

Cheap to consume: `grep '"level": "ERROR"' campaign.log` costs near-zero
tokens vs reading a full log dump.
"""
import json
import os
import time

MAX_BYTES = 20 * 1024 * 1024  # 20MB per file
MAX_ROTATED = 5               # keep at most this many old files


class CampaignLog:
    def __init__(self, path):
        self.path = path

    def _rotate_if_needed(self):
        if not os.path.exists(self.path) or os.path.getsize(self.path) < MAX_BYTES:
            return
        for i in range(MAX_ROTATED - 1, 0, -1):
            src, dst = f"{self.path}.{i}", f"{self.path}.{i + 1}"
            if os.path.exists(src):
                os.rename(src, dst)
        os.rename(self.path, f"{self.path}.1")

    def log(self, level, msg, **extra):
        assert level in ("INFO", "WARN", "ERROR"), level
        self._rotate_if_needed()
        line = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "level": level, "msg": msg, **extra}
        with open(self.path, "a") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def info(self, msg, **extra): self.log("INFO", msg, **extra)
    def warn(self, msg, **extra): self.log("WARN", msg, **extra)
    def error(self, msg, **extra): self.log("ERROR", msg, **extra)


if __name__ == "__main__":
    import tempfile

    d = tempfile.mkdtemp()
    p = os.path.join(d, "campaign.log")
    lg = CampaignLog(p)
    lg.info("started", seed="w999")
    lg.warn("retrying", attempt=2)
    lg.error("gate failed", gate="windability", reason="pierces sheet")
    lines = open(p).read().splitlines()
    assert len(lines) == 3
    parsed = [json.loads(l) for l in lines]
    assert parsed[2]["level"] == "ERROR" and parsed[2]["gate"] == "windability"

    # force rotation with a tiny cap so the test doesn't need 20MB of writes.
    # (rebind the module-level global directly -- `import camplog` here would
    # create a SECOND module instance when this file is run as __main__,
    # whose MAX_BYTES the running CampaignLog class would never see)
    MAX_BYTES = 200  # noqa: F811 -- intentional rebind of the module global
    p2 = os.path.join(d, "rot.log")
    lg2 = CampaignLog(p2)
    for i in range(50):
        lg2.info(f"padding line number {i} to force rotation past the tiny cap")
    assert os.path.exists(p2), "current log should exist"
    assert os.path.exists(p2 + ".1"), "should have rotated at least once"
    n_rotated = sum(1 for i in range(1, MAX_ROTATED + 2) if os.path.exists(f"{p2}.{i}"))
    assert 1 <= n_rotated <= MAX_ROTATED, f"rotated file count out of bounds: {n_rotated}"
    assert not os.path.exists(f"{p2}.{MAX_ROTATED + 1}"), "should not exceed MAX_ROTATED"

    print("camplog.py self-test PASSED")
