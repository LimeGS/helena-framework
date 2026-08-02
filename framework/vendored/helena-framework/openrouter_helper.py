#!/usr/bin/env python3
"""Escalation-only OpenRouter helper. Called by executor_loop.py ONLY when
a systemic failure is detected (5 consecutive distinct seeds failed) --
NOT per-iteration. The deterministic grow-gate pipeline handles the normal
case; this is for genuinely novel situations where a second opinion on
"what's actually going wrong" is worth it.

LOGGING: every call -- success or failure -- is recorded in FULL (prompt,
response, latency, model, success flag) to a dedicated JSON-lines log via
camplog.CampaignLog, separate from the general campaign.log so the LLM
audit trail can be reviewed on its own
(`grep '"kind": "llm_call"' llm_calls.log`) without the operational noise
of every seed attempt. This is the complete record of every LLM
interaction in the campaign -- there is currently only one call site
(this function), but any future one should follow the same convention:
log inside the call, not leave it to the caller to remember.

SECURITY: the API key is read from the OPENROUTER_API_KEY environment
variable ONLY -- never hardcoded, never logged, never committed. Set it on
the remote instance's environment before starting the loop, e.g.:
    export OPENROUTER_API_KEY="sk-or-v1-..."
(A live key was shared in plaintext chat earlier -- worth rotating it at
some point, independent of how carefully this script handles it from here.)
The prompt/response ARE logged (that's the point of this file), but the key
itself never appears in any log line -- verified by the self-test below.
"""
import json
import os
import time
import urllib.request

MODEL = "tencent/hy3:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"


def build_prompt(context, recent_errors):
    return (
        "A VC3D segmentation pipeline hit a systemic failure (5+ consecutive "
        "distinct seeds failed). Context: " + context + "\n"
        "Recent errors:\n" + "\n".join(f"- {e}" for e in recent_errors[-5:]) + "\n\n"
        "In 3-5 sentences: what's the most likely root cause, and what's one "
        "concrete thing to try differently on the next batch of seeds? Do not "
        "suggest modifying the MCP server code -- that requires human review."
    )


def escalate(context, recent_errors, log=None, timeout_s=30, transport=None):
    """context: short string describing the systemic failure.
    recent_errors: list of the last few error strings.
    log: a camplog.CampaignLog instance (or any object with .log(level, msg,
      **extra)) pointed at the dedicated llm_calls.log -- if given, the FULL
      prompt/response/timing is recorded here regardless of outcome.
    transport: optional callable(req_body_dict) -> response_dict, used ONLY
      for QA (inject a fake transport so the exact request/response/logging
      path is exercised without a live network call or a real key). Default
      None means "make the real HTTPS call".
    Returns the model's suggestion as plain text, or None if no key is
    configured. Never raises -- an escalation helper that itself crashes the
    loop would defeat its purpose; failures are logged and returned as a
    bracketed error string instead."""
    key = os.environ.get("OPENROUTER_API_KEY")
    prompt = build_prompt(context, recent_errors)
    t0 = time.time()

    if not key:
        if log:
            log.log("WARN", "escalation skipped: no OPENROUTER_API_KEY set", kind="llm_call",
                     model=MODEL, prompt=prompt, response=None, success=False,
                     latency_s=0.0, error="no API key configured")
        return None

    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 300}
    response_text, error = None, None
    try:
        if transport is not None:
            data = transport(body)  # QA injection point -- no real network/key use
        else:
            req = urllib.request.Request(
                API_URL, data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read())
        response_text = data["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001 -- must never crash the caller
        error = f"{type(e).__name__}: {e}"

    latency = round(time.time() - t0, 3)
    if log:
        log.log("ERROR" if error else "INFO",
                 "escalation call failed" if error else "escalation call succeeded",
                 kind="llm_call", model=MODEL, prompt=prompt, response=response_text,
                 success=error is None, latency_s=latency, error=error)

    if error:
        return f"[escalation call failed: {error}]"
    return response_text


if __name__ == "__main__":
    import tempfile

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from camplog import CampaignLog

    d = tempfile.mkdtemp()
    log_path = os.path.join(d, "llm_calls.log")
    log = CampaignLog(log_path)

    # 1) no-key path: logs a WARN, returns None, no network attempted
    saved = os.environ.pop("OPENROUTER_API_KEY", None)
    result = escalate("test context", ["error 1", "error 2"], log=log)
    assert result is None, "should return None cleanly when no key is set"
    lines = [json.loads(l) for l in open(log_path)]
    assert len(lines) == 1 and lines[0]["level"] == "WARN" and lines[0]["kind"] == "llm_call"
    print("  [PASS] no-key path: logged, returned None, no network call")

    # 2) successful call via injected transport (no real network/key) --
    #    verifies full prompt+response get logged
    os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-fake-for-qa-only"

    def fake_ok_transport(body):
        assert body["model"] == MODEL
        return {"choices": [{"message": {"content": "try seed offset +5mm axially"}}]}

    result = escalate("systemic failure test", ["job_failed x5"], log=log, transport=fake_ok_transport)
    assert result == "try seed offset +5mm axially"
    lines = [json.loads(l) for l in open(log_path)]
    assert len(lines) == 2
    rec = lines[1]
    assert rec["kind"] == "llm_call" and rec["success"] is True
    assert "systemic failure test" in rec["prompt"]
    assert rec["response"] == "try seed offset +5mm axially"
    assert rec["latency_s"] >= 0
    print("  [PASS] successful call: full prompt+response logged")

    # 3) failing transport -- verifies errors are logged too, not swallowed silently
    def fake_broken_transport(body):
        raise TimeoutError("simulated network timeout")

    result = escalate("systemic failure test 2", ["job_failed x5"], log=log, transport=fake_broken_transport)
    assert result is not None and "escalation call failed" in result
    lines = [json.loads(l) for l in open(log_path)]
    assert len(lines) == 3
    rec = lines[2]
    assert rec["success"] is False and rec["level"] == "ERROR"
    assert "TimeoutError" in rec["error"]
    print("  [PASS] failed call: error logged, not swallowed, caller doesn't crash")

    # 4) the key itself must NEVER appear in any log line
    full_log_text = open(log_path).read()
    assert "sk-or-v1-fake-for-qa-only" not in full_log_text, "API KEY LEAKED INTO LOGS"
    print("  [PASS] API key never appears in the log file")

    if saved:
        os.environ["OPENROUTER_API_KEY"] = saved
    else:
        os.environ.pop("OPENROUTER_API_KEY", None)
    print("openrouter_helper.py self-test PASSED (4/4)")
