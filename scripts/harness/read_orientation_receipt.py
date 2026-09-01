"""Read the orientation proof's own receipt for the latest control surface."""
import json, os, sys
sys.path.insert(0, "/workspace/campaign-x/scripts/harness")
from panel_client import Panel, PanelError

MISSION, SAMPLE = "control-fl-pherc0139-dev-20260806", "PHerc0139"
JOB = "p3-e31e99be9d6d4b"
c = Panel("https://localhost:8800", insecure=True)
c.sign_in(os.environ["U"], os.environ["HELENA_PANEL_PASSWORD"])

rows = c.call("GET", f"/api/flattening?sample={SAMPLE}&mission={MISSION}").get("rows", [])
mine = next((r for r in rows if r.get("requested_by_job_id") == JOB), None)
if not mine:
    print("no flattening row for", JOB, "| rows:", len(rows)); raise SystemExit(0)
surface = mine["surface_id"]
print("surface:", surface)

try:
    proof = c.call("GET", "/api/geometry/orientation-proof?"
                   f"mission={MISSION}&sample={SAMPLE}&surface={surface}&p3_job={JOB}")
except PanelError as e:
    print("status:", e.status, "body:", e.body[:400]); raise SystemExit(0)

print("proof keys:", sorted(proof))
for k in ("state", "verdict", "proven", "reason", "reason_code", "non_claim"):
    if k in proof:
        print(f"  {k}: {json.dumps(proof[k])[:220]}")
# The verdict and why, not the whole document.
for k, v in proof.items():
    if isinstance(v, (str, bool, int, float)) or k.lower().endswith(("state","verdict","reason","status")):
        print(f"  {k} = {json.dumps(v)[:200]}")
ev = proof.get("evidence") or proof.get("checks") or {}
if isinstance(ev, dict):
    for k, v in ev.items():
        print(f"  evidence.{k} = {json.dumps(v)[:180]}")
