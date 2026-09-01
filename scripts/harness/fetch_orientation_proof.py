"""The orientation proof the panel serves for one control surface.

Asks the HTTP API rather than the database on purpose: this is the answer a
client actually receives, so a proof that exists but does not serve shows up
here as a status instead of a row. Prints the error body on failure, which is
where the reason lives.

The mission, surface and job ids are from the August 11-12 PHerc0139 orientation
investigation and are that run's, not a fixture. Needs U and
HELENA_PANEL_PASSWORD in the environment.
"""
import json, os, sys
sys.path.insert(0, "/workspace/campaign-x/scripts/harness")
from panel_client import Panel, PanelError
c = Panel("https://localhost:8800", insecure=True)
c.sign_in(os.environ["U"], os.environ["HELENA_PANEL_PASSWORD"])
q = ("mission=control-fl-pherc0139-dev-20260806&sample=PHerc0139"
     "&surface=407fb91f-6b5e-555a-933c-3fc96ac59726&p3_job=p3-ae5ec64efc8240")
try:
    print(json.dumps(c.call("GET", "/api/geometry/orientation-proof?" + q), indent=1)[:600])
except PanelError as e:
    print("status:", e.status)
    print("body  :", e.body[:700])
