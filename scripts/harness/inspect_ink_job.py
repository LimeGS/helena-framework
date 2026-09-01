"""What argv a stored job would actually run, without running it.

Reads one P3 job out of the control plane and asks job_store for the runner and
the command it would build. The gap between what a job records and what the
worker executes is where a mis-specified parameter hides -- it costs nothing to
print, and a job that raises here would have failed on the GPU instead.

The job id is from the August 11-12 PHerc0139 orientation investigation and is
that run's, not a fixture. Point CX_DB at the control plane and edit the id.
"""
import json, sys
sys.path.insert(0, "/workspace/campaign-x/framework/stages/03-ink/fleet")
sys.path.insert(0, "/workspace/campaign-x")
from job_store import InkJobStore, command_for, runner_for
import os
store = InkJobStore(os.environ["CX_DB"])
job = store.job("p3-af8a53db1aa54c")
print("phase      :", job.get("phase"))
print("parameters :", json.dumps(job.get("parameters"))[:400])
try:
    runner = runner_for(job)
    print("runner     :", runner)
    argv = command_for(job, runner=str(runner), output_dir="/tmp/probe")
    print("argv       :")
    for a in argv: print("   ", a)
except Exception as e:
    import traceback; traceback.print_exc()
