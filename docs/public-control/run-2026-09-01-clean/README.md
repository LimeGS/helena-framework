# The same control, on a machine installed with one command

The run beside this one needed twenty-five fixes between a rented machine and a
receipt. This is the check that they hold: a new machine, `install.sh --gpu`
from the published repository, and nothing edited by hand.

    control_state: CONTROL_PASS       first_nonpassing_boundary: null
    content_sha256: eb90c16981202803c4b76a0bb3db96c2cf9fc49078bacc4db75634c9dba7545f

    through:  helena-queue
    job_id:   p5-2e942b941a3349
    artifact: /artifacts/ink-maps-v1/surfaces/PHerc0139/ink-maps/p5-2e942b941a3349

One command brought up nine of ten containers and left four workers polling with
this host's own name and no database errors. The tenth is the surface-QC
runtime, which wants a checkpoint nothing downloads -- P2's, not the ink lane's,
so the control does not touch it. It was the one finding on the list still
open when this was written; the GPU deploy fetches that checkpoint itself now
(`REPRODUCE.md`, step 2).

Two prerequisites were installed by hand, and neither is Helena's: the NVIDIA
open kernel modules, which Blackwell requires and this VM template does not
ship, and the NVIDIA Container Toolkit, without which Docker cannot reach the
card. The installer warns about the second now rather than letting the workers
start and find no device.

## What the digests say

    ours, twice, nine days apart   fc9f91da…   bit-identical
    rented 5090, twice             f4abc1b4…   bit-identical
    across the two                             different bytes

Same GPU, same bytes. Different GPU, different bytes and the same statistics to
four decimals -- p50 0.2784, p99 0.7765, std 0.1674 across all five runs. That
is what floating-point accumulation does, and it is why the digest is a fair
check only against a run on the same card.
