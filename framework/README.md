# Helena Framework framework

This directory is the reusable, campaign-independent contract of Helena Framework.
It contains no CT cache, trained weights, finds, run logs, campaign receipts,
or target-specific conclusions. A run may read this directory, but no run may
write into it.

The numbered stages describe transformations of material, not the history of
research milestones. Each stage accepts immutable inputs declared by a
manifest and writes a new run directory plus a receipt. The stage descriptors
are intentionally small machine-readable indexes; implementation remains in
`scripts/` until each implementation has a container-equivalence receipt.

`contracts/` owns schemas and invariants. `profiles/` is where versioned,
scientific execution profiles belong. `registries/` separates published
methods from locally runnable adapters and from methods whose transfer has
actually passed compatible controls. `stages/` declares the interfaces among
segmenting, flattening, ink inference, validation, reconstruction, and
discovery.

Stage 03 provides reusable, checkpoint-bound ink lane profiles plus
`route_ink_methods.py`. The router consumes a campaign-owned policy; therefore
new campaigns and new rolls can choose different validated lanes without
embedding target IDs or experimental decisions in the framework.
