# Capability registries

These registries describe what the framework knows how to run and what it
only knows exists. They are not leaderboards and they never select a model by
recency or name alone.

Registries are scroll- and campaign-independent. Training scrolls may appear
because they are an intrinsic part of a published model's provenance, but a
target-specific decision, exception, threshold or benchmark result belongs in
`workspace/campaigns/<campaign-id>/`, never here.

Every executable plan must resolve a method through a registry entry and then
freeze the exact source revision, checkpoint SHA-256, preprocessing, physical
voxel scale, axis/layer order, and adapter version in its run manifest. An
entry marked `KNOWN_NOT_INTEGRATED` or `EXPERIMENTAL_BLOCKED` cannot become a
production default without a new adapter and a prospectively frozen control
evaluation.

The registry separates three questions that used to be conflated:

1. Is the method published?
2. Can this repository execute it reproducibly?
3. Has it transferred on controls compatible with the target scan?

Only the third question can justify target routing. A target activation never
counts as its own calibration evidence.
