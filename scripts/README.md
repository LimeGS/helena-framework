# Shared scripts and harnesses

`scripts/` is intentionally small. It contains only code that is shared by
more than one material stage or that orchestrates declared stage commands.

- `harness/`: command-line entry points that compose already-declared stages;
  they do not contain scientific transformations of their own.
- `container/`: receipt and runtime-capture helpers used by any containerized
  stage.
- `models/`: `install_ink_weights.py`, which fetches the checkpoints the weight
  manifest names and verifies each against its digest.
  It writes into the models volume from outside, which the panel is meant to
  be the one process doing; `harness/install_declared_weights.py` asks the
  panel what it lacks and has it fetch each checkpoint against the profile's
  digest, through the same API the Models page uses.
- `migrations/`: one-off scripts that move existing control-plane records onto
  a rule that arrived after them.

Patches to third-party tooling live under `containers/patches/`, beside the
images that apply them.

Stage-specific implementation belongs beside the stage interface:

```text
framework/stages/01-segmentation/scripts/
framework/stages/02-flattening/scripts/
framework/stages/03-ink/scripts/
framework/stages/04-validation/scripts/
framework/stages/05-reconstruction/scripts/
framework/stages/06-discovery/scripts/
```

The historical research scripts retain their original names inside those
folders. They are not renamed during this migration, so a receipt can still
identify the exact implementation through its Git blob hash. Historical
commands from before this layout change remain reproducible by checking out
their recorded commit.
