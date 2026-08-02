# Shared scripts and harnesses

`scripts/` is intentionally small. It contains only code that is shared by
more than one material stage or that orchestrates declared stage commands.

- `harness/`: command-line entry points that compose already-declared stages;
  they do not contain scientific transformations of their own.
- `container/`: receipt and runtime-capture helpers used by any containerized
  stage.
- `patches/`: narrowly scoped patches to third-party tooling.

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
