# Catalog

The catalog is the campaign-independent inventory of source CT volumes,
coordinate contracts, published surfaces, and availability metadata. It must
not contain generated TIFF stacks, model maps, private credentials, or
campaign conclusions. During migration, canonical legacy records are exposed
as symlinks here rather than duplicated.

## Current TIFXYZ catalogue

`geometry_surface_catalog_v2/` is archive-first: every complete
`workspace/surfaces/campaign-x/<scroll>/<surface>/` directory is exactly one
Campaign X row. Growth receipts are attached as provenance, but a receipt
cannot invent a surface whose TIFXYZ is absent. The SQLite database is the
query interface and the JSON summary is the human/audit interface.

Regenerate it after archiving a new TIFXYZ:

```sh
python3 framework/stages/01-segmentation/scripts/build_geometry_surface_catalog_v2.py \
  --root . \
  --archive-root workspace/surfaces/campaign-x \
  --public-inventory workspace/archive/campaign-x-2026/legacy-phases/phase4/public_target_tifxyz_v1/PUBLIC_TARGET_TIFXYZ_INVENTORY.json \
  --database workspace/catalog/geometry_surface_catalog_v2/GEOMETRY_SURFACE_CATALOG.sqlite \
  --summary workspace/catalog/geometry_surface_catalog_v2/GEOMETRY_SURFACE_CATALOG_SUMMARY.json
```

Counts and gross areas are inventory measurements, not deduplicated physical
sheet coverage. Every own surface remains `UNVALIDATED` until Module 04
performs physical geometry QC.

## Historical growth exclusions

`historical_growth_exclusions_v1/HISTORICAL_GROWTH_EXCLUSIONS.json` answers a
different question: **which 3D regions have already been grown at least once?**
It includes only PASSED geometry-recovery receipts whose local `meta.json`
still matches the SHA-256 frozen in that receipt. Its rows are used to avoid
duplicate seed selection.

Two preservation states are intentionally distinct:

- `SOURCE_TIFXYZ_COMPLETE`: the historical runtime still has all four source
  files and may be copied to the durable archive after re-verification;
- `VERIFIED_META_ONLY`: only the hash-verified bounding box survives. It is a
  spatial exclusion, never a reusable surface.

Regenerate and safely backfill with:

```sh
python3 framework/stages/01-segmentation/scripts/build_historical_growth_exclusions_v1.py \
  --root . \
  --output workspace/catalog/historical_growth_exclusions_v1/HISTORICAL_GROWTH_EXCLUSIONS.json

python3 framework/stages/01-segmentation/scripts/backfill_historical_geometry_surfaces_v1.py \
  --root . \
  --manifest workspace/catalog/historical_growth_exclusions_v1/HISTORICAL_GROWTH_EXCLUSIONS.json \
  --archive-root workspace/surfaces/campaign-x \
  --output workspace/catalog/historical_growth_exclusions_v1/HISTORICAL_SURFACE_BACKFILL_RECEIPT.json
```

The backfill receipt is idempotent: a repeated run reports already archived
byte-identical files. A mismatch, incomplete destination, or path escape fails
closed. Neither catalogue represents physical geometry acceptance.
