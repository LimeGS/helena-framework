# Catalog

The catalog is the campaign-independent inventory of source CT volumes,
coordinate contracts, published surfaces, and availability metadata. It must
not contain generated TIFF stacks, model maps, private credentials, or
campaign conclusions. During migration, canonical legacy records are exposed
as symlinks here rather than duplicated.

## Eligible volumes

`eligible_volumes.json` is the list of scrolls P0 can start from: for each,
the CT volume, the m7 surface prediction over it, the voxel size and the beam
energy. The committed file is the seed a new deployment starts from. The
panel rebuilds it from the open-data bucket on startup and once a day
(`CX_CATALOG_REFRESH=1`, the default; `0` pins it to the file) and keeps the
result in its cache, falling back to the committed file when the bucket
cannot be reached.
`framework/stages/01-segmentation/scripts/build_eligible_volumes_catalog.py`
is the builder both use.

## Current TIFXYZ catalogue

`geometry_surface_catalog_v2/` is what the panel reads by default
(`CX_GEOMETRY_CATALOG`); `geometry_surface_catalog_v3/` and `_v4/` are later
builds that individual scripts under
`framework/stages/01-segmentation/scripts/` name explicitly. It is
archive-first: every complete
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

The scripts that built these two files left the framework when the search
moved out of it; the files stay as the record they were.
`HISTORICAL_SURFACE_BACKFILL_RECEIPT.json` beside the exclusions says, per
surface, whether byte-identical files were already in the archive; a
mismatch, an incomplete destination or a path escape failed closed. Neither
catalogue represents physical geometry acceptance.
