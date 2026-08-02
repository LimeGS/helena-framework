# Scientific profiles

A profile is a versioned, immutable declaration of scientific parameters:
model and checkpoint hashes, voxel scale, renderer settings, inference
replicas, thresholds, and retention requirements. Profiles never contain a CT
cache, a secret, or a run result.

The legacy Helena Framework profiles remain in the archived campaign evidence until
they are extracted one by one with equivalence tests. New work must reference
a profile by path and SHA-256 from its stage manifest.

`03-ink/` profiles bind one runnable adapter to an exact checkpoint SHA-256,
physical input normalization, output contract, and conservative defaults.
They never decide whether a method is appropriate for a particular roll.
That decision belongs to a campaign routing policy and its frozen controls.

Extracted frozen profiles:

- `validation/ct-fiber-localization-gate-v1.json`: the prospective v1 gate
  separating surface-localized CT response from depth-diffuse fiber/laminar
  confounds. It is copied byte-for-byte from the archived Helena Framework freeze;
  SHA-256 `d0ac3eb2d518ebefc544db069078c08868da902323cbf8cea2e1bfd8e4dd122b`.
- `validation/ct-fiber-supported-window-router-v4.1.json`: the non-destructive
  physical-support router validated on MULTISCROLL_TRANSFER_V2.
- `validation/ct-fiber-texture-priority-router-v4.2.json`: the development-
  optimized B1/B2 ordering layer. V1/V2 are development-only for this profile;
  external transfer remains pending MULTISCROLL_TRANSFER_V3.
