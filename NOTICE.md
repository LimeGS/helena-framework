# Notice — what MIT covers here, and what it does not

The code in this repository is MIT (see `LICENSE`). Two things it cannot cover,
because they are not ours to license:

## Scan data and anything derived from it

The CT volumes, official segmentations and published ink maps this framework
reads come from the **Vesuvius Challenge** and are **CC BY-NC 4.0** —
non-commercial. That licence follows the data into whatever is made from it: a
rendered layer stack, a flattened sheet, a probability map or a plate is a
derivative of a CC BY-NC source, whatever the code that produced it is licensed
under.

So: run this commercially if you like — the *tools* are MIT. What you produce
from Vesuvius Challenge data is not, and this repository makes no claim over it
in either direction.

The Vesuvius Challenge data agreement is the authority, not this file.

## Model checkpoints

No weights are in this repository. The ink profiles under
`framework/profiles/03-ink/` name checkpoints and describe how to fetch them;
each comes with its own terms from whoever trained it. A profile is a pointer,
not a redistribution.

## volume-cartographer, which the worker images contain

`helena-villa` is [volume-cartographer](https://github.com/ScrollPrize/villa)
compiled from source, and `helena-worker-cpp` is built on it; `helena-gpu-runtime`
carries three of its binaries with their library closure, taken out of
`helena-villa` in a build stage. **volume-cartographer is
GPL-3.0**, not MIT — the villa repository's own root LICENSE is MIT, and the
subdirectory this framework compiles is not covered by it.

Nothing here modifies it. The build clones one pinned commit, verifies the tree
hash it received and compiles it unchanged; `containers/images/Containerfile.villa`
and `containers/images/scrollfiesta/locks/source-lock.json` are the whole recipe.

The Python that drives it is a separate work: the `vc_*` tools are standalone
executables run as subprocesses, so this repository's MIT licence is unaffected.

What that costs, if you distribute a built image rather than only running one:
GPL-3.0 §6 asks that whoever conveys the binaries can supply the corresponding
source. The commit is pinned and public and the build scripts are here, so the
answer to "which source" is already recorded — but the obligation belongs to
whoever publishes the image, not to upstream, and it outlives upstream's
repository. Mirror the commit before publishing.

None of this applies to building and running it yourself, which is what
`deploy-platform.sh` does by default.

## Third-party code vendored here

Everything under `framework/vendored/` is recorded in `INDEX.json` with the
commit it was taken at. All of *that* is MIT (volume-cartographer above is not
vendored here -- it is fetched and compiled) — most was written for this project
and extracted for reuse:

| Component | Origin | Code |
|---|---|---|
| `helena-framework` | ours, extracted from the PHerc0139 campaign | MIT |
| `hf-proxy-v4-dataset` | ours (`vesuvius-experiments`) | MIT; the dataset card it ships is CC BY-NC 4.0 |
| `pherc0139-column-atlas-gh` | ours ([LimeGS/pherc0139-column-atlas](https://github.com/LimeGS/pherc0139-column-atlas)) | MIT; bundled plates governed by their sources |
| `ppm_from_tifxyz` | ours | MIT |
| `reference-strips` | ours | MIT; referenced scan data CC BY-NC 4.0 |
| `scroll-streaming-tools` | ours | MIT; referenced scan data CC BY-NC 4.0 |
| `scroll-tracing-benchmark-v4` | ours | MIT; referenced scroll data CC BY-NC 4.0 |
| `vetting-card` | ours ([LimeGS/vetting-card](https://github.com/LimeGS/vetting-card)) | MIT; bundled sample is a crop of an official ink map, CC BY-NC 4.0 |

`containers/patches/` holds patches applied to upstream tools at build time. They
are diffs against projects that carry their own licences; the patch is ours, the
code it patches is not. Each patch names the project and commit it applies to.

## Tools this orchestrates but does not contain

VC3D/m7, `vc_render_tifxyz`, `vc_flatten` and the ink models run inside the
container images this repository builds. Their licences are their own — the
Containerfiles under `containers/images/` show exactly what is installed and
from where.
