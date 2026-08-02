# Vetting Card

**Automated anti-pareidolia checks for "I found letters" claims on Vesuvius
papyri ink maps. Drop a map, mark the region, get a calibrated pass/fail
card — entirely in your browser, no upload, no neural network.**

Pareidolia (posting noise as letterforms) is the community's #1 reviewer
time-sink. This tool runs the standard refutation checks by machine and
produces an attachable **vetting card** with the SHA-256 hashes a reviewer
needs to replay the exact claim. The rule it enables:

> **No priority review without a hash-bearing card and its input.**

A green card does **not** mean "confirmed ancient text" — it means the
supplied bytes survived a first automated round of refutation attempts, so
human attention goes to claims that cleared the easy bar.

## Use it in 60 seconds

**Web (no install):** serve this repo and open the page — every check runs
locally in your browser via Pyodide; the image never leaves your machine
(enforced by the page's Content-Security-Policy, not just promised).

```bash
python -m http.server 8000        # from this repo's ROOT, not from web/
# open http://localhost:8000/web/
```

Drop your ink map (PNG/JPG), drag a box over the claimed letters (or type
exact pixel coordinates), set your scan's µm/pixel (presets: 19.2 ds8 maps /
7.91 full-res / 2.4 native), hit *Run the checks*, download the card. A
one-click bundled sample (a human-verified clear-text window from PHerc
0139's official map) shows the full flow. First visit downloads the ~79 MB
vendored runtime once, with a progress bar; after that it's cached. The
page makes **zero external requests** and works on an air-gapped LAN.

**CLI (one command):**

```bash
pip install -r requirements.txt
python vet_map.py --map claim_map.npy --bbox x0,y0,x1,y1 \
    --px-um 19.2 --out verdict.json --card VETTING_CARD.md
```

`--map` takes a `.npy` float array or 8-bit `.png`; `--px-um` is your scan's
real microns-per-pixel. Web and CLI run the *same files* — the page fetches
`card_config.py` / `vet_map.py` / `make_card.py` from this repo at load, so
the two can never drift (verified: bit-identical check values on the bundled
sample). ~1 s per verdict.

## The four checks — all must pass

| Check | Question it refutes | Gate |
|---|---|---|
| `degenerate` | Is the box blank, saturated, or the map constant? | hard fail |
| `letter_energy` | Is there band-pass (DoG) energy at the 1.5–4 mm letter scale, relative to the box's own variance? | `E_FRAC_MIN` |
| `structure` | Do letter-*sized* connected components actually cover the box? | `STRUCTURE_AREAFRAC_MIN` |
| `contrast_bimodality` | Is the box bimodal (ink + background) rather than a smooth blob (the classic oversmoothed-output pareidolia)? | `OTSU_SEP_MIN` |

Plus two guards that return *"cannot evaluate"*, never a misleading content
FAIL, when the input can't honestly be judged at all:

- **Window size**: a box smaller than one letter at your µm/px (the #1
  usage error, usually a wrong `--px-um`) — a too-small crop can't
  masquerade as "no letters here". Use a ~1 cm window.
- **Render family** (v0.4): does the *submitted map* look like the raw
  ink-detection render the four gates above were calibrated on? A
  photo-style composite (ink map painted onto a papyrus-texture
  background for publication readability) fails those gates for reasons
  that have nothing to do with whether the text is real — this catches
  that case explicitly instead of returning a flat, misleading FAIL. Only
  active when the submitted image has enough area around the claim to
  trust the signal (whole maps/plates; usually skipped for the web tool's
  own tight, padded claim crops). See CALIBRATION.md.

`--local-rarity` optionally reports the old top-1%-of-this-map percentile
as non-gating context (expensive at native resolution).

## Calibration, in numbers

Thresholds were **fit on 324 labeled windows from two scrolls** (PHerc 0139 +
Scroll 1: 174 human-verified letterform positives, 150 clean negatives;
frozen 70/30 split, chosen on train only), then **validated without
re-fitting on 96 windows from Scroll 4 / PHerc 1667** — the fully-read
scroll, i.e. ground truth from published papyrology:

- Held-out on the fitted scrolls: **55/56 positives pass, 1/45 negatives**.
- Scroll-4 lacunae (published "no text here"): **0/60 pass**.
- Scroll-4 gold-in-dense-wrap positives: **27/36** — reported, not hidden;
  the label is wrap-level, looser than the fitted scrolls' window-level
  review. Do not pool the two percentages.
- Synthetic noise / blur / fiber: **0/20 pass**.

Full derivation, dataset provenance, honest limits, and replay tooling:
[CALIBRATION.md](CALIBRATION.md). The `config_hash` in every card footer
changes if any threshold moves.

## What the card records

- SHA-256 of the evaluated input (CLI: the map file; web: the exact
  grayscale crop evaluated in-browser, with its coordinates in the source).
- SHA-256 of the evaluator source (`vet_map.py`) and the config hash.
- Every check's value against its absolute threshold, and the verdict.

A hash binds **bytes, not authorship** — the card is self-attested evidence
until a reviewer replays the published evaluator against matching bytes.

## Pipeline check (`vet_pipeline.py`)

Audits output maps a claimant produced on a fixture set of known-blank and
known-text windows (`pareidolia_rate` / `sensitivity`). **Currently fails
closed**: the shipped manifest's three blank fixtures are not publicly
fetchable yet, so the CLI refuses to issue a formal pipeline PASS rather
than pretend third parties could reproduce it. The single-map claim check
above is unaffected. Fixture sources and their exact public/non-public
status: [fixtures/PROVENANCE.md](fixtures/PROVENANCE.md).

## Fixture manifest

`fixtures/manifest.json` lists the fixture windows (`blank` = human-reviewed
no-text, `positive` = known transcribed text), each with full citations and
sha256 pins. `python fetch_fixtures.py` live-downloads and verifies the
public one; see [fixtures/PROVENANCE.md](fixtures/PROVENANCE.md).

## Scope and limitations

- Calibrated on official ds8 renders at 15–19 µm/px, one reviewer;
  spot-checked passing at 2.4 µm/px; **unvalidated on other ink-detection
  render families**.
- A pass is "survived an automated first pass" — never a transcription, a
  confidence interval, or proof a named model produced the map.
- No angle-shuffle / render-jitter controls yet (see Roadmap) — the tool
  sees one static map at one stated scale.
- Checks are distilled from the parent project's internal vetting battery
  (blank-patch, shuffle, same-scale discipline); this is a from-scratch
  standalone reimplementation, not a copy of internal code.

## Roadmap

Specified but not implemented (they need the claimant's rendering/inference
harness, not just an output image): angle-shuffle control, render-jitter
control, broader independently-adjudicated calibration, a fully public blank
fixture set (the current blanks' bytes are already sha256-pinned so a future
upload is verifiable), and an FFT-based `letter_energy` for native-resolution
scans.

## Deploy the web tool on GitHub Pages

1. Push this repo public.
2. Settings → Pages → "Deploy from a branch", branch `main`, folder
   `/ (root)` — root, not `/web`: the page fetches the real `.py` files
   from the repo root. `.nojekyll` is already included.
3. Live in ~1 min at `https://<user>.github.io/vetting-card/web/`.

Everything is vendored in-repo (Pyodide runtime + wheels + fonts): no CDN,
no external requests. The local `http.server` step exists only because
browsers block `fetch()`/WASM over `file://`.

## Tests

```bash
python -m unittest discover -s tests          # fast suite (~30 s)
```

A checkout with the byte-pinned project data tree can additionally run
`VETTING_CARD_REAL_CALIBRATION=1 python -m unittest tests.test_calibration`
(real-raster probes) and `python calibration/validate_tool.py` (exhaustive
release-time sweep).

## License

MIT. See `LICENSE`. The bundled sample window is a crop of an official
Vesuvius Challenge ink map (CC BY-NC 4.0).
