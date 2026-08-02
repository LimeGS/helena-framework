#!/usr/bin/env python3
"""Regenerate web/sample_claim.png -- the demo claim bundled with the web tool.

A real, human-verified clear-text window from PHerc 0139's official ink map
(segment 20260115000000-w044_2026011522, window y=1904 x=1632 win=544 at
~18.07 um/px; rated "clear text" in the published review), cropped to
bbox + the 3-sigma bandpass margin. Verified: this crop yields bit-identical
check values to the full map (affine-invariant gates). Source map CC BY-NC
4.0 (Vesuvius Challenge official ink-detection output).

Sample metadata the page uses: bbox 333,333,877,877  px_um 18.07
"""
import glob
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "..")
import card_config

SEG = "20260115000000-w044_2026011522"
X, Y, WIN = 1632, 1904, 544
PX_UM = 9830.0 / WIN

maps_dir = sys.argv[1] if len(sys.argv) > 1 else "../../../data/index_s5_0139/maps/pherc0139"
jpg = glob.glob(f"{maps_dir}/{SEG}/*ds8.jpg")[0]
full = np.array(Image.open(jpg).convert("L"))
_, sig_hi = card_config.bandpass_sigmas(PX_UM)
pad = int(np.ceil(3 * sig_hi))
crop = full[Y - pad:Y + WIN + pad, X - pad:X + WIN + pad]
Image.fromarray(crop).save("sample_claim.png", optimize=True)
print(f"sample_claim.png {crop.shape} | bbox en el crop: {pad},{pad},{pad+WIN},{pad+WIN} | px_um {PX_UM:.2f}")
