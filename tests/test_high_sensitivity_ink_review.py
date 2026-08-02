import importlib.util
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).parents[1] / "framework/stages/04-validation/scripts/helena_build_high_sensitivity_ink_review.py"
SPEC = importlib.util.spec_from_file_location("high_sensitivity_review", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
bounded_crop = MODULE.bounded_crop
nearest_hotspot = MODULE.nearest_hotspot


def test_bounded_crop_preserves_requested_shape_and_pads() -> None:
    image = Image.new("L", (4, 4), 255)
    result = bounded_crop(image, center_x=0, center_y=0, size=4)
    assert result.size == (4, 4)
    assert result.getpixel((0, 0)) == (0, 0, 0)
    assert result.getpixel((2, 2)) == (255, 255, 255)


def test_nearest_hotspot_uses_analysis_coordinates() -> None:
    chosen = nearest_hotspot(
        [{"rank": 1, "center_y_x": [10, 10]}, {"rank": 2, "center_y_x": [20, 20]}],
        18,
        19,
    )
    assert chosen is not None
    assert chosen["rank"] == 2
