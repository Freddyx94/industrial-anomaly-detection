"""Shared test fixtures.

Tests never use the real MVTec images (licence, and CI does not have them). Instead they build a
tiny dataset with the same folder layout from generated 8x8 images.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


@pytest.fixture
def tiny_mvtec(tmp_path: Path) -> Path:
    """One category "widget": 2 train good, 1 test good, 2 test "scratch" with masks."""
    root = tmp_path / "mvtec_ad"
    gray = np.full((8, 8, 3), 128, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:4, 2:4] = 255  # a 2x2 "defect"
    for name in ("000", "001"):
        _save_png(root / "widget/train/good" / f"{name}.png", gray)
    _save_png(root / "widget/test/good/000.png", gray)
    for name in ("000", "001"):
        _save_png(root / "widget/test/scratch" / f"{name}.png", gray)
        _save_png(root / "widget/ground_truth/scratch" / f"{name}_mask.png", mask)
    (root / ".DS_Store").write_text("")  # hidden file that must be ignored
    return root
