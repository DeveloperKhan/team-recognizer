"""Image loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_rgb(path: str | Path) -> np.ndarray:
    """An image as an RGB uint8 array (alpha flattened onto black)."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img)
    return np.asarray(img.convert("RGB"))


def find_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
