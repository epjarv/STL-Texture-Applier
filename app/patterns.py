"""Tileable height maps used as built-in surface finishes."""

from __future__ import annotations

import numpy as np
from PIL import Image

PATTERN_SIZE = 512


def _axes(n: int = PATTERN_SIZE) -> tuple[np.ndarray, np.ndarray]:
    t = (np.arange(n, dtype=np.float32) + 0.5) / n
    y, x = np.meshgrid(t, t, indexing="ij")
    return x, y


def wood(n: int = PATTERN_SIZE) -> np.ndarray:
    x, y = _axes(n)
    warp = 0.07 * np.sin(2 * np.pi * y) + 0.035 * np.sin(2 * np.pi * (2.0 * y) + 1.3)
    grain = np.sin(2 * np.pi * (3.2 * x + 1.8 * warp))
    soft = np.sin(2 * np.pi * (y + 0.15 * x))
    grooves = 0.5 + 0.5 * grain
    height = 0.34 + 0.52 * np.power(grooves, 1.35) + 0.06 * soft
    return np.clip(height, 0, 1).astype(np.float32)


def knurl(n: int = PATTERN_SIZE) -> np.ndarray:
    x, y = _axes(n)
    freq = 6.0
    ridge_a = np.abs(np.sin(np.pi * (x + y) * freq))
    ridge_b = np.abs(np.sin(np.pi * (x - y) * freq))
    diamonds = np.minimum(ridge_a, ridge_b)
    height = np.clip((diamonds - 0.12) / 0.72, 0, 1)
    return height.astype(np.float32)


def brushed(n: int = PATTERN_SIZE) -> np.ndarray:
    x, y = _axes(n)
    drift = 0.25 * np.sin(2 * np.pi * 2.0 * y)
    lines = np.sin(2 * np.pi * (7.0 * x + drift))
    broken = np.sin(2 * np.pi * (2.0 * y + 0.25 * x))
    height = 0.5 + 0.2 * lines + 0.05 * broken
    return np.clip(height, 0, 1).astype(np.float32)


def dimpled(n: int = PATTERN_SIZE) -> np.ndarray:
    x, y = _axes(n)
    freq = 5.0
    cell_x = np.mod(x * freq, 1.0) - 0.5
    cell_y = np.mod(y * freq, 1.0) - 0.5
    dist = np.sqrt(cell_x * cell_x + cell_y * cell_y)
    cup = np.clip(dist / 0.46, 0, 1)
    height = 0.15 + 0.85 * np.power(cup, 0.85)
    return height.astype(np.float32)


def pebble(n: int = PATTERN_SIZE) -> np.ndarray:
    x, y = _axes(n)
    height = np.zeros_like(x)
    layers = (
        (3.0, 4.0, 0.24, 0.4),
        (5.0, 7.0, 0.16, 1.2),
        (8.0, 6.0, 0.10, 2.1),
    )
    for fx, fy, amp, phase in layers:
        height += amp * np.sin(2 * np.pi * (fx * x + phase)) * np.sin(
            2 * np.pi * (fy * y + phase * 1.37)
        )
    return np.clip(0.55 + height, 0, 1).astype(np.float32)


BUILT_IN = (
    ("Wood", "Wood grain", wood),
    ("Knurl", "Diamond knurl", knurl),
    ("Brushed", "Brushed metal", brushed),
    ("Dimpled", "Round dimples", dimpled),
    ("Pebble", "Pebbled grain", pebble),
)


def build_patterns() -> dict[str, np.ndarray]:
    return {name: fn() for name, _tip, fn in BUILT_IN}


def load_image(path: str, max_size: int = 1024) -> np.ndarray:
    """Brightness of an image, used as a height map. 0 is dark, 1 is light."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        arr = np.asarray(rgb).astype(np.float32) / 255.0
    linear = np.power(np.clip(arr, 0.0, 1.0), 2.2)
    height = 0.2126 * linear[:, :, 0] + 0.7152 * linear[:, :, 1] + 0.0722 * linear[:, :, 2]
    return height.astype(np.float32)


def swatch(height: np.ndarray, size: int = 72) -> np.ndarray:
    """Warm grayscale preview of a height map, uint8 RGB."""
    image = Image.fromarray(np.clip(height * 255.0, 0, 255).astype(np.uint8), mode="L")
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    tone = np.asarray(image).astype(np.float32) / 255.0
    dark = np.array([38, 36, 33], dtype=np.float32)
    light = np.array([236, 230, 220], dtype=np.float32)
    rgb = dark * (1.0 - tone[..., None]) + light * tone[..., None]
    return np.ascontiguousarray(rgb.astype(np.uint8))
