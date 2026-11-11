"""Small deterministic synthetic data for pipeline smoke tests only."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .preprocessing import save_rgb_image


def create_synthetic_dataset(
    output_dir: str | Path, *, count: int = 8, image_size: int = 32, seed: int = 42
) -> Path:
    if count < 3:
        raise ValueError("Synthetic split needs at least 3 images")
    if image_size < 8 or image_size % 8:
        raise ValueError("Synthetic image_size must be >= 8 and divisible by 8")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    for index in range(count):
        phase = (index + 1) / count
        red = (xx / max(image_size - 1, 1) + phase) % 1.0
        green = (yy / max(image_size - 1, 1) + phase / 2.0) % 1.0
        blue = 0.5 + 0.35 * np.sin((xx + yy) / image_size * np.pi * (index + 1))
        image = np.stack((red, green, blue), axis=-1)
        image += generator.normal(0.0, 0.01, image.shape)
        save_rgb_image(root / f"synthetic_{index:02d}.png", np.clip(image, 0.0, 1.0))
    return root
