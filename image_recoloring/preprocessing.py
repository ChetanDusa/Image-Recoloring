"""Shared RGB/LAB preprocessing for training, inference, and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps
from skimage.color import lab2rgb, rgb2lab
from skimage.transform import resize


AB_SCALE = 128.0
L_RANGE = (0.0, 100.0)
SUPPORTED_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


class ImageReadError(ValueError):
    """Raised when an input image cannot be decoded or normalized."""


def _float_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in (1, 2, 3, 4):
        raise ImageReadError(f"Expected HxW grayscale/RGB/RGBA image; got shape {array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float32) / float(np.iinfo(array.dtype).max)
    else:
        array = array.astype(np.float32)
        if not np.all(np.isfinite(array)):
            raise ImageReadError("Image contains NaN or infinite values")
        if array.size and (array.min() < 0.0 or array.max() > 1.0):
            raise ImageReadError("Floating-point RGB input must be scaled to [0, 1]")
    if array.shape[-1] in (2, 4):
        if array.shape[-1] == 2:
            gray, alpha = array[..., :1], array[..., 1:2]
            color = np.repeat(gray, 3, axis=-1)
        else:
            color, alpha = array[..., :3], array[..., 3:4]
        array = color * alpha + (1.0 - alpha)
    return np.clip(array[..., :3], 0.0, 1.0).astype(np.float32)


def read_rgb_image(path: str | Path) -> np.ndarray:
    """Read an image as float RGB in [0, 1], honoring EXIF orientation."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ImageReadError(f"Input image does not exist: {resolved}")
    try:
        with Image.open(resolved) as opened:
            image = ImageOps.exif_transpose(opened)
            if image.mode in {"RGBA", "LA"}:
                array = np.asarray(image)
            else:
                array = np.asarray(image.convert("RGB"))
    except Exception as exc:
        raise ImageReadError(f"Could not decode image {resolved}: {exc}") from exc
    return _float_rgb(array)


def resize_rgb(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Resize RGB using bilinear interpolation and anti-aliasing."""
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image_size dimensions must be positive")
    rgb = _float_rgb(image)
    if rgb.shape[:2] == (height, width):
        return rgb.copy()
    return resize(
        rgb,
        (height, width, 3),
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)


def rgb_to_model_inputs(
    image: np.ndarray, image_size: tuple[int, int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Convert RGB to unscaled L and normalized a/b model tensors."""
    rgb = _float_rgb(image)
    if image_size is not None:
        rgb = resize_rgb(rgb, image_size)
    lab = rgb2lab(rgb).astype(np.float32)
    lightness = lab[..., :1]
    chroma = lab[..., 1:] / AB_SCALE
    return lightness, chroma.astype(np.float32)


def load_lab_pair(
    path: str | Path, image_size: tuple[int, int] = (256, 256)
) -> tuple[np.ndarray, np.ndarray]:
    """Load one image and return (L, a/b divided by 128)."""
    return rgb_to_model_inputs(read_rgb_image(path), image_size)


def model_output_to_rgb(lightness: np.ndarray, normalized_ab: np.ndarray) -> np.ndarray:
    """Combine L with normalized predicted a/b and return clipped RGB."""
    l_channel = np.asarray(lightness, dtype=np.float32)
    ab_channels = np.asarray(normalized_ab, dtype=np.float32)
    if l_channel.ndim == 2:
        l_channel = l_channel[..., None]
    if l_channel.ndim != 3 or l_channel.shape[-1] != 1:
        raise ValueError(f"Expected L shape HxWx1; got {l_channel.shape}")
    if ab_channels.ndim != 3 or ab_channels.shape[-1] != 2:
        raise ValueError(f"Expected a/b shape HxWx2; got {ab_channels.shape}")
    if l_channel.shape[:2] != ab_channels.shape[:2]:
        raise ValueError("L and a/b spatial shapes must match")
    lab = np.concatenate((l_channel, ab_channels * AB_SCALE), axis=-1)
    return np.clip(lab2rgb(lab), 0.0, 1.0).astype(np.float32)


def color_neutral_rgb(lightness: np.ndarray) -> np.ndarray:
    """Create the fair LAB color-neutral (a=b=0) baseline."""
    l_channel = np.asarray(lightness)
    if l_channel.ndim == 2:
        l_channel = l_channel[..., None]
    zeros = np.zeros((*l_channel.shape[:2], 2), dtype=np.float32)
    return model_output_to_rgb(l_channel, zeros)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Explicit OpenCV BGR-to-RGB conversion without changing the value range."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 BGR image; got {array.shape}")
    return array[..., ::-1].copy()


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Explicit RGB-to-OpenCV-BGR conversion without changing the value range."""
    return bgr_to_rgb(image)


def save_rgb_image(path: str | Path, image: np.ndarray) -> Path:
    """Save float RGB [0, 1] as an 8-bit image."""
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    rgb = _float_rgb(image)
    encoded = np.round(rgb * 255.0).astype(np.uint8)
    Image.fromarray(encoded, mode="RGB").save(resolved)
    return resolved


def find_images(root: str | Path) -> list[Path]:
    """Return supported images below a file or directory in stable order."""
    resolved = Path(root).expanduser().resolve()
    if resolved.is_file():
        if resolved.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ImageReadError(f"Unsupported image extension: {resolved.suffix}")
        return [resolved]
    if not resolved.is_dir():
        raise ImageReadError(f"Input path does not exist: {resolved}")
    images: Iterable[Path] = (
        path for path in resolved.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return sorted(images, key=lambda item: item.as_posix().lower())
