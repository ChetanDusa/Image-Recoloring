"""Image colorization metrics with explicit ranges and conventions."""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


METRIC_NAMES = ("psnr_db", "ssim", "rgb_mse", "rgb_mae", "lab_ab_mae")


def compute_image_metrics(reference_rgb: np.ndarray, predicted_rgb: np.ndarray) -> dict[str, float]:
    """Compute full-reference RGB metrics plus LAB chrominance MAE."""
    reference = np.clip(np.asarray(reference_rgb, dtype=np.float32), 0.0, 1.0)
    predicted = np.clip(np.asarray(predicted_rgb, dtype=np.float32), 0.0, 1.0)
    if reference.shape != predicted.shape or reference.ndim != 3 or reference.shape[-1] != 3:
        raise ValueError(
            f"Expected matching HxWx3 arrays; got {reference.shape} and {predicted.shape}"
        )
    difference = predicted - reference
    reference_lab = rgb2lab(reference)
    predicted_lab = rgb2lab(predicted)
    with np.errstate(divide="ignore", invalid="ignore"):
        psnr_db = float(peak_signal_noise_ratio(reference, predicted, data_range=1.0))
    return {
        "psnr_db": psnr_db,
        "ssim": float(
            structural_similarity(reference, predicted, data_range=1.0, channel_axis=-1)
        ),
        "rgb_mse": float(np.mean(np.square(difference), dtype=np.float64)),
        "rgb_mae": float(np.mean(np.abs(difference), dtype=np.float64)),
        "lab_ab_mae": float(
            np.mean(np.abs(predicted_lab[..., 1:] - reference_lab[..., 1:]), dtype=np.float64)
        ),
    }


def summarize_metrics(rows: list[dict[str, float]]) -> dict[str, object]:
    """Return population mean/std and disclose any non-finite values."""
    if not rows:
        raise ValueError("Cannot summarize zero metric rows")
    summary: dict[str, object] = {"count": len(rows), "standard_deviation": "population (ddof=0)"}
    for metric in METRIC_NAMES:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[metric] = {
            "mean": float(np.mean(finite)) if finite.size else None,
            "std": float(np.std(finite, ddof=0)) if finite.size else None,
            "non_finite_count": int(values.size - finite.size),
        }
    return summary
