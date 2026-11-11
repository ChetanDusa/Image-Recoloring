import numpy as np

from image_recoloring.metrics import compute_image_metrics, summarize_metrics


def test_metrics_for_identical_images() -> None:
    reference = np.full((8, 8, 3), 0.4, dtype=np.float32)
    metrics = compute_image_metrics(reference, reference.copy())
    assert np.isinf(metrics["psnr_db"])
    assert metrics["ssim"] == 1.0
    assert metrics["rgb_mse"] == 0.0
    assert metrics["rgb_mae"] == 0.0
    assert metrics["lab_ab_mae"] == 0.0


def test_known_rgb_error_and_summary() -> None:
    reference = np.zeros((8, 8, 3), dtype=np.float32)
    predicted = np.full((8, 8, 3), 0.5, dtype=np.float32)
    metrics = compute_image_metrics(reference, predicted)
    assert np.isclose(metrics["rgb_mse"], 0.25)
    assert np.isclose(metrics["rgb_mae"], 0.5)
    assert np.isclose(metrics["psnr_db"], 10.0 * np.log10(4.0))
    summary = summarize_metrics([metrics, metrics])
    assert summary["count"] == 2
    assert summary["rgb_mse"]["mean"] == 0.25
    assert summary["rgb_mse"]["std"] == 0.0
