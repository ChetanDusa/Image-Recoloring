import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from image_recoloring.model import (  # noqa: E402
    BASELINE_NON_TRAINABLE_PARAMS,
    BASELINE_TOTAL_PARAMS,
    BASELINE_TRAINABLE_PARAMS,
    build_baseline_model,
    compile_baseline_model,
)


def test_baseline_parameter_counts_and_synthetic_forward_pass() -> None:
    model = build_baseline_model((32, 32, 1))
    assert model.count_params() == BASELINE_TOTAL_PARAMS
    assert sum(int(tf.size(weight)) for weight in model.trainable_weights) == BASELINE_TRAINABLE_PARAMS
    assert (
        sum(int(tf.size(weight)) for weight in model.non_trainable_weights)
        == BASELINE_NON_TRAINABLE_PARAMS
    )
    output = model(np.zeros((1, 32, 32, 1), dtype=np.float32), training=False)
    assert tuple(output.shape) == (1, 32, 32, 2)
    assert np.all(np.isfinite(output.numpy()))


def test_compile_uses_regression_metrics_not_accuracy() -> None:
    model = compile_baseline_model(build_baseline_model((32, 32, 1)))
    assert model.loss.name == "loss_mse"
    assert all("accuracy" not in metric.name for metric in model.metrics)
