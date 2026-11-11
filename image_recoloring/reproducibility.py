"""Random-seed and deterministic-execution helpers."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_global_determinism(seed: int) -> dict[str, Any]:
    """Seed Python, NumPy, and TensorFlow and request deterministic ops."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    deterministic_ops = True
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        deterministic_ops = False
    return {
        "seed": seed,
        "tensorflow_version": tf.__version__,
        "deterministic_ops_requested": deterministic_ops,
    }
