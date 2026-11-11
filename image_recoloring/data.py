"""Streaming TensorFlow datasets backed by manifest image paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .preprocessing import load_lab_pair


def _decode_path(value) -> str:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def make_tf_dataset(
    paths: list[Path],
    *,
    image_size: tuple[int, int],
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    """Build a deterministic tf.data pipeline without loading the dataset into RAM."""
    if not paths:
        raise ValueError("Cannot build a dataset from an empty path list")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    import tensorflow as tf

    height, width = image_size
    string_paths = [str(path) for path in paths]
    dataset = tf.data.Dataset.from_tensor_slices(string_paths)
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(string_paths), seed=seed, reshuffle_each_iteration=True
        )

    def load_with_numpy(path_tensor):
        lightness, chroma = load_lab_pair(_decode_path(path_tensor), image_size)
        return lightness.astype(np.float32), chroma.astype(np.float32)

    def load_tensor(path_tensor):
        lightness, chroma = tf.numpy_function(
            load_with_numpy, [path_tensor], [tf.float32, tf.float32]
        )
        lightness.set_shape((height, width, 1))
        chroma.set_shape((height, width, 2))
        return lightness, chroma

    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    dataset = dataset.map(load_tensor, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    return dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
