"""The preserved baseline convolutional encoder-decoder."""

from __future__ import annotations


BASELINE_TOTAL_PARAMS = 6_260_178
BASELINE_TRAINABLE_PARAMS = 6_258_258
BASELINE_NON_TRAINABLE_PARAMS = 1_920


def build_baseline_model(input_shape: tuple[int, int, int] = (256, 256, 1)):
    """Build the original 14-Conv2D architecture with named layers."""
    if len(input_shape) != 3 or input_shape[-1] != 1:
        raise ValueError("input_shape must be (height, width, 1)")
    if input_shape[0] % 8 or input_shape[1] % 8:
        raise ValueError("input height and width must be divisible by 8")

    import tensorflow as tf

    layers = tf.keras.layers
    model = tf.keras.Sequential(
        [
            layers.Input(shape=input_shape, name="l_input"),
            layers.Conv2D(64, 3, activation="relu", padding="same", name="enc_conv_1"),
            layers.Conv2D(64, 3, activation="relu", padding="same", name="enc_conv_2"),
            layers.BatchNormalization(name="enc_bn_1"),
            layers.Conv2D(128, 3, strides=2, activation="relu", padding="same", name="enc_down_1"),
            layers.Conv2D(128, 3, activation="relu", padding="same", name="enc_conv_3"),
            layers.BatchNormalization(name="enc_bn_2"),
            layers.Conv2D(256, 3, strides=2, activation="relu", padding="same", name="enc_down_2"),
            layers.Conv2D(256, 3, activation="relu", padding="same", name="enc_conv_4"),
            layers.Conv2D(512, 3, strides=2, activation="relu", padding="same", name="enc_down_3"),
            layers.BatchNormalization(name="bottleneck_bn"),
            layers.Conv2D(512, 3, activation="relu", padding="same", name="bottleneck_conv"),
            layers.Conv2D(256, 3, activation="relu", padding="same", name="dec_conv_1"),
            layers.BatchNormalization(name="dec_bn_1"),
            layers.Conv2D(128, 3, activation="relu", padding="same", name="dec_conv_2"),
            layers.UpSampling2D(2, interpolation="nearest", name="dec_up_1"),
            layers.Conv2D(64, 3, activation="relu", padding="same", name="dec_conv_3"),
            layers.UpSampling2D(2, interpolation="nearest", name="dec_up_2"),
            layers.Conv2D(32, 3, activation="relu", padding="same", name="dec_conv_4"),
            layers.Conv2D(16, 3, activation="relu", padding="same", name="dec_conv_5"),
            layers.Conv2D(2, 3, activation="tanh", padding="same", name="ab_output_half_resolution"),
            layers.UpSampling2D(2, interpolation="nearest", name="ab_output"),
        ],
        name="baseline_lab_colorizer",
    )
    return model


def compile_baseline_model(model, learning_rate: float = 1e-3):
    """Compile for a/b regression; pixelwise classification accuracy is omitted."""
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.MeanSquaredError(name="loss_mse"),
        metrics=[
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )
    return model
