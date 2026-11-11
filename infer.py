#!/usr/bin/env python3
"""Colorize one image or a directory using a trained baseline model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from image_recoloring.artifacts import ArtifactError, validate_model_artifact
from image_recoloring.preprocessing import ImageReadError, find_images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colorize grayscale/RGB images with a trained model.")
    parser.add_argument("--model", required=True, help="Real .keras artifact (not a Git LFS pointer)")
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--output-dir", default="artifacts/predictions")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--show", action="store_true", help="Also open each result interactively")
    return parser


def run(args: argparse.Namespace) -> list[Path]:
    model_path = validate_model_artifact(args.model)
    images = find_images(args.input)
    if not images:
        raise ImageReadError(f"No supported images found below {Path(args.input).resolve()}")
    if args.image_size <= 0 or args.image_size % 8:
        raise ValueError("--image-size must be positive and divisible by 8")

    import tensorflow as tf

    from image_recoloring.preprocessing import (
        model_output_to_rgb,
        read_rgb_image,
        resize_rgb,
        rgb_to_model_inputs,
        save_rgb_image,
    )

    model = tf.keras.models.load_model(model_path, compile=False)
    expected = model.input_shape
    if len(expected) != 4 or expected[-1] != 1:
        raise ValueError(f"Model input must be [batch, height, width, 1]; got {expected}")
    if expected[1] is not None and (expected[1] != args.image_size or expected[2] != args.image_size):
        raise ValueError(
            f"Model expects {expected[1]}x{expected[2]} inputs, but --image-size is {args.image_size}"
        )

    input_root = Path(args.input).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    saved: list[Path] = []
    for image_path in images:
        original = read_rgb_image(image_path)
        lightness, _ = rgb_to_model_inputs(original, (args.image_size, args.image_size))
        predicted_ab = model.predict(lightness[None, ...], verbose=0)[0]
        if predicted_ab.shape != (args.image_size, args.image_size, 2):
            raise ValueError(
                f"Model output must be {args.image_size}x{args.image_size}x2; got {predicted_ab.shape}"
            )
        prediction = model_output_to_rgb(lightness, predicted_ab)
        prediction = resize_rgb(prediction, original.shape[:2])
        relative = image_path.relative_to(input_root) if input_root.is_dir() else Path(image_path.name)
        destination = output_root / relative.parent / f"{relative.stem}_colorized.png"
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace it: {destination}")
        saved.append(save_rgb_image(destination, prediction))
        if args.show:
            import matplotlib.pyplot as plt

            plt.imshow(prediction)
            plt.title(image_path.name)
            plt.axis("off")
            plt.show()
    return saved


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        saved = run(args)
    except (ArtifactError, ImageReadError, ValueError, FileExistsError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for path in saved:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
