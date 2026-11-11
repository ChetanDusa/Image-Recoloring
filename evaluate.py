#!/usr/bin/env python3
"""Evaluate colorization on an untouched manifest test split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from image_recoloring.artifacts import ArtifactError, validate_model_artifact
from image_recoloring.manifest import (
    ManifestError,
    load_manifest,
    paths_for_split,
    sha256_file,
    verify_manifest_hashes,
)
from image_recoloring.metrics import METRIC_NAMES, compute_image_metrics, summarize_metrics
from image_recoloring.preprocessing import ImageReadError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a trained colorizer and LAB-neutral baseline on the held-out test split."
    )
    parser.add_argument("--model", help="Trained .keras model; required unless --baseline-only")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="artifacts/evaluation")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-grids", type=int, default=10)
    parser.add_argument("--skip-hash-verification", action="store_true")
    return parser


def _save_grid(
    destination: Path,
    original: np.ndarray,
    neutral: np.ndarray,
    predicted: np.ndarray,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    error = np.mean(np.abs(predicted - original), axis=-1)
    figure, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    panels = [
        (original, "Original RGB", None),
        (neutral, "Color-neutral input", None),
        (predicted, "Predicted RGB", None),
        (error, "Mean absolute RGB error", "magma"),
    ]
    for axis, (image, label, color_map) in zip(axes, panels):
        axis.imshow(image, cmap=color_map, vmin=0.0, vmax=1.0)
        axis.set_title(label)
        axis.axis("off")
    figure.suptitle(title)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    if not args.baseline_only and not args.model:
        raise ValueError("--model is required unless --baseline-only is supplied")
    if args.baseline_only and args.model:
        raise ValueError("Use either --model or --baseline-only, not both")
    if args.image_size <= 0 or args.image_size % 8:
        raise ValueError("--image-size must be positive and divisible by 8")
    if args.max_grids < 0:
        raise ValueError("--max-grids must be non-negative")

    manifest = load_manifest(args.manifest)
    if not args.skip_hash_verification:
        verify_manifest_hashes(manifest, args.dataset_dir)
    test_paths = paths_for_split(manifest, args.dataset_dir, "test")
    if not test_paths:
        raise ManifestError("Manifest has no held-out test images")

    model = None
    model_path = None
    if not args.baseline_only:
        model_path = validate_model_artifact(args.model)
        import tensorflow as tf

        model = tf.keras.models.load_model(model_path, compile=False)
        expected = model.input_shape
        if len(expected) != 4 or expected[-1] != 1:
            raise ValueError(f"Model input must be [batch, height, width, 1]; got {expected}")
        if expected[1] is not None and (expected[1] != args.image_size or expected[2] != args.image_size):
            raise ValueError(
                f"Model expects {expected[1]}x{expected[2]}, but --image-size is {args.image_size}"
            )

    from image_recoloring.preprocessing import (
        color_neutral_rgb,
        model_output_to_rgb,
        read_rgb_image,
        resize_rgb,
        rgb_to_model_inputs,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_dir).expanduser().resolve()
    rows: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, float]]] = {"color_neutral_baseline": []}
    if model is not None:
        grouped["model"] = []

    for index, path in enumerate(test_paths):
        original = resize_rgb(read_rgb_image(path), (args.image_size, args.image_size))
        lightness, _ = rgb_to_model_inputs(original)
        neutral = color_neutral_rgb(lightness)
        relative = path.relative_to(dataset_root).as_posix()
        neutral_metrics = compute_image_metrics(original, neutral)
        grouped["color_neutral_baseline"].append(neutral_metrics)
        rows.append({"path": relative, "system": "color_neutral_baseline", **neutral_metrics})

        if model is not None:
            predicted_ab = model.predict(lightness[None, ...], verbose=0)[0]
            if predicted_ab.shape != (args.image_size, args.image_size, 2):
                raise ValueError(
                    f"Model output must be {args.image_size}x{args.image_size}x2; "
                    f"got {predicted_ab.shape}"
                )
            predicted = model_output_to_rgb(lightness, predicted_ab)
            model_metrics = compute_image_metrics(original, predicted)
            grouped["model"].append(model_metrics)
            rows.append({"path": relative, "system": "model", **model_metrics})
            if index < args.max_grids:
                safe_name = relative.replace("/", "__")
                _save_grid(
                    output_dir / "comparison_grids" / f"{Path(safe_name).stem}.png",
                    original,
                    neutral,
                    predicted,
                    relative,
                )

    csv_path = output_dir / "per_image_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "system", *METRIC_NAMES])
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "evaluation_split": "test",
        "image_size": args.image_size,
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "source_hashes_verified": not args.skip_hash_verification,
        "metric_definitions": {
            "psnr_db": "RGB [0,1], data_range=1.0; higher is better",
            "ssim": "RGB [0,1], data_range=1.0, channel_axis=-1; higher is better",
            "rgb_mse": "mean squared RGB error on clipped [0,1]; lower is better",
            "rgb_mae": "mean absolute RGB error on clipped [0,1]; lower is better",
            "lab_ab_mae": "mean absolute error over unnormalized LAB a/b channels; lower is better",
        },
        "model": (
            {"path": str(model_path), "sha256": sha256_file(model_path)} if model_path else None
        ),
        "notes": [
            "The LAB color-neutral baseline is reported separately from model scores.",
            "Metrics compare against resized reference images and do not measure semantic plausibility.",
            "Population standard deviation (ddof=0) is reported across images.",
        ],
        "results": {name: summarize_metrics(values) for name, values in grouped.items()},
    }
    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, json_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        csv_path, json_path = run(args)
    except (
        ArtifactError,
        ManifestError,
        ImageReadError,
        ValueError,
        OSError,
        ImportError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Per-image metrics: {csv_path}")
    print(f"Summary: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
