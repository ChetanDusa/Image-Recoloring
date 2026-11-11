#!/usr/bin/env python3
"""Train the preserved baseline model reproducibly."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

from image_recoloring.manifest import (
    ManifestError,
    create_manifest,
    load_manifest,
    paths_for_split,
    verify_manifest_hashes,
)
from image_recoloring.smoke import create_synthetic_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the baseline LAB colorization CNN.")
    parser.add_argument("--dataset-dir", help="Original RGB images, searched recursively")
    parser.add_argument("--manifest", help="Existing split manifest; created in output dir if omitted")
    parser.add_argument("--output-dir", default="artifacts/run")
    parser.add_argument("--epochs", type=int, help="Default: 100, or 1 in smoke mode")
    parser.add_argument("--batch-size", type=int, help="Default: 16, or 2 in smoke mode")
    parser.add_argument("--image-size", type=int, help="Default: 256, or 32 in smoke mode")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--dataset-source", default="unspecified")
    parser.add_argument("--dataset-license", default="verify per source image")
    parser.add_argument(
        "--skip-hash-verification",
        action="store_true",
        help="Skip re-hashing files from an existing manifest (faster, less reproducible)",
    )
    parser.add_argument(
        "--legacy-hyperparameters",
        action="store_true",
        help="Use the source script's configured 400 epochs and batch size 25 (does not prove they ran)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Generate eight synthetic images and run one tiny pipeline check; not a benchmark",
    )
    return parser


def _package_versions(names: list[str]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = args.image_size if args.image_size is not None else (32 if args.smoke_test else 256)
    epochs = args.epochs if args.epochs is not None else (1 if args.smoke_test else 100)
    batch_size = args.batch_size if args.batch_size is not None else (2 if args.smoke_test else 16)
    if args.legacy_hyperparameters:
        epochs = args.epochs if args.epochs is not None else 400
        batch_size = args.batch_size if args.batch_size is not None else 25
    if image_size <= 0 or image_size % 8:
        raise ValueError("--image-size must be positive and divisible by 8")
    if epochs <= 0 or batch_size <= 0 or args.patience < 0:
        raise ValueError("epochs and batch size must be positive; patience must be non-negative")

    if args.smoke_test:
        dataset_dir = create_synthetic_dataset(
            output_dir / "synthetic_dataset",
            count=8,
            image_size=image_size,
            seed=args.seed,
        )
        dataset_source = "deterministic synthetic smoke data (not for reported performance)"
        dataset_license = "generated"
    elif args.dataset_dir:
        dataset_dir = Path(args.dataset_dir).expanduser().resolve()
        dataset_source = args.dataset_source
        dataset_license = args.dataset_license
    else:
        raise ValueError("--dataset-dir is required unless --smoke-test is used")

    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else output_dir / "split_manifest.json"
    )
    if args.manifest:
        manifest = load_manifest(manifest_path)
        if not args.skip_hash_verification:
            verify_manifest_hashes(manifest, dataset_dir)
    else:
        manifest = create_manifest(
            dataset_dir,
            manifest_path,
            seed=args.seed,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            source=dataset_source,
            license_name=dataset_license,
        )
    train_paths = paths_for_split(manifest, dataset_dir, "train")
    validation_paths = paths_for_split(manifest, dataset_dir, "validation")
    test_paths = paths_for_split(manifest, dataset_dir, "test")
    if not validation_paths or not test_paths:
        raise ManifestError("Training requires non-empty validation and untouched test splits")

    from image_recoloring.reproducibility import set_global_determinism

    determinism = set_global_determinism(args.seed)
    import tensorflow as tf

    from image_recoloring.data import make_tf_dataset
    from image_recoloring.model import build_baseline_model, compile_baseline_model

    training_data = make_tf_dataset(
        train_paths,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=args.seed,
    )
    validation_data = make_tf_dataset(
        validation_paths,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
        seed=args.seed,
    )
    test_data = make_tf_dataset(
        test_paths,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
        seed=args.seed,
    )
    model = compile_baseline_model(
        build_baseline_model((image_size, image_size, 1)), args.learning_rate
    )
    checkpoint = output_dir / "best_model.keras"
    configuration = {
        "architecture": "preserved 14-Conv2D baseline",
        "batch_size": batch_size,
        "checkpoint": str(checkpoint),
        "dataset_dir": str(dataset_dir),
        "determinism": determinism,
        "epochs_requested": epochs,
        "image_size": image_size,
        "learning_rate": args.learning_rate,
        "legacy_hyperparameters": args.legacy_hyperparameters,
        "manifest": str(manifest_path),
        "model_parameters": {
            "total": model.count_params(),
            "trainable": sum(int(tf.size(weight)) for weight in model.trainable_weights),
            "non_trainable": sum(int(tf.size(weight)) for weight in model.non_trainable_weights),
        },
        "packages": _package_versions(["tensorflow", "numpy", "Pillow", "scikit-image"]),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "seed": args.seed,
        "source_hashes_verified": not bool(args.manifest and args.skip_hash_verification),
        "smoke_test": args.smoke_test,
        "split_counts": manifest["counts"],
    }
    _write_json(output_dir / "config.json", configuration)
    with (output_dir / "model_summary.txt").open("w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(line + "\n"))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint, monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(output_dir / "training_log.csv"),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    started = time.perf_counter()
    history = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=False,
        verbose=2,
    )
    duration_seconds = time.perf_counter() - started
    history_payload = {
        "duration_seconds": duration_seconds,
        "epochs_completed": len(history.epoch),
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
    }
    _write_json(output_dir / "history.json", history_payload)
    test_metrics = {
        key: float(value) for key, value in model.evaluate(test_data, verbose=0, return_dict=True).items()
    }
    _write_json(
        output_dir / "test_tensor_metrics.json",
        {
            "note": "Normalized LAB a/b tensor metrics; use evaluate.py for RGB PSNR/SSIM/MSE/MAE.",
            "count": len(test_paths),
            "metrics": test_metrics,
        },
    )
    return checkpoint


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        checkpoint = run(args)
    except (ManifestError, ValueError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Training complete. Best checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
