#!/usr/bin/env python3
"""Create a reproducible train/validation/test manifest."""

from __future__ import annotations

import argparse
import sys

from image_recoloring.manifest import ManifestError, create_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hash, deduplicate, and split original color images."
    )
    parser.add_argument("--dataset-dir", required=True, help="Directory searched recursively")
    parser.add_argument("--output", default="artifacts/split_manifest.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--source", default="unspecified", help="Dataset provenance note")
    parser.add_argument(
        "--license", dest="license_name", default="verify per source image", help="License note"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = create_manifest(
            args.dataset_dir,
            args.output,
            seed=args.seed,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            source=args.source,
            license_name=args.license_name,
        )
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Wrote {args.output}: {manifest['counts']} from "
        f"{manifest['unique_images']} unique images; "
        f"excluded {len(manifest['duplicates_excluded'])} exact duplicate(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
