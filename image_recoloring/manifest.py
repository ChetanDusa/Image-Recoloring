"""Deterministic, leakage-resistant dataset manifests."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .preprocessing import find_images


class ManifestError(ValueError):
    """Raised for invalid datasets or split manifests."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_counts(total: int, validation_ratio: float, test_ratio: float) -> dict[str, int]:
    if total < 3:
        raise ManifestError("At least 3 unique images are required for train/validation/test splits")
    validation = max(1, int(round(total * validation_ratio))) if validation_ratio else 0
    test = max(1, int(round(total * test_ratio))) if test_ratio else 0
    while validation + test >= total:
        if validation >= test and validation > 1:
            validation -= 1
        elif test > 1:
            test -= 1
        else:
            raise ManifestError("Ratios leave no training images")
    return {"train": total - validation - test, "validation": validation, "test": test}


def create_manifest(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    seed: int = 42,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    source: str = "unspecified",
    license_name: str = "verify per source image",
) -> dict[str, Any]:
    """Hash, deduplicate, split, and save a dataset manifest."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise ManifestError(f"Dataset directory does not exist: {root}")
    if seed < 0:
        raise ManifestError("seed must be non-negative")
    if validation_ratio < 0 or test_ratio < 0 or validation_ratio + test_ratio >= 1:
        raise ManifestError("validation_ratio and test_ratio must be >= 0 and sum to less than 1")

    images = find_images(root)
    if not images:
        raise ManifestError(f"No supported image files found below {root}")

    unique_by_hash: dict[str, Path] = {}
    duplicate_records: list[dict[str, str]] = []
    sizes: dict[Path, int] = {}
    for image in images:
        digest = sha256_file(image)
        canonical = unique_by_hash.get(digest)
        if canonical is None:
            unique_by_hash[digest] = image
            sizes[image] = image.stat().st_size
        else:
            duplicate_records.append(
                {
                    "duplicate_path": image.relative_to(root).as_posix(),
                    "canonical_path": canonical.relative_to(root).as_posix(),
                    "sha256": digest,
                }
            )

    unique = [(path, digest) for digest, path in unique_by_hash.items()]
    unique.sort(key=lambda item: item[0].relative_to(root).as_posix().lower())
    counts = _split_counts(len(unique), validation_ratio, test_ratio)
    shuffled = unique.copy()
    random.Random(seed).shuffle(shuffled)

    assignments: dict[Path, str] = {}
    test_end = counts["test"]
    validation_end = test_end + counts["validation"]
    for index, (path, _) in enumerate(shuffled):
        if index < test_end:
            split = "test"
        elif index < validation_end:
            split = "validation"
        else:
            split = "train"
        assignments[path] = split

    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": digest,
            "bytes": sizes[path],
            "split": assignments[path],
        }
        for path, digest in unique
    ]
    manifest: dict[str, Any] = {
        "format_version": 1,
        "dataset": {
            "source": source,
            "license": license_name,
            "root_at_creation": str(root),
        },
        "seed": seed,
        "ratios": {
            "train": 1.0 - validation_ratio - test_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "counts": counts,
        "files_discovered": len(images),
        "unique_images": len(entries),
        "duplicates_excluded": duplicate_records,
        "images": entries,
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ManifestError(f"Manifest does not exist: {resolved}")
    try:
        manifest = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Could not read manifest {resolved}: {exc}") from exc
    if manifest.get("format_version") != 1 or not isinstance(manifest.get("images"), list):
        raise ManifestError(f"Unsupported or malformed manifest: {resolved}")
    seen_hash_splits: dict[str, str] = {}
    for entry in manifest["images"]:
        if not {"path", "sha256", "split"} <= entry.keys():
            raise ManifestError("Each manifest image needs path, sha256, and split")
        if entry["split"] not in {"train", "validation", "test"}:
            raise ManifestError(f"Unknown split: {entry['split']}")
        prior = seen_hash_splits.get(entry["sha256"])
        if prior is not None and prior != entry["split"]:
            raise ManifestError("Identical image content appears in multiple splits")
        seen_hash_splits[entry["sha256"]] = entry["split"]
    return manifest


def paths_for_split(
    manifest: dict[str, Any], dataset_root: str | Path, split: str
) -> list[Path]:
    if split not in {"train", "validation", "test"}:
        raise ManifestError(f"Unknown split: {split}")
    root = Path(dataset_root).expanduser().resolve()
    paths = [root / entry["path"] for entry in manifest["images"] if entry["split"] == split]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        sample = ", ".join(str(path) for path in missing[:3])
        raise ManifestError(f"{len(missing)} manifest image(s) are missing; first: {sample}")
    return paths


def verify_manifest_hashes(manifest: dict[str, Any], dataset_root: str | Path) -> None:
    """Fail if an image changed after the manifest was created."""
    root = Path(dataset_root).expanduser().resolve()
    for entry in manifest["images"]:
        path = root / entry["path"]
        if not path.is_file():
            raise ManifestError(f"Manifest image is missing: {path}")
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise ManifestError(
                f"Manifest hash mismatch for {path}; recreate the manifest before training/evaluation"
            )
