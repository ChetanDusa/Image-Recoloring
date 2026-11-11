"""Validation helpers for model artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


LFS_HEADER = "version https://git-lfs.github.com/spec/v1"


class ArtifactError(ValueError):
    """Raised when a requested model artifact cannot be loaded."""


@dataclass(frozen=True)
class LfsPointer:
    oid: str
    size: int


def read_lfs_pointer(path: Path) -> LfsPointer | None:
    """Return pointer metadata when *path* is a Git LFS pointer."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
    if not text.startswith(LFS_HEADER):
        return None
    oid_match = re.search(r"^oid sha256:([0-9a-f]{64})$", text, re.MULTILINE)
    size_match = re.search(r"^size (\d+)$", text, re.MULTILINE)
    if not oid_match or not size_match:
        raise ArtifactError(f"Malformed Git LFS pointer: {path}")
    return LfsPointer(oid=oid_match.group(1), size=int(size_match.group(1)))


def validate_model_artifact(path: str | Path) -> Path:
    """Resolve a model path and reject missing files and LFS pointers."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ArtifactError(
            f"Model does not exist: {resolved}. Supply a trained .keras model with --model."
        )
    if not resolved.is_file():
        raise ArtifactError(f"Model path is not a file: {resolved}")
    pointer = read_lfs_pointer(resolved)
    if pointer:
        raise ArtifactError(
            f"{resolved} is a Git LFS pointer, not trained weights "
            f"(oid sha256:{pointer.oid}, expected size {pointer.size:,} bytes). "
            "Retrieve the LFS object from the original Git remote before inference."
        )
    if resolved.stat().st_size == 0:
        raise ArtifactError(f"Model file is empty: {resolved}")
    return resolved
