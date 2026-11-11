from pathlib import Path

import pytest

from image_recoloring.artifacts import ArtifactError, read_lfs_pointer, validate_model_artifact


def test_lfs_pointer_is_detected_and_rejected(tmp_path: Path) -> None:
    pointer_path = tmp_path / "model.keras"
    pointer_path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "a" * 64 + "\n"
        "size 12345\n",
        encoding="utf-8",
    )
    pointer = read_lfs_pointer(pointer_path)
    assert pointer is not None
    assert pointer.size == 12345
    with pytest.raises(ArtifactError, match="Git LFS pointer"):
        validate_model_artifact(pointer_path)


def test_missing_model_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="Supply a trained"):
        validate_model_artifact(tmp_path / "missing.keras")
