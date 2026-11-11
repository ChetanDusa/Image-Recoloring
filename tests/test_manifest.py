import json
from pathlib import Path

import numpy as np
from PIL import Image

import pytest

from image_recoloring.manifest import (
    ManifestError,
    create_manifest,
    load_manifest,
    verify_manifest_hashes,
)


def _write_image(path: Path, value: int) -> None:
    image = np.full((8, 8, 3), value, dtype=np.uint8)
    Image.fromarray(image).save(path)


def test_manifest_is_deterministic_deduplicated_and_leakage_free(tmp_path: Path) -> None:
    dataset = tmp_path / "images"
    dataset.mkdir()
    for index in range(10):
        _write_image(dataset / f"image_{index}.png", index * 20)
    (dataset / "duplicate.png").write_bytes((dataset / "image_0.png").read_bytes())

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first = create_manifest(dataset, first_path, seed=123)
    second = create_manifest(dataset, second_path, seed=123)
    assert first == second
    assert first_path.read_text() == second_path.read_text()
    assert first["files_discovered"] == 11
    assert first["unique_images"] == 10
    assert len(first["duplicates_excluded"]) == 1
    assert first["counts"] == {"train": 8, "validation": 1, "test": 1}

    loaded = load_manifest(first_path)
    hashes_by_split = {
        split: {entry["sha256"] for entry in loaded["images"] if entry["split"] == split}
        for split in ("train", "validation", "test")
    }
    assert hashes_by_split["train"].isdisjoint(hashes_by_split["validation"])
    assert hashes_by_split["train"].isdisjoint(hashes_by_split["test"])
    assert hashes_by_split["validation"].isdisjoint(hashes_by_split["test"])
    json.loads(first_path.read_text())
    verify_manifest_hashes(loaded, dataset)
    _write_image(dataset / "image_1.png", 255)
    with pytest.raises(ManifestError, match="hash mismatch"):
        verify_manifest_hashes(loaded, dataset)
