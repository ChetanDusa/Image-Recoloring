import json
from pathlib import Path

from evaluate import main
from image_recoloring.manifest import create_manifest
from image_recoloring.smoke import create_synthetic_dataset


def test_baseline_only_evaluation_writes_csv_and_json(tmp_path: Path) -> None:
    dataset = create_synthetic_dataset(tmp_path / "dataset", count=6, image_size=16, seed=5)
    manifest = tmp_path / "manifest.json"
    create_manifest(dataset, manifest, seed=5)
    output = tmp_path / "evaluation"
    result = main(
        [
            "--baseline-only",
            "--dataset-dir",
            str(dataset),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--image-size",
            "16",
            "--max-grids",
            "0",
        ]
    )
    assert result == 0
    assert (output / "per_image_metrics.csv").is_file()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["results"]["color_neutral_baseline"]["count"] == 1
    assert summary["model"] is None
