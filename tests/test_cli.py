import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_inference_rejects_repository_lfs_pointer() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "infer.py"),
            "--model",
            str(ROOT / "Image colorizer.keras"),
            "--input",
            "missing.png",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Git LFS pointer" in result.stderr


def test_prepare_dataset_reports_missing_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "prepare_dataset.py"),
            "--dataset-dir",
            str(tmp_path / "missing"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Dataset directory does not exist" in result.stderr
