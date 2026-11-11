# Reproducible CNN Image Colorization

This project learns to predict color from luminance. Each RGB image is converted to CIE LAB: the CNN receives the `L` channel at `256x256x1` (kept in LAB's `[0, 100]` range) and regresses the `a/b` chrominance channels at `256x256x2` (targets divided by `128`). The output uses `tanh`, then inference multiplies a/b by `128` and converts LAB back to clipped RGB `[0, 1]`.

The refactor preserves the original CNN. It adds deterministic train/validation/test manifests, streaming input, checkpoints, early stopping, evaluation, portable CLIs, tests, and explicit failure for Git LFS pointer files. Content-preserving copies of the original source are under [`original/`](original/).

## Artifact and results status

`Image colorizer.keras` in this folder is **not a model**. It is a 133-byte Git LFS pointer for object `sha256:a2f64ace3e9f4bc105b5ef0e104eabdbbf82b588f33839117437a50aae7f7e86`, whose declared size is 75,214,128 bytes. During final verification, the full public LFS object was recovered from the [original GitHub repository](https://github.com/yaswanth891/Image-Recoloring) without replacing the local pointer. Its downloaded SHA-256 exactly matched the LFS OID, its Keras archive passed integrity checks, and its architecture matched the refactored baseline. Use a recovered copy—or train a new checkpoint—with `--model`; do not pass the pointer file.

The repository still contains no original training dataset, split manifest, or historical experiment log. The two screenshots linked by the old README remain qualitative only: they have no matching input identity, split, or metric record. A new five-image demonstration using provenance-documented scikit-image samples is described below and in [`FINAL_VERIFICATION.md`](FINAL_VERIFICATION.md).

| System | Evaluation data | PSNR | SSIM | RGB MSE | RGB MAE | LAB a/b MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original trained model | Held-out real test set | Not measured—original held-out dataset unavailable | Not measured | Not measured | Not measured | Not measured |
| LAB color-neutral baseline | Held-out real test set | Not measured—dataset unavailable | Not measured | Not measured | Not measured | Not measured |
| Optional architecture experiments | Held-out real test set | Not run—baseline cannot yet be established | Not run | Not run | Not run | Not run |

Synthetic smoke data exists only to verify that the software runs. Its one-image test result is recorded in `PROJECT_AUDIT.md`; it is not evidence of colorization quality.

### Recovered-model demonstration (not a held-out benchmark)

The recovered model was evaluated on five `256x256` scikit-image photographs: astronaut, Chelsea the cat, coffee, Hubble Deep Field, and a Falcon 9 launch. These were chosen after training, and the original training membership is unknown, so the results are a small functionality demonstration—not a properly held-out test-set estimate.

| System | Images | PSNR dB | SSIM | RGB MSE | RGB MAE | LAB a/b MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Recovered CNN | 5 | 22.4287 | 0.8804 | 0.010215 | 0.067167 | 12.2571 |
| LAB color-neutral baseline | 5 | 21.7016 | 0.8920 | 0.014441 | 0.079528 | 14.2090 |

On a MacBook Air M1 CPU, warm-up excluded, 15 timed forward passes averaged **118.36 ms** (population standard deviation 3.65 ms); estimated preprocessing + model + postprocessing averaged **155.72 ms** per image. The CNN improved average PSNR/MSE/MAE and LAB a/b MAE, while the neutral baseline had higher average SSIM. It beat the neutral baseline on PSNR/MSE/MAE for three of five images, illustrating both useful learned color and substantial domain-dependent failure.

## Baseline architecture

All convolutions use `3x3` kernels and same padding. Encoder/decoder feature layers use ReLU; the a/b output uses `tanh`. Nearest-neighbor upsampling matches the original script.

| Stage | Operation | Output shape |
| --- | --- | --- |
| Input | LAB L | `256x256x1` |
| Encoder 1 | Conv 64, Conv 64, BatchNorm | `256x256x64` |
| Down 1 | stride-2 Conv 128, Conv 128, BatchNorm | `128x128x128` |
| Down 2 | stride-2 Conv 256, Conv 256 | `64x64x256` |
| Down 3 | stride-2 Conv 512, BatchNorm | `32x32x512` |
| Bottleneck | Conv 512 | `32x32x512` |
| Decoder 1 | Conv 256, BatchNorm, Conv 128, upsample | `64x64x128` |
| Decoder 2 | Conv 64, upsample | `128x128x64` |
| Output head | Conv 32, Conv 16, Conv 2 (`tanh`), upsample | `256x256x2` |

Verified counts: **14 Conv2D layers**, **6,260,178 total parameters**, **6,258,258 trainable parameters**, and **1,920 non-trainable BatchNorm parameters**. There are three downsampling and three upsampling stages. The final chrominance field is predicted at `128x128` and upsampled, which can contribute to blocky color boundaries.

## Setup

The pinned environment was tested on Python 3.13.4, TensorFlow 2.21.0, NumPy 2.3.4, Pillow 12.2.0, scikit-image 0.26.0, and matplotlib 3.10.8 on Apple Silicon macOS. TensorFlow hardware availability varies by platform.

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Run all tests:

```bash
python -m pytest
```

## Dataset and deterministic splits

Place original color images anywhere below one directory. Class subdirectories are not required:

```text
dataset/
├── source_a/
│   ├── image_001.jpg
│   └── image_002.png
└── source_b/
    └── image_003.webp
```

Create the default 80/10/10 split:

```bash
python prepare_dataset.py \
  --dataset-dir /path/to/dataset \
  --output artifacts/split_manifest.json \
  --seed 42 \
  --source "Pexels and Unsplash; record exact download provenance" \
  --license "verify license for every image"
```

The splitter hashes every original file, removes exact byte-identical duplicates before assignment, and stores relative paths and SHA-256 values. A fixed seed makes assignment repeatable. Training and evaluation verify existing-manifest hashes by default so changed source images fail fast. This detects exact duplicates, not visually near-duplicate crops or re-encodings; those need a perceptual-hash or manual provenance review.

Pexels and Unsplash were named in the original README, but no downloaded files or per-image license records were included. Confirm each source's current license and keep attribution/provenance with the manifest before publishing or training.

## Training

Train the preserved baseline with streaming `tf.data` batches:

```bash
python train.py \
  --dataset-dir /path/to/dataset \
  --manifest artifacts/split_manifest.json \
  --output-dir artifacts/baseline_run \
  --epochs 100 \
  --batch-size 16 \
  --seed 42
```

If `--manifest` is omitted, training creates one in the output directory. Outputs include:

- `best_model.keras`: lowest-validation-loss checkpoint only;
- `config.json`: paths, package/platform versions, seed, split sizes, and parameter counts;
- `training_log.csv` and `history.json`: train/validation metrics, epochs actually completed, and runtime;
- `model_summary.txt`;
- `test_tensor_metrics.json`: one post-selection check of normalized LAB a/b MSE and MAE.

Early stopping monitors `val_loss`, restores the best weights, and defaults to patience 10. MSE remains the training objective because this is continuous chrominance regression. MAE and MSE are reported; pixelwise `accuracy` was removed because exact/categorical correctness is not meaningful for continuous a/b values.

The old source configured batch size 25, a 7% array-level validation split, and 400 requested epochs. That does **not** prove 400 epochs ran. `--legacy-hyperparameters` restores batch size 25 and the 400-epoch request while retaining the corrected data splits and callbacks; the exact unsafe source remains in `original/Model_Architecture.py` for comparison.

Run the CPU-friendly synthetic integration check (one epoch, `32x32`, eight generated images):

```bash
python train.py --smoke-test --output-dir artifacts/smoke
```

## Inference

One image:

```bash
python infer.py \
  --model artifacts/baseline_run/best_model.keras \
  --input /path/to/image.jpg \
  --output-dir artifacts/predictions
```

A directory is processed recursively with the same command. Add `--overwrite` to replace outputs or `--show` for an interactive preview. Grayscale and RGBA files are normalized to RGB; transparent pixels are composited over white. OpenCV BGR arrays must be converted explicitly with `bgr_to_rgb`; file loading itself uses Pillow RGB consistently.

Passing the repository's pointer file produces an actionable error rather than a Keras deserialization failure.

## Evaluation

Evaluate exactly the manifest's held-out test split:

```bash
python evaluate.py \
  --model artifacts/baseline_run/best_model.keras \
  --dataset-dir /path/to/dataset \
  --manifest artifacts/split_manifest.json \
  --output-dir artifacts/evaluation
```

To measure only the trivial LAB color-neutral baseline without model weights:

```bash
python evaluate.py \
  --baseline-only \
  --dataset-dir /path/to/dataset \
  --manifest artifacts/split_manifest.json \
  --output-dir artifacts/neutral_evaluation
```

Evaluation writes per-image CSV, aggregate JSON (count, population mean, and standard deviation), and comparison grids containing original RGB, color-neutral input, predicted RGB, and an absolute-error heatmap.

- **PSNR (dB):** RGB fidelity derived from MSE; higher is better. Uses clipped float RGB and `data_range=1.0`.
- **SSIM:** local structural/luminance similarity; higher is better. Uses `data_range=1.0` and `channel_axis=-1`.
- **RGB MSE:** average squared RGB error; lower is better and penalizes large errors strongly.
- **RGB MAE:** average absolute RGB error; lower is better and is easier to interpret linearly.
- **LAB a/b MAE:** chrominance error in unnormalized LAB units; lower is better.

These reference metrics can penalize plausible alternative colors. Always inspect qualitative grids and report the neutral baseline separately.

## Interview-ready design explanation

The encoder learns spatial/semantic features from luminance while reducing resolution by `8x`; the decoder predicts chrominance and restores spatial size. LAB separates brightness from color, so one input channel maps naturally to two regression targets. Fixed manifests prevent a source image from leaking across train, validation, and test. Validation loss selects checkpoints and controls early stopping; the test split is used only after selection.

The main architectural limitation is loss of spatial detail: no skip connections carry encoder edges to the decoder, and the final two chrominance channels are computed at half resolution before nearest-neighbor upsampling. Plain MSE also tends toward averaged, desaturated colors when several colors are plausible. After a real baseline is measured, two controlled experiments are reasonable: (1) add encoder-to-decoder skip connections while keeping the LAB objective, or (2) compare MSE with a weighted L1/SSIM-style objective. Neither is implemented or claimed superior because the real data and baseline weights are absent.

See [`PROJECT_AUDIT.md`](PROJECT_AUDIT.md) for the complete audit, verified smoke-test evidence, limitations, and resume bullets.
