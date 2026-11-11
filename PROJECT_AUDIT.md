# Image Recoloring Technical Audit and Implementation Report

Audit date: 2026-11-11

Repository audited: extracted project root

> Post-audit update: the final verification phase later recovered the full public LFS model after the original GitHub URL was supplied. The Phase 0 statements below describe the repository state and information available at audit time. See `FINAL_VERIFICATION.md` for checksum, compatibility, real-image demonstration metrics, and latency.

## Executive result

The original proof of concept has been converted into a portable, deterministic, test-covered TensorFlow pipeline without changing its core CNN. The repository now supports manifest-based train/validation/test splits, streaming training, best-checkpoint selection, early stopping, single/directory inference, held-out evaluation, per-image and aggregate metrics, and saved qualitative grids.

No held-out model-performance claim can be verified. Final verification recovered the original weights, but the repository still has no training dataset, original split, or run logs, and its tracked `Image colorizer.keras` remains an LFS pointer. The later five-image real-photo demonstration and the earlier synthetic smoke run verify operation, not generalization.

## Phase 0: original repository audit

The extracted repository contained only:

- `.gitattributes`;
- `Image colorizer.keras` (133 bytes);
- `README.md`;
- `codes/Model_Architecture.py`;
- `codes/image_recolor.py`.

There was no dataset, model directory, package manifest, test, experiment log, split record, local example image, Git metadata, or license file. Content-preserving baseline copies now live in `original/`.

### Model artifact verification

The nominal model file contains:

```text
version https://git-lfs.github.com/spec/v1
oid sha256:a2f64ace3e9f4bc105b5ef0e104eabdbbf82b588f33839117437a50aae7f7e86
size 75214128
```

This proves the checked-in 133-byte text file is not a Keras archive. The intended object declares a 75,214,128-byte size. At initial audit time, Git LFS retrieval from this folder could not be attempted because:

1. `git status` reports that the directory is not a Git repository;
2. no remote URL is available;
3. `git lfs version` reports that `git-lfs` is not installed.

Inference and model evaluation were therefore initially blocked. The later final verification recovered the same object through GitHub's public raw LFS endpoint without replacing the pointer. In an original clone with access to its remote, `git lfs pull --include="Image colorizer.keras"` remains the standard retrieval route. The new CLIs inspect pointer files and fail with the OID and expected size instead of passing them to Keras.

### Verified architecture

The original network is a sequential encoder-decoder with 14 `Conv2D` layers, four BatchNorm layers, three stride-2 convolutions, and three nearest-neighbor `2x` upsamplings. It maps `256x256x1` LAB L to `256x256x2` normalized a/b.

| Section | Spatial progression | Channels |
| --- | --- | --- |
| Encoder | `256 -> 128 -> 64 -> 32` | `1 -> 64 -> 128 -> 256 -> 512` |
| Bottleneck/decoder | `32 -> 64 -> 128` | `512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 2` |
| Output | `128 -> 256` via final upsample | `2` |

Parameter counts were independently computed from convolution kernels/biases and BatchNorm state, then cross-checked with TensorFlow `model.count_params()`:

- convolution parameters: 6,256,338;
- BatchNorm parameters: 3,840 total (1,920 trainable scale/offset and 1,920 non-trainable moving state);
- **total: 6,260,178**;
- **trainable: 6,258,258**;
- **non-trainable: 1,920**.

The original training script configured Adam, MSE, pixelwise `accuracy`, batch size 25, `validation_split=0.07`, and `epochs=400`. These are configured values only. There is no history proving training began, completed 400 epochs, or produced the pointer's referenced artifact.

### Original data flow

Training used `ImageDataGenerator(rescale=1/255).flow_from_directory(..., target_size=(256,256), class_mode=None)`. Each RGB batch was converted with `skimage.color.rgb2lab`; L was left in `[0,100]`, and a/b was divided by 128. Every image was appended to Python lists and converted to full in-memory NumPy arrays before `model.fit`.

Inference loaded an image as RGB float values near `[0,255]`, resized to `256x256`, divided by 255, converted to LAB, and passed only L to the model. Predicted a/b was multiplied by 128, combined with L, converted with `lab2rgb`, and resized to the original spatial size. `cv.imread` read BGR but its pixels were not used—only its shape—so it was confusing and unnecessary rather than a direct BGR reconstruction bug.

The refactor deliberately retains the compatibility-critical numeric contract: unscaled LAB L, a/b divided by 128, `tanh` output, and multiplication by 128 during reconstruction.

## Problems identified

### Correctness and reproducibility

- Both scripts contained user-specific Windows absolute paths.
- `flow_from_directory` expects class subdirectories and could report zero images for a flat folder, even though colorization has no class labels.
- The generator shuffled without a recorded seed, then all data was materialized in RAM.
- `validation_split=0.07` operated on the accumulated array; there was no stable original-file manifest, duplicate check, or held-out test set.
- No random seed, deterministic TensorFlow option, configuration record, package version record, or source-image hash was stored.
- Pixelwise `accuracy` is inappropriate for continuous a/b regression and can be misleading.
- The checkpoint callback did not monitor validation performance or set `save_best_only=True`; no early stopping existed.
- The configured checkpoint parent directory was calculated but never created.
- Training caught broad per-image exceptions, printed them, and continued without a structured failed-file record.
- TensorFlow's bundled Keras and standalone `keras` imports were mixed.

### Inference and evaluation

- Inference ran at import time, had no CLI, did not validate input/model paths, and loaded the model with compilation/custom loss unnecessarily.
- The output array was allocated and filled twice.
- Output was only displayed interactively; it was not saved.
- There was no directory mode or explicit grayscale/RGBA behavior.
- No PSNR, SSIM, RGB MSE, RGB MAE, LAB chrominance MAE, neutral baseline, per-image result, aggregate statistic, or comparison grid existed.
- The README's claimed model/code locations did not match the archive.

### Evidence and maintenance

- The `.keras` path was only a pointer.
- No dependency file or tested environment was provided.
- No automated tests existed.
- Two remote screenshots were presented without a model hash, input identity, split, or reproducible command, so they are not quantitative evidence.
- Dataset provenance named Pexels and Unsplash generally but did not list images, download dates, attribution, or per-image licensing.

## Improvements implemented

### Data and preprocessing

- Centralized RGB loading, EXIF orientation, grayscale/RGBA normalization, alpha compositing, resize, LAB conversion, and LAB reconstruction in `image_recoloring/preprocessing.py`.
- Made RGB/BGR conversion explicit and test-covered; OpenCV is no longer a runtime dependency.
- Preserved the baseline L and a/b scaling contract.
- Added recursive file discovery for common image formats.
- Added fixed-seed, SHA-256 manifest generation with default 80/10/10 train/validation/test splits.
- Exact byte-identical files are deduplicated before splitting, and existing manifests are hash-verified by default.
- Added deterministic streaming `tf.data` batches, avoiding a full image dataset in memory.

### Training

- Rebuilt the same layer sequence as a named, reusable constructor.
- Consolidated on `tensorflow.keras`.
- Replaced regression `accuracy` with MSE and MAE.
- Added `ModelCheckpoint(monitor="val_loss", save_best_only=True)`.
- Added `EarlyStopping(monitor="val_loss", restore_best_weights=True)` and `TerminateOnNaN`.
- Added Python, NumPy, and TensorFlow seeds plus deterministic TensorFlow ops.
- Added configuration, model summary, CSV log, JSON history/runtime, and final normalized-a/b test records.
- Added `--legacy-hyperparameters` for the source's batch-size/epoch configuration without reintroducing unsafe splitting.
- Added an eight-image synthetic smoke mode that defaults to one epoch at `32x32`.

### Inference and evaluation

- Added portable file/directory inference with `pathlib` and `argparse`.
- Added actionable missing-input, wrong-shape, existing-output, and LFS-pointer failures.
- Added held-out evaluation for RGB PSNR, SSIM, MSE, MAE, and unnormalized LAB a/b MAE.
- SSIM explicitly uses `data_range=1.0` and `channel_axis=-1`; RGB is clipped to `[0,1]`.
- Added separate LAB color-neutral baseline rows.
- Added per-image CSV, aggregate JSON with count/mean/population standard deviation, model SHA-256, and source-hash verification status.
- Added saved original/neutral/prediction/error comparison grids.

### Dependency and documentation work

- Added exact runtime and development requirements.
- Added a tested cross-platform setup and command reference in `README.md`.
- Preserved the old files in `original/`; the old `codes/` paths are compatibility wrappers.
- Added a 14-test suite covering transforms, formats, resizing, round-trip behavior, RGB/BGR, manifests, duplicate/leakage protection, artifact validation, CLI failures, metrics, baseline evaluation, parameters, and a synthetic forward pass.

## Changed-file summary

| File(s) | Reason |
| --- | --- |
| `original/Model_Architecture.py`, `original/image_recolor.py`, `original/README.md` | Content-preserving review baseline from the extracted project |
| `codes/Model_Architecture.py`, `codes/image_recolor.py` | Backward-compatible wrappers for the new CLIs |
| `image_recoloring/artifacts.py` | Detect missing/empty models and Git LFS pointers |
| `image_recoloring/preprocessing.py` | Shared RGB/LAB contract, format handling, resize, save, RGB/BGR helpers |
| `image_recoloring/model.py` | Preserved, reusable 14-Conv2D model and regression compilation |
| `image_recoloring/manifest.py` | Hashing, exact deduplication, deterministic splits, integrity verification |
| `image_recoloring/data.py` | Streaming deterministic `tf.data` input |
| `image_recoloring/metrics.py` | PSNR, SSIM, RGB MSE/MAE, LAB a/b MAE, aggregate statistics |
| `image_recoloring/reproducibility.py` | Python/NumPy/TensorFlow seeds and deterministic ops |
| `image_recoloring/smoke.py` | Clearly labeled deterministic synthetic integration data |
| `prepare_dataset.py` | Dataset-manifest CLI |
| `train.py` | Reproducible training, callbacks, logs, and post-selection test tensor metrics |
| `infer.py` | Portable file/directory colorization CLI |
| `evaluate.py` | Held-out model/neutral evaluation and comparison grids |
| `requirements*.txt`, `pyproject.toml`, `.gitignore` | Environment, test configuration, generated-file exclusions |
| `tests/` | Fast automated verification |
| `README.md` | Architecture, setup, commands, results status, metrics, limitations, interview explanation |

## Verification executed

Environment used: macOS 26.5.2 Apple Silicon, Python 3.13.4, TensorFlow 2.21.0, NumPy 2.3.4, Pillow 12.2.0, scikit-image 0.26.0, matplotlib 3.10.8, pytest 9.2.0.

| Command/check | Result |
| --- | --- |
| `python -m compileall ...` | Passed; all new Python files compiled |
| `python -m pytest -q` | **14 passed**, no failures |
| Baseline model construction and synthetic forward pass | Output `1x32x32x2`; all values finite |
| TensorFlow parameter cross-check | `6,260,178` total; `6,258,258` trainable; `1,920` non-trainable |
| `python train.py --smoke-test ...` | Completed 1 epoch; best `.keras` checkpoint, logs, config, and test record saved |
| Identical smoke training repeated with seed 42 | Histories exactly equal; maximum absolute weight difference **0.0** |
| `evaluate.py` on smoke test split | CSV, JSON, model hash, and one comparison grid saved |
| `infer.py` on a smoke image | Saved a colorized PNG successfully |
| `infer.py` with repository pointer | Exited 2 with an actionable Git LFS pointer error |

The first smoke run recorded 3.816 seconds inside `model.fit` on this machine. This is not a `256x256` throughput benchmark.

## Verified dataset and metric statistics

### Real project data

- Real training/validation/test images in repository: **0**.
- Actual trained model files in repository at audit start: **0**.
- Git LFS pointer files: **1**, 133 bytes, declaring a 75,214,128-byte object.
- Historical held-out metrics: **none**.

### Synthetic pipeline check

The smoke generator produced eight deterministic `32x32` images and split them into 6 train, 1 validation, and 1 test image. One epoch was requested and completed. The following values are real measurements but are intentionally **not** project-quality or resume performance results: the data is synthetic, the test sample count is one, the input is `32x32`, and training lasted one epoch.

| System | Count | PSNR dB | SSIM | RGB MSE | RGB MAE | LAB a/b MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| One-epoch smoke model | 1 | 11.096899 | 0.291867 | 0.077680 | 0.208744 | 37.687157 |
| LAB color-neutral baseline | 1 | 11.746753 | 0.384465 | 0.066884 | 0.197312 | 36.553583 |

The neutral baseline outperforming the one-epoch smoke model is expected and reinforces that smoke results must not be presented as learned performance.

## Remaining limitations and blockers

1. **The tracked model path remains a pointer.** Final verification recovered and validated a durable copy outside this repository, but fresh checkouts still need the public LFS object; the 133-byte pointer cannot be loaded.
2. **The real dataset is missing.** No defensible dataset size, source distribution, license status, split counts, failure modes, or real metric can be reported.
3. **No expensive real training was run.** The audit brief explicitly prohibits it without the data/hardware and approval. The original 400-epoch setting is not a verified completed run.
4. **Exact deduplication is not near-duplicate detection.** Re-encoded copies, crops, or burst frames may still leak without perceptual grouping.
5. **Determinism is platform-dependent.** The same environment produced byte-equivalent weights across two local smoke runs, but different TensorFlow/platform kernels can vary.
6. **MSE encourages average colors.** Colorization is multimodal, so valid alternative colors can score poorly and predictions can desaturate.
7. **Spatial detail is constrained.** There are no skip connections, and the two-channel result is generated at half resolution before nearest upsampling.
8. **No semantic/perceptual metric or user study is included.** PSNR/SSIM/MAE alone do not capture plausibility.
9. **Historical screenshots remain unverified.** They should be replaced with grids generated by `evaluate.py` after recovering or training a model.

## Optional experiments (deferred, not claimed)

Architectural changes were intentionally not implemented because there is no real baseline to compare against. After recovering data and establishing a frozen baseline:

1. Add lightweight encoder-to-decoder skip connections to restore boundaries. This increases activation memory and may copy luminance edges without resolving semantic color ambiguity.
2. Compare the current MSE loss with a controlled L1 plus structural/perceptual term. This can preserve sharper color transitions but requires careful loss weighting and a complete retrain.

Use the same manifest, select only on validation data, and evaluate the untouched test split once for each finalized experiment.

## Concise interview explanation

The system separates lightness from chrominance in LAB. A 14-layer CNN compresses the L channel through three stride-2 stages, learns a 512-channel representation, and decodes two normalized a/b channels through three upsampling stages. MSE is a simple stable regression loss, while pixelwise accuracy is invalid because a/b values are continuous rather than discrete classes. A SHA-256 manifest fixes train/validation/test membership and blocks exact duplicates from crossing splits. Best-validation-loss checkpointing and early stopping control overfitting, and the test split is scored only after model selection. The largest technical risks are multimodal color ambiguity, MSE desaturation, missing skip connections, half-resolution chrominance prediction, dataset-domain bias, and the absent historical artifact/data.

## ATS-friendly resume bullets (verified only)

- Refactored a **14-layer, 6,260,178-parameter** TensorFlow LAB colorization CNN into a reproducible training/inference pipeline with **3-stage** encoding and decoding, deterministic seeds, streaming `tf.data`, best-model checkpointing, and early stopping.
- Implemented SHA-256 dataset manifests with exact duplicate exclusion and fixed **80/10/10** train/validation/test splits, plus **5** held-out image metrics (PSNR, SSIM, RGB MSE, RGB MAE, and LAB a/b MAE) and per-image CSV/JSON reporting.
- Built a **14-test** automated suite covering preprocessing, leakage controls, model shapes/parameters, CLI failures, metrics, and end-to-end smoke execution; achieved **14/14 passing tests** and **0.0 maximum weight difference** across two same-seed smoke-training runs.
