# Final Recovered-Model Verification

Verification date: 2026-11-11

Hardware: MacBook Air M1

Environment: macOS 26.5.2, Python 3.13.4, TensorFlow 2.21.0, scikit-image 0.26.0

## Outcome

The original trained CNN was successfully recovered from `https://github.com/yaswanth891/Image-Recoloring` at commit `5d6c04906b041c62549ca2296ae2db806ae571ce`. The refactored repository's 133-byte pointer was not replaced. A durable recovered copy was saved outside the project as:

```text
Image-Recoloring-original-a2f64ace.keras
```

## Model integrity and compatibility

| Check | Verified result |
| --- | --- |
| File size | 75,214,128 bytes |
| Git LFS OID | `a2f64ace3e9f4bc105b5ef0e104eabdbbf82b588f33839117437a50aae7f7e86` |
| Downloaded SHA-256 | Exact match to the LFS OID |
| Archive integrity | Keras ZIP tested successfully; `metadata.json`, `config.json`, and `model.weights.h5` valid |
| Original save metadata | Keras 3.4.1; saved 2024-07-25 21:31:45 |
| Input/output | `(None,256,256,1)` to `(None,256,256,2)` |
| Layers | 14 Conv2D, 4 BatchNorm, 3 UpSampling2D |
| Parameters | 6,260,178 total; 6,258,258 trainable; 1,920 non-trainable |
| Weight validity | All finite |
| Refactored architecture signature | Exact match, ignoring generated layer names |
| Forward pass | Valid finite `1x256x256x2` output |

No compatibility bug or architectural change was required.

## Demonstration images and provenance

Five real photographs bundled with scikit-image 0.26.0 were used to minimize downloads and provide explicit provenance:

| Image | Provenance/license |
| --- | --- |
| Astronaut Eileen Collins | NASA Great Images; public domain/no known restrictions |
| Chelsea the cat | Stefan van der Walt; CC0 |
| Coffee cup | Rachel Michetti/Pikolo Espresso Bar; CC0 |
| Hubble eXtreme Deep Field | NASA/HubbleSite; public domain |
| Falcon 9 DSCOVR launch | SpaceX Photos; public domain |

Official provenance: `https://scikit-image.org/docs/stable/api/skimage.data.html`.

This is a five-image convenience demonstration, not a held-out benchmark. The original training dataset and membership are unavailable, selection occurred after training, and overlap cannot be ruled out.

## Aggregate results

Metrics compare resized `256x256` RGB references against predictions. PSNR/SSIM use clipped RGB `[0,1]` with `data_range=1.0`; SSIM uses `channel_axis=-1`. Standard deviations in the JSON report are population values across images.

| System | Count | PSNR dB | SSIM | RGB MSE | RGB MAE | LAB a/b MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Recovered CNN | 5 | 22.428665 | 0.880444 | 0.010215 | 0.067167 | 12.257101 |
| LAB color-neutral baseline | 5 | 21.701644 | 0.892036 | 0.014441 | 0.079528 | 14.209016 |

The CNN improved average PSNR, MSE, RGB MAE, and LAB a/b MAE, but the neutral baseline produced higher average SSIM. The CNN beat the baseline on PSNR/MSE/MAE for astronaut, Chelsea, and coffee; the baseline won those metrics for Hubble and rocket. Visual inspection shows muted colors and broad color bleeding, consistent with the no-skip, half-resolution chrominance design.

## Latency

TensorFlow detected CPU only. Model loading and one warm-up call were excluded. Each of five inputs received three timed forward passes:

- 15 forward calls: **118.362 ms mean**, 3.654 ms standard deviation;
- minimum: 113.816 ms; maximum: 127.566 ms;
- estimated preprocessing + one mean model call + postprocessing: **155.718 ms/image mean**, 11.111 ms standard deviation.

These are local MacBook Air M1 measurements, not cross-platform benchmarks.

## Output files

All generated material is under:

```text
artifacts/final_verification_2026-11-11/
```

- `comparisons/`: grayscale input, original RGB, CNN prediction, and error heatmap for all five images;
- `inference_native_size/`: outputs produced by the refactored `infer.py` CLI;
- `inputs/original/` and `inputs/grayscale_256/`;
- `predictions_256/`;
- `per_image_metrics.csv`;
- `aggregate_results.json`;
- `provenance.json`.

The `artifacts/` directory is intentionally Git-ignored. Commit this report and source/documentation changes; publish selected generated grids separately only if desired.

## Quality and limitations

- The recovered model is authentic, intact, loadable, and architecture-compatible.
- The refactored pipeline produces finite, correctly shaped outputs.
- These five images do not establish generalization or replace a held-out test set.
- The original training data, split, epoch history, and training-time validation metrics remain unavailable.
- No claim about 400 completed epochs is supported.
- Git metadata is absent from the refactored local directory, so `git status` and the exact commit diff cannot be checked here; the source tree itself passes its documented tests.

## Resume-bullet check

The three bullets in `PROJECT_AUDIT.md` remain factually accurate. They cite verified architecture statistics, implemented pipeline features, 14 automated tests, and the reproducible synthetic smoke check. None depends on an unverified held-out performance claim, so no resume change is required.
