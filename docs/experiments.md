# Experiments log

Every training run is recorded here by hand after it finishes. A run writes into `runs/<name>/`
(the real directory name can get a numeric suffix, for example `name2`, when it already exists):

- `results.csv` - per-epoch losses and metrics,
- `resolved_config.yaml` - the final config actually used (absolute paths),
- `run_summary.json` - library versions, GPU, peak VRAM, seconds per epoch, best-epoch metrics,
- `weights/best.pt`, `weights/last.pt`.

Copy the numbers from `run_summary.json` into the table. `runs/` is not committed.

| run name | model | imgsz | batch | epochs | box mAP50-95 | mask mAP50-95 | min/epoch | peak VRAM | where trained | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo11n_seg_baseline | yolo11n-seg | 640 | 16 | 50 | 0.510 | 0.474 | 0.93 | 2.7 GB | RTX 3050 6 GB laptop, local | Val split; best epoch 50 (still improving), box mAP50 0.683, mask mAP50 0.662, 46.7 min total. |
| yolo11n_seg_negatives | yolo11n-seg | 640 | 16 | 50 | 0.518 | 0.478 | 1.16 | 2.7 GB | RTX 3050 6 GB laptop, local | Baseline + 200 undamaged-car negatives (seed 42, 7% of training). Val split; best epoch 50, box mAP50 0.683, mask mAP50 0.658, 58.0 min total. False detections on 100 held-out intact cars (images with at least one): conf 0.25 72 -> 4, conf 0.5 39 -> 2 (vs. baseline). Held-out images come from the same source as the training negatives. |
| yolo26m_seg_negatives_colab | yolo26m-seg | 640 | 16 | 50 | 0.557 | 0.523 | 3.19 | 12.2 GB | Google Colab, Tesla T4 16 GB (free) | Same data as yolo11n_seg_negatives (200 negatives). Val split; best epoch 50, box mAP50 0.741, mask mAP50 0.728, 159.5 min total. |

## Test split (374 images)

Produced by `python -m cardd.training.evaluate --weights <best.pt> --split test --heldout data/processed/intact_heldout/images`
(`runs/eval/<run>_test/evaluation.json`). Use the test split only for finished models, never for tuning.
False positives are counted on the 100 held-out undamaged-car images (images with at least one detection).

| run name | box mAP50 | box mAP50-95 | mask mAP50 | mask mAP50-95 | FP images @0.25 | FP images @0.5 |
|---|---|---|---|---|---|---|
| yolo11n_seg_baseline | 0.710 | 0.524 | 0.700 | 0.468 | 72 / 100 | 39 / 100 |
| yolo11n_seg_negatives | 0.699 | 0.514 | 0.700 | 0.478 | 4 / 100 | 2 / 100 |
| yolo26m_seg_negatives_colab | 0.747 | 0.553 | 0.733 | 0.513 | 1 / 100 | 1 / 100 |

Per-class mask mAP50-95 (baseline -> negatives): dent 0.326 -> 0.335, scratch 0.269 -> 0.279,
crack 0.163 -> 0.160, glass_shatter 0.898 -> 0.902, lamp_broken 0.682 -> 0.716. `crack` and the
thin `scratch` class are the weakest, with recall around 0.4-0.57.

Per-class mask mAP50-95 of `yolo26m_seg_negatives_colab`: dent 0.375, scratch 0.311, crack 0.215,
glass_shatter 0.910, lamp_broken 0.754. Inference time on the RTX 3050 (ms per image, test split):
yolo11n 6.9, yolo26m 26.6.
