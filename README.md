# Car damage detection & segmentation

[![CI](https://github.com/Alojz-s/car-damage-detection-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/Alojz-s/car-damage-detection-segmentation/actions/workflows/ci.yml)

Engineering thesis: detection and instance segmentation of car body damage on the
[CarDD](https://cardd-ustc.github.io/) dataset with YOLO-seg, plus a Damage Severity Index (DSI).

Classes: dent, scratch, crack, glass shatter, lamp broken (tire flat is excluded).

## Setup (Windows, Python 3.12, NVIDIA GPU)

Run in PowerShell from the repo root:

```powershell
winget install Python.Python.3.12   # once; reopen the terminal afterwards
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev,app]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `True` and your GPU name. The NVIDIA driver must support CUDA 12.8 or newer
(check with `nvidia-smi`). Verified with torch 2.11.0+cu128, Ultralytics 8.4.212 on an
RTX 3050 6 GB Laptop GPU.

Run the checks with `ruff check .`, `ruff format --check .` and `pytest` (the same steps as CI).
Optionally enable the pre-commit hook (ruff lint and format on every commit):

```powershell
pre-commit install
```

## Data

Unpack the original CarDD download and keep only the COCO part (`CarDD_COCO/`) as
`data/raw/CarDD/` (it contains `train2017/`, `val2017/`, `test2017/` and `annotations/`).
The SOD part is not used. The data is not committed, the licence forbids redistribution.
Then convert it to YOLO-seg format:

```bash
python -m cardd.data.convert
```

Output goes to `data/processed/cardd_yolo/` (original file names kept). Options: `--raw`, `--out`,
`--force` (replace a previous output), `--drop-empty` (drop images without labels),
`--multipart merge|split` (default `merge`), `--link copy|hardlink`.

## Training

Prerequisite: the converted dataset in `data/processed/cardd_yolo/` (see Data above). Start the
baseline (YOLO11n-seg, 50 epochs) from the repo root:

```powershell
python -m cardd.training.train --config configs/train_yolo11n_seg.yaml
```

Override config values with `--set key=value` (repeatable), for example `--set epochs=5 --set batch=8`.
To continue an interrupted run (the one named in the config, or pass `--set name=...`):

```powershell
python -m cardd.training.train --config configs/train_yolo11n_seg.yaml --resume
```

Outputs go to `runs/<name>/` (`weights/`, `results.csv`, `resolved_config.yaml`,
`run_summary.json`). Record finished runs in `docs/experiments.md`. To watch progress, follow
`results.csv` or start TensorBoard with `tensorboard --logdir runs`, and monitor the GPU with
`nvidia-smi -l 2`.

### Undamaged-car negatives (fewer false positives)

CarDD contains almost only damaged cars, so a model trained on it alone marks reflections and
intact lamps as damage. `cardd.data.add_negatives` adds photos of undamaged cars (the
`F_Normal`/`R_Normal` classes of [DrBimmer/comprehensive-car-damage](https://huggingface.co/datasets/DrBimmer/comprehensive-car-damage),
MIT, saved in `data/raw/comprehensive_car_damage_normal/`) to the training split with empty
label files, and keeps a separate held-out set of intact cars in `data/processed/intact_heldout/`:

```powershell
python -m cardd.data.add_negatives            # 200 into training, 100 held out, seed 42
python -m cardd.training.train --config configs/train_yolo11n_seg_negatives.yaml
```

Images listed in `configs/negatives_exclude.txt` (slight damage, duplicates) are skipped. Re-run
the script after every `python -m cardd.data.convert --force`. The chosen files are listed in
`data/processed/cardd_yolo/negatives_manifest.json`.

## Evaluation

```powershell
python -m cardd.training.evaluate --weights runs/yolo11n_seg_negatives/weights/best.pt --split test --heldout data/processed/intact_heldout/images
```

Writes `runs/eval/<run>_test/` with `evaluation.json` (box and mask mAP, per-class metrics, speed,
false positives on the undamaged-car images), `evaluation_per_class.csv`, the confusion matrix and
precision-recall curves. It takes about a minute and needs little VRAM. Use `--split val` while
choosing models and the test split only for the final numbers.

## Layout

- `src/cardd/` - package (data, training, dsi, inference, viz)
- `configs/` - dataset, training and DSI configs
- `app/` - Gradio demo (Polish/English)
- `tests/`, `notebooks/`, `docs/`

## Status

Work in progress.
