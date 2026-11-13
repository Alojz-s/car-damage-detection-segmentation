# Car damage detection & segmentation

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

Run the checks with `ruff check .` and `pytest`.

## Data

Unpack the original CarDD download so that it lives in `data/raw/CarDD_release/`
(`CarDD_COCO/` is used, `CarDD_SOD/` is ignored). It is not committed, the licence forbids
redistribution. Then convert it to YOLO-seg format:

```bash
python -m cardd.data.convert
```

## Layout

- `src/cardd/` - package (data, training, dsi, inference, viz)
- `configs/` - dataset, training and DSI configs
- `app/` - Gradio demo (Polish/English)
- `tests/`, `notebooks/`, `docs/`

## Status

Work in progress.
