# Car damage detection & segmentation

Engineering thesis: detection and instance segmentation of car body damage on the
[CarDD](https://cardd-ustc.github.io/) dataset with YOLO-seg, plus a Damage Severity Index (DSI).

Classes: dent, scratch, crack, glass shatter, lamp broken (tire flat is excluded).

## Setup (Windows, Python 3.12, NVIDIA GPU)

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev,app]"
python -c "import torch; print(torch.cuda.is_available())"
```

## Data

Place the original CarDD download in `data/raw/CarDD/` (not committed, the licence forbids
redistribution), then convert it to YOLO-seg format:

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
