# Design decisions

- `tire flat` class dropped from CarDD (5 classes remain).
- Code and repo in English; thesis text and demo UI in Polish.
- Python 3.12 (dependency compatibility).
- Images without label lines are kept by default as background images with an empty label file
  (`keep_empty=True`). This covers tire-only images (counts on full CarDD: train 172 / val 42 /
  test 25, reported as `images_emptied`) and images whose annotations were all skipped
  (`images_without_labels`). `--drop-empty` removes all of them.
- Multi-polygon CarDD instances are merged into one label line by default (`--multipart merge`,
  via Ultralytics `merge_multi_segment`), so the number of val/test ground-truth objects equals the
  number of CarDD instances. `--multipart split` writes one line per part; its metrics are not
  directly comparable with the CarDD paper because one instance becomes several GT objects.
- Converted files keep their original CarDD file names; `<split>_images.csv` maps them to image ids.
- Negatives: the baseline model (trained on CarDD only) flagged reflections on glass and paint and
  intact lamps as damage (`glass_shatter` 0.90-0.95 on an intact windscreen). Undamaged-car photos
  are added to training as background images (empty label files): the `F_Normal`/`R_Normal`
  classes of the Hugging Face dataset `DrBimmer/comprehensive-car-damage` (MIT licence; the image
  source is not stated, 800 images, version `355b0990`). Slightly damaged images found in a manual
  review and duplicates are listed in `configs/negatives_exclude.txt`. 100 other images are held
  out for measuring the false positive rate and are never used for training. Compare
  `yolo11n_seg_baseline` and `yolo11n_seg_negatives` (same settings, same seed).
