# Temporary run: raw images, no preprocessing

This is a temporary experiment setup. Instead of training on the Stage 2.2
preprocessed images (`data/processed/`), it trains on the **raw** CEDAR + BHSig260
images (`data/raw/`) exactly as they are, to see whether the preprocessing was
hurting accuracy.

The images are still brought to the model's fixed input size, otherwise they cannot
be batched, but **no** Stage 2.2 step runs: no ink crop, no background whitening,
no binarization. Only a grayscale conversion and a resize to 192x384.

## What changed

| File | Change |
|---|---|
| `training/stage3_metric_learning.py` | `SignatureDataset` gained a `resize_only` mode (`resize_mode` = `aspect` or `stretch`). |
| `training/train_siamese.py` | New flags `--resize-only`, `--resize-mode`, `--negatives`. |
| `training/evaluate.py` | Same three flags; metrics are now tolerant of a missing skilled-forgery set. |
| `tests/test_resize_dataset.py` | Unit tests for the new mode. |

`--resize-mode aspect` (default) preserves the aspect ratio and pads with white;
`--resize-mode stretch` forces the image to 192x384.

`--negatives` chooses the negative score set for validation/test EER and accuracy:
`skilled` (forgeries), `random` (genuine signatures of other signers), or `auto`
(forgeries when present, otherwise random). The CEDAR + BHSig260 raw manifest has
forgeries, so `auto` resolves to `skilled` — the same numbers as before.

## Manifest

No new manifest is needed: `data/manifest.csv` already points at the raw images
(`raw/...` relative to `data/`). It holds 16,632 rows for 314 signers.

## Run locally

```bash
.venv/bin/python -m training.train_siamese \
  --manifest data/manifest.csv --resize-only --negatives auto \
  --output-dir checkpoints/raw --device cuda --workers 2 \
  --phase1-epochs 5 --phase2-epochs 30
```

```bash
.venv/bin/python -m training.evaluate \
  --checkpoint checkpoints/raw/best_model.pt \
  --manifest data/manifest.csv --resize-only --negatives auto --device cuda
```

## Run on Kaggle

1. Upload `sigver_raw.zip` as a Kaggle Dataset (it extracts to
   `manifest.csv` + `raw/...`).
2. Use `kaggle_raw_run.ipynb`. It clones the repo, trains with `--resize-only`,
   saves the log to `training_log.txt`, plots `curves.png` (loss + per-threshold
   accuracy), evaluates on the test signers (`eval_default.json`), sweeps fixed
   thresholds `0.7 / 0.8 / 0.85 / 0.9`, and lists the saved checkpoint. The
   equivalent commands are:

```python
!git clone <your-repo-url> /kaggle/working/sigver
%cd /kaggle/working/sigver
!pip install -q timm opencv-python-headless
DATA = "/kaggle/input/sigver-raw"
!python -m training.train_siamese --manifest $DATA/manifest.csv \
  --resize-only --negatives auto --output-dir /kaggle/working/ckpt_raw \
  --device cuda --workers 2 --phase1-epochs 5 --phase2-epochs 30
!python -m training.evaluate --checkpoint /kaggle/working/ckpt_raw/best_model.pt \
  --manifest $DATA/manifest.csv --resize-only --negatives auto --device cuda
```

Turn on the GPU and Internet (ImageNet weights). Use **Save Version -> Save & Run
All (Commit)** for a long run, then download `best_model.pt` from the Output tab.

## Zip contents

`sigver_raw.zip` (about 3.9 GB) contains:

```
manifest.csv
raw/BHSig260/BHSig100_Bengali/{Genuine,Forged}/*.tif
raw/BHSig260/BHSig160_Hindi/{Genuine,Forged}/*.tif
raw/cedar/full_org/*.png
raw/cedar/full_forg/*.png
```

Keep `manifest.csv` and `raw/` together: the manifest resolves its `raw/...` paths
from its own folder.

## Reverting

This is a temporary experiment and can be reverted with `git checkout` on the three
modified files (the raw-image mode is additive and does not change the default
behaviour, so the existing tests still pass either way).
