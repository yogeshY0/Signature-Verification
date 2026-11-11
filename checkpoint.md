# Checkpoint — working log

Running log of the temporary "raw image" experiments in this repo. Update this file
whenever something changes, so a later session can pick up without re-deriving the state.

Date of last update: 2026-11-11

## Conventions / commands

| Thing | Value |
|---|---|
| Repo root | `/Users/aacs/Desktop/Signature-Verification` |
| Python | `.venv/bin/python` |
| Tests | `.venv/bin/python -m unittest discover -s tests -t .` — **48 tests, all pass** |
| Raw data | `data/raw/` (CEDAR 55 signers; BHSig260 100 Bengali + 160 Hindi) |
| Raw manifest | `data/manifest.csv` — 16,632 rows, 314 signers, 9,096 forgeries |
| Old preprocessed data | `data/processed/` — 192x384 **black-and-white** PNGs + its own manifest |

Note: `python -m ...` scripts must run from the repo root (cwd must contain
`preprocessing/`, `training/`, `tests/`).

## Story so far (chronological)

1. The `dataset/` folder (a separate download) was **deleted** by the user, so "raw
   dataset" now means `data/raw/`. No new manifest was needed — `data/manifest.csv`
   already describes the raw images (`raw/...` relative to `data/`).
2. Added **`--resize-only`**: feed raw images straight to the network
   (grayscale + resize to 192x384 + ImageNet normalize; **no** crop, whitening or
   binarization).
3. Made validation/evaluation tolerant of a missing skilled-forgery set and added
   `--negatives {auto,skilled,random}`.
4. Zipped the raw dataset + manifest → **`sigver_raw.zip` (3.9 GB)** for Kaggle upload
   (`manifest.csv` at the top level, then `raw/...`).
5. Wrote `kaggle_raw_run.ipynb`, `TEMPORARY_RAW_RUN.md`, `tests/test_resize_dataset.py`.
6. Explained what "no crop" means, with a side-by-side montage →
   `raw_vs_preprocessed_sample.png` (RAW | Stage 2.2 grayscale | data/processed binary |
   --resize-only). Crop = `crop_to_ink()`, which cuts the frame to the ink bounding box
   plus a 5% margin, so the signature fills the frame instead of shrinking with the page.
7. Extended the Kaggle notebook with `2>&1 | tee {LOG}`, `analyze_log --plot` +
   `IPython.display`, default evaluation (`eval_default.json`) and a fixed-threshold
   sweep over 0.7 / 0.8 / 0.85 / 0.9.
8. Added **`--crop-grayscale`**: grayscale + the ink-crop step + resize, but **no**
   background whitening and **no** binarization (so the input is not black-and-white).
9. Wrote `kaggle_raw_crop_run.ipynb` (uses `--crop-grayscale`) and this `checkpoint.md`.

## Input modes (SignatureDataset — at most one; mutually exclusive)

| mode | CLI flag | pipeline |
|---|---|---|
| default | *(none)* | grayscale + illumination whitening + border-artifact removal + `crop_to_ink` + resize (`binarize=False`) |
| already preprocessed | `--already-preprocessed` | file is already the 192x384 grayscale Stage 2.2 output |
| resize only | `--resize-only` | grayscale + resize to 192x384 (**no** crop / whiten / binarize) |
| crop only | `--crop-grayscale` | grayscale + `crop_to_ink` + resize (**no** whiten / binarize) |

`--resize-mode aspect` (default: preserve aspect ratio, pad with white) or `stretch`.

`--negatives auto` (default) uses skilled forgeries when the split has any, otherwise
random (different-signer) pairs. This raw manifest has forgeries, so `auto` → `skilled`.

## Files

**Modified**
- `training/stage3_metric_learning.py` — `resize_only` + `crop_only` modes in `SignatureDataset`
- `training/train_siamese.py` — `--resize-only`, `--resize-mode`, `--crop-grayscale`, `--negatives`
- `training/evaluate.py` — same flags + metrics tolerant of an empty skilled set

**Added**
- `tests/test_resize_dataset.py` — tests for the resize-only and crop-only modes
- `TEMPORARY_RAW_RUN.md` — runbook for the raw (no-preprocessing) experiment
- `kaggle_raw_run.ipynb` — Kaggle run: `--resize-only` + logs/curves/eval/sweep
- `kaggle_raw_crop_run.ipynb` — Kaggle run: `--crop-grayscale` + logs/curves/eval/sweep
- `sigver_raw.zip` — 3.9 GB raw dataset + manifest, for Kaggle
- `raw_vs_preprocessed_sample.png` — illustration of what the crop does (throwaway)
- `checkpoint.md` — this file

## Quick commands (local)

```bash
# A) raw, no crop, grayscale
.venv/bin/python -m training.train_siamese --manifest data/manifest.csv \
  --resize-only --negatives auto --output-dir checkpoints/raw \
  --device cuda --workers 2 --phase1-epochs 5 --phase2-epochs 30

# B) ink crop + grayscale (no black-and-white)
.venv/bin/python -m training.train_siamese --manifest data/manifest.csv \
  --crop-grayscale --negatives auto --output-dir checkpoints/cropgray \
  --device cuda --workers 2 --phase1-epochs 5 --phase2-epochs 30

# evaluate (must match the training input mode)
.venv/bin/python -m training.evaluate --checkpoint checkpoints/raw/best_model.pt \
  --manifest data/manifest.csv --resize-only --negatives auto --device cuda
```

Kaggle: upload `sigver_raw.zip` as a dataset, then run `kaggle_raw_run.ipynb` (A) or
`kaggle_raw_crop_run.ipynb` (B).

## Gotchas

- Raw `.tif` scans make OpenCV print a `TIFF Software tag` warning per file → set
  `OPENCV_LOG_LEVEL=ERROR` (the notebooks do this).
- `analyze_log --plot` needs `matplotlib` (not in `requirements.txt`).
- `crop_to_ink` can raise `ValueError` on a blank/faint scan; `crop_only` falls back to
  the full frame instead of crashing (0 failures found in a 600-image sample, and
  `data/processed` kept all 16,632 rows).
- `data/processed/` is **binarized** because `preprocess_dataset` runs
  `preprocess_signature(binarize=not grayscale)`. Training on raw with the *default*
  path uses `binarize=False`, so it is grayscale (plus whitening) — different from
  `data/processed`.
- Revert everything with `git checkout` on the three modified files and delete the added
  ones; the new modes are additive and leave the default behaviour unchanged.
