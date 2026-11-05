# Usage

Step-by-step commands, from raw data to a trained and evaluated model. Run everything from
the repository root with the project's virtual environment (`.venv/bin/python`, or
`python` if the environment is activated). Scripts run as modules (`python -m ...`), so
the working directory must contain `preprocessing/`, `training/` and `tests/`.

## 0. Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install matplotlib        # only needed for the training-curve plot
```

Check the code:

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

## 1. Put the raw data in place

```
data/raw/cedar/full_org/        genuine signatures
data/raw/cedar/full_forg/       forgeries
data/raw/BHSig260/BHSig100_Bengali/{Genuine,Forged}/
data/raw/BHSig260/BHSig160_Hindi/{Genuine,Forged}/
data/raw/SOHI/P001/{Genuine,Forged}/ ... P123   (Bengali, 16 + 16 per signer)
```

`data/` and `checkpoints/` are Git-ignored.

## 2. Build the manifests

Each dataset has its own signer-disjoint split in `data/manifests/<name>.csv`, and all of
them are merged into `data/manifest.csv`. The CEDAR manifest (`data/manifests/cedar.csv`)
already exists. This command writes the BHSig260 and SOHI ones and merges everything:

```bash
.venv/bin/python -m preprocessing.build_manifests
```

Manifest columns: `image_path,signer_id,is_forgery,split`. A signer belongs to exactly
one split. Forgeries carry the ID of the signer they imitate.

### Augmenting CEDAR (optional)

Only CEDAR is augmented, with three mild, geometry-preserving operations: stroke-thickness
jitter (dilation/erosion), small spatial shifts, and a very light elastic deformation.
Rotation, shear, and scaling are deliberately **not** used, because tilting or stretching a
signature changes its spatial geometry and stroke orientation and introduces unrealistic
artifacts. Only **train**-split signers are augmented; validation and test signers stay
pristine. Augmented raw images sit next to their originals as
`augmented_<type>_<variant>_<original>.png`.

```bash
# Writes the augmented raw images and refreshes data/manifests/cedar.csv + data/manifest.csv.
.venv/bin/python -m preprocessing.augment_cedar

# Preprocess CEDAR (originals + augmented) and refresh data/processed/manifest.csv.
# --prune drops stale CEDAR PNGs; BHSig260 and SOHI are preserved as-is.
.venv/bin/python -m preprocessing.preprocess_dataset \
  --manifest data/manifest.csv --output-dir data/processed \
  --datasets cedar --prune
```

## 3. Preprocess once

Writes 192x384 black-and-white PNGs to `data/processed/` and a matching
`data/processed/manifest.csv`:

```bash
.venv/bin/python -m preprocessing.preprocess_dataset \
  --manifest data/manifest.csv --output-dir data/processed
```

Add `--grayscale` to keep grayscale strokes instead. Training and later comparison must
use the same mode. Skipped images (no ink found) are listed at the end of the run.

Add `--datasets <name>[,<name>...]` to (re)process only those dataset folders; every other
dataset already present in `data/processed/manifest.csv` is copied over unchanged, so no
other dataset is reprocessed. Combine it with `--prune` to delete stale PNGs left inside the
processed dataset folders.

## 4. Smoke test (optional, about 2 minutes)

```bash
.venv/bin/python -m training.train_siamese \
  --manifest data/processed/manifest.csv --already-preprocessed \
  --output-dir checkpoints/smoke --device cpu --no-pretrained \
  --phase1-epochs 1 --phase2-epochs 1 \
  --signers-per-batch 4 --genuine-per-signer 2
```

## 5. Train

Training starts from ImageNet weights, freezes the backbone for the warm-up phase, then
unfreezes the last two stages. Augmentation is on by default for the training split. Use
`--device cuda` on a GPU machine.

```bash
.venv/bin/python -m training.train_siamese \
  --manifest data/processed/manifest.csv --already-preprocessed \
  --output-dir checkpoints/v2 --device cuda --workers 2 \
  --phase1-epochs 5 --phase2-epochs 30
```

| Option | Default | Meaning |
|---|---|---|
| `--phase1-epochs` / `--phase2-epochs` | 5 / 20 | Warm-up and fine-tune epochs |
| `--signers-per-batch` | 12 | Signers per batch; lower it if the GPU runs out of memory |
| `--genuine-per-signer` | 4 | Genuine signatures per signer in a batch |
| `--forgeries-per-signer` | 2 | Forgeries per signer in a batch |
| `--margin` | 0.3 | Triplet-loss margin |
| `--augment` | off | Turn augmentation on (it raised skilled EER from about 12% to about 16% in our runs) |
| `--select-by` | eer | Pick the saved checkpoint by validation skilled EER (`eer`) or validation loss (`loss`) |
| `--lr-schedule` | cosine | Cosine decay within each phase to 1% of its rate; `constant` for the old behaviour |
| `--no-pretrained` | off | Random weights instead of ImageNet (smoke tests only) |

The best checkpoint (by validation EER by default) is saved as `<output-dir>/best_model.pt`.
One JSON line per epoch is printed, including `lr`, `validation_eer_skilled` and
`validation_accuracy_skilled` (accuracy at each `--accuracy-thresholds` value, default
`0.7 0.8 0.9`). Computing
the EER adds a few seconds per epoch on a GPU. Because the checkpoint and the evaluation
threshold are both chosen on the validation signers, the validation EER is slightly optimistic;
trust the test numbers.

### On Kaggle

```python
!git clone <your-repo-url> /kaggle/working/sigver
%cd /kaggle/working/sigver
!pip install -q timm opencv-python-headless
DATA = "/kaggle/input/<your-dataset>"      # must contain manifest.csv at its top level
!python -m training.train_siamese --manifest $DATA/manifest.csv --already-preprocessed \
  --output-dir /kaggle/working/ckpt_v2 --device cuda --workers 2 \
  --phase1-epochs 5 --phase2-epochs 40
```

Turn on the GPU and Internet (for the ImageNet weights). For a long run, use
**Save Version -> Save & Run All (Commit)** so the run survives closing the tab, then
download `best_model.pt` from the version's Output tab.

## 6. Check the training curves

Save the printed epoch lines (or the Kaggle log export) to `training_log.txt`:

```bash
.venv/bin/python -m training.analyze_log training_log.txt --plot training_curves.png
```

It prints loss per epoch and says whether validation loss is still falling (train longer),
or training loss falls while validation rises (overfitting). If the log holds several runs,
it analyzes the last one. The plot has two panels: train/validation loss, and validation
accuracy per epoch, one line per threshold (needs a log from a run that printed
`validation_accuracy_skilled`).

## 7. Evaluate on the held-out test signers

```bash
.venv/bin/python -m training.evaluate \
  --checkpoint checkpoints/v2/best_model.pt \
  --manifest data/processed/manifest.csv --already-preprocessed
```

The threshold is chosen on the validation signers and applied to the test signers. The
output reports FAR, FRR and EER against skilled forgeries and against random forgeries.

The JSON has three parts: `test` (all test signers pooled), `test_per_dataset` (CEDAR,
Bengali, Hindi separately; random negatives come from the same dataset only), and
`confidence_95` inside each (signer-level bootstrap intervals). `--bootstrap N` sets the
number of resamples (default 1000, `0` disables). Evaluation takes several minutes on CPU.
Each signer is enrolled with the first 5 genuine signatures (`--references` changes this).
Add `--device cpu` if there is no GPU.

`--threshold 0.9` applies a fixed similarity threshold instead of the validation EER point
(the output's `threshold_used` and `threshold_source` show which was applied). A threshold
picked by looking at test results is not an honest estimate; choose it on validation.
`accuracy_skilled` is correct decisions divided by all decisions over genuine and
skilled-forgery queries (random pairs excluded). `balanced_accuracy_skilled` is the mean of
the genuine-accept rate and the skilled-forgery reject rate at that threshold.

## 8. Compare your own signature images

Preprocesses the images, saves them, and prints the cosine similarity of every pair:

```bash
.venv/bin/python -m training.compare_signatures \
  --checkpoint checkpoints/v2/best_model.pt \
  --images test_signs/1.1.png test_signs/1.5.png test_signs/2.1.png test_signs/2.2.png \
  --output-dir preprocessed_test_signs
```

Quote paths that contain spaces. Add `--grayscale` only if the model was trained on
grayscale data. As a rough guide from the evaluation, a similarity of about 0.8 or higher
suggests the same signer; use the threshold printed by step 7 for your model.

## Other commands

Demo of the preprocessing on the two sample JPEGs (writes `preprocessed_outputs/`):

```bash
.venv/bin/python -m preprocessing.demo_outputs
```

## Common problems

| Problem | Cause and fix |
|---|---|
| `No module named 'preprocessing'` | Not in the repository root. `cd` to it. |
| `Path must be within the project` from `unittest` | Same cause. Run from the repository root. |
| `could not read signature image` | A file listed in the manifest is missing or renamed. Check the path in the error. |
| Out of GPU memory | Lower `--signers-per-batch`, for example to 8. |
| `best_model.zip` has no `.pt` inside | A PyTorch checkpoint is a zip file. Rename it to `best_model.pt`; don't unzip it. |
