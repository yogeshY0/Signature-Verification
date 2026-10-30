# Offline Signature Verification

This guide describes the project pipeline and the Stage 3 implementation added
for deep metric learning. The project is intended to compare cheque signatures
using learned image embeddings; it is not, by itself, a production fraud
decision system.

## Pipeline status

| Stage | Responsibility | Status |
| --- | --- | --- |
| 1. Signature extraction | Detect and crop signatures from cheque images, currently upstream (for example, YOLO). | Upstream |
| 2.1 Artifact cleaning | Remove stamps, lines, and noise from the crop. | Upstream |
| 2.2 Image preprocessing | Convert to black ink on white, crop around ink, resize without stretching, and normalize for the model. | Implemented in [`signature_preprocessing.py`](./signature_preprocessing.py) |
| 3. Deep metric learning | Train a ConvNeXt-Tiny encoder to map signatures into an embedding space. | Implemented in [`stage3_metric_learning.py`](./stage3_metric_learning.py) and [`train_siamese.py`](./train_siamese.py) |
| 4. Scoring and verification | Compare embeddings with cosine similarity and evaluate a policy on representative data. | Similarity helper implemented; thresholds and deployment evaluation require project data |

The Stage 2 output expected by Stage 3 is a 384 px wide, 192 px high image
(`H x W = 192 x 384`, `target_size = (width, height) = (384, 192)`) with black ink
on a white background. The preprocessing module converts this to an
ImageNet-normalized PyTorch tensor in `C x H x W` order: 3 channels, 192 high,
384 wide. Keep the same preprocessing options for training and inference.

## Environment setup

The development environment was created with Python 3.11 in the repository's
`.venv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code, select `.venv/bin/python` as the workspace interpreter. The
dependencies are listed in [`requirements.txt`](./requirements.txt): PyTorch,
torchvision, timm, OpenCV, scikit-learn, and Albumentations.

Use `--device auto` to select CUDA when available and CPU otherwise. An MPS
device can be selected explicitly with `--device mps` on Apple Silicon, but the
installed PyTorch build crashed during an MPS ConvNeXt forward-pass check.
Use CPU unless MPS is confirmed to work on the local macOS/PyTorch combination.
The first pretrained run may download ConvNeXt weights.

## Stage 3 implementation

### Model and embeddings

- Backbone: `convnext_tiny` from timm, initialized with ImageNet weights by
  default.
- Projection head: `Linear → BatchNorm → ReLU → Dropout → Linear`.
- Embedding: 512 dimensions, L2-normalized.
- `SiameseSignatureModel.compare(reference, query)` returns pairwise cosine
  similarities. Similarity is not a calibrated probability.

### Sampling and loss

Training batches are signer-balanced: sample `P` signers, `K` genuine samples
per signer, and up to `F` skilled forgeries that imitate each signer. Genuine
signatures are anchors and positives. Genuine signatures from other signers
and skilled forgeries of the same signer are negatives; forgeries are never
anchors or positives.

The loss computes pairwise Euclidean distances over normalized embeddings.
Warm-up uses semi-hard negative mining; fine-tuning uses batch-hard mining.
The training output reports loss, mean selected positive and negative
distances, and the fraction of active triplets.

### Two training phases

Defaults follow the architecture plan:

1. Warm-up: freeze ConvNeXt, train the projection head for 5 epochs with
   semi-hard mining and learning rate `1e-3`.
2. Fine-tune: unfreeze the last two ConvNeXt stages for 20 epochs with
   batch-hard mining and learning rate `1e-4`.

The default batch uses 12 signers, 4 genuine signatures per signer, and up to
2 skilled forgeries per signer. Adjust epochs, batch composition, and device
for the dataset and available hardware. The best checkpoint is selected by
signer-disjoint validation loss.

## Dataset manifest

The repository does not include signature images. Create a CSV manifest with
these columns:

| Column | Meaning |
| --- | --- |
| `image_path` | Path to an image; absolute or relative to the manifest directory, unless `--image-root` is provided. |
| `signer_id` | Signer identifier. For a skilled forgery, use the identifier of the signer being imitated. |
| `is_forgery` | `true`/`false` or `1`/`0`. |
| `split` | `train`, `val` (or `validation`), or optional `test`. |

Example:

```csv
image_path,signer_id,is_forgery,split
signatures/signer_001/genuine_01.png,signer_001,false,train
signatures/signer_001/genuine_02.png,signer_001,false,train
signatures/signer_002/forged_01.png,signer_002,true,train
signatures/validation/signer_101/genuine_01.png,signer_101,false,val
signatures/validation/signer_102/genuine_01.png,signer_102,false,val
```

Keep signers disjoint across train, validation, and test. Test rows are not
used to train or validate; they are checked for signer overlap. The loader
requires at least one train row and one validation row. Training needs enough
signers with the requested number of genuine images per batch. Validation
requires at least two distinct signers with two genuine images each.

By default, the manifest points to upstream Stage 2.1 output images, and the
dataset applies the project's Stage 2.2 preprocessing path. `--ink auto`
detects the minority pixel class as ink; choose `--ink light` for white ink on
black or `--ink dark` for dark ink on a light background if needed. If files
are already preprocessed 192 x 384 grayscale images, pass
`--already-preprocessed`.

## Train

From the activated environment:

```bash
python train_siamese.py \
  --manifest data/manifest.csv \
  --output-dir checkpoints/stage3
```

The script saves the best checkpoint to
`checkpoints/stage3/best_model.pt`. The checkpoint contains model weights,
embedding dimension, validation loss, epoch and phase, seed, and preprocessing
settings.

Useful options:

```text
--image-root PATH                 Base directory for relative image paths
--device auto|cpu|cuda|mps        Compute device (auto: CUDA, otherwise CPU)
--ink auto|light|dark             Input ink polarity
--already-preprocessed            Require 192x384 grayscale input files
--no-pretrained                   Initialize ConvNeXt without pretrained weights
--phase1-epochs N                 Warm-up epochs (default 5)
--phase2-epochs N                 Fine-tuning epochs (default 20)
--signers-per-batch N             P (default 12)
--genuine-per-signer N            K (default 4)
--forgeries-per-signer N          F (default 2)
--margin FLOAT                    Triplet margin (default 0.3)
--workers N                       DataLoader workers (default 0)
--seed N                          Random seed (default 42)
```

For example, to run a quick CPU trial without downloading ImageNet weights:

```bash
python train_siamese.py \
  --manifest data/manifest.csv \
  --output-dir checkpoints/stage3-smoke \
  --device cpu \
  --no-pretrained \
  --phase1-epochs 1 \
  --phase2-epochs 0
```

The trial still requires a valid manifest and enough samples for the selected
batch sizes.

## Tests and checks

Run the focused unit tests:

```bash
python -m unittest test_stage3_metric_learning.py
```

The tests cover manifest signer-split validation, P-K batch composition,
skilled-forgery negatives, semi-hard loss, and invalid triplet batches. During
implementation, all five tests passed, the training CLI started, and a
synthetic dataset completed a one-epoch warm-up and a one-epoch fine-tuning
run, saving a checkpoint. A separate CPU forward-pass check produced a
unit-normalized 512-dimensional embedding.

These are implementation checks only. Synthetic data does not establish
signature-recognition quality, select operating thresholds, or replace a
signer-disjoint evaluation on representative real signatures.

## Before using real data or deployment

1. Prepare and review genuine and, where available, skilled-forgery labels.
   Verify that forgery `signer_id` values identify the imitated signer.
2. Split by signer (and by acquisition session/scanner where feasible) before
   training. Never tune on the test set.
3. Check upstream signature crops for clipped strokes and choose the correct
   ink polarity.
4. Train on real data, inspect validation loss and embedding diagnostics, and
   compare performance separately for genuine, random-forgery, and
   skilled-forgery queries.
5. Select thresholds on validation data. Report FAR/FRR and signer-level
   uncertainty on an untouched test set before considering any operating
   policy.
6. Treat cosine similarity as a score, not a probability or proof of
   authenticity. Use human review and institution-approved decision policies.

For the full pipeline design, evaluation protocol, and scoring-policy
discussion, see [`offline_signature_verification_architecture.md`](./offline_signature_verification_architecture.md).
