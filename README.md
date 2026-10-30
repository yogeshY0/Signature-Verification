# Offline Signature Verification

## Stage 3: ConvNeXt-Tiny metric learning

Stage 3 is implemented as a shared-weight ConvNeXt-Tiny encoder with a normalized
512-dimensional projection head, signer-balanced batches, and semi-hard then
batch-hard triplet mining. Training data is described by a CSV manifest; the
repository does not include signature images.

### Environment

Use Python 3.11 and an isolated virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code, select `.venv/bin/python` as the workspace interpreter. `--device
auto` selects CUDA when available and otherwise CPU. You can explicitly pass
`--device mps` on Apple Silicon, but the MPS forward pass crashed with the
installed PyTorch build during validation here; use CPU unless MPS works on
your macOS/PyTorch combination. The first pretrained run may download
ConvNeXt weights.

### Manifest

Create a CSV with the required columns:

```csv
image_path,signer_id,is_forgery,split
signatures/signer_001/genuine_01.png,signer_001,false,train
signatures/signer_001/genuine_02.png,signer_001,false,train
signatures/signer_002/forged_01.png,signer_002,true,train
signatures/validation/signer_101/genuine_01.png,signer_101,false,val
```

Relative image paths are resolved from the manifest's directory (or
`--image-root`). `signer_id` on a skilled forgery must identify the signer being
imitated. Keep each signer in exactly one split; optional `test` rows are not
used for training or validation. Include at least two distinct validation
signers with two genuine images each, and enough training signers with the
requested number of genuine images for each batch.

By default, images are read as Stage 2.1 outputs and passed through
`preprocess_signature` using the same Stage 2.2 path as inference. For already
preprocessed 192x384 grayscale images, pass `--already-preprocessed`.
`--ink auto` is the default; set `--ink light` for white ink on black from the
upstream pipeline or `--ink dark` for dark ink on a light background.

### Train

```bash
python train_siamese.py \
  --manifest data/manifest.csv \
  --output-dir checkpoints/stage3
```

Defaults follow the plan: 5 frozen-backbone warm-up epochs with semi-hard
mining, then 20 epochs with the final two backbone stages unfrozen and
batch-hard mining. The default batch uses 12 signers, 4 genuine samples per
signer, and up to 2 skilled forgeries per signer. Adjust batch size and epoch
counts for available data and hardware. Checkpoints are selected by
signer-disjoint validation loss and written to `checkpoints/stage3/best_model.pt`.

Run the focused checks with:

```bash
python -m unittest test_stage3_metric_learning.py
```
