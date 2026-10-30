# Offline Signature Verification (OSV) System Architecture

**Version:** 1.2  
**Core technologies:** YOLO (v8/v11), ConvNeXt, PyTorch, Siamese networks, triplet loss with hard mining

## 1. System overview

This document outlines the architecture and implementation steps for an offline signature verification system. The pipeline takes a scanned bank cheque, extracts the signature region, preprocesses the image, and compares it with stored reference signatures using deep metric learning.

The architecture has four main stages:

1. **Extraction (YOLO):** Locate and crop the signature. *Implemented upstream.*
2. **Preprocessing:** (2.1) Reduce banking artifacts such as stamps, lines, and noise (*implemented upstream*), then (2.2) convert the signature image to black and white and resize it to the agreed wide aspect ratio (*implemented in [signature_preprocessing.py](./signature_preprocessing.py)*).
3. **Feature extraction (ConvNeXt Siamese):** Convert the preprocessed image into a dense vector.
4. **Scoring engine:** Calculate a similarity score for the pair.

**Scope of this document:** Stage 1 and step 2.1 are treated as given. This document specifies the input contract they must satisfy (Section 3) and covers step 2.2, model training, scoring, evaluation, and decision policy.

## 2. Environment setup

Install the core deep-learning and image-processing libraries:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Select `.venv/bin/python` as the VS Code workspace interpreter. On Apple
Silicon, pass `--device mps` to opt into Metal Performance Shaders. The current
installed PyTorch build crashed during an MPS ConvNeXt forward pass, so automatic
device selection uses CUDA when available and CPU otherwise.

## 3. Pipeline stages

### Stage 1: Signature extraction (done upstream)

- **Model:** YOLO object detector.
- **Input:** Full scanned cheque image (RGB or grayscale).
- **Output:** Signature bounding box coordinates: `[x_min, y_min, x_max, y_max]`.
- **Action:** Crop the original high-resolution image using the predicted coordinates.

### Stage 2: Preprocessing

Stage 2 has two steps:

| Step | Responsibility | Status |
| --- | --- | --- |
| 2.1 Artifact cleaning | Remove stamps, printed lines, and noise from the YOLO crop. | Done upstream (outside this repository) |
| 2.2 Black-and-white conversion and aspect-aware resize | Turn the cleaned signature image into a binary, fixed-size, wide-aspect model input. | **Implemented** in [signature_preprocessing.py](./signature_preprocessing.py) |

Gate crops with low YOLO confidence or multiple detections before Stage 2. Empty images (no ink) are rejected in step 2.2 with a `ValueError`, so they can be routed to review instead of being scored.

#### Step 2.1: Artifact cleaning (done upstream)

This system only relies on the following contract for the upstream output:

- A single signature image as a `uint8` array (grayscale `H × W`, or 3-channel BGR). Any size, aspect ratio, and ink polarity is accepted by step 2.2.
- **Current output:** a binarized image with **white ink (255) on a black background (0)**, on a square canvas (observed: 512 × 512) in which the signature fills only a part of the area (observed: about 427 × 188 px, roughly 3% ink pixels). Ink strokes are about 1–2 px wide.

**Upstream check:** In the sample output the lower ends of several strokes end on a flat horizontal edge at the bottom of the ink area, which suggests the signature may be clipped by the YOLO box. Verify this on more samples and add margin to the detection box if confirmed, because clipped strokes cannot be recovered downstream.

#### Step 2.2: Black-and-white conversion and aspect-aware resize (implemented)

**Input:** the signature image from step 2.1 (any size, grayscale or BGR, either ink polarity).  
**Output:** a `192 × 384` (H × W) `uint8` image with **black ink (0) on a white background (255)**, ready for Stage 3.

Signatures are wide, so a square canvas wastes most of the input area and forces heavy downscaling of thin strokes. The agreed input is a **wide-aspect, higher-resolution canvas**:

| Setting | Previous | Agreed |
| --- | --- | --- |
| Canvas (W × H) | `224 × 224` | **`384 × 192`** (2:1) |
| Pixels | 50,176 | 73,728 (about 1.5×) |
| Divisible by 32 | Yes | Yes (ConvNeXt total stride) |

Procedure (`preprocess_signature`):

1. **Black-and-white conversion** (`to_black_and_white`): convert to grayscale, then binarize with Otsu's method. The result is normalized to black ink on white.
   - `ink="light"` is the current upstream output (white ink on black) and is inverted. `ink="dark"` leaves the polarity unchanged.
   - `ink="auto"` (default) treats the minority class as ink, because a signature covers far less than half of its crop. Use an explicit value in production.
   - An optional Gaussian blur (`blur_ksize`, disabled by default) suppresses scanner noise before thresholding. Leave it off for already-clean binary input, where it would only soften 1–2 px strokes.
2. **Crop tightly to the ink** (`crop_to_ink`): bounding box of the ink plus a 5% margin. This is required: in the sample output the signature occupies only about 427 × 188 px of a 512 × 512 canvas, so resizing the whole square would shrink it to about 160 × 70 px with 1 px strokes. Tight cropping gives about 350 × 154 px, roughly 5× more ink pixels, on the same canvas.
3. **Aspect-aware resize** (`aspect_aware_resize`):
   - Compute `scale = min(W_target / w, H_target / h)`. The image is scaled uniformly, so no stretching occurs.
   - Resize with `INTER_AREA` when shrinking (avoids aliasing of thin strokes) and `INTER_CUBIC` when enlarging.
   - Center the result on a `384 × 192` white canvas (`pad_value = 255`, matching the background).
4. **Model tensor** (`to_model_input`, called by Stage 3): replicate the single channel to 3 channels and normalize with ImageNet mean and standard deviation, to use the pretrained weights unchanged.

```python
from signature_preprocessing import preprocess_signature, to_model_input

image = preprocess_signature(stage2_1_output, target_size=(384, 192), ink="light")  # 192 x 384 uint8, black on white
tensor = to_model_input(image)                                                     # 3 x 192 x 384 float32
```

Notes:

- The image is binarized *before* resizing, so thresholding sees full-resolution strokes. Resizing then anti-aliases the stroke edges, so the final image has a few gray levels along the strokes. It is deliberately not re-thresholded, because that would erase 1–2 px strokes.
- Otsu uses one global threshold. If real scans show uneven illumination, an adaptive threshold is an alternative to evaluate; it is not implemented.
- Tight cropping discards the absolute position and size of the signature within the box. Relative proportions (aspect ratio) are kept by the uniform scaling. If absolute size is found to be informative, compare against a variant that skips the ink crop in the ablation study (Section 5.4).
- `convnext_tiny` is fully convolutional with global average pooling, so non-square inputs need no architectural change.
- The canvas size is a hyperparameter. Compare `224 × 224`, `384 × 192`, and a larger option such as `512 × 256` in the ablation study (Section 5), because the best size depends on the scan DPI and the stroke thickness after cropping.
- Use the identical step at training and inference time. Any difference in threshold method, padding value, or interpolation is a train/serve skew. Apply augmentation before this step (Section 4).

### Stage 3: Deep metric learning (Siamese network)

Rather than classifying signatures into identity classes, the network learns an embedding space in which genuine signatures from the same person are close together and dissimilar signatures are farther apart. The network is trained with **triplet loss and batch-hard mining** (Section 4) and used pairwise at inference time.

The initial training implementation is in [stage3_metric_learning.py](./stage3_metric_learning.py) and [train_siamese.py](./train_siamese.py). It uses the manifest, input preprocessing, two-phase training, and batch composition described in [README.md](./README.md). Training requires a signer-disjoint validation split and labeled signature images; none are included in this repository.

#### Architecture: ConvNeXt-Tiny

- **Backbone:** `convnext_tiny` from the `timm` library, initialized with ImageNet pre-trained weights.
- **Rationale:** ConvNeXt is a modern convolutional architecture designed to combine strong spatial feature extraction with computational efficiency.
- **Projection head:** The backbone produces a feature representation (commonly 768 dimensions for ConvNeXt-Tiny). Pass it through a custom head:
  `Linear → BatchNorm → ReLU → Dropout → Linear`.
- **Embedding:** L2-normalize the final 512-dimensional output vector, `v`.

#### Siamese forward pass

1. Pass the **reference signature** (`S_ref`) through the ConvNeXt network to get `v_ref`.
2. Pass the **test signature** (`S_test`) through the exact same network, with shared weights, to get `v_test`.

During training, the same shared-weight network embeds a whole batch at once (all anchors, positives, and negatives pass through one forward call), and triplets are formed from the embeddings inside the batch.

### Stage 4: Scoring and verification

Calculate cosine similarity between the embeddings:

$$
\operatorname{Similarity}(v_{ref}, v_{test}) = \frac{v_{ref} \cdot v_{test}}{\|v_{ref}\|\,\|v_{test}\|}
$$

For normalized vectors, cosine similarity ranges from `-1` to `1`. A linearly rescaled score can be calculated as:

$$
\operatorname{Score} = \left(\frac{\operatorname{Similarity}+1}{2}\right) \times 100
$$

**Important:** This rescaled value is a similarity score, not a calibrated probability of authenticity. Calibrate decision thresholds on representative, appropriately separated validation data; use a probability-calibration method if a probability estimate is required.

#### Multi-reference comparison

Banks may store multiple baseline signatures (for example, `N = 3`). Compare the test cheque against each reference, then aggregate the comparisons. Averaging can be a starting point, but the aggregation method and resulting thresholds should be validated against the deployment data and evaluation objective.

## 4. Training strategy

### Dataset preparation

Training uses **signer-balanced batches** from which triplets are mined online (see the loss function below), instead of pre-built random pairs. Each sample is `(image, signer_id, is_forgery)`.

- **Genuine signatures:** Positives for the same signer, and negatives for every other signer.
- **Skilled forgeries (when available):** Used only as negatives, and only for the signer they imitate. They are the hardest and most realistic negatives, so keep them in every batch where possible. A forgery is never an anchor or a positive.
- **Random forgeries:** Genuine signatures of other signers. They come for free from the batch.

**PK batch sampler:** Each batch holds `P` signers, with `K` genuine signatures and up to `F` skilled forgeries per signer. A starting point is `P = 12`, `K = 4`, `F = 2` (72 images); tune to GPU memory. Every anchor then has at least `K - 1` positives and many negatives in the batch, which hard mining requires.

Split data by signer and, where possible, by acquisition session. This helps avoid leakage between training and evaluation sets and better measures generalization to unseen samples. See Section 5 for the full evaluation protocol.

### Loss function

Use **triplet loss with batch-hard mining** on the L2-normalized embeddings. For an anchor `a`, the loss uses its hardest positive and its hardest negative within the batch:

$$
d(x, y) = \|v_x - v_y\|_2
$$

$$
L = \frac{1}{|B|} \sum_{a \in B} \max\left(0,\; \max_{p \in P(a)} d(a,p) \;-\; \min_{n \in N(a)} d(a,n) \;+\; m\right)
$$

- `P(a)`: other genuine signatures of the same signer as `a`.
- `N(a)`: genuine signatures of other signers, plus skilled forgeries of the same signer as `a`.
- `m`: margin. On unit-norm embeddings, `d ∈ [0, 2]`; start with `m = 0.3` and tune on the validation set.
- Distances are computed for the whole batch as one pairwise matrix, so mining adds no extra forward passes.

**Consistency with scoring:** For unit vectors, `d² = 2 − 2·cos`, so ranking by Euclidean distance during training is identical to ranking by cosine similarity at inference (Stage 4).

**Why this replaces contrastive loss:** Random negative pairs are mostly easy, so the loss quickly stops producing gradient. Batch-hard mining always works on the most confusing positives and negatives, and skilled forgeries are naturally the hardest negatives.

**Stability notes:**

- Hardest-example mining can collapse embeddings or latch onto mislabeled samples early in training. Use **semi-hard mining** (negatives with `d(a,p) < d(a,n) < d(a,p) + m`) during Phase 1, then switch to batch-hard in Phase 2.
- Inspect the data for mislabeled genuine or forgery samples before training. A near-duplicate "forgery" mined as the hardest negative will damage the embedding.
- Monitor the fraction of triplets with non-zero loss and the mean `d(a,p)` and `d(a,n)`. If all embeddings collapse to a point (`d ≈ 0` everywhere), reduce the learning rate or return to semi-hard mining.

### Fine-tuning steps

1. **Phase 1 (warm-up):** Freeze the ConvNeXt backbone. Train only the projection head for 5–10 epochs with a learning rate of `1 × 10⁻³`, using semi-hard mining.
2. **Phase 2 (unfreezing):** Unfreeze the last two ConvNeXt stages. Lower the learning rate to `1 × 10⁻⁴` and train for another 20–30 epochs, using batch-hard mining.
3. **Data augmentation:** Consider small rotations (for example, `±5°`), elastic transformations, and brightness adjustments. Validate augmentations to ensure they preserve meaningful handwriting characteristics and do not create unrealistic samples. Apply augmentation *before* step 2.2 (black-and-white conversion and resize), so the final input is produced by the same path as at inference.

Treat epoch counts and learning rates as initial settings; tune them using signer-disjoint validation data.

## 5. Evaluation plan

The goal is to estimate how the system behaves on **unseen signers, from real cheque scans, against realistic attacks**. A single accuracy number on random pairs is not acceptable for this task.

### 5.1 Data splits

| Split | Purpose | Rules |
| --- | --- | --- |
| Train | Fit the model | Signers disjoint from validation and test. |
| Validation | Tune hyperparameters, choose thresholds | Signers disjoint from train and test. |
| Test | Final, one-time report | Signers disjoint from train and validation, ideally from different acquisition sessions, branches, or scanners. Never used for tuning. |

- Split **by signer**, never by image. If session or scanner metadata exists, hold out whole sessions or scanners for part of the test set.
- Include a test subset of **real cheque crops that went through the actual Stage 1 and Stage 2 pipeline**, not only clean public-dataset images.
- Fix the random seeds and store the split lists with the model version, so results are reproducible.

### 5.2 Verification protocol

Evaluation mimics deployment:

1. For each test signer, choose `N` genuine signatures as the **enrolled references** (report `N = 1, 3, 5`).
2. Compare every remaining query against the references and aggregate (see multi-reference comparison in Stage 4).
3. Queries are drawn from three categories, which are **reported separately**:

| Query category | Label | Meaning |
| --- | --- | --- |
| Genuine | Accept | Held-out genuine signatures of the enrolled signer. |
| Random forgery | Reject | Genuine signatures of other signers. |
| Skilled forgery | Reject | Deliberate imitations of the enrolled signer. This is the main attack and the hardest case. |

If no skilled forgeries exist for the test signers, state this clearly in the report. Results then cover random forgeries only and must not be presented as protection against skilled forgery.

### 5.3 Metrics

| Metric | Definition | Notes |
| --- | --- | --- |
| FAR (random) | Random forgeries accepted / random forgeries | Report at fixed operating points. |
| FAR (skilled) | Skilled forgeries accepted / skilled forgeries | The key security metric. |
| FRR | Genuine rejected / genuine | The key customer-friction metric. |
| EER | Operating point where FAR = FRR | Compute separately for random and skilled forgeries. |
| ROC / AUC | Threshold-independent separation | One curve per forgery category. |
| FAR at fixed FRR, FRR at fixed FAR | For example FRR at `FAR = 1%` and FAR at `FRR = 5%` | Closer to how the policy thresholds are actually set. |
| Weighted cost | `C_FA · FAR + C_FR · FRR` | Cost weights come from the institution's risk policy. |
| Manual-review rate | Share of queries falling into the review zone | Operational load of the three-zone policy. |
| Automation rate at target error | Share auto-decided while FAR (skilled) stays below the target | The business value of the model. |

Rules for reporting:

- **Select thresholds on the validation set**, then apply them unchanged to the test set and report the resulting FAR and FRR. Do not report test-set EER computed with a test-tuned threshold as the deployment performance.
- Provide **confidence intervals** with a signer-level bootstrap (resample signers, not images), because errors are correlated within a signer.
- Report **per-signer** FAR and FRR distributions (median, 90th percentile, worst decile). A good average can hide signers whose signatures are very variable or very simple.

### 5.4 Ablation studies

Run each change against the baseline on the validation set, and confirm the final choice once on the test set.

| Question | Variants |
| --- | --- |
| Input resolution and shape | `224 × 224` (padded square) vs. `384 × 192` (current) vs. `512 × 256` |
| Tight ink crop | With (current) vs. without (whole Stage 2 canvas resized) |
| Loss | Contrastive vs. triplet (random) vs. triplet (semi-hard) vs. triplet (batch-hard) |
| Skilled forgeries as negatives | With vs. without forgeries in training |
| Margin `m` | `0.2`, `0.3`, `0.5` |
| Batch composition | Several `(P, K, F)` settings |
| Multi-reference aggregation | Mean vs. max vs. top-k mean vs. comparison with the reference centroid |
| Number of references `N` | `1`, `3`, `5` |
| Backbone fine-tuning depth | Head only vs. last two stages vs. full |

### 5.5 Robustness and error analysis

Evaluate slices of the test set and report FAR and FRR for each:

- Scanner, branch, or DPI.
- Image quality: low contrast, residual stamp or line overlap, thin or faint ink, small crop size.
- Signature type: short or simple signatures, initials-only signatures, highly variable signers.
- Time gap between enrolment and query, if known.

Review the **highest-scoring skilled forgeries** and the **lowest-scoring genuine signatures** manually. Group them into failure causes (bad crop, residual artifacts, unusual signer variability, labeling errors) to decide whether the fix belongs in the data, the upstream stages, or the model.

Also run **embedding diagnostics** during training: mean intra-signer and inter-signer distance, the share of active triplets, and a 2-D projection (for example UMAP) of validation embeddings colored by signer.

### 5.6 Acceptance gates and monitoring

- Define pass/fail criteria *before* running the test set, for example "FAR (skilled) ≤ X% at FRR ≤ Y%, with the 95% confidence interval upper bound under the policy limit". The actual values must come from the institution's risk policy.
- Do not release a new model unless it passes the gates on the full test set and does not regress on any robustness slice by more than an agreed tolerance.
- After deployment, monitor the score distribution, the review-zone rate, the rejection rate by branch and scanner, and the confirmed-fraud and reviewer-override rates. A shift in these is a signal to recalibrate thresholds or retrain.
- Feed reviewer decisions back as a labeled dataset (with audit logs), and re-run Sections 5.2–5.3 on it before changing any threshold.

## 6. Production and business logic

A three-zone decision policy can route scores for review. The values below are **illustrative starting points only** and must be calibrated on representative historical data before operational use.

| Score | Action | Description |
| --- | --- | --- |
| `≥ 85%` | Auto-approve candidate | High similarity; only eligible for automated processing if validation supports this operating point and other required controls pass. |
| `≥ 60%` and `< 85%` | Manual review | Ambiguous match or possible deviation; route for human review with a side-by-side visual comparison. |
| `< 60%` | Reject or flag for review | Low similarity; flag for investigation rather than treating the score alone as proof of forgery. |

The exact thresholds should be calibrated against a representative, signer-disjoint dataset using ROC analysis and operating metrics such as the False Acceptance Rate (FAR) and False Rejection Rate (FRR). Select thresholds according to the institution's risk policy. A signature score should not be the sole basis for blocking a transaction or declaring fraud; include human review, auditability, and appropriate fallback procedures.
