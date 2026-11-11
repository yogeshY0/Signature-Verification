"""Stage 3 ConvNeXt-Tiny embeddings and signer-balanced triplet learning."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Literal, Mapping, Sequence

import cv2
import timm
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

from preprocessing.augmentation import augment_signature
from preprocessing.signature_preprocessing import (
    DEFAULT_TARGET_SIZE,
    aspect_aware_resize,
    crop_to_ink,
    preprocess_signature,
    to_model_input,
)

MiningMode = Literal["semi-hard", "batch-hard"]
# How raw images are brought to the model's fixed input size in ``resize_only`` mode.
ResizeMode = Literal["aspect", "stretch"]

# Signer ids are prefixed by their source dataset when manifests are built (see
# preprocessing/build_manifests.py): "cedar_007", "bhsig_bn_003", "bhsig_hi_012",
# "sohi_042". Grouping signers by this prefix lets the sampler balance each batch
# across datasets so a large dataset (e.g. BHSig260, ~14k images) cannot crowd out
# a small one (e.g. CEDAR, ~2.6k images) and starve it of gradient updates.
DATASET_PREFIXES: tuple[tuple[str, str], ...] = (
    ("cedar", "CEDAR"),
    ("bhsig", "BHSig"),
    ("sohi", "SOHI"),
)
# Bucket for signer ids that do not carry a known dataset prefix.
OTHER_DATASET = "OTHER"


def dataset_for_signer(signer_id: str) -> str:
    """Return the source dataset for a signer id from its manifest prefix.

    ``"cedar_007" -> "CEDAR"``, ``"bhsig_bn_003"`` and ``"bhsig_hi_012" -> "BHSig"``,
    ``"sohi_042" -> "SOHI"``. Ids without a known prefix fall back to ``OTHER`` so
    datasets that do not follow the convention still train as one balanced group.
    """
    normalized = signer_id.strip().lower()
    for prefix, dataset in DATASET_PREFIXES:
        if normalized == prefix or normalized.startswith(f"{prefix}_"):
            return dataset
    return OTHER_DATASET


def _allocate_signers(
    available: Mapping[str, int],
    weights: Mapping[str, float] | None,
    total: int,
) -> dict[str, int]:
    """Split ``total`` per-batch signer slots across datasets by weight.

    Uses largest-remainder rounding so the integer allocations sum to
    ``min(total, sum(available.values()))``. Datasets are weighted by ``weights``
    (missing entries default to equal weight, zero or negative weights exclude the
    dataset). Allocations never exceed a dataset's available signer count, and any
    surplus is redistributed to datasets that still have spare capacity.
    """
    if total <= 0:
        return {}
    weights = weights or {}

    def weight_of(dataset: str) -> float:
        value = float(weights.get(dataset, 1.0))
        return value if value > 0 else 0.0

    active = {
        dataset: count
        for dataset, count in available.items()
        if count > 0 and weight_of(dataset) > 0
    }
    if not active:
        return {}

    total_weight = sum(weight_of(dataset) for dataset in active)
    target = min(total, sum(active.values()))
    counts: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for dataset, count in active.items():
        quota = target * weight_of(dataset) / total_weight
        counts[dataset] = min(int(quota), count)
        remainders.append((quota - int(quota), dataset))

    remaining = target - sum(counts.values())
    # Hand out leftover slots to the largest fractional quotas that still fit.
    for _, dataset in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        spare = active[dataset] - counts[dataset]
        if spare > 0:
            added = min(spare, remaining)
            counts[dataset] += added
            remaining -= added
    return {dataset: count for dataset, count in counts.items() if count > 0}


@dataclass(frozen=True)
class SignatureRecord:
    """One manifest example and the signer whose identity it represents."""

    image_path: Path
    signer_id: str
    is_forgery: bool


def _parse_bool(value: str, *, row_number: int) -> bool:
    """Parse common CSV boolean spellings and report invalid values with row context."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(
        f"manifest row {row_number}: is_forgery must be true/false or 1/0"
    )


def load_manifest(
    manifest_path: str | Path,
    image_root: str | Path | None = None,
) -> tuple[list[SignatureRecord], list[SignatureRecord]]:
    """Load train/validation rows from a signer-disjoint CSV manifest.

    Required columns: ``image_path,signer_id,is_forgery,split``. Relative image
    paths are resolved from ``image_root`` or the manifest's directory. Forgeries
    must use the ID of the signer they imitate.
    """
    manifest = Path(manifest_path).expanduser().resolve()
    # Relative paths are rooted at image_root when provided, otherwise beside the CSV.
    root = Path(image_root).expanduser().resolve() if image_root else manifest.parent
    required = {"image_path", "signer_id", "is_forgery", "split"}
    train: list[SignatureRecord] = []
    validation: list[SignatureRecord] = []
    test_signers: set[str] = set()

    with manifest.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest is missing required columns: {', '.join(sorted(missing))}")

        for row_number, row in enumerate(reader, start=2):
            image_path = row["image_path"].strip()
            signer_id = row["signer_id"].strip()
            split = row["split"].strip().lower()
            if not image_path or not signer_id:
                raise ValueError(
                    f"manifest row {row_number}: image_path and signer_id cannot be empty"
                )
            if split not in {"train", "val", "validation", "test"}:
                raise ValueError(
                    f"manifest row {row_number}: split must be train, val, or test"
                )

            path = Path(image_path).expanduser()
            if not path.is_absolute():
                path = root / path
            record = SignatureRecord(
                image_path=path.resolve(),
                signer_id=signer_id,
                is_forgery=_parse_bool(row["is_forgery"], row_number=row_number),
            )
            if split == "train":
                train.append(record)
            elif split in {"val", "validation"}:
                validation.append(record)
            else:
                test_signers.add(signer_id)

    if not train or not validation:
        raise ValueError("manifest must contain at least one train row and one val row")

    train_signers = {record.signer_id for record in train}
    validation_signers = {record.signer_id for record in validation}
    # Keep every signer confined to one split so validation does not leak identities.
    overlap = (train_signers & validation_signers) | (train_signers & test_signers) | (
        validation_signers & test_signers
    )
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"signers must be disjoint across train, val, and test: {examples}")
    return train, validation


class SignatureDataset(Dataset[tuple[Tensor, int, bool]]):
    """Load signatures, preprocess them, and return model-ready tensors and labels.

    Four input modes are supported:
    * default: read the raw crop and run the Stage 2.2 ``preprocess_signature`` path.
    * ``already_preprocessed``: the files are Stage 2.2 output (192x384 grayscale).
    * ``resize_only``: the files are treated as already-prepared signatures and fed to
      the network as-is. The image is read at any size/channel count, converted to
      grayscale and resized to the model input, with no Stage 2.2 crop, background
      whitening or binarization.
    * ``crop_only``: add the ink-crop step on top of the raw grayscale image (crop to
      the ink bounding box, then resize). No background whitening and no binarization,
      so stroke intensity is preserved and the output is not black-and-white.
    """

    def __init__(
        self,
        records: Sequence[SignatureRecord],
        signer_to_index: dict[str, int],
        *,
        already_preprocessed: bool = False,
        resize_only: bool = False,
        crop_only: bool = False,
        resize_mode: ResizeMode = "aspect",
        ink: Literal["auto", "light", "dark"] = "auto",
        augment: bool = False,
    ) -> None:
        inputs = {
            "already_preprocessed": already_preprocessed,
            "resize_only": resize_only,
            "crop_only": crop_only,
        }
        enabled = [name for name, on in inputs.items() if on]
        if len(enabled) > 1:
            raise ValueError(f"{' and '.join(enabled)} are mutually exclusive")
        if resize_mode not in {"aspect", "stretch"}:
            raise ValueError(f"unsupported resize_mode: {resize_mode}")
        self.records = list(records)
        # Augmentation is for the training split only; evaluation data stays untouched.
        self.augment = augment
        self.signer_to_index = signer_to_index
        self.already_preprocessed = already_preprocessed
        self.resize_only = resize_only
        self.crop_only = crop_only
        self.resize_mode = resize_mode
        self.ink = ink

    def __len__(self) -> int:
        return len(self.records)

    def _read_resized(self, path: Path) -> np.ndarray:
        """Read any signature image and resize it to the model input size.

        The Stage 2.2 pipeline is skipped entirely: no ink crop, background whitening
        or binarization. Only a grayscale conversion and a resize to
        ``DEFAULT_TARGET_SIZE`` happen, so the raw image can be fed to the network.
        """
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read signature image: {path}")
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        width, height = DEFAULT_TARGET_SIZE
        if self.resize_mode == "stretch":
            return cv2.resize(grayscale, (width, height), interpolation=cv2.INTER_AREA)
        # Preserve the aspect ratio and pad with white, matching the model's input.
        return aspect_aware_resize(grayscale, DEFAULT_TARGET_SIZE, pad_value=255)

    def _read_cropped(self, path: Path) -> np.ndarray:
        """Grayscale + crop to the ink box + resize (no whitening, no binarization).

        This adds only the Stage 2.2 ink-crop step on top of the raw grayscale image.
        Strokes keep their original intensity and the paper background is not whitened,
        so the result stays grayscale rather than black-and-white.
        """
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read signature image: {path}")
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        try:
            cropped = crop_to_ink(grayscale, margin=0.05)
        except ValueError:
            # Faint/blank scan: keep the whole frame instead of crashing the run.
            cropped = grayscale
        width, height = DEFAULT_TARGET_SIZE
        if self.resize_mode == "stretch":
            return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)
        return aspect_aware_resize(cropped, DEFAULT_TARGET_SIZE, pad_value=255)

    def __getitem__(self, index: int) -> tuple[Tensor, int, bool]:
        record = self.records[index]
        if self.resize_only:
            preprocessed = self._read_resized(record.image_path)
        elif self.crop_only:
            preprocessed = self._read_cropped(record.image_path)
        else:
            # Already-preprocessed files are grayscale at the model's fixed input size.
            read_mode = cv2.IMREAD_GRAYSCALE if self.already_preprocessed else cv2.IMREAD_COLOR
            image = cv2.imread(str(record.image_path), read_mode)
            if image is None:
                raise FileNotFoundError(f"could not read signature image: {record.image_path}")

            if self.already_preprocessed:
                expected_shape = (DEFAULT_TARGET_SIZE[1], DEFAULT_TARGET_SIZE[0])
                if image.shape != expected_shape:
                    raise ValueError(
                        f"preprocessed image {record.image_path} has shape {image.shape}; "
                        f"expected {expected_shape}"
                    )
                preprocessed = image
            else:
                preprocessed = preprocess_signature(image, ink=self.ink)

        if self.augment:
            preprocessed = augment_signature(preprocessed)

        # The model accepts three channels; preprocessing repeats grayscale for RGB.
        tensor = torch.from_numpy(to_model_input(preprocessed))
        return tensor, self.signer_to_index[record.signer_id], record.is_forgery


class PKBatchSampler(Sampler[list[int]]):
    """Build batches with P signers, K genuine samples, and up to F forgeries each.

    Signers are grouped by their source dataset and the per-batch signer budget
    ``P`` is split across the active datasets (equal weights by default, or
    ``dataset_weights``) instead of being sampled globally. This is
    dataset-balanced PK sampling: it stops a large dataset (e.g. BHSig260) from
    dominating every batch and starving a small dataset (e.g. CEDAR) of the
    gradient supplied by online batch-hard triplet mining. ``P`` and its image
    budget ``P * (K + F)`` are unchanged, only the composition is balanced.
    Within each sampled signer the standard PK structure is preserved: ``K``
    genuine samples and up to ``F`` skilled forgeries. Forgeries are always
    negatives associated with the signer they imitate.
    """

    def __init__(
        self,
        records: Sequence[SignatureRecord],
        *,
        signers_per_batch: int,
        genuine_per_signer: int,
        forgeries_per_signer: int = 0,
        seed: int = 42,
        dataset_weights: Mapping[str, float] | None = None,
        dataset_resolver: Callable[[str], str] = dataset_for_signer,
    ) -> None:
        if signers_per_batch < 2:
            raise ValueError("signers_per_batch must be at least 2")
        if genuine_per_signer < 2:
            raise ValueError("genuine_per_signer must be at least 2")
        if forgeries_per_signer < 0:
            raise ValueError("forgeries_per_signer cannot be negative")

        self.signers_per_batch = signers_per_batch
        self.genuine_per_signer = genuine_per_signer
        self.forgeries_per_signer = forgeries_per_signer
        self.seed = seed
        self.epoch = 0
        self.dataset_weights = dict(dataset_weights) if dataset_weights else {}
        self.dataset_resolver = dataset_resolver
        # Keep genuine and forged record indices separate for each signer.
        self.by_signer: dict[str, dict[bool, list[int]]] = {}
        for index, record in enumerate(records):
            groups = self.by_signer.setdefault(record.signer_id, {False: [], True: []})
            groups[record.is_forgery].append(index)

        self.eligible_signers = [
            signer
            for signer, groups in self.by_signer.items()
            if len(groups[False]) >= genuine_per_signer
        ]
        # Each batch needs enough genuine samples per signer to form positives.
        if len(self.eligible_signers) < signers_per_batch:
            raise ValueError(
                f"need at least {signers_per_batch} signers with "
                f"{genuine_per_signer} genuine images; found {len(self.eligible_signers)}"
            )

        # Map every eligible signer to its source dataset (e.g. CEDAR/BHSig/SOHI).
        self.dataset_to_signers: dict[str, list[str]] = {}
        for signer in self.eligible_signers:
            self.dataset_to_signers.setdefault(self.dataset_resolver(signer), []).append(signer)

        # Split the per-batch signer budget across datasets (equal weights by default).
        self.dataset_allocation = _allocate_signers(
            {dataset: len(signers) for dataset, signers in self.dataset_to_signers.items()},
            self.dataset_weights,
            self.signers_per_batch,
        )
        self.active_signers_per_batch = sum(self.dataset_allocation.values())
        if self.active_signers_per_batch < 2:
            raise ValueError(
                "dataset_weights leave fewer than two signers to sample per batch"
            )

    def __len__(self) -> int:
        # Number of batches per epoch; signers are sampled anew for each batch.
        return max(
            1,
            (len(self.eligible_signers) + self.active_signers_per_batch - 1)
            // self.active_signers_per_batch,
        )

    def set_epoch(self, epoch: int) -> None:
        """Change the deterministic sampling sequence for the next epoch."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        # A local seeded RNG makes each epoch reproducible without changing global RNG.
        rng = random.Random(self.seed + self.epoch)
        for _ in range(len(self)):
            indices: list[int] = []
            # Draw each dataset's signer quota so no single dataset dominates the batch.
            for dataset, count in self.dataset_allocation.items():
                signer_batch = rng.sample(self.dataset_to_signers[dataset], count)
                for signer in signer_batch:
                    groups = self.by_signer[signer]
                    indices.extend(rng.sample(groups[False], self.genuine_per_signer))
                    # Not every signer has forgeries; use all available up to the cap.
                    forgery_count = min(self.forgeries_per_signer, len(groups[True]))
                    if forgery_count:
                        indices.extend(rng.sample(groups[True], forgery_count))
            yield indices


class SiameseSignatureModel(nn.Module):
    """Map each signature to a normalized embedding using a shared encoder.

    ConvNeXt extracts visual features. The projection head maps them to the
    embedding space, and L2 normalization makes cosine and Euclidean distances
    useful for comparing signatures.
    """

    def __init__(self, *, pretrained: bool = True, embedding_dim: int = 512) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.projection = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            # LayerNorm acts per sample, independent of PK batch composition.
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, images: Tensor) -> Tensor:
        """Encode a batch and return one unit-length embedding per image."""
        return F.normalize(self.projection(self.backbone(images)), p=2, dim=1)

    def compare(self, reference: Tensor, query: Tensor) -> Tensor:
        """Return pairwise cosine similarities; these are not calibrated probabilities."""
        reference_embedding = self(reference)
        query_embedding = self(query)
        return F.cosine_similarity(reference_embedding, query_embedding, dim=1)

    def freeze_backbone(self) -> None:
        """Stop gradient updates to the pretrained feature extractor."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_last_stages(self, count: int = 2) -> None:
        """Freeze the backbone, then enable gradients only in its last stages."""
        stages = getattr(self.backbone, "stages", None)
        if stages is None or count < 1:
            raise ValueError("backbone must expose stages and count must be positive")
        self.freeze_backbone()
        for stage in stages[-count:]:
            for parameter in stage.parameters():
                parameter.requires_grad = True


def triplet_loss(
    embeddings: Tensor,
    signer_ids: Tensor,
    is_forgery: Tensor,
    *,
    margin: float = 0.3,
    mining: MiningMode = "batch-hard",
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute triplet loss for genuine anchors and return detached diagnostics.

    Positives are other genuine samples from the same signer. Negatives are
    genuine samples from different signers or forgeries labeled as imitating
    the anchor's signer. Forged samples never act as anchors or positives.
    """
    if embeddings.ndim != 2 or signer_ids.ndim != 1 or is_forgery.ndim != 1:
        raise ValueError("expected 2D embeddings and 1D signer/forgery labels")
    if embeddings.shape[0] != signer_ids.shape[0] or embeddings.shape[0] != is_forgery.shape[0]:
        raise ValueError("embeddings and labels must have the same batch size")
    if mining not in {"semi-hard", "batch-hard"}:
        raise ValueError(f"unsupported mining mode: {mining}")

    # Pairwise distances are used to select examples for each genuine anchor.
    distances = torch.cdist(embeddings.float(), embeddings.float(), p=2)
    same_signer = signer_ids[:, None].eq(signer_ids[None, :])
    genuine = ~is_forgery.bool()
    not_self = ~torch.eye(len(signer_ids), dtype=torch.bool, device=signer_ids.device)
    # Positives exclude the anchor itself and must be genuine.
    positives = same_signer & genuine[None, :] & not_self
    # A same-signer forgery is a valid negative because it imitates the anchor.
    negatives = ((~same_signer) & genuine[None, :]) | (
        same_signer & is_forgery.bool()[None, :]
    )
    anchors = torch.where(genuine)[0]

    positive_mask = positives[anchors]
    negative_mask = negatives[anchors]
    if torch.any(positive_mask.sum(dim=1) == 0):
        raise ValueError("each genuine anchor needs another genuine sample from its signer")
    if torch.any(negative_mask.sum(dim=1) == 0):
        raise ValueError("each genuine anchor needs at least one valid negative")

    anchor_distances = distances[anchors]
    # Batch-hard loss takes the farthest valid positive and closest valid negative.
    positive_distances = anchor_distances.masked_fill(~positive_mask, float("-inf"))
    hardest_positive = positive_distances.max(dim=1).values
    negative_distances = anchor_distances.masked_fill(~negative_mask, float("inf"))
    hardest_negative = negative_distances.min(dim=1).values

    if mining == "semi-hard":
        # Choose the nearest negative farther away than the hardest positive but
        # still within the margin. Fall back to the hardest negative if none qualify.
        semi_hard_mask = (
            negative_mask
            & (anchor_distances > hardest_positive[:, None])
            & (anchor_distances < (hardest_positive + margin)[:, None])
        )
        semi_hard_distances = anchor_distances.masked_fill(
            ~semi_hard_mask, float("inf")
        )
        selected_negative = torch.where(
            semi_hard_mask.any(dim=1),
            semi_hard_distances.min(dim=1).values,
            hardest_negative,
        )
    else:
        selected_negative = hardest_negative

    per_anchor_loss = F.relu(hardest_positive - selected_negative + margin)
    # Metrics are detached so logging them does not retain the autograd graph.
    return per_anchor_loss.mean(), {
        "mean_positive": hardest_positive.mean().detach(),
        "mean_negative": selected_negative.mean().detach(),
        "active_fraction": (per_anchor_loss > 0).float().mean().detach(),
    }
