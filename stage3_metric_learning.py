"""Stage 3 ConvNeXt-Tiny embeddings and signer-balanced triplet learning."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Sequence

import cv2
import timm
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

from signature_preprocessing import (
    DEFAULT_TARGET_SIZE,
    preprocess_signature,
    to_model_input,
)

MiningMode = Literal["semi-hard", "batch-hard"]


@dataclass(frozen=True)
class SignatureRecord:
    image_path: Path
    signer_id: str
    is_forgery: bool


def _parse_bool(value: str, *, row_number: int) -> bool:
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
    overlap = (train_signers & validation_signers) | (train_signers & test_signers) | (
        validation_signers & test_signers
    )
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"signers must be disjoint across train, val, and test: {examples}")
    return train, validation


class SignatureDataset(Dataset[tuple[Tensor, int, bool]]):
    def __init__(
        self,
        records: Sequence[SignatureRecord],
        signer_to_index: dict[str, int],
        *,
        already_preprocessed: bool = False,
        ink: Literal["auto", "light", "dark"] = "auto",
    ) -> None:
        self.records = list(records)
        self.signer_to_index = signer_to_index
        self.already_preprocessed = already_preprocessed
        self.ink = ink

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int, bool]:
        record = self.records[index]
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

        tensor = torch.from_numpy(to_model_input(preprocessed))
        return tensor, self.signer_to_index[record.signer_id], record.is_forgery


class PKBatchSampler(Sampler[list[int]]):
    """Sample P distinct signers, K genuine images, and up to F matching forgeries."""

    def __init__(
        self,
        records: Sequence[SignatureRecord],
        *,
        signers_per_batch: int,
        genuine_per_signer: int,
        forgeries_per_signer: int = 0,
        seed: int = 42,
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
        self.by_signer: dict[str, dict[bool, list[int]]] = {}
        for index, record in enumerate(records):
            groups = self.by_signer.setdefault(record.signer_id, {False: [], True: []})
            groups[record.is_forgery].append(index)

        self.eligible_signers = [
            signer
            for signer, groups in self.by_signer.items()
            if len(groups[False]) >= genuine_per_signer
        ]
        if len(self.eligible_signers) < signers_per_batch:
            raise ValueError(
                f"need at least {signers_per_batch} signers with "
                f"{genuine_per_signer} genuine images; found {len(self.eligible_signers)}"
            )

    def __len__(self) -> int:
        return max(1, (len(self.eligible_signers) + self.signers_per_batch - 1) // self.signers_per_batch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        for _ in range(len(self)):
            signer_batch = rng.sample(self.eligible_signers, self.signers_per_batch)
            indices: list[int] = []
            for signer in signer_batch:
                groups = self.by_signer[signer]
                indices.extend(rng.sample(groups[False], self.genuine_per_signer))
                forgery_count = min(self.forgeries_per_signer, len(groups[True]))
                if forgery_count:
                    indices.extend(rng.sample(groups[True], forgery_count))
            yield indices


class SiameseSignatureModel(nn.Module):
    """Shared ConvNeXt-Tiny encoder with a normalized 512-D projection head."""

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
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, images: Tensor) -> Tensor:
        return F.normalize(self.projection(self.backbone(images)), p=2, dim=1)

    def compare(self, reference: Tensor, query: Tensor) -> Tensor:
        """Return pairwise cosine similarities; these are not calibrated probabilities."""
        reference_embedding = self(reference)
        query_embedding = self(query)
        return F.cosine_similarity(reference_embedding, query_embedding, dim=1)

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_last_stages(self, count: int = 2) -> None:
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
    """Mine genuine anchors, genuine positives, and valid random/skilled negatives."""
    if embeddings.ndim != 2 or signer_ids.ndim != 1 or is_forgery.ndim != 1:
        raise ValueError("expected 2D embeddings and 1D signer/forgery labels")
    if embeddings.shape[0] != signer_ids.shape[0] or embeddings.shape[0] != is_forgery.shape[0]:
        raise ValueError("embeddings and labels must have the same batch size")
    if mining not in {"semi-hard", "batch-hard"}:
        raise ValueError(f"unsupported mining mode: {mining}")

    distances = torch.cdist(embeddings.float(), embeddings.float(), p=2)
    same_signer = signer_ids[:, None].eq(signer_ids[None, :])
    genuine = ~is_forgery.bool()
    not_self = ~torch.eye(len(signer_ids), dtype=torch.bool, device=signer_ids.device)
    positives = same_signer & genuine[None, :] & not_self
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
    positive_distances = anchor_distances.masked_fill(~positive_mask, float("-inf"))
    hardest_positive = positive_distances.max(dim=1).values
    negative_distances = anchor_distances.masked_fill(~negative_mask, float("inf"))
    hardest_negative = negative_distances.min(dim=1).values

    if mining == "semi-hard":
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
    return per_anchor_loss.mean(), {
        "mean_positive": hardest_positive.mean().detach(),
        "mean_negative": selected_negative.mean().detach(),
        "active_fraction": (per_anchor_loss > 0).float().mean().detach(),
    }
