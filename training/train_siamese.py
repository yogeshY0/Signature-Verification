"""Train the Stage 3 ConvNeXt-Tiny signature embedding model."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from preprocessing.signature_preprocessing import DEFAULT_TARGET_SIZE
from training.evaluate import (
    _accuracy,
    _choose_negatives,
    _embed,
    _eer_threshold,
    _pool,
    _scores,
)
from training.stage3_metric_learning import (
    PKBatchSampler,
    SignatureDataset,
    SiameseSignatureModel,
    load_manifest,
    triplet_loss,
)


def _device_from_argument(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _parse_dataset_weights(items: list[str] | None) -> dict[str, float]:
    """Parse ``DATASET=WEIGHT`` CLI entries into a weight mapping.

    Negative weights are rejected and entries without ``=`` are reported clearly
    so a typo cannot silently fall back to equal weighting.
    """
    weights: dict[str, float] = {}
    for item in items or []:
        name, separator, value = item.partition("=")
        if not separator or not name.strip():
            raise ValueError(
                f"--dataset-weights entries must look like DATASET=WEIGHT, got {item!r}"
            )
        try:
            weight = float(value)
        except ValueError as error:
            raise ValueError(
                f"--dataset-weights weight for {name.strip()!r} is not a number: {value!r}"
            ) from error
        if weight < 0:
            raise ValueError(
                f"--dataset-weights weight for {name.strip()!r} cannot be negative"
            )
        weights[name.strip()] = weight
    return weights


def _run_epoch(
    model: SiameseSignatureModel,
    loader: DataLoader,
    device: torch.device,
    *,
    mining: Literal["semi-hard", "batch-hard"],
    margin: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "mean_positive": 0.0, "mean_negative": 0.0, "active_fraction": 0.0}
    batch_count = 0

    for images, signer_ids, is_forgery in loader:
        images = images.to(device)
        signer_ids = signer_ids.to(device)
        is_forgery = is_forgery.to(device=device, dtype=torch.bool)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            embeddings = model(images)
            loss, metrics = triplet_loss(
                embeddings,
                signer_ids,
                is_forgery,
                margin=margin,
                mining=mining,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    max_norm=5.0,
                )
                optimizer.step()

        totals["loss"] += loss.detach().item()
        for name, value in metrics.items():
            totals[name] += value.item()
        batch_count += 1

    if batch_count == 0:
        raise RuntimeError("data loader produced no batches")
    return {name: value / batch_count for name, value in totals.items()}


def _validation_eer(
    model: SiameseSignatureModel,
    records,
    *,
    already_preprocessed: bool,
    device: torch.device,
    resize_only: bool = False,
    resize_mode: str = "aspect",
    negatives: str = "auto",
    references: int = 5,
    thresholds: tuple[float, ...] = (),
) -> tuple[float, dict[str, float]]:
    """Validation EER and accuracy at fixed thresholds on validation signers.

    Uses the same protocol as evaluate.py. ``negatives`` selects the negative score
    set: skilled forgeries, random (different-signer) pairs, or ``auto`` (forgeries
    when present, otherwise random). Accuracy counts genuine and the selected
    negatives, so a genuine-only dataset can still be scored.
    """
    embeddings = _embed(
        model,
        records,
        already_preprocessed,
        device,
        64,
        0,
        resize_only=resize_only,
        resize_mode=resize_mode,
    )
    scores = _scores(records, embeddings, references)
    genuine, skilled, random = _pool(scores, sorted(scores))
    negative_scores = _choose_negatives(genuine, skilled, random, negatives)
    accuracy = {f"{t:g}": _accuracy(genuine, negative_scores, t) for t in thresholds}
    return _eer_threshold(genuine, negative_scores)[1], accuracy


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/stage3"))
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--ink", choices=("auto", "light", "dark"), default="auto")
    parser.add_argument(
        "--already-preprocessed",
        action="store_true",
        help="Input files are already 192x384 Stage 2 output images",
    )
    parser.add_argument(
        "--resize-only",
        action="store_true",
        help="Feed raw images directly (no Stage 2.2 preprocessing): read any "
        "size/channel count, convert to grayscale, resize to 192x384, normalize.",
    )
    parser.add_argument(
        "--crop-grayscale",
        action="store_true",
        help="Add the ink-crop step but keep grayscale: read raw images, convert to "
        "grayscale, crop to the ink box, resize to 192x384. No background whitening "
        "and no binarization, so the output is not black-and-white.",
    )
    parser.add_argument(
        "--resize-mode",
        choices=("aspect", "stretch"),
        default="aspect",
        help="Resize strategy for --resize-only: 'aspect' preserves the aspect ratio and "
        "pads with white; 'stretch' forces the image to 192x384.",
    )
    parser.add_argument(
        "--negatives",
        choices=("auto", "skilled", "random"),
        default="auto",
        help="Negative set for validation EER/accuracy: 'skilled' forgeries, 'random' "
        "genuine signatures of other signers, or 'auto' (forgeries when present, else "
        "random). Genuine-only datasets need 'auto' or 'random'.",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable training-time augmentation (off by default; it hurt EER in our runs; validation never uses it).",
    )
    parser.add_argument(
        "--select-by",
        choices=("eer", "loss"),
        default="eer",
        help="Metric that decides the saved best checkpoint (validation skilled EER or loss).",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("cosine", "constant"),
        default="cosine",
        help="Per-phase learning-rate schedule; cosine decays to 1%% of the phase rate.",
    )
    parser.add_argument("--phase1-epochs", type=int, default=5)
    parser.add_argument("--phase2-epochs", type=int, default=20)
    parser.add_argument("--signers-per-batch", type=int, default=12)
    parser.add_argument("--genuine-per-signer", type=int, default=4)
    parser.add_argument("--forgeries-per-signer", type=int, default=2)
    parser.add_argument(
        "--dataset-weights",
        nargs="*",
        default=None,
        metavar="DATASET=WEIGHT",
        help=(
            "Per-dataset signer weight for dataset-balanced PK sampling, e.g. "
            "CEDAR=1 BHSig=1 SOHI=1. Datasets are matched by signer-id prefix. "
            "Omitted datasets default to equal weight; a weight of 0 excludes a "
            "dataset. Default: equal weight for every active dataset."
        ),
    )
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--accuracy-thresholds",
        type=float,
        nargs="*",
        default=[0.7, 0.8, 0.9],
        help="Similarity thresholds at which validation accuracy is logged each epoch.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.phase1_epochs < 0 or args.phase2_epochs < 0:
        raise ValueError("epoch counts cannot be negative")
    if args.phase1_epochs + args.phase2_epochs == 0:
        raise ValueError("at least one training phase must have a positive epoch count")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if args.margin <= 0:
        raise ValueError("margin must be positive")
    dataset_weights = _parse_dataset_weights(args.dataset_weights)

    _set_seed(args.seed)
    train_records, validation_records = load_manifest(args.manifest, args.image_root)
    all_signers = sorted(
        {record.signer_id for record in train_records + validation_records}
    )
    signer_to_index = {signer: index for index, signer in enumerate(all_signers)}

    train_dataset = SignatureDataset(
        train_records,
        signer_to_index,
        already_preprocessed=args.already_preprocessed,
        resize_only=args.resize_only,
        crop_only=args.crop_grayscale,
        resize_mode=args.resize_mode,
        ink=args.ink,
        augment=args.augment,
    )
    validation_dataset = SignatureDataset(
        validation_records,
        signer_to_index,
        already_preprocessed=args.already_preprocessed,
        resize_only=args.resize_only,
        crop_only=args.crop_grayscale,
        resize_mode=args.resize_mode,
        ink=args.ink,
    )

    train_sampler = PKBatchSampler(
        train_records,
        signers_per_batch=args.signers_per_batch,
        genuine_per_signer=args.genuine_per_signer,
        forgeries_per_signer=args.forgeries_per_signer,
        seed=args.seed,
        dataset_weights=dataset_weights,
    )
    # Report the dataset-balanced batch composition so runs are easy to compare.
    print(
        "Dataset-balanced sampling | signers/dataset: "
        f"{ {dataset: len(signers) for dataset, signers in train_sampler.dataset_to_signers.items()} }"
        f" -> signers per batch: {train_sampler.dataset_allocation}"
    )
    validation_eligible_signers = {
        record.signer_id
        for record in validation_records
        if not record.is_forgery
    }
    genuine_counts = {
        signer: sum(
            record.signer_id == signer and not record.is_forgery
            for record in validation_records
        )
        for signer in validation_eligible_signers
    }
    eligible_validation_signers = sum(count >= 2 for count in genuine_counts.values())
    if eligible_validation_signers < 2:
        raise ValueError(
            "validation needs at least two signers with two genuine images each"
        )
    validation_batch_signers = min(args.signers_per_batch, eligible_validation_signers)
    validation_sampler = PKBatchSampler(
        validation_records,
        signers_per_batch=validation_batch_signers,
        genuine_per_signer=2,
        forgeries_per_signer=args.forgeries_per_signer,
        seed=args.seed,
        dataset_weights=dataset_weights,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=validation_sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = _device_from_argument(args.device)
    print(f"Using device: {device}")
    model = SiameseSignatureModel(pretrained=not args.no_pretrained).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pt"
    best_score = math.inf
    # EER needs enough genuine signatures per signer for the 5-reference protocol.
    genuine_per_validation_signer: dict[str, int] = {}
    for record in validation_records:
        if not record.is_forgery:
            genuine_per_validation_signer[record.signer_id] = (
                genuine_per_validation_signer.get(record.signer_id, 0) + 1
            )
    eer_records = [
        record
        for record in validation_records
        if genuine_per_validation_signer.get(record.signer_id, 0) > 5
    ]
    if args.select_by == "eer" and not eer_records:
        raise ValueError("no validation signer has more than 5 genuine signatures for EER selection")

    phases: list[tuple[str, int, float, Literal["semi-hard", "batch-hard"]]] = [
        ("warmup", args.phase1_epochs, 1e-3, "semi-hard"),
        ("fine_tune", args.phase2_epochs, 1e-4, "batch-hard"),
    ]
    global_epoch = 0
    for phase_name, epoch_count, learning_rate, mining in phases:
        if epoch_count == 0:
            continue
        if phase_name == "warmup":
            model.freeze_backbone()
        else:
            model.unfreeze_last_stages(count=2)

        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epoch_count, eta_min=learning_rate * 0.01
            )
            if args.lr_schedule == "cosine"
            else None
        )
        for phase_epoch in range(epoch_count):
            train_sampler.set_epoch(global_epoch)
            train_metrics = _run_epoch(
                model,
                train_loader,
                device,
                mining=mining,
                margin=args.margin,
                optimizer=optimizer,
            )
            # Validate with batch-hard in every phase: semi-hard and batch-hard
            # losses are not comparable, which would skew best-checkpoint selection.
            validation_metrics = _run_epoch(
                model,
                validation_loader,
                device,
                mining="batch-hard",
                margin=args.margin,
            )
            global_epoch += 1
            validation_eer, validation_accuracy = _validation_eer(
                model,
                eer_records,
                already_preprocessed=args.already_preprocessed,
                resize_only=args.resize_only,
                resize_mode=args.resize_mode,
                negatives=args.negatives,
                device=device,
                thresholds=tuple(args.accuracy_thresholds),
            ) if eer_records else (math.nan, {})
            learning_rate_used = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()
            print(
                json.dumps(
                    {
                        "phase": phase_name,
                        "epoch": phase_epoch + 1,
                        "lr": learning_rate_used,
                        "train": train_metrics,
                        "validation": validation_metrics,
                        "validation_eer_skilled": validation_eer,
                        "validation_accuracy_skilled": validation_accuracy,
                    }
                )
            )

            score = validation_eer if args.select_by == "eer" else validation_metrics["loss"]
            if score < best_score:
                best_score = score
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "embedding_dim": model.embedding_dim,
                        "best_validation_loss": validation_metrics["loss"],
                        "best_validation_eer": validation_eer,
                        "selected_by": args.select_by,
                        "epoch": global_epoch,
                        "phase": phase_name,
                        "seed": args.seed,
                        "preprocessing": {
                            "already_preprocessed": args.already_preprocessed,
                            "resize_only": args.resize_only,
                            "crop_only": args.crop_grayscale,
                            "resize_mode": args.resize_mode,
                            "negatives": args.negatives,
                            "ink": args.ink,
                            "augmentation": args.augment,
                            "target_size": list(DEFAULT_TARGET_SIZE),
                        },
                    },
                    checkpoint_path,
                )

    print(f"Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
