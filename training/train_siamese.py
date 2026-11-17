"""Train the Stage 3 ConvNeXt-Tiny signature embedding model."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from preprocessing.signature_preprocessing import DEFAULT_TARGET_SIZE
from training.evaluate import reference_metrics
from training.stage3_metric_learning import (
    PKBatchSampler,
    SignatureDataset,
    SignatureRecord,
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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reference_selection_records(
    records: Sequence[SignatureRecord], references: int
) -> tuple[list[SignatureRecord], list[int]]:
    """Records (and their indices) for signers with enough genuine samples to enroll.

    The reference protocol scores a query against ``references`` enrolled genuine
    signatures, so a signer needs strictly more than that many genuine samples. The
    returned indices let callers align a full-length embedding array to the subset.
    """
    genuine_counts: dict[str, int] = defaultdict(int)
    for record in records:
        if not record.is_forgery:
            genuine_counts[record.signer_id] += 1
    eligible = {signer for signer, count in genuine_counts.items() if count > references}
    indices = [index for index, record in enumerate(records) if record.signer_id in eligible]
    return [records[index] for index in indices], indices


def _can_score(
    records: Sequence[SignatureRecord], references: int, negatives_mode: str = "auto"
) -> bool:
    """Whether the signers can produce EER/AUC under the reference protocol.

    Random negatives need a second signer; skilled negatives need a forgery. With
    neither, only genuine scores exist and EER/AUC are undefined.
    """
    eligible, _ = _reference_selection_records(records, references)
    if not eligible:
        return False
    signers = {record.signer_id for record in eligible}
    has_forgery = any(record.is_forgery for record in eligible)
    if negatives_mode == "skilled":
        return has_forgery
    if negatives_mode == "random":
        return len(signers) >= 2
    return len(signers) >= 2 or has_forgery


@torch.no_grad()
def _embed_records(
    model: SiameseSignatureModel,
    dataset: SignatureDataset,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    """Embed a dataset once, in record order, for the reference protocol."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    was_training = model.training
    model.eval()
    chunks = [model(images.to(device)).cpu().numpy() for images, _, _ in loader]
    model.train(was_training)
    return np.concatenate(chunks)


def _improves(metric: str, candidate: float, best: float) -> bool:
    """Whether ``candidate`` beats ``best`` under the chosen selection metric."""
    if math.isnan(candidate):
        return False
    if metric == "auc":
        return candidate > best
    return candidate < best


def _save_checkpoint(
    path: Path,
    model: SiameseSignatureModel,
    *,
    select_metric: str,
    select_value: float,
    selection: dict[str, float],
    validation_loss: float,
    epoch: int,
    phase: str,
    seed: int,
    already_preprocessed: bool,
    ink: str,
) -> None:
    """Write the checkpoint dict that consumers reconstruct the model from.

    ``best_validation_loss`` is kept for backwards compatibility; ``select_metric``
    and ``select_value`` record which epoch was kept and why.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_dim": model.embedding_dim,
            "best_validation_loss": validation_loss,
            "select_metric": select_metric,
            "select_value": select_value,
            "selection_metrics": selection,
            "epoch": epoch,
            "phase": phase,
            "seed": seed,
            "preprocessing": {
                "already_preprocessed": already_preprocessed,
                "ink": ink,
                "target_size": list(DEFAULT_TARGET_SIZE),
            },
        },
        path,
    )


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
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--phase1-epochs", type=int, default=5)
    parser.add_argument("--phase2-epochs", type=int, default=20)
    parser.add_argument("--signers-per-batch", type=int, default=12)
    parser.add_argument("--genuine-per-signer", type=int, default=4)
    parser.add_argument("--forgeries-per-signer", type=int, default=2)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--select-metric",
        choices=("loss", "eer", "auc"),
        default="eer",
        help="Checkpoint criterion: lowest validation loss, lowest EER, or highest "
        "AUC. EER/AUC use the signer-disjoint reference protocol from training.evaluate. "
        "Average loss is only a proxy that saturates once the margin is met, so EER "
        "(the reported metric) is the default.",
    )
    parser.add_argument(
        "--select-references",
        type=int,
        default=1,
        help="Genuine signatures enrolled per signer when scoring EER/AUC. A signer "
        "needs more genuine samples than this to be scored.",
    )
    parser.add_argument(
        "--select-negatives",
        choices=("auto", "skilled", "random"),
        default="auto",
        help="Negative set for EER/AUC: skilled forgeries, cross-signer genuines, or auto.",
    )
    parser.add_argument(
        "--select-batch-size",
        type=int,
        default=64,
        help="Batch size for the per-epoch EER/AUC embedding pass.",
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
    if args.select_references < 1:
        raise ValueError("select-references must be at least 1")
    if args.select_batch_size < 1:
        raise ValueError("select-batch-size must be at least 1")

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
        ink=args.ink,
    )
    validation_dataset = SignatureDataset(
        validation_records,
        signer_to_index,
        already_preprocessed=args.already_preprocessed,
        ink=args.ink,
    )

    train_sampler = PKBatchSampler(
        train_records,
        signers_per_batch=args.signers_per_batch,
        genuine_per_signer=args.genuine_per_signer,
        forgeries_per_signer=args.forgeries_per_signer,
        seed=args.seed,
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
    )

    select_metric = args.select_metric
    if select_metric != "loss" and not _can_score(
        validation_records, args.select_references, args.select_negatives
    ):
        print(
            f"warning: validation set cannot support '{select_metric}' selection "
            f"(need two eligible signers or a forgery, each with more than "
            f"{args.select_references} genuine signatures, for negatives "
            f"'{args.select_negatives}'); using loss instead",
            file=sys.stderr,
        )
        select_metric = "loss"

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
    # AUC improves as it rises; loss and EER improve as they fall.
    best_selection_value = -math.inf if select_metric == "auc" else math.inf
    saved_checkpoint = False
    print(f"Checkpoint selection metric: {select_metric}")

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

            # EER/AUC score the validation signers with the same reference protocol
            # the evaluation reports, so the checkpoint is chosen on the target metric
            # instead of the average loss, which saturates once the margin is met.
            selection: dict[str, float] = {"loss": validation_metrics["loss"]}
            negatives_source: str | None = None
            if select_metric != "loss":
                embeddings = _embed_records(
                    model,
                    validation_dataset,
                    device,
                    args.select_batch_size,
                    args.workers,
                )
                eligible_records, eligible_indices = _reference_selection_records(
                    validation_records, args.select_references
                )
                reference = reference_metrics(
                    eligible_records,
                    embeddings[eligible_indices],
                    args.select_references,
                    args.select_negatives,
                )
                selection["eer"] = reference["eer"]
                selection["auc"] = reference["auc"]
                selection["eer_threshold"] = reference["eer_threshold"]
                negatives_source = reference["negatives_source"]
            candidate = selection["loss"] if select_metric == "loss" else selection[select_metric]

            global_epoch += 1
            epoch_record: dict[str, object] = {
                "phase": phase_name,
                "epoch": phase_epoch + 1,
                "train": train_metrics,
                "validation": validation_metrics,
                "selection": selection,
            }
            if negatives_source is not None:
                epoch_record["selection_negatives"] = negatives_source
            print(json.dumps(epoch_record))

            if _improves(select_metric, candidate, best_selection_value):
                best_selection_value = candidate
                saved_checkpoint = True
                _save_checkpoint(
                    checkpoint_path,
                    model,
                    select_metric=select_metric,
                    select_value=best_selection_value,
                    selection=selection,
                    validation_loss=validation_metrics["loss"],
                    epoch=global_epoch,
                    phase=phase_name,
                    seed=args.seed,
                    already_preprocessed=args.already_preprocessed,
                    ink=args.ink,
                )

    if not saved_checkpoint:
        # A metric that was undefined or never improved should not leave the output
        # directory without a checkpoint that downstream tools can load.
        _save_checkpoint(
            checkpoint_path,
            model,
            select_metric=select_metric,
            select_value=candidate,
            selection=selection,
            validation_loss=validation_metrics["loss"],
            epoch=global_epoch,
            phase=phase_name,
            seed=args.seed,
            already_preprocessed=args.already_preprocessed,
            ink=args.ink,
        )

    print(f"Best checkpoint: {checkpoint_path} ({select_metric}={candidate})")


if __name__ == "__main__":
    main()
