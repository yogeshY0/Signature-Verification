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

from signature_preprocessing import DEFAULT_TARGET_SIZE
from stage3_metric_learning import (
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
    best_validation_loss = math.inf

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
            validation_metrics = _run_epoch(
                model,
                validation_loader,
                device,
                mining=mining,
                margin=args.margin,
            )
            global_epoch += 1
            print(
                json.dumps(
                    {
                        "phase": phase_name,
                        "epoch": phase_epoch + 1,
                        "train": train_metrics,
                        "validation": validation_metrics,
                    }
                )
            )

            if validation_metrics["loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["loss"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "embedding_dim": model.embedding_dim,
                        "best_validation_loss": best_validation_loss,
                        "epoch": global_epoch,
                        "phase": phase_name,
                        "seed": args.seed,
                        "preprocessing": {
                            "already_preprocessed": args.already_preprocessed,
                            "ink": args.ink,
                            "target_size": list(DEFAULT_TARGET_SIZE),
                        },
                    },
                    checkpoint_path,
                )

    print(f"Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
