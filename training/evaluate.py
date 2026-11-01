"""Evaluate a trained checkpoint with FAR / FRR / EER on signer-disjoint splits.

Protocol (per signer): the first ``--references`` genuine signatures are enrolled as
references. A query's score is its mean cosine similarity to those references.
  * positives:        the signer's remaining genuine signatures
  * skilled negatives: forgeries of that signer
  * random negatives:  genuine signatures of other signers
The decision threshold is chosen at the EER on the validation signers and then
applied unchanged to the test signers, so the test numbers are not tuned on test data.

Usage:
    python -m training.evaluate --checkpoint checkpoints/cedar_v1/best_model.pt \
        --manifest data/processed/manifest.csv --already-preprocessed
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.stage3_metric_learning import (
    SignatureDataset,
    SignatureRecord,
    SiameseSignatureModel,
)


def _read_split(manifest: Path, root: Path, splits: set[str]) -> list[SignatureRecord]:
    records = []
    with manifest.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["split"].strip().lower() not in splits:
                continue
            path = Path(row["image_path"].strip())
            records.append(
                SignatureRecord(
                    image_path=(path if path.is_absolute() else root / path).resolve(),
                    signer_id=row["signer_id"].strip(),
                    is_forgery=row["is_forgery"].strip().lower() in {"1", "true", "yes"},
                )
            )
    if not records:
        raise ValueError(f"no manifest rows for splits {sorted(splits)}")
    return records


@torch.no_grad()
def _embed(model, records, already_preprocessed, device, batch_size, workers) -> np.ndarray:
    dataset = SignatureDataset(
        records, {r.signer_id: 0 for r in records}, already_preprocessed=already_preprocessed
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    model.eval()
    chunks = [model(images.to(device)).cpu().numpy() for images, _, _ in loader]
    return np.concatenate(chunks)


def _scores(records, embeddings, references: int):
    """Return (genuine, skilled, random) score arrays under the reference protocol."""
    by_signer = defaultdict(lambda: {"gen": [], "forg": []})
    for index, record in enumerate(records):
        by_signer[record.signer_id]["forg" if record.is_forgery else "gen"].append(index)

    refs = {}
    for signer, groups in by_signer.items():
        if len(groups["gen"]) <= references:
            raise ValueError(f"{signer} has too few genuine signatures for {references} references")
        refs[signer] = embeddings[groups["gen"][:references]]

    genuine, skilled, random = [], [], []
    for signer, groups in by_signer.items():
        def score(index: int, ref_signer: str = signer) -> float:
            return float((refs[ref_signer] @ embeddings[index]).mean())

        genuine += [score(i) for i in groups["gen"][references:]]
        skilled += [score(i) for i in groups["forg"]]
        for other, other_groups in by_signer.items():
            if other != signer:
                random += [score(i) for i in other_groups["gen"][references:]]
    return np.array(genuine), np.array(skilled), np.array(random)


def _rates(genuine, negatives, threshold):
    frr = float((genuine < threshold).mean())
    far = float((negatives >= threshold).mean())
    return far, frr


def _eer_threshold(genuine, negatives):
    candidates = np.unique(np.concatenate([genuine, negatives]))
    gaps = [abs(_rates(genuine, negatives, t)[0] - _rates(genuine, negatives, t)[1]) for t in candidates]
    threshold = float(candidates[int(np.argmin(gaps))])
    far, frr = _rates(genuine, negatives, threshold)
    return threshold, (far + frr) / 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--already-preprocessed", action="store_true")
    parser.add_argument("--references", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    root = (args.image_root or args.manifest.parent).resolve()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = SiameseSignatureModel(
        pretrained=False, embedding_dim=checkpoint["embedding_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    results = {}
    for name, splits in (("val", {"val", "validation"}), ("test", {"test"})):
        records = _read_split(args.manifest, root, splits)
        embeddings = _embed(
            model, records, args.already_preprocessed, args.device, args.batch_size, args.workers
        )
        results[name] = _scores(records, embeddings, args.references)

    # Threshold from validation signers; "skilled" is the harder and more relevant case.
    val_gen, val_skilled, val_random = results["val"]
    threshold, val_eer = _eer_threshold(val_gen, val_skilled)

    test_gen, test_skilled, test_random = results["test"]
    _, test_eer_skilled = _eer_threshold(test_gen, test_skilled)
    _, test_eer_random = _eer_threshold(test_gen, test_random)
    far_skilled, frr = _rates(test_gen, test_skilled, threshold)
    far_random, _ = _rates(test_gen, test_random, threshold)

    print(
        json.dumps(
            {
                "threshold_from_validation": threshold,
                "validation_eer_skilled": val_eer,
                "test": {
                    "far_skilled_at_threshold": far_skilled,
                    "far_random_at_threshold": far_random,
                    "frr_at_threshold": frr,
                    "eer_skilled": test_eer_skilled,
                    "eer_random": test_eer_random,
                    "genuine_queries": len(test_gen),
                    "skilled_forgeries": len(test_skilled),
                    "random_pairs": len(test_random),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
