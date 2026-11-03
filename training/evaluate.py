"""Evaluate a trained checkpoint with FAR / FRR / EER on signer-disjoint splits.

Protocol (per signer): the first ``--references`` genuine signatures are enrolled as
references. A query's score is its mean cosine similarity to those references.
  * positives:        the signer's remaining genuine signatures
  * skilled negatives: forgeries of that signer
  * random negatives:  genuine signatures of other signers
The decision threshold is chosen at the EER on the validation signers and then
applied unchanged to the test signers, so the test numbers are not tuned on test data.

Besides the pooled numbers, the output has a per-dataset breakdown (CEDAR, Bengali,
Hindi; random negatives come from the same dataset) and 95% confidence intervals from
resampling test signers with replacement (``--bootstrap``; 0 disables). With only a few
dozen test signers, differences smaller than the interval are not reliable.

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


def _scores(records, embeddings, references: int) -> dict[str, dict]:
    """Per-signer score arrays under the reference protocol.

    Returns ``{signer: {"gen", "skilled", "random": {other_signer: scores}}}`` where
    every score is the mean similarity of a query to that signer's references.
    """
    by_signer = defaultdict(lambda: {"gen": [], "forg": []})
    for index, record in enumerate(records):
        by_signer[record.signer_id]["forg" if record.is_forgery else "gen"].append(index)

    refs = {}
    for signer, groups in by_signer.items():
        if len(groups["gen"]) <= references:
            raise ValueError(f"{signer} has too few genuine signatures for {references} references")
        refs[signer] = embeddings[groups["gen"][:references]]

    def score(ref_signer: str, indices: list[int]) -> np.ndarray:
        if not indices:
            return np.zeros(0)
        return (refs[ref_signer] @ embeddings[indices].T).mean(axis=0)

    scores = {}
    for signer, groups in by_signer.items():
        scores[signer] = {
            "gen": score(signer, groups["gen"][references:]),
            "skilled": score(signer, groups["forg"]),
            "random": {
                other: score(signer, other_groups["gen"][references:])
                for other, other_groups in by_signer.items()
                if other != signer
            },
        }
    return scores


def _dataset(signer_id: str) -> str:
    """Dataset name from the signer ID prefix, e.g. ``bhsig_hi_012`` -> ``bhsig_hi``."""
    return signer_id.rsplit("_", 1)[0]


def _pool(scores, signers, dataset_only: bool = False):
    """Concatenate (genuine, skilled, random) scores for the given signers (repeats allowed)."""
    genuine = [scores[s]["gen"] for s in signers]
    skilled = [scores[s]["skilled"] for s in signers]
    random = []
    for signer in signers:
        for other, values in scores[signer]["random"].items():
            if not dataset_only or _dataset(other) == _dataset(signer):
                random.append(values)
    cat = lambda parts: np.concatenate(parts) if parts else np.zeros(0)
    return cat(genuine), cat(skilled), cat(random)


def _rates(genuine, negatives, threshold):
    frr = float((genuine < threshold).mean())
    far = float((negatives >= threshold).mean())
    return far, frr


def _eer_threshold(genuine, negatives):
    """Threshold where FAR and FRR are closest, and the EER there (sort-based, O(n log n))."""
    candidates = np.unique(np.concatenate([genuine, negatives]))
    gen_sorted, neg_sorted = np.sort(genuine), np.sort(negatives)
    frr = np.searchsorted(gen_sorted, candidates, side="left") / len(genuine)
    far = 1 - np.searchsorted(neg_sorted, candidates, side="left") / len(negatives)
    best = int(np.argmin(np.abs(far - frr)))
    return float(candidates[best]), float((far[best] + frr[best]) / 2)


def _metrics(genuine, skilled, random, threshold) -> dict[str, float]:
    far_skilled, frr = _rates(genuine, skilled, threshold)
    far_random, _ = _rates(genuine, random, threshold)
    return {
        "eer_skilled": _eer_threshold(genuine, skilled)[1],
        "eer_random": _eer_threshold(genuine, random)[1],
        "far_skilled_at_threshold": far_skilled,
        "far_random_at_threshold": far_random,
        "frr_at_threshold": frr,
    }


def _bootstrap(scores, signers, threshold, resamples, dataset_only=False, seed=0):
    """95% intervals from resampling signers with replacement."""
    if resamples <= 0:
        return None
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        chosen = [signers[i] for i in rng.integers(0, len(signers), len(signers))]
        genuine, skilled, random = _pool(scores, chosen, dataset_only)
        if len(genuine) and len(skilled) and len(random):
            draws.append(_metrics(genuine, skilled, random, threshold))
    return {
        key: [float(np.percentile([d[key] for d in draws], q)) for q in (2.5, 97.5)]
        for key in draws[0]
    }


def _summary(scores, signers, threshold, resamples, dataset_only=False) -> dict:
    genuine, skilled, random = _pool(scores, signers, dataset_only)
    return {
        "signers": len(signers),
        **_metrics(genuine, skilled, random, threshold),
        "genuine_queries": len(genuine),
        "skilled_forgeries": len(skilled),
        "random_pairs": len(random),
        "confidence_95": _bootstrap(scores, signers, threshold, resamples, dataset_only),
    }


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
    parser.add_argument(
        "--bootstrap", type=int, default=1000, help="Signer resamples for 95%% intervals; 0 disables."
    )
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
    val_signers = sorted(results["val"])
    val_gen, val_skilled, _ = _pool(results["val"], val_signers)
    threshold, val_eer = _eer_threshold(val_gen, val_skilled)

    test_scores = results["test"]
    test_signers = sorted(test_scores)
    per_dataset = {
        name: _summary(
            test_scores,
            [s for s in test_signers if _dataset(s) == name],
            threshold,
            args.bootstrap,
            dataset_only=True,
        )
        for name in sorted({_dataset(s) for s in test_signers})
    }

    print(
        json.dumps(
            {
                "threshold_from_validation": threshold,
                "validation_eer_skilled": val_eer,
                "test": _summary(test_scores, test_signers, threshold, args.bootstrap),
                "test_per_dataset": per_dataset,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
