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
def _embed(
    model,
    records,
    already_preprocessed,
    device,
    batch_size,
    workers,
    *,
    resize_only: bool = False,
    crop_only: bool = False,
    resize_mode: str = "aspect",
) -> np.ndarray:
    dataset = SignatureDataset(
        records,
        {r.signer_id: 0 for r in records},
        already_preprocessed=already_preprocessed,
        resize_only=resize_only,
        crop_only=crop_only,
        resize_mode=resize_mode,  # type: ignore[arg-type]
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
    """Return (FAR, FRR) at ``threshold``; ``nan`` when a side has no scores."""
    frr = float((genuine < threshold).mean()) if len(genuine) else float("nan")
    far = float((negatives >= threshold).mean()) if len(negatives) else float("nan")
    return far, frr


def _accuracy(genuine, negatives, threshold) -> float:
    """Correct decisions / all decisions over genuine and negative queries only."""
    total = len(genuine) + len(negatives)
    if total == 0:
        return float("nan")
    correct = (genuine >= threshold).sum() + (negatives < threshold).sum()
    return float(correct / total)


def _eer_threshold(genuine, negatives):
    """Threshold where FAR and FRR are closest, and the EER there (sort-based, O(n log n)).

    Returns ``(nan, nan)`` when either side is empty (e.g. a genuine-only dataset with
    no forgeries), so callers can fall back to another negative set.
    """
    if len(genuine) == 0 or len(negatives) == 0:
        return float("nan"), float("nan")
    candidates = np.unique(np.concatenate([genuine, negatives]))
    gen_sorted, neg_sorted = np.sort(genuine), np.sort(negatives)
    frr = np.searchsorted(gen_sorted, candidates, side="left") / len(genuine)
    far = 1 - np.searchsorted(neg_sorted, candidates, side="left") / len(negatives)
    best = int(np.argmin(np.abs(far - frr)))
    return float(candidates[best]), float((far[best] + frr[best]) / 2)


def _choose_negatives(genuine, skilled, random, mode: str = "auto"):
    """Pick the negative score set used for EER / accuracy.

    ``skilled`` uses skilled forgeries, ``random`` uses genuine signatures of other
    signers, and ``auto`` uses skilled forgeries when the evaluation set has any and
    otherwise falls back to random pairs, so a genuine-only dataset can still be
    scored (random forgeries).
    """
    if mode == "skilled":
        return skilled
    if mode == "random":
        return random
    if mode != "auto":
        raise ValueError(f"unsupported negatives mode: {mode}")
    return skilled if len(skilled) else random


def _metrics(genuine, skilled, random, threshold, negatives_mode: str = "auto") -> dict[str, float]:
    """Metrics against skilled forgeries, random pairs, or the selected negative set.

    ``negatives_mode`` selects which set the headline keys (``accuracy``, ``eer``,
    ``balanced_accuracy``, ``far_at_threshold``) use. The skilled- and random-specific
    keys are kept for backwards compatibility and are ``nan`` when that set is empty.
    """
    source = negatives_mode
    if source == "auto":
        source = "skilled" if len(skilled) else "random"
    if source not in {"skilled", "random"}:
        raise ValueError(f"unsupported negatives mode: {negatives_mode}")
    negatives = skilled if source == "skilled" else random

    far_negative, frr = _rates(genuine, negatives, threshold)
    far_skilled, _ = _rates(genuine, skilled, threshold)
    far_random, _ = _rates(genuine, random, threshold)
    return {
        # Negative set actually used, and the headline metrics against it.
        "negatives_source": source,
        "accuracy": _accuracy(genuine, negatives, threshold),
        "balanced_accuracy": ((1 - frr) + (1 - far_negative)) / 2,
        "eer": _eer_threshold(genuine, negatives)[1],
        "far_at_threshold": far_negative,
        # Backwards-compatible skilled / random specific metrics.
        "accuracy_skilled": _accuracy(genuine, skilled, threshold),
        "balanced_accuracy_skilled": ((1 - frr) + (1 - far_skilled)) / 2,
        "eer_skilled": _eer_threshold(genuine, skilled)[1],
        "eer_random": _eer_threshold(genuine, random)[1],
        "far_skilled_at_threshold": far_skilled,
        "far_random_at_threshold": far_random,
        "frr_at_threshold": frr,
    }


def _bootstrap(
    scores, signers, threshold, resamples, dataset_only=False, negatives_mode="auto", seed=0
):
    """95% intervals from resampling signers with replacement."""
    if resamples <= 0:
        return None
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        chosen = [signers[i] for i in rng.integers(0, len(signers), len(signers))]
        genuine, skilled, random = _pool(scores, chosen, dataset_only)
        if len(genuine) and len(_choose_negatives(genuine, skilled, random, negatives_mode)):
            draws.append(_metrics(genuine, skilled, random, threshold, negatives_mode))
    if not draws:
        return None
    return {
        key: [float(np.percentile([d[key] for d in draws], q)) for q in (2.5, 97.5)]
        for key in draws[0]
    }


def _summary(scores, signers, threshold, resamples, dataset_only=False, negatives_mode="auto") -> dict:
    genuine, skilled, random = _pool(scores, signers, dataset_only)
    return {
        "signers": len(signers),
        **_metrics(genuine, skilled, random, threshold, negatives_mode),
        "genuine_queries": len(genuine),
        "skilled_forgeries": len(skilled),
        "random_pairs": len(random),
        "confidence_95": _bootstrap(
            scores, signers, threshold, resamples, dataset_only, negatives_mode
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--already-preprocessed", action="store_true")
    parser.add_argument(
        "--resize-only",
        action="store_true",
        help="Feed raw images directly: read any size/channels, grayscale, resize to "
        "192x384 (no Stage 2.2 crop/whitening/binarization).",
    )
    parser.add_argument(
        "--crop-grayscale",
        action="store_true",
        help="Add the ink-crop step but keep grayscale: grayscale, crop to the ink box, "
        "resize to 192x384. No background whitening and no binarization.",
    )
    parser.add_argument(
        "--resize-mode",
        choices=("aspect", "stretch"),
        default="aspect",
        help="Resize strategy for --resize-only: 'aspect' preserves the aspect ratio and "
        "pads with white; 'stretch' forces 192x384.",
    )
    parser.add_argument(
        "--negatives",
        choices=("auto", "skilled", "random"),
        default="auto",
        help="Negative score set for EER/accuracy: 'skilled' forgeries, 'random' genuine "
        "signatures of other signers, or 'auto' (forgeries when present, else random). "
        "Genuine-only datasets need 'auto' or 'random'.",
    )
    parser.add_argument("--references", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--threshold",
        type=float,
        help="Fixed cosine-similarity threshold. Default: the validation skilled-EER point.",
    )
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
            model,
            records,
            args.already_preprocessed,
            args.device,
            args.batch_size,
            args.workers,
            resize_only=args.resize_only,
            crop_only=args.crop_grayscale,
            resize_mode=args.resize_mode,
        )
        results[name] = _scores(records, embeddings, args.references)

    # Threshold from validation signers. "auto" uses skilled forgeries when the
    # validation set has any and otherwise random (different-signer) pairs.
    val_signers = sorted(results["val"])
    val_gen, val_skilled, val_random = _pool(results["val"], val_signers)
    val_negatives = _choose_negatives(val_gen, val_skilled, val_random, args.negatives)
    eer_threshold, val_eer = _eer_threshold(val_gen, val_negatives)
    threshold = eer_threshold if args.threshold is None else args.threshold

    test_scores = results["test"]
    test_signers = sorted(test_scores)
    per_dataset = {
        name: _summary(
            test_scores,
            [s for s in test_signers if _dataset(s) == name],
            threshold,
            args.bootstrap,
            dataset_only=True,
            negatives_mode=args.negatives,
        )
        for name in sorted({_dataset(s) for s in test_signers})
    }

    print(
        json.dumps(
            {
                "negatives": args.negatives,
                "threshold_from_validation": eer_threshold,
                "threshold_used": threshold,
                "threshold_source": "validation_eer" if args.threshold is None else "user",
                "validation_eer": val_eer,
                # Kept for backwards compatibility; equals `validation_eer` above.
                "validation_eer_skilled": val_eer,
                "test": _summary(
                    test_scores, test_signers, threshold, args.bootstrap, negatives_mode=args.negatives
                ),
                "test_per_dataset": per_dataset,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
