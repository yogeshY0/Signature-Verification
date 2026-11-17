"""Tests for the EER/AUC checkpoint-selection metrics and their helpers."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from training.evaluate import _roc_auc, reference_metrics
from training.stage3_metric_learning import SignatureRecord
from training.train_siamese import (
    _can_score,
    _improves,
    _reference_selection_records,
)


def _records(signers: list[str], genuine: int, forgery: int) -> list[SignatureRecord]:
    records: list[SignatureRecord] = []
    for signer in signers:
        for index in range(genuine):
            records.append(SignatureRecord(Path(f"{signer}_g{index}.png"), signer, False))
        for index in range(forgery):
            records.append(SignatureRecord(Path(f"{signer}_f{index}.png"), signer, True))
    return records


def _one_hot(records: list[SignatureRecord], *, forgery_to_other_signer: bool) -> np.ndarray:
    """One direction per signer; forgeries either copy it or point at a neighbour."""
    order = sorted({record.signer_id for record in records})
    index = {signer: position for position, signer in enumerate(order)}
    vectors = np.zeros((len(records), len(order)), dtype=np.float32)
    for row, record in enumerate(records):
        position = index[record.signer_id]
        if record.is_forgery and forgery_to_other_signer:
            position = (position + 1) % len(order)
        vectors[row, position] = 1.0
    return vectors


class RocAucTests(unittest.TestCase):
    def test_nan_when_a_side_is_empty(self) -> None:
        self.assertTrue(math.isnan(_roc_auc(np.zeros(0), np.ones(3))))
        self.assertTrue(math.isnan(_roc_auc(np.ones(3), np.zeros(0))))

    def test_perfect_and_chance_separation(self) -> None:
        genuine = np.array([1.0, 1.0])
        self.assertEqual(_roc_auc(genuine, np.array([0.0, 0.0])), 1.0)
        self.assertEqual(_roc_auc(genuine, np.array([1.0, 1.0])), 0.5)


class ReferenceMetricsTests(unittest.TestCase):
    def test_separable_forgeries_give_zero_eer(self) -> None:
        records = _records(["a", "b", "c", "d"], genuine=4, forgery=2)
        # Forgeries imitate a different signer, so they score far from their own refs.
        embeddings = _one_hot(records, forgery_to_other_signer=True)

        metrics = reference_metrics(records, embeddings, references=2)

        self.assertEqual(metrics["negatives_source"], "skilled")
        self.assertAlmostEqual(metrics["eer"], 0.0, places=6)
        self.assertAlmostEqual(metrics["auc"], 1.0, places=6)

    def test_indistinguishable_forgeries_give_chance_eer(self) -> None:
        records = _records(["a", "b", "c", "d"], genuine=4, forgery=2)
        # Forgeries copy their signer's direction, so skilled negatives are unscorable.
        embeddings = _one_hot(records, forgery_to_other_signer=False)

        metrics = reference_metrics(records, embeddings, references=2)

        self.assertAlmostEqual(metrics["eer"], 0.5, places=6)
        self.assertAlmostEqual(metrics["auc"], 0.5, places=6)

    def test_negatives_mode_selects_the_negative_set(self) -> None:
        records = _records(["a", "b", "c", "d"], genuine=4, forgery=2)
        embeddings = _one_hot(records, forgery_to_other_signer=False)

        auto = reference_metrics(records, embeddings, references=2, negatives_mode="auto")
        random = reference_metrics(records, embeddings, references=2, negatives_mode="random")

        self.assertEqual(auto["negatives_source"], "skilled")
        self.assertAlmostEqual(auto["eer"], 0.5, places=6)
        # Other signers stay far apart, so random pairs are perfectly separable.
        self.assertAlmostEqual(random["eer"], 0.0, places=6)
        self.assertAlmostEqual(random["auc"], 1.0, places=6)


class SelectionHelperTests(unittest.TestCase):
    def test_reference_records_keep_only_signers_with_spare_genuine(self) -> None:
        records = _records(["a"], genuine=3, forgery=0) + _records(["b"], genuine=2, forgery=0)

        eligible, indices = _reference_selection_records(records, references=2)

        self.assertEqual([record.signer_id for record in eligible], ["a", "a", "a"])
        self.assertEqual(indices, [0, 1, 2])

    def test_can_score_requires_negatives(self) -> None:
        two_signers = _records(["a"], genuine=3, forgery=0) + _records(["b"], genuine=3, forgery=0)
        self.assertTrue(_can_score(two_signers, references=2))

        one_signer_genuine_only = _records(["a"], genuine=3, forgery=0)
        self.assertFalse(_can_score(one_signer_genuine_only, references=2))

        one_signer_with_forgery = _records(["a"], genuine=3, forgery=1)
        self.assertTrue(_can_score(one_signer_with_forgery, references=2, negatives_mode="auto"))
        self.assertTrue(_can_score(one_signer_with_forgery, references=2, negatives_mode="skilled"))
        self.assertFalse(_can_score(one_signer_with_forgery, references=2, negatives_mode="random"))

    def test_can_score_rejects_too_few_genuine_per_signer(self) -> None:
        records = _records(["a"], genuine=2, forgery=0) + _records(["b"], genuine=2, forgery=0)
        self.assertFalse(_can_score(records, references=2))

    def test_improves_respects_metric_direction(self) -> None:
        self.assertTrue(_improves("loss", 0.2, 0.3))
        self.assertFalse(_improves("loss", 0.4, 0.3))
        self.assertTrue(_improves("eer", 0.1, 0.2))
        self.assertFalse(_improves("eer", 0.3, 0.2))
        self.assertTrue(_improves("auc", 0.9, 0.8))
        self.assertFalse(_improves("auc", 0.7, 0.8))

    def test_improves_ignores_nan(self) -> None:
        self.assertFalse(_improves("loss", float("nan"), 0.3))
        self.assertFalse(_improves("eer", float("nan"), 0.2))
        self.assertFalse(_improves("auc", float("nan"), 0.8))


if __name__ == "__main__":
    unittest.main()
