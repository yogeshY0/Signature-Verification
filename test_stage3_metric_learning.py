"""Focused tests for Stage 3 sampling and triplet mining."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from stage3_metric_learning import (
    PKBatchSampler,
    SignatureRecord,
    load_manifest,
    triplet_loss,
)


class ManifestTests(unittest.TestCase):
    def test_rejects_signer_overlap_with_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["image_path", "signer_id", "is_forgery", "split"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "image_path": "train.png",
                            "signer_id": "signer_a",
                            "is_forgery": "false",
                            "split": "train",
                        },
                        {
                            "image_path": "val.png",
                            "signer_id": "signer_b",
                            "is_forgery": "false",
                            "split": "val",
                        },
                        {
                            "image_path": "test.png",
                            "signer_id": "signer_a",
                            "is_forgery": "false",
                            "split": "test",
                        },
                    ]
                )

            with self.assertRaisesRegex(ValueError, "disjoint across train, val, and test"):
                load_manifest(manifest)


class PKBatchSamplerTests(unittest.TestCase):
    def test_samples_genuine_and_matching_forgery_per_signer(self) -> None:
        records = [
            SignatureRecord(Path(f"{signer}-{index}.png"), signer, False)
            for signer in ("a", "b")
            for index in range(3)
        ] + [
            SignatureRecord(Path(f"{signer}-forgery.png"), signer, True)
            for signer in ("a", "b")
        ]
        sampler = PKBatchSampler(
            records,
            signers_per_batch=2,
            genuine_per_signer=2,
            forgeries_per_signer=1,
            seed=7,
        )

        batch = next(iter(sampler))
        selected = [records[index] for index in batch]
        self.assertEqual(len(batch), 6)
        for signer in ("a", "b"):
            signer_records = [record for record in selected if record.signer_id == signer]
            self.assertEqual(sum(not record.is_forgery for record in signer_records), 2)
            self.assertEqual(sum(record.is_forgery for record in signer_records), 1)


class TripletLossTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embeddings = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.2, 0.0], [2.2, 0.0], [0.2, 0.0]],
            requires_grad=True,
        )
        self.signer_ids = torch.tensor([0, 0, 1, 1, 0])
        self.is_forgery = torch.tensor([False, False, False, False, True])

    def test_batch_hard_uses_matching_skilled_forgery_as_negative(self) -> None:
        loss, metrics = triplet_loss(
            self.embeddings,
            self.signer_ids,
            self.is_forgery,
            margin=0.3,
            mining="batch-hard",
        )
        self.assertAlmostEqual(loss.item(), 0.85, places=5)
        self.assertGreater(metrics["active_fraction"].item(), 0)
        loss.backward()
        self.assertIsNotNone(self.embeddings.grad)

    def test_semi_hard_returns_finite_loss(self) -> None:
        loss, _ = triplet_loss(
            self.embeddings,
            self.signer_ids,
            self.is_forgery,
            margin=0.3,
            mining="semi-hard",
        )
        self.assertTrue(torch.isfinite(loss).item())

    def test_requires_a_positive_per_genuine_anchor(self) -> None:
        with self.assertRaisesRegex(ValueError, "another genuine sample"):
            triplet_loss(
                torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
                torch.tensor([0, 1]),
                torch.tensor([False, False]),
            )


if __name__ == "__main__":
    unittest.main()
