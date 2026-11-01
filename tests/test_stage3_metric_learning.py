"""Focused tests for Stage 3 sampling and triplet mining."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from training.stage3_metric_learning import (
    PKBatchSampler,
    SiameseSignatureModel,
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

    def test_epoch_controls_reproducible_sampling(self) -> None:
        records = [
            SignatureRecord(Path(f"{signer}-{index}.png"), signer, False)
            for signer in "abcd"
            for index in range(3)
        ]

        def first_batch(epoch: int) -> list[int]:
            sampler = PKBatchSampler(
                records, signers_per_batch=2, genuine_per_signer=2, seed=1
            )
            sampler.set_epoch(epoch)
            return next(iter(sampler))

        self.assertEqual(first_batch(0), first_batch(0))

    def test_rejects_too_few_eligible_signers(self) -> None:
        records = [SignatureRecord(Path("a.png"), "a", False)]
        with self.assertRaisesRegex(ValueError, "need at least"):
            PKBatchSampler(records, signers_per_batch=2, genuine_per_signer=2)


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


    def test_forgery_is_never_an_anchor_or_positive(self) -> None:
        # Moving the forgery far away must not change anchors' positives: only the
        # genuine-anchor loss terms may depend on it, as a negative.
        embeddings = self.embeddings.detach().clone()
        loss_near, _ = triplet_loss(embeddings, self.signer_ids, self.is_forgery)
        embeddings[4] = torch.tensor([50.0, 0.0])
        loss_far, metrics = triplet_loss(embeddings, self.signer_ids, self.is_forgery)
        self.assertLess(loss_far.item(), loss_near.item())
        self.assertLess(metrics["mean_positive"].item(), 2.0)

    def test_semi_hard_picks_negative_within_margin_else_hardest(self) -> None:
        signer_ids = torch.tensor([0, 0, 1, 1])
        is_forgery = torch.tensor([False] * 4)
        # Anchor 0: positive at 1.0; negatives at 1.2 (semi-hard) and 0.5 (hardest).
        embeddings = torch.tensor([[0.0], [1.0], [1.2], [-0.5]])
        _, metrics = triplet_loss(
            embeddings, signer_ids, is_forgery, margin=0.3, mining="semi-hard"
        )
        _, hard_metrics = triplet_loss(
            embeddings, signer_ids, is_forgery, margin=0.3, mining="batch-hard"
        )
        self.assertGreater(
            metrics["mean_negative"].item(), hard_metrics["mean_negative"].item()
        )

    def test_rejects_unknown_mining_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported mining mode"):
            triplet_loss(
                self.embeddings, self.signer_ids, self.is_forgery, mining="random"  # type: ignore[arg-type]
            )


class SiameseModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SiameseSignatureModel(pretrained=False)

    def test_embeddings_are_unit_norm_and_batch_size_one_works(self) -> None:
        self.model.train()
        embedding = self.model(torch.randn(1, 3, 192, 384))
        self.assertEqual(tuple(embedding.shape), (1, 512))
        self.assertAlmostEqual(embedding.norm(dim=1).item(), 1.0, places=5)

    def test_projection_head_uses_layernorm_not_batchnorm(self) -> None:
        kinds = [type(layer) for layer in self.model.projection]
        self.assertEqual(
            kinds,
            [
                torch.nn.Linear,
                torch.nn.LayerNorm,
                torch.nn.ReLU,
                torch.nn.Dropout,
                torch.nn.Linear,
            ],
        )

    def test_unfreeze_last_two_stages_only(self) -> None:
        self.model.unfreeze_last_stages(count=2)
        stages = self.model.backbone.stages
        trainable = [
            any(parameter.requires_grad for parameter in stage.parameters())
            for stage in stages
        ]
        self.assertEqual(trainable, [False, False, True, True])
        self.assertTrue(
            all(parameter.requires_grad for parameter in self.model.projection.parameters())
        )

    def test_compare_returns_cosine_similarity(self) -> None:
        self.model.eval()
        image = torch.randn(2, 3, 192, 384)
        similarity = self.model.compare(image, image)
        self.assertTrue(torch.allclose(similarity, torch.ones(2), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
