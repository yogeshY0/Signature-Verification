"""Tests for training-time augmentation and its use in SignatureDataset."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from preprocessing.augmentation import augment_signature
from training.stage3_metric_learning import SignatureDataset, SignatureRecord


def _signature() -> np.ndarray:
    image = np.full((192, 384), 255, np.uint8)
    cv2.line(image, (100, 140), (280, 50), 0, 3)
    cv2.line(image, (120, 60), (260, 130), 0, 3)
    return image


class AugmentationTests(unittest.TestCase):
    def test_keeps_shape_dtype_and_binary_values(self) -> None:
        random.seed(0)
        np.random.seed(0)
        for _ in range(50):
            out = augment_signature(_signature())
            self.assertEqual(out.shape, (192, 384))
            self.assertEqual(out.dtype, np.uint8)
            self.assertTrue(set(np.unique(out)) <= {0, 255})
            self.assertGreater((out == 0).sum(), 100)

    def test_grayscale_input_stays_grayscale(self) -> None:
        random.seed(1)
        image = _signature()
        image[image == 0] = 90
        out = augment_signature(image)
        self.assertIn(90, set(np.unique(out)) | {90})
        self.assertGreater(len(np.unique(out)), 2)

    def test_produces_varied_outputs_and_is_reproducible(self) -> None:
        random.seed(2)
        np.random.seed(2)
        first = [augment_signature(_signature()) for _ in range(5)]
        self.assertGreater(len({a.tobytes() for a in first}), 1)
        random.seed(2)
        np.random.seed(2)
        second = [augment_signature(_signature()) for _ in range(5)]
        for a, b in zip(first, second):
            self.assertTrue(np.array_equal(a, b))

    def test_rejects_color_input(self) -> None:
        with self.assertRaises(ValueError):
            augment_signature(np.zeros((192, 384, 3), np.uint8))

    def test_dataset_augments_only_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "s.png"
            cv2.imwrite(str(path), _signature())
            record = SignatureRecord(path, "s1", False)
            plain = SignatureDataset([record], {"s1": 0}, already_preprocessed=True)
            augmented = SignatureDataset(
                [record], {"s1": 0}, already_preprocessed=True, augment=True
            )
            base = plain[0][0]
            self.assertTrue(np.array_equal(base.numpy(), plain[0][0].numpy()))
            random.seed(3)
            self.assertFalse(np.array_equal(base.numpy(), augmented[0][0].numpy()))


if __name__ == "__main__":
    unittest.main()
