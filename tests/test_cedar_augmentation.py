"""Tests for the raw-image CEDAR augmentation (preprocessing/augment_cedar.py)."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from preprocessing.augment_cedar import (
    AUGMENT_TYPES,
    augment_raw,
    augmented_name,
)


def _signature() -> np.ndarray:
    image = np.full((200, 400), 255, np.uint8)
    cv2.line(image, (90, 160), (320, 60), 0, 4)
    cv2.line(image, (110, 70), (300, 150), 0, 4)
    return image


def _bbox(image: np.ndarray) -> tuple[int, int]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray < 128)
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


class CedarAugmentationTests(unittest.TestCase):
    def test_preserves_shape_dtype_and_channels(self) -> None:
        for base in (_signature(), cv2.cvtColor(_signature(), cv2.COLOR_GRAY2BGR)):
            for kind in AUGMENT_TYPES:
                out = augment_raw(base, kind, np.random.default_rng(0))
                self.assertEqual(out.shape, base.shape)
                self.assertEqual(out.dtype, np.uint8)

    def test_is_deterministic_per_seed(self) -> None:
        base = _signature()
        for kind in AUGMENT_TYPES:
            first = augment_raw(base, kind, np.random.default_rng(123))
            second = augment_raw(base, kind, np.random.default_rng(123))
            self.assertTrue(np.array_equal(first, second))
            different = augment_raw(base, kind, np.random.default_rng(124))
            self.assertFalse(np.array_equal(first, different))

    def test_does_not_rotate_shear_or_scale(self) -> None:
        # No rotation / shear / scaling means the ink bounding box keeps its aspect
        # ratio and its size changes only by a few pixels.
        base = _signature()
        base_w, base_h = _bbox(base)
        for kind in AUGMENT_TYPES:
            for seed in range(8):
                out = augment_raw(base, kind, np.random.default_rng(seed))
                width, height = _bbox(out)
                self.assertAlmostEqual(width / height, base_w / base_h, delta=0.12)
                self.assertLessEqual(abs(width - base_w), 10)
                self.assertLessEqual(abs(height - base_h), 10)

    def test_shift_is_pure_translation(self) -> None:
        base = cv2.cvtColor(_signature(), cv2.COLOR_GRAY2BGR)
        out = augment_raw(base, "shift", np.random.default_rng(5))
        self.assertEqual(out.shape, base.shape)
        self.assertEqual(_bbox(out), _bbox(base))  # translation only

    def test_shift_keeps_background_white(self) -> None:
        base = _signature()
        out = augment_raw(base, "shift", np.random.default_rng(0))
        self.assertEqual(set(np.unique(out)) - {255, 0}, set())

    def test_augmented_name(self) -> None:
        self.assertEqual(
            augmented_name("thickness", 2, "original_1_1"),
            "augmented_thickness_2_original_1_1.png",
        )


if __name__ == "__main__":
    unittest.main()
