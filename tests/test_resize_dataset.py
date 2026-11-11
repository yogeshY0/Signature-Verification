"""Tests for the raw-image ``resize_only`` feed mode of SignatureDataset."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from training.stage3_metric_learning import SignatureDataset, SignatureRecord


def _raw_signature(height: int, width: int, *, color: bool = True) -> np.ndarray:
    """A dark stroke on a white background at an arbitrary size."""
    if color:
        image = np.full((height, width, 3), 255, np.uint8)
        cv2.line(image, (width // 6, height // 2), (width * 5 // 6, height // 3), (0, 0, 0), 3)
        return image
    image = np.full((height, width), 255, np.uint8)
    cv2.line(image, (width // 6, height // 2), (width * 5 // 6, height // 3), 0, 3)
    return image


class ResizeOnlyDatasetTests(unittest.TestCase):
    def test_reads_any_size_and_returns_model_input(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "raw.png"
            cv2.imwrite(str(path), _raw_signature(300, 900))
            dataset = SignatureDataset(
                [SignatureRecord(path, "s1", False)], {"s1": 0}, resize_only=True
            )
            tensor, signer, is_forgery = dataset[0]
            self.assertEqual(tuple(tensor.shape), (3, 192, 384))
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertEqual(signer, 0)
            self.assertFalse(is_forgery)

    def test_handles_grayscale_and_four_channel_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            gray = Path(folder) / "g.png"
            cv2.imwrite(str(gray), _raw_signature(200, 400, color=False))
            rgba = Path(folder) / "rgba.png"
            bgr = _raw_signature(220, 700)
            alpha = np.full((220, 700, 1), 255, np.uint8)
            cv2.imwrite(str(rgba), np.concatenate([bgr, alpha], axis=2))
            datasets = [
                SignatureDataset([SignatureRecord(p, "s", False)], {"s": 0}, resize_only=True)
                for p in (gray, rgba)
            ]
            for dataset in datasets:
                self.assertEqual(tuple(dataset[0][0].shape), (3, 192, 384))

    def test_aspect_mode_pads_with_white_without_stretching(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "square.png"
            # A square image scaled to fit 192x384 leaves white padding left and right.
            cv2.imwrite(str(path), _raw_signature(200, 200, color=False))
            dataset = SignatureDataset(
                [SignatureRecord(path, "s", False)], {"s": 0}, resize_only=True
            )
            resized = dataset._read_resized(path)
            self.assertEqual(resized.shape, (192, 384))
            self.assertTrue((resized[:, 0] == 255).all())
            self.assertTrue((resized[:, -1] == 255).all())
            self.assertLess(resized[:, 192].min(), 255)

    def test_stretch_mode_fills_the_full_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "wide.png"
            cv2.imwrite(str(path), _raw_signature(100, 800, color=False))
            dataset = SignatureDataset(
                [SignatureRecord(path, "s", False)],
                {"s": 0},
                resize_only=True,
                resize_mode="stretch",
            )
            self.assertEqual(dataset._read_resized(path).shape, (192, 384))

    def test_rejects_conflicting_modes_and_unknown_resize_mode(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "s.png"
            cv2.imwrite(str(path), _raw_signature(192, 384, color=False))
            records = [SignatureRecord(path, "s", False)]
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                SignatureDataset(
                    records, {"s": 0}, already_preprocessed=True, resize_only=True
                )
            with self.assertRaisesRegex(ValueError, "unsupported resize_mode"):
                SignatureDataset(
                    records, {"s": 0}, resize_only=True, resize_mode="nearest"  # type: ignore[arg-type]
                )


class CropOnlyDatasetTests(unittest.TestCase):
    def test_crop_keeps_grayscale_tones(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "g.png"
            image = np.full((240, 600), 235, np.uint8)
            cv2.line(image, (100, 180), (500, 60), 90, 5)   # dark-gray stroke
            cv2.line(image, (120, 60), (480, 190), 150, 4)  # lighter-gray stroke
            cv2.imwrite(str(path), image)
            dataset = SignatureDataset(
                [SignatureRecord(path, "s", False)], {"s": 0}, crop_only=True
            )
            out = dataset._read_cropped(path)
            self.assertEqual(out.shape, (192, 384))
            # Not binarized: more than two distinct levels survive, and the paper
            # background (235) is not whitened to pure 255. (White padding added by
            # the aspect-preserving resize is expected, so we check the paper tone.)
            self.assertGreater(len(np.unique(out)), 2)
            self.assertGreater(int(((out >= 230) & (out <= 240)).sum()), 0)

    def test_crop_fills_more_of_the_frame_than_resize_only(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frame.png"
            image = np.full((200, 1000), 255, np.uint8)
            cv2.rectangle(image, (420, 80), (580, 120), 0, -1)  # small ink in a big frame
            cv2.imwrite(str(path), image)
            record = SignatureRecord(path, "s", False)
            cropped = SignatureDataset([record], {"s": 0}, crop_only=True)._read_cropped(path)
            plain = SignatureDataset([record], {"s": 0}, resize_only=True)._read_resized(path)
            self.assertGreater(int((cropped < 128).sum()), int((plain < 128).sum()))

    def test_crop_mode_is_mutually_exclusive_with_other_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "s.png"
            cv2.imwrite(str(path), _raw_signature(192, 384, color=False))
            records = [SignatureRecord(path, "s", False)]
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                SignatureDataset(records, {"s": 0}, crop_only=True, resize_only=True)


if __name__ == "__main__":
    unittest.main()
