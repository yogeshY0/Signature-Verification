"""Tests for dataset-scoped preprocessing (preprocessing/preprocess_dataset.py)."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from preprocessing.preprocess_dataset import process


def _write_signature(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((120, 300, 3), 255, np.uint8)
    cv2.line(image, (60, 90), (240, 40), (0, 0, 0), 4)
    cv2.imwrite(str(path), image)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_path", "signer_id", "is_forgery", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)


class DatasetScopedPreprocessTests(unittest.TestCase):
    def _setup(self, root: Path) -> Path:
        _write_signature(root / "raw/cedar/full_org/original_1_1.png")
        _write_signature(root / "raw/cedar/full_forg/forgeries_1_1.png")
        _write_signature(root / "raw/other/genuine/a.png")
        manifest = root / "manifest.csv"
        _write_manifest(
            manifest,
            [
                {
                    "image_path": "raw/cedar/full_org/original_1_1.png",
                    "signer_id": "cedar_001",
                    "is_forgery": "false",
                    "split": "train",
                },
                {
                    "image_path": "raw/cedar/full_forg/forgeries_1_1.png",
                    "signer_id": "cedar_001",
                    "is_forgery": "true",
                    "split": "train",
                },
                {
                    "image_path": "raw/other/genuine/a.png",
                    "signer_id": "other_001",
                    "is_forgery": "false",
                    "split": "train",
                },
            ],
        )
        return manifest

    def test_selecting_one_dataset_keeps_the_others_as_is(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._setup(root)
            out = root / "processed"

            # First pass processes only the "other" dataset.
            process(manifest, out, datasets={"other"})
            self.assertTrue((out / "other/genuine/a.png").exists())
            self.assertFalse((out / "cedar").exists())

            # Second pass processes only cedar and must preserve the "other" row.
            process(manifest, out, datasets={"cedar"})
            self.assertTrue((out / "cedar/full_org/original_1_1.png").exists())
            self.assertTrue((out / "cedar/full_forg/forgeries_1_1.png").exists())

            with (out / "manifest.csv").open(newline="") as handle:
                paths = {row["image_path"] for row in csv.DictReader(handle)}
            self.assertIn("other/genuine/a.png", paths)
            self.assertIn("cedar/full_org/original_1_1.png", paths)
            self.assertIn("cedar/full_forg/forgeries_1_1.png", paths)

    def test_output_is_binary_192x384(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._setup(root)
            out = root / "processed"
            process(manifest, out, datasets={"cedar"})
            image = cv2.imread(
                str(out / "cedar/full_org/original_1_1.png"), cv2.IMREAD_UNCHANGED
            )
            self.assertEqual(image.shape, (192, 384))
            self.assertTrue(set(np.unique(image)) <= {0, 255})

    def test_prune_removes_stale_files_in_the_selected_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._setup(root)
            out = root / "processed"
            process(manifest, out, datasets={"cedar"})
            stale = out / "cedar/full_org/original_7_1.png"
            stale.write_bytes(b"stale")
            process(manifest, out, datasets={"cedar"}, prune=True)
            self.assertFalse(stale.exists())
            self.assertTrue((out / "cedar/full_org/original_1_1.png").exists())


if __name__ == "__main__":
    unittest.main()
