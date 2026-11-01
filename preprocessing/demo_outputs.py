"""Demo: preprocess sample JPEGs and save grayscale and np.repeat RGB copies.

Usage (from the repository root):
    python -m preprocessing.demo_outputs
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from preprocessing.signature_preprocessing import preprocess_signature


def save_test_outputs(
    image_paths: tuple[str, ...] = ("sample_images/images.jpeg", "sample_images/images (1).jpeg"),
    output_dir: str = "preprocessed_outputs",
) -> None:
    """Preprocess JPEG signatures and save grayscale and repeated RGB outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read image: {image_path}")

        grayscale = preprocess_signature(image)
        repeated_rgb = np.repeat(grayscale[..., None], 3, axis=2)
        stem = Path(image_path).stem.replace(" ", "_")

        for suffix, array in (("grayscale", grayscale), ("rgb_repeat", repeated_rgb)):
            target = output_path / f"{stem}_{suffix}.png"
            if not cv2.imwrite(str(target), array):
                raise OSError(f"could not write output: {target}")
        print(f"{image_path}: grayscale={grayscale.shape}, rgb_repeat={repeated_rgb.shape}")


if __name__ == "__main__":
    save_test_outputs()
