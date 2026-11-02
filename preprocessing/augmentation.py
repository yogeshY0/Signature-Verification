"""Training-time augmentation for preprocessed signature images.

Applied to 192x384 uint8 dark-ink-on-white images, after preprocessing and before
``to_model_input``. It applies:
- rotation up to +-5 degrees
- shear
- 0.9-1.1 scale
- shifts up to 5%
- occasional thicker or thinner strokes
- light noise
- no flips
- binary images stay strictly 0/255

All changes are mild, because slant and proportions are part of a signer's identity.
Randomness comes from the ``random`` and ``numpy.random`` modules, which PyTorch
seeds separately in every DataLoader worker.
"""

from __future__ import annotations

import random

import cv2
import numpy as np


def augment_signature(
    image: np.ndarray,
    *,
    max_rotation: float = 5.0,
    max_shear: float = 0.10,
    scale_range: tuple[float, float] = (0.90, 1.10),
    max_shift: float = 0.05,
    thickness_prob: float = 0.3,
    noise_prob: float = 0.2,
) -> np.ndarray:
    """Return a randomly perturbed copy of a single-channel uint8 signature image."""
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("expected a single-channel uint8 image")

    height, width = image.shape
    # Binary inputs (only 0 and 255) are re-thresholded so they stay binary.
    binary = bool(((image == 0) | (image == 255)).all())

    angle = np.deg2rad(random.uniform(-max_rotation, max_rotation))
    shear = random.uniform(-max_shear, max_shear)
    scale = random.uniform(*scale_range)
    shift_x = random.uniform(-max_shift, max_shift) * width
    shift_y = random.uniform(-max_shift, max_shift) * height

    # Compose rotation, shear and scale about the image center, then translate.
    cos, sin = np.cos(angle), np.sin(angle)
    linear = scale * np.array([[cos, -sin + shear], [sin, cos]])
    center = np.array([width / 2, height / 2])
    offset = center - linear @ center + np.array([shift_x, shift_y])
    matrix = np.hstack([linear, offset[:, None]]).astype(np.float32)
    out = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    if random.random() < thickness_prob:
        kernel = np.ones((2, 2), np.uint8)
        # Ink is dark, so erosion thickens strokes and dilation thins them.
        out = cv2.erode(out, kernel) if random.random() < 0.5 else cv2.dilate(out, kernel)

    if random.random() < noise_prob:
        noise = np.random.normal(0, 8, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if binary:
        out = np.where(out < 128, 0, 255).astype(np.uint8)
    return out
