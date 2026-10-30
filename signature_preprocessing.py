"""Stage 2 preprocessing step: grayscale normalization + aspect-aware resize.

Input:  a signature image (grayscale or BGR, any size, with a white or whitish
        background), i.e. the output of the upstream YOLO crop and artifact
        cleaning.
Output: a fixed-size ``(height, width)`` uint8 grayscale image, dark ink on a
        white background, at the agreed wide aspect ratio. Stroke intensity is
        preserved and strokes are never stretched.

``preprocess_signature`` runs the whole step; the individual functions are exposed for
testing and ablations.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

# Agreed input size (width, height): 2:1 and a multiple of 32 (ConvNeXt total stride).
DEFAULT_TARGET_SIZE = (384, 192)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

InkPolarity = Literal["auto", "light", "dark"]  # retained for CLI compatibility


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image is None or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a non-empty 2D or 3D array")
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8")
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[..., 0]
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("image must have 1 or 3 channels")


def flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Whiten slowly varying background (shadows, lighting gradients).

    The input must have dark ink on a lighter background. The background is
    estimated by morphological closing with a kernel larger than the stroke
    width, then smoothed. Returns an image where the background is ~255 and
    grayscale ink contrast is preserved.
    """
    k = max(15, (min(gray.shape[:2]) // 40) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), k / 2)

    # Subtract the estimated background variation without quantizing the ink.
    # Pixels matching the local background are explicitly made pure white.
    background_gap = background.astype(np.int16) - gray.astype(np.int16)
    corrected = np.clip(255 - background_gap, 0, 255).astype(np.uint8)
    corrected[background_gap <= 8] = 255
    return corrected


def to_grayscale(
    image: np.ndarray,
    ink: InkPolarity = "auto",
    blur_ksize: int = 0,
    remove_shadow: bool = True,
) -> np.ndarray:
    """Normalize to grayscale dark ink on a white background without binarizing.

    Args:
        image: Grayscale or BGR uint8 image.
        ink: Retained for compatibility with existing callers. The preprocessing
            assumes dark ink on a white or whitish background and does not inspect
            or invert ink polarity.
        blur_ksize: Odd Gaussian kernel size applied before resizing. ``0``
            disables it so pressure-dependent intensity is retained.
        remove_shadow: Flatten uneven illumination while preserving grayscale
            stroke contrast.
    """
    gray = _to_gray(image).copy()
    # Ink polarity is intentionally not checked or inverted. Inputs are expected
    # to have dark signatures on a white or whitish background.
    if remove_shadow:
        gray = flatten_illumination(gray)
    else:
        # Also support callers that skip illumination correction while removing
        # small residual tint from an otherwise white background.
        gray[gray >= 245] = 255
    if blur_ksize:
        if blur_ksize % 2 == 0:
            raise ValueError("blur_ksize must be odd")
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    return gray

def crop_to_ink(
    image: np.ndarray,
    margin: float = 0.05,
    ink_threshold: int = 128,
    min_ink_pixels: int = 5,
) -> np.ndarray:
    """Crop a grayscale dark-ink-on-white image to the ink box plus a margin.

    Args:
        image: Single-channel uint8 image with dark ink on a white background.
        margin: Padding around the ink box, as a fraction of the box's larger side.
        ink_threshold: Pixels below this value count as ink for locating the
            signature. The original grayscale values are returned unchanged.
        min_ink_pixels: If fewer ink pixels are found the image is treated as empty
            and ``ValueError`` is raised, so the caller can route it to review.
    """
    mask = image < ink_threshold
    if int(mask.sum()) < min_ink_pixels:
        raise ValueError("no signature ink found in image")

    ys, xs = np.where(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad = int(round(margin * max(x1 - x0, y1 - y0)))
    h, w = image.shape[:2]
    return image[max(0, y0 - pad) : min(h, y1 + pad), max(0, x0 - pad) : min(w, x1 + pad)]


def aspect_aware_resize(
    image: np.ndarray,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    pad_value: int = 255,
) -> np.ndarray:
    """Scale ``image`` to fit inside ``target_size`` and pad it to the exact size.

    The aspect ratio is preserved: the image is scaled uniformly, centered, and the
    remaining area is filled with ``pad_value`` (255 for normalized grayscale
    images).

    Args:
        image: ``H x W`` or ``H x W x C`` uint8 array.
        target_size: ``(width, height)`` of the output canvas.
        pad_value: Background value used for padding.

    Returns:
        Array of shape ``(height, width)`` (2D input) or ``(height, width, C)``.
    """
    if image is None or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a non-empty 2D or 3D array")

    target_w, target_h = target_size
    h, w = image.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = max(1, min(target_w, round(w * scale)))
    new_h = max(1, min(target_h, round(h * scale)))

    # INTER_AREA avoids aliasing of thin strokes when shrinking.
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    canvas_shape = (target_h, target_w) + image.shape[2:]
    canvas = np.full(canvas_shape, pad_value, dtype=image.dtype)

    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def preprocess_signature(
    image: np.ndarray,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    ink: InkPolarity = "auto",
    margin: float = 0.05,
    blur_ksize: int = 0,
) -> np.ndarray:
    """Grayscale normalization, tight crop, and aspect-aware resize.

    The threshold is used only to locate the signature for cropping. The output
    remains grayscale, so stroke and pressure variations are preserved.

    Returns:
        ``target_h x target_w`` uint8 grayscale image, dark ink on white.

    Raises:
        ValueError: If the input is invalid or contains no ink.
    """
    grayscale = to_grayscale(image, ink=ink, blur_ksize=blur_ksize)
    cropped = crop_to_ink(grayscale, margin=margin)
    return aspect_aware_resize(cropped, target_size, pad_value=255)


def to_model_input(image: np.ndarray) -> np.ndarray:
    """Convert a preprocessed uint8 image to a ``3 x H x W`` float32 ImageNet-normalized array.

    The single channel is replicated to 3 channels so ImageNet-pretrained weights can
    be used unchanged.
    """
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError("expected the single-channel output of preprocess_signature")

    array = np.repeat(image[..., None], 3, axis=2).astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(array.transpose(2, 0, 1))
