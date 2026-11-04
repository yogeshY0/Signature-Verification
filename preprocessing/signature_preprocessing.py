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

def remove_border_artifacts(
    gray: np.ndarray,
    ink_threshold: int = 200,
    max_thickness: float = 0.03,
    min_elongation: float = 5.0,
    border_zone: float = 0.10,
) -> np.ndarray:
    """Whiten thin straight ink lines that lie in the border zone of the image.

    Scan edges, page shadows and ruled lines show up as long thin lines near the
    border. They are not part of the signature but would stretch the crop box.
    Binarization often breaks them into pieces that no longer touch the border, so a
    component is removed when it lies within ``border_zone`` (fraction of the image
    size) of an edge, is at most ``max_thickness`` of the image's short side thick, is
    at least ``min_elongation`` times longer than thick, and is fairly solid (a
    straight line fills its bounding box; handwriting strokes do not).
    """
    mask = (gray < ink_threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = gray.shape[:2]
    limit = max_thickness * min(h, w)
    cleaned = gray.copy()
    for label in range(1, count):
        x, y, bw, bh, area = stats[label]
        thin, long_ = min(bw, bh), max(bw, bh)
        near_border = (
            x <= border_zone * w
            or x + bw >= (1 - border_zone) * w
            or y <= border_zone * h
            or y + bh >= (1 - border_zone) * h
        )
        straight = area >= 0.3 * bw * bh
        if near_border and thin <= limit and long_ >= min_elongation * thin and straight:
            cleaned[labels == label] = 255
    return _remove_attached_border_lines(cleaned, ink_threshold, max_thickness)


def _remove_attached_border_lines(
    gray: np.ndarray,
    ink_threshold: int,
    max_thickness: float,
    zone: float = 0.06,
    min_length: float = 0.30,
    edge_band: float = 0.03,
    min_fragment_elongation: float = 4.0,
) -> np.ndarray:
    """Erase long straight runs near the image edge even when joined to the writing.

    Edge lines are often bridged to the signature by a few pixels, so they form one
    big component and the component test cannot isolate them. Here a horizontal (or
    vertical) opening with a long kernel keeps only runs at least ``min_length`` of the
    image width (height) long, restricted to the strip within ``zone`` of the matching
    edges. The kernel is long so ordinary strokes and stems are never matched; thick
    results are kept too. Leftover fragments lying entirely in the outermost
    ``edge_band`` are removed only if they are elongated along that edge, so writing
    that merely touches the edge is kept.
    """
    h, w = gray.shape[:2]
    mask = (gray < ink_threshold).astype(np.uint8)
    limit = max_thickness * min(h, w)
    lines = np.zeros_like(mask)
    for horizontal in (True, False):
        length = max(3, int(round(min_length * (w if horizontal else h))))
        kernel = np.ones((1, length) if horizontal else (length, 1), np.uint8)
        runs = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        strip = np.zeros_like(mask)
        if horizontal:
            strip[: int(zone * h) + 1] = 1
            strip[int((1 - zone) * h) :] = 1
        else:
            strip[:, : int(zone * w) + 1] = 1
            strip[:, int((1 - zone) * w) :] = 1
        runs = runs & strip
        count, labels, stats, _ = cv2.connectedComponentsWithStats(runs, connectivity=8)
        for label in range(1, count):
            bw, bh = stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
            if min(bw, bh) <= limit:
                lines[labels == label] = 1

    if lines.any():
        # Dilate to catch the anti-aliased edge of the line without touching far strokes.
        lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
        mask = mask & (1 - lines)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    band_y, band_x = edge_band * h, edge_band * w
    remove = np.zeros(count, bool)
    for label in range(1, count):
        x, y, bw, bh, _ = stats[label]
        horizontal_edge = y + bh <= band_y or y >= h - band_y
        vertical_edge = x + bw <= band_x or x >= w - band_x
        remove[label] = (horizontal_edge and bw >= min_fragment_elongation * bh) or (
            vertical_edge and bh >= min_fragment_elongation * bw
        )
    cleaned = gray.copy()
    cleaned[(lines > 0) | remove[labels]] = 255
    return cleaned


def _drop_distant_small_marks(
    stats: np.ndarray,
    labels: list[int],
    scale: int,
    small_fraction: float = 0.02,
    max_gap: float = 0.08,
) -> list[int]:
    """Drop small ink components that sit far from the rest of the writing.

    A component is a stray mark if its ink is under ``small_fraction`` of all ink and the
    gap between its box and the nearest other component's box exceeds ``max_gap`` of the
    image's larger side. Dots and marks near the writing (i-dots, punctuation) are kept.
    """
    areas = {i: int(stats[i, cv2.CC_STAT_AREA]) for i in labels}
    total = sum(areas.values())

    def box(i: int) -> tuple[int, int, int, int]:
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        return x, y, x + int(stats[i, cv2.CC_STAT_WIDTH]), y + int(stats[i, cv2.CC_STAT_HEIGHT])

    def gap(a: int, b: int) -> float:
        ax0, ay0, ax1, ay1 = box(a)
        bx0, by0, bx1, by1 = box(b)
        return float(max(0, bx0 - ax1, ax0 - bx1, by0 - ay1, ay0 - by1))

    strays = {i for i in labels if areas[i] < small_fraction * total}
    anchors = [i for i in labels if i not in strays]
    if not anchors:
        return labels
    keep = list(anchors)
    for i in strays:
        if min(gap(i, j) for j in anchors) <= max_gap * scale:
            keep.append(i)
    return keep


def crop_to_ink(
    image: np.ndarray,
    margin: float = 0.05,
    ink_threshold: int = 200,
    min_ink_pixels: int = 5,
    min_speck_fraction: float = 0.0001,
) -> np.ndarray:
    """Crop a grayscale dark-ink-on-white image to the ink box plus a margin.

    Args:
        image: Single-channel uint8 image with dark ink on a white background.
        margin: Padding around the ink box, as a fraction of the box's larger side.
        ink_threshold: Pixels below this value count as ink for locating the
            signature. The original grayscale values are returned unchanged.
        min_ink_pixels: If fewer ink pixels are found the image is treated as empty
            and ``ValueError`` is raised, so the caller can route it to review.
        min_speck_fraction: Connected ink components smaller than this fraction of
            the image area (dust, scan specks) are ignored when locating the ink box,
            so isolated specks do not stretch the crop. Ignored if nothing else is left.
    """
    mask = image < ink_threshold
    # Faint scans may have no pixels below the threshold; relax it step by step.
    for relaxed in (225, 240):
        if int(mask.sum()) >= min_ink_pixels:
            break
        mask = image < relaxed
    if int(mask.sum()) < min_ink_pixels:
        raise ValueError("no signature ink found in image")

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    big = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_speck_fraction * mask.size]
    if big:
        keep = _drop_distant_small_marks(stats, big, max(image.shape[:2]))
        mask = np.isin(labels, keep)

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
    binarize: bool = False,
) -> np.ndarray:
    """Grayscale normalization, tight crop, and aspect-aware resize.

    With ``binarize=True`` the image is Otsu-thresholded to pure black ink (0) on
    white (255) before cropping, and the resized result is re-thresholded so the
    output contains only 0 and 255.

    The threshold is used only to locate the signature for cropping. The output
    remains grayscale, so stroke and pressure variations are preserved.

    Returns:
        ``target_h x target_w`` uint8 grayscale image, dark ink on white.

    Raises:
        ValueError: If the input is invalid or contains no ink.
    """
    grayscale = to_grayscale(image, ink=ink, blur_ksize=blur_ksize)
    if binarize:
        _, grayscale = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cropped = crop_to_ink(remove_border_artifacts(grayscale), margin=margin)
    resized = aspect_aware_resize(cropped, target_size, pad_value=255)
    if binarize:
        # Area resampling blurs thin strokes into gray; a high cut keeps them as ink.
        resized = np.where(resized < 200, 0, 255).astype(np.uint8)
    return resized


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
