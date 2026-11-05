"""Generate mild, CEDAR-only raw-image augmentations and refresh the CEDAR manifest.

Only three targeted, geometry-preserving operations are applied. Rotation, shear and
scaling are deliberately **not** used, because tilting or stretching a signature
changes its spatial layout and stroke orientation and introduces unrealistic
artifacts:

- ``thickness``: stroke-thickness jitter (dilation / erosion). Slightly thicker or
  thinner pen strokes mimic different pen types and ink flow.
- ``shift``: small spatial shift. A few pixels of horizontal / vertical translation
  make the model tolerant to slight placement differences.
- ``elastic``: very light elastic deformation. A subtle smooth local warp simulates
  natural hand tremor and day-to-day writing fluctuations.

Augmented raw images are written next to their source under ``data/raw/cedar`` named
``augmented_<type>_<variant>_<original>.png``. Only **train**-split images are
augmented, so the validation and test signers stay pristine. The script then rewrites
``data/manifests/cedar.csv`` and merges every per-dataset manifest into
``data/manifest.csv``. Run ``python -m preprocessing.preprocess_dataset`` afterwards
to produce the processed PNGs and the processed manifest.

Usage:
    python -m preprocessing.augment_cedar
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np

from preprocessing.build_manifests import FIELDS, merge

AUGMENT_TYPES = ("thickness", "shift", "elastic")
PREFIX = "augmented"
DEFAULT_VARIANTS = 2


def augmented_name(kind: str, variant: int, stem: str) -> str:
    """Return the raw file name for one augmented image."""
    return f"{PREFIX}_{kind}_{variant}_{stem}.png"


def _border_value(channels: int):
    """White background value matching the image's channel count."""
    return 255 if channels == 1 else tuple([255] * channels)


def _seed_for(stem: str, kind: str, variant: int) -> int:
    """Stable per-image seed so a re-run reproduces the same augmentations."""
    digest = hashlib.sha256(f"{stem}|{kind}|{variant}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def augment_thickness(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Slightly thicken or thin the dark strokes.

    Ink is dark on white, so erosion thickens strokes and dilation thins them.
    """
    size = int(rng.integers(2, 4))
    kernel = np.ones((size, size), np.uint8)
    if rng.random() < 0.5:
        return cv2.erode(image, kernel)
    return cv2.dilate(image, kernel)


def augment_shift(image: np.ndarray, rng: np.random.Generator, max_shift: int = 5) -> np.ndarray:
    """Translate the whole image by a few pixels; the border stays white.

    Integer shifts are used so no interpolation blur is introduced.
    """
    channels = 1 if image.ndim == 2 else image.shape[2]
    dx, dy = (int(v) for v in rng.integers(-max_shift, max_shift + 1, size=2))
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_border_value(channels),
    )


def augment_elastic(
    image: np.ndarray,
    rng: np.random.Generator,
    max_displacement: float = 4.0,
    sigma_range: tuple[float, float] = (12.0, 16.0),
) -> np.ndarray:
    """Apply a very light smooth local warp.

    A low-frequency displacement field is scaled so the strongest local movement is
    ``max_displacement`` pixels, which stays within hand-tremor range.
    """
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    sigma = float(rng.uniform(*sigma_range))

    def displacement() -> np.ndarray:
        field = cv2.GaussianBlur(
            rng.uniform(-1.0, 1.0, (height, width)).astype(np.float32), (0, 0), sigma
        )
        return field / (np.abs(field).max() + 1e-8) * max_displacement

    dx, dy = displacement(), displacement()
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    return cv2.remap(
        image,
        grid_x + dx,
        grid_y + dy,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_border_value(channels),
    )


AUGMENTERS = {
    "thickness": augment_thickness,
    "shift": augment_shift,
    "elastic": augment_elastic,
}


def augment_raw(image: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Dispatch to the augmenter for ``kind``."""
    if kind not in AUGMENTERS:
        raise ValueError(f"unknown augmentation type: {kind}")
    return AUGMENTERS[kind](image, rng)


def generate(
    data_root: Path,
    manifest: Path,
    variants: int = DEFAULT_VARIANTS,
) -> None:
    """Write augmented raw images and refresh the CEDAR and merged manifests."""
    data_root = data_root.resolve()
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    # Drop rows from an earlier run so re-running is idempotent.
    originals = [r for r in rows if not Path(r["image_path"]).name.startswith(f"{PREFIX}_")]

    augmented_rows: list[dict[str, str]] = []
    failures: list[str] = []
    written = 0
    for row in originals:
        if row["split"].strip().lower() != "train":
            continue
        source = data_root / row["image_path"]
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            failures.append(f"{source}: unreadable")
            continue
        stem = Path(row["image_path"]).stem
        folder = Path(row["image_path"]).parent
        for kind in AUGMENT_TYPES:
            for variant in range(1, variants + 1):
                rng = np.random.default_rng(_seed_for(stem, kind, variant))
                augmented = augment_raw(image, kind, rng)
                relative = (folder / augmented_name(kind, variant, stem)).as_posix()
                target = data_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(target), augmented):
                    failures.append(f"{target}: could not write")
                    continue
                augmented_rows.append({**row, "image_path": relative})
                written += 1

    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(originals)
        writer.writerows(augmented_rows)

    merge()

    train_originals = sum(r["split"].strip().lower() == "train" for r in originals)
    print(f"wrote {written} augmented raw images to {data_root / 'raw' / 'cedar'}")
    print(f"  train originals: {train_originals}")
    print(f"  {manifest}: {len(originals)} original + {len(augmented_rows)} augmented rows")
    if failures:
        print(f"{len(failures)} skipped:")
        for message in failures:
            print(f"  {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/cedar.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--variants", type=int, default=DEFAULT_VARIANTS)
    args = parser.parse_args()
    if args.variants < 1:
        raise ValueError("--variants must be at least 1")
    generate(args.data_root, args.manifest, args.variants)


if __name__ == "__main__":
    main()

