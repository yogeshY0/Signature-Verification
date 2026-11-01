"""One-time preprocessing: raw manifest -> 192x384 black-and-white PNGs + processed manifest
(pass --grayscale to keep grayscale strokes).

Usage:
    python -m preprocessing.preprocess_dataset --manifest data/manifest.csv --output-dir data/processed

Output layout mirrors the raw layout (``raw/cedar/...`` -> ``processed/cedar/...``) and
``<output-dir>/manifest.csv`` is written with paths relative to itself. Train with:
    python -m training.train_siamese --manifest data/processed/manifest.csv --already-preprocessed
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from preprocessing.signature_preprocessing import preprocess_signature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--image-root", type=Path, help="Defaults to the manifest's directory.")
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Keep grayscale strokes instead of the default black-and-white output.",
    )
    args = parser.parse_args()

    root = (args.image_root or args.manifest.parent).resolve()
    out_dir = args.output_dir.resolve()

    with args.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    out_rows: list[dict[str, str]] = []
    failures: list[str] = []
    for i, row in enumerate(rows, 1):
        source = Path(row["image_path"])
        source = source if source.is_absolute() else root / source
        # Drop the leading "raw/" so processed/cedar mirrors raw/cedar.
        parts = Path(row["image_path"]).parts
        relative = Path(*parts[1:]) if parts and parts[0] == "raw" else Path(*parts)
        target = (out_dir / relative).with_suffix(".png")

        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            failures.append(f"{source}: unreadable")
            continue
        try:
            processed = preprocess_signature(image, binarize=not args.grayscale)
        except ValueError as error:
            failures.append(f"{source}: {error}")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), processed)
        out_rows.append({**row, "image_path": target.relative_to(out_dir).as_posix()})
        if i % 200 == 0 or i == len(rows):
            print(f"{i}/{len(rows)} processed")

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"wrote {len(out_rows)} images and {out_dir / 'manifest.csv'}")
    if failures:
        print(f"{len(failures)} skipped:")
        for message in failures:
            print(f"  {message}")


if __name__ == "__main__":
    main()
