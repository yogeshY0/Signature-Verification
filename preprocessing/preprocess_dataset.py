"""One-time preprocessing: raw manifest -> 192x384 black-and-white PNGs + processed manifest
(pass --grayscale to keep grayscale strokes).

Usage:
    python -m preprocessing.preprocess_dataset --manifest data/manifest.csv --output-dir data/processed

Use ``--datasets`` to (re)process only the named dataset folders while leaving every other
dataset already present in the output manifest untouched. This is how CEDAR is augmented and
re-preprocessed without touching BHSig260 or SOHI:

    python -m preprocessing.preprocess_dataset --manifest data/manifest.csv \\
        --output-dir data/processed --datasets cedar --prune

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


def _relative_path(image_path: str) -> Path:
    """Drop a leading ``raw/`` so processed/cedar mirrors raw/cedar."""
    parts = Path(image_path).parts
    return Path(*parts[1:]) if parts and parts[0] == "raw" else Path(*parts)


def _dataset_of(relative: Path) -> str:
    """Top-level dataset folder of a manifest path, e.g. ``cedar`` or ``BHSig260``."""
    return relative.parts[0] if relative.parts else ""


def process(
    manifest: Path,
    out_dir: Path,
    image_root: Path | None = None,
    *,
    grayscale: bool = False,
    datasets: set[str] | None = None,
    prune: bool = False,
) -> tuple[int, list[str]]:
    """Preprocess ``manifest`` into ``out_dir`` and refresh ``out_dir/manifest.csv``.

    When ``datasets`` is given, only rows whose top-level dataset folder is listed are
    processed; the rows for every other dataset are copied unchanged from the existing
    ``out_dir/manifest.csv`` so unrelated datasets are never reprocessed. With ``prune``
    the stale PNGs left behind inside the freshly processed dataset folders are deleted.

    Returns the number of images written and the list of skipped sources.
    """
    manifest = Path(manifest)
    root = (image_root or manifest.parent).resolve()
    out_dir = Path(out_dir)
    selected = {name.strip().lower() for name in datasets} if datasets else None

    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected_rows = [
        row
        for row in rows
        if selected is None or _dataset_of(_relative_path(row["image_path"])).lower() in selected
    ]

    out_rows: list[dict[str, str]] = []
    failures: list[str] = []
    produced: dict[str, set[Path]] = {}
    for i, row in enumerate(selected_rows, 1):
        relative = _relative_path(row["image_path"])
        source = Path(row["image_path"])
        source = source if source.is_absolute() else root / source
        target = (out_dir / relative).with_suffix(".png")

        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            failures.append(f"{source}: unreadable")
            continue
        try:
            processed = preprocess_signature(image, binarize=not grayscale)
        except ValueError as error:
            failures.append(f"{source}: {error}")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), processed)
        produced.setdefault(_dataset_of(relative), set()).add(target.resolve())
        out_rows.append({**row, "image_path": target.relative_to(out_dir).as_posix()})
        if i % 200 == 0 or i == len(selected_rows):
            print(f"{i}/{len(selected_rows)} processed")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = out_dir / "manifest.csv"

    # Preserve rows for datasets that were not selected in this run.
    merged = out_rows
    preserved_count = 0
    if selected is not None and manifest_out.exists():
        with manifest_out.open(newline="") as handle:
            existing = list(csv.DictReader(handle))
        preserved = [r for r in existing if _dataset_of(Path(r["image_path"])).lower() not in selected]
        preserved_count = len(preserved)
        merged = preserved + out_rows

    fieldnames = list(rows[0].keys()) if rows else ["image_path", "signer_id", "is_forgery", "split"]
    with manifest_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    if prune and selected is not None:
        for dataset, keep in produced.items():
            for stale in (out_dir / dataset).rglob("*.png"):
                if stale.resolve() not in keep:
                    stale.unlink()

    print(f"wrote {len(out_rows)} images and {manifest_out}")
    if selected is not None:
        print(f"  preserved {preserved_count} rows for other datasets")
    if failures:
        print(f"{len(failures)} skipped:")
        for message in failures:
            print(f"  {message}")
    return len(out_rows), failures


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
    parser.add_argument(
        "--datasets",
        help="Comma-separated dataset folders to (re)process (e.g. 'cedar'). Rows for any "
        "other dataset already present in the output manifest are kept as-is.",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="With --datasets, delete stale PNGs under the processed dataset folders.",
    )
    args = parser.parse_args()

    datasets = None
    if args.datasets:
        datasets = {name.strip() for name in args.datasets.split(",") if name.strip()}
        if not datasets:
            raise ValueError("--datasets must list at least one dataset name")

    process(
        args.manifest,
        args.output_dir,
        args.image_root,
        grayscale=args.grayscale,
        datasets=datasets,
        prune=args.prune,
    )


if __name__ == "__main__":
    main()
