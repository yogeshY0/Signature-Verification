"""Build the BHSig260 manifest and merge all per-dataset manifests.

Per-dataset manifests live in ``data/manifests/<name>.csv`` with image paths relative
to ``data/``. Each has its own signer-disjoint split. This script writes
``data/manifests/bhsig260.csv`` and then merges every file in that folder into
``data/manifest.csv``.

Usage:
    python -m preprocessing.build_manifests
"""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path

DATA = Path("data")
FIELDS = ["image_path", "signer_id", "is_forgery", "split"]
NAME = re.compile(r"^([BH])-S-(\d+)-([GF])-\d+\.tif$", re.IGNORECASE)
LANGUAGE = {"B": "bn", "H": "hi"}


def split_signers(signers: list[str], seed: int = 42) -> dict[str, str]:
    """Assign whole signers to train/val/test (about 70/15/15)."""
    shuffled = sorted(signers)
    random.Random(seed).shuffle(shuffled)
    n_val = n_test = round(len(shuffled) * 0.15)
    labels = ["test"] * n_test + ["val"] * n_val
    labels += ["train"] * (len(shuffled) - len(labels))
    return dict(zip(shuffled, labels))


def build_bhsig260() -> None:
    root = DATA / "raw" / "BHSig260"
    rows: list[dict[str, str]] = []
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        dataset_rows = []
        for path in sorted(dataset_dir.rglob("*.tif")):
            match = NAME.match(path.name)
            if not match:
                raise ValueError(f"unexpected file name: {path}")
            language, number, kind = match.groups()
            is_forgery = kind.upper() == "F"
            # The folder must agree with the filename, or labels would be wrong.
            if is_forgery != (path.parent.name.lower() == "forged"):
                raise ValueError(f"folder and file name disagree: {path}")
            dataset_rows.append(
                {
                    "image_path": path.relative_to(DATA).as_posix(),
                    "signer_id": f"bhsig_{LANGUAGE[language.upper()]}_{int(number):03d}",
                    "is_forgery": str(is_forgery).lower(),
                }
            )
        splits = split_signers(sorted({r["signer_id"] for r in dataset_rows}))
        rows += [{**r, "split": splits[r["signer_id"]]} for r in dataset_rows]
    write(DATA / "manifests" / "bhsig260.csv", rows)


def write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def merge() -> None:
    merged: list[dict[str, str]] = []
    for path in sorted((DATA / "manifests").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            merged += list(csv.DictReader(handle))
    write(DATA / "manifest.csv", merged)


if __name__ == "__main__":
    build_bhsig260()
    merge()
