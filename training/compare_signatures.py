"""Preprocess signature images and print pairwise cosine similarity.

Usage:
    python -m training.compare_signatures --checkpoint best_model.pt \
        --images test_signs/1.1.png test_signs/1.5.png --output-dir preprocessed_test_signs

    cd /Users/aacs/Desktop/Signature-Verification
.venv/bin/python -m training.compare_signatures \
  --checkpoint best_model.pt \
  --images test_signs/1.1.png test_signs/1.5.png test_signs/2.1.png test_signs/2.2.png
  
Images use the same black-and-white preprocessing as the training data. The
preprocessed images are saved to ``--output-dir`` as 192x384 PNGs.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np
import torch

from preprocessing.signature_preprocessing import preprocess_signature, to_model_input
from training.stage3_metric_learning import SiameseSignatureModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("preprocessed_test_signs"))
    parser.add_argument("--grayscale", action="store_true", help="Match grayscale-trained models.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = SiameseSignatureModel(pretrained=False, embedding_dim=checkpoint["embedding_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device).eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    names, tensors = [], []
    for path in args.images:
        # IMREAD_COLOR drops any alpha channel.
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read image: {path}")
        processed = preprocess_signature(image, binarize=not args.grayscale)
        target = args.output_dir / f"{path.stem}.png"
        if not cv2.imwrite(str(target), processed):
            raise OSError(f"could not write {target}")
        names.append(path.stem)
        tensors.append(torch.from_numpy(to_model_input(processed)))

    with torch.no_grad():
        embeddings = model(torch.stack(tensors).to(args.device)).cpu().numpy()

    # Embeddings are L2-normalized, so the dot product is the cosine similarity.
    print(f"Saved {len(names)} preprocessed images to {args.output_dir}/")
    print("\nPairwise cosine similarity:")
    for (i, a), (j, b) in itertools.combinations(enumerate(names), 2):
        print(f"  {a} vs {b}: {float(embeddings[i] @ embeddings[j]):.4f}")

    print("\nSimilarity matrix:")
    matrix = embeddings @ embeddings.T
    print("        " + "".join(f"{n:>8}" for n in names))
    for name, row in zip(names, matrix):
        print(f"{name:>8}" + "".join(f"{v:8.3f}" for v in row))


if __name__ == "__main__":
    main()
