"""Summarize a training log: loss per epoch and an undertrained/overfitting check.

The log is the text printed by ``training.train_siamese`` (one JSON line per epoch).
Save the notebook output to a file, then run:
    python -m training.analyze_log training_log.txt [--plot curves.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_epochs(path: Path) -> list[dict]:
    """Return the epochs of the last training run found in the log.

    Handles Kaggle log exports, where a timestamp and line number precede the JSON.
    A new run starts whenever the warm-up epoch counter restarts at 1.
    """
    runs: list[list[dict]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        start = line.find('{"phase"')
        if start < 0:
            continue
        epoch = json.loads(line[start:])
        if epoch["phase"] == "warmup" and epoch["epoch"] == 1:
            runs.append([])
        if not runs:
            runs.append([])
        runs[-1].append(epoch)
    if not runs:
        raise ValueError("no epoch lines found; expected the JSON lines printed during training")
    if len(runs) > 1:
        print(f"log contains {len(runs)} runs; analyzing the last one ({len(runs[-1])} epochs)\n")
    return runs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", type=Path)
    parser.add_argument("--plot", type=Path, help="Optional PNG path (needs matplotlib).")
    parser.add_argument("--window", type=int, default=5, help="Epochs used for the trend.")
    args = parser.parse_args()

    epochs = read_epochs(args.log)
    train = [e["train"]["loss"] for e in epochs]
    val = [e["validation"]["loss"] for e in epochs]

    print(f"{'epoch':>5} {'phase':<10} {'train_loss':>10} {'val_loss':>9} {'val_gap':>8}")
    for i, e in enumerate(epochs, 1):
        v = e["validation"]
        print(f"{i:>5} {e['phase']:<10} {train[i-1]:>10.4f} {val[i-1]:>9.4f} "
              f"{v['mean_negative'] - v['mean_positive']:>8.3f}")

    best = min(range(len(val)), key=val.__getitem__)
    print(f"\nbest validation loss {val[best]:.4f} at epoch {best + 1} of {len(val)}")

    # Compare the mean of the last `window` epochs with the `window` before them.
    w = min(args.window, len(val) // 2)
    if w >= 2:
        val_change = sum(val[-w:]) / w - sum(val[-2 * w : -w]) / w
        train_change = sum(train[-w:]) / w - sum(train[-2 * w : -w]) / w
        print(f"last {w} epochs vs previous {w}: val {val_change:+.4f}, train {train_change:+.4f}")
        if best >= len(val) - w and val_change < -0.002:
            print("Verdict: validation loss is still falling -> undertrained; train longer.")
        elif train_change < 0 and val_change > 0.002:
            print("Verdict: train falls while validation rises -> overfitting; try a lower learning rate, fewer epochs, or more data.")
        else:
            print("Verdict: validation loss has flattened; more epochs alone will help little.")

    if args.plot:
        import matplotlib.pyplot as plt

        plt.plot(range(1, len(val) + 1), train, label="train loss")
        plt.plot(range(1, len(val) + 1), val, label="validation loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(args.plot, dpi=120)
        print(f"saved plot to {args.plot}")


if __name__ == "__main__":
    main()
