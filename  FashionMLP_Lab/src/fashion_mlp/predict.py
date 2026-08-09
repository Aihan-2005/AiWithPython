from __future__ import annotations

import argparse
from pathlib import Path

import keras
import numpy as np

from . import models as _models  
from .constants import CLASS_NAMES
from .data import load_fashion_mnist
from .training import main_probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a Fashion-MNIST test example.")
    parser.add_argument("model_path")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_fashion_mnist(seed=args.seed)

    if not 0 <= args.index < len(data.x_test):
        raise IndexError(f"index must be between 0 and {len(data.x_test) - 1}")

    model = keras.models.load_model(Path(args.model_path))
    raw = model.predict(data.x_test[args.index : args.index + 1], verbose=0)
    probabilities = main_probabilities(raw)[0]

    k = max(1, min(args.top_k, len(CLASS_NAMES)))
    top_indices = np.argsort(probabilities)[::-1][:k]

    print(f"True class: {CLASS_NAMES[int(data.y_test[args.index])]}")
    print("Top predictions:")
    for idx in top_indices:
        print(f"  {CLASS_NAMES[int(idx)]:<14} {float(probabilities[idx]):.4f}")


if __name__ == "__main__":
    main()
