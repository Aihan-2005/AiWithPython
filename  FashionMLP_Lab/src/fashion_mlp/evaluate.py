from __future__ import annotations

import argparse
from pathlib import Path

import keras
import numpy as np
from sklearn.metrics import classification_report

from . import models as _models  
from .constants import CLASS_NAMES
from .data import load_fashion_mnist
from .plots import save_confusion_matrix, save_prediction_grid
from .training import infer_model_kind, main_probabilities
from .utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved Fashion-MNIST model.")
    parser.add_argument("model_path")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir) if args.output_dir else model_path.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(model_path)
    data = load_fashion_mnist(seed=args.seed)

    model_kind = infer_model_kind(model)
    if model_kind == "functional_aux":
        y_test_for_eval = {"main_output": data.y_test, "aux_output": data.y_test}
    else:
        y_test_for_eval = data.y_test

    metrics = model.evaluate(data.x_test, y_test_for_eval, verbose=0, return_dict=True)
    predictions = main_probabilities(model.predict(data.x_test, verbose=0))
    y_pred = np.argmax(predictions, axis=1)

    report = classification_report(
        data.y_test,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    save_json(metrics, output_dir / "metrics.json")
    save_json(report, output_dir / "classification_report.json")
    save_confusion_matrix(data.y_test, y_pred, output_dir / "confusion_matrix.png", normalize=True)
    save_prediction_grid(
        data.x_test[:16],
        data.y_test[:16],
        predictions[:16],
        output_dir / "sample_predictions.png",
    )

    print("Evaluation complete")
    for key, value in metrics.items():
        print(f"  {key}: {float(value):.6f}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
