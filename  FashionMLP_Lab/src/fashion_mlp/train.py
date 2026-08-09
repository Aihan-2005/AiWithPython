from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .callbacks import build_callbacks
from .config import TrainConfig
from .data import load_fashion_mnist
from .models import build_model
from .plots import plot_history
from .training import compile_classifier, targets_for_model
from .utils import save_json, set_global_seed, timestamped_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Fashion-MNIST MLP experiment.")
    parser.add_argument(
        "--model",
        choices=["sequential", "functional", "functional_aux", "subclassed"],
        default="sequential",
    )
    parser.add_argument("--hidden-units", nargs="+", type=int, default=[300, 100])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", default="runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        model=args.model,
        hidden_units=tuple(args.hidden_units),
        dropout=args.dropout,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        validation_size=args.validation_size,
        seed=args.seed,
    )

    set_global_seed(config.seed)
    data = load_fashion_mnist(config.validation_size, config.seed)
    run_dir = timestamped_dir(args.runs_dir, config.model)
    save_json(config.to_dict(), run_dir / "config.json")

    model = build_model(config.model, config.hidden_units, config.dropout)
    compile_classifier(model, config.model, config.optimizer, config.learning_rate)

    # Build subclassed model before summary for complete parameter information.
    if config.model == "subclassed":
        _ = model(data.x_train[:1], training=False)
    model.summary()

    callbacks = build_callbacks(run_dir, patience=config.patience)
    y_train = targets_for_model(config.model, data.y_train)
    y_valid = targets_for_model(config.model, data.y_valid)
    y_test = targets_for_model(config.model, data.y_test)

    history = model.fit(
        data.x_train,
        y_train,
        validation_data=(data.x_valid, y_valid),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    pd.DataFrame(history.history).to_csv(run_dir / "history.csv", index_label="epoch")
    plot_history(history, run_dir)

    evaluation = model.evaluate(data.x_test, y_test, verbose=0, return_dict=True)
    save_json(evaluation, run_dir / "metrics.json")

    model.save(run_dir / "final_model.keras")

    print("\nTraining complete")
    print(f"Run directory: {run_dir}")
    print("Test metrics:")
    for key, value in evaluation.items():
        print(f"  {key}: {float(value):.6f}")


if __name__ == "__main__":
    main()
