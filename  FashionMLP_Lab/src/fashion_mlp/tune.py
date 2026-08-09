from __future__ import annotations

import argparse
from pathlib import Path

import keras
import keras_tuner as kt

from .callbacks import build_callbacks
from .data import load_fashion_mnist
from .models import build_sequential_classifier
from .plots import plot_history
from .training import make_optimizer
from .utils import save_json, set_global_seed, timestamped_dir


class FashionHyperModel(kt.HyperModel):
    """Tune architecture/optimizer in build() and batch size in fit()."""

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        n_hidden = hp.Int("n_hidden", min_value=1, max_value=5, default=2)
        units = hp.Int("units", min_value=64, max_value=512, step=64, default=256)
        dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1, default=0.1)
        optimizer_name = hp.Choice("optimizer", ["adam", "sgd"], default="adam")
        learning_rate = hp.Float(
            "learning_rate",
            min_value=1e-4,
            max_value=5e-2,
            sampling="log",
            default=1e-3,
        )

        model = build_sequential_classifier(
            hidden_units=(units,) * n_hidden,
            dropout=dropout,
        )
        model.compile(
            optimizer=make_optimizer(optimizer_name, learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, hp, model, x, y, **kwargs):
        batch_size = hp.Choice("batch_size", [32, 64, 128, 256], default=128)
        return model.fit(x, y, batch_size=batch_size, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Fashion-MNIST MLP hyperparameters.")
    parser.add_argument("--strategy", choices=["random", "hyperband", "bayesian"], default="random")
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--directory", default="tuner_runs")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_tuner(args, hypermodel, project_name: str):
    common = dict(
        hypermodel=hypermodel,
        objective="val_accuracy",
        seed=args.seed,
        directory=args.directory,
        project_name=project_name,
        overwrite=args.overwrite,
    )
    if args.strategy == "random":
        return kt.RandomSearch(max_trials=args.max_trials, **common)
    if args.strategy == "bayesian":
        return kt.BayesianOptimization(max_trials=args.max_trials, **common)
    return kt.Hyperband(
        max_epochs=args.epochs,
        factor=3,
        hyperband_iterations=2,
        **common,
    )


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    data = load_fashion_mnist(seed=args.seed)

    project_name = f"fashion_{args.strategy}"
    hypermodel = FashionHyperModel()
    tuner = build_tuner(args, hypermodel, project_name)

    search_log_dir = Path(tuner.project_dir) / "tensorboard_search"
    search_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        ),
        keras.callbacks.TensorBoard(log_dir=search_log_dir),
        keras.callbacks.TerminateOnNaN(),
    ]

    tuner.search(
        data.x_train,
        data.y_train,
        validation_data=(data.x_valid, data.y_valid),
        epochs=args.epochs,
        callbacks=search_callbacks,
        verbose=2,
    )

    tuner.results_summary(num_trials=min(10, args.max_trials))
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_values = dict(best_hp.values)

    result_dir = timestamped_dir(args.directory, f"best_{args.strategy}")
    save_json(best_values, result_dir / "best_hyperparameters.json")

    # Retrain a fresh model with the best hyperparameters using the full training workflow.
    best_model = hypermodel.build(best_hp)
    callbacks = build_callbacks(result_dir, patience=max(args.patience, 5))
    history = best_model.fit(
        data.x_train,
        data.y_train,
        validation_data=(data.x_valid, data.y_valid),
        epochs=args.epochs,
        batch_size=int(best_values.get("batch_size", 128)),
        callbacks=callbacks,
        verbose=2,
    )
    plot_history(history, result_dir)

    metrics = best_model.evaluate(data.x_test, data.y_test, verbose=0, return_dict=True)
    save_json(metrics, result_dir / "test_metrics.json")
    best_model.save(result_dir / "final_model.keras")

    print("\nBest hyperparameters:")
    for key, value in best_values.items():
        print(f"  {key}: {value}")
    print("Test metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {float(value):.6f}")
    print(f"Artifacts: {result_dir}")


if __name__ == "__main__":
    main()
