from __future__ import annotations

import argparse
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import load_fashion_mnist
from .models import build_sequential_classifier
from .training import make_optimizer
from .utils import set_global_seed, timestamped_dir


class LearningRateFinder(keras.callbacks.Callback):
    """Exponentially increase LR after every training batch and record a smoothed loss."""

    def __init__(self, start_lr: float, end_lr: float, max_steps: int, beta: float = 0.98):
        super().__init__()
        self.start_lr = float(start_lr)
        self.end_lr = float(end_lr)
        self.max_steps = int(max_steps)
        self.beta = float(beta)
        self.multiplier = (self.end_lr / self.start_lr) ** (1.0 / max(1, self.max_steps - 1))
        self.step = 0
        self.avg_loss = 0.0
        self.best_loss = np.inf
        self.learning_rates: list[float] = []
        self.losses: list[float] = []

    def _set_lr(self, value: float) -> None:
        lr = self.model.optimizer.learning_rate
        if hasattr(lr, "assign"):
            lr.assign(value)
        else:
            self.model.optimizer.learning_rate = value

    def _get_lr(self) -> float:
        return float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))

    def on_train_begin(self, logs=None):
        self._set_lr(self.start_lr)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        raw_loss = float(logs["loss"])
        self.step += 1

        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * raw_loss
        smooth_loss = self.avg_loss / (1 - self.beta ** self.step)

        lr = self._get_lr()
        self.learning_rates.append(lr)
        self.losses.append(smooth_loss)
        self.best_loss = min(self.best_loss, smooth_loss)

        if self.step >= self.max_steps or (self.step > 20 and smooth_loss > 4 * self.best_loss):
            self.model.stop_training = True
            return

        self._set_lr(lr * self.multiplier)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an exponential learning-rate range test.")
    parser.add_argument("--start-lr", type=float, default=1e-5)
    parser.add_argument("--end-lr", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-units", nargs="+", type=int, default=[300, 100])
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="lr_runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_lr <= 0 or args.end_lr <= args.start_lr:
        raise ValueError("Require 0 < start_lr < end_lr")

    set_global_seed(args.seed)
    data = load_fashion_mnist(seed=args.seed)

    model = build_sequential_classifier(tuple(args.hidden_units))
    model.compile(
        optimizer=make_optimizer(args.optimizer, args.start_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    finder = LearningRateFinder(args.start_lr, args.end_lr, args.steps)
    steps_per_epoch = int(np.ceil(len(data.x_train) / args.batch_size))
    epochs = int(np.ceil(args.steps / steps_per_epoch)) + 1

    model.fit(
        data.x_train,
        data.y_train,
        epochs=epochs,
        batch_size=args.batch_size,
        callbacks=[finder],
        verbose=0,
    )

    run_dir = timestamped_dir(args.output_dir, "lr_find")
    df = pd.DataFrame({"learning_rate": finder.learning_rates, "loss": finder.losses})
    df.to_csv(run_dir / "lr_find.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(finder.learning_rates, finder.losses)
    ax.set_xlabel("Learning rate (log scale)")
    ax.set_ylabel("Smoothed training loss")
    ax.set_title("Learning-rate range test")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "lr_finder.png", dpi=170)
    plt.close(fig)

    if len(finder.losses) >= 5:
        best_idx = int(np.argmin(finder.losses))
        best_observed_lr = finder.learning_rates[best_idx]
        conservative_lr = best_observed_lr / 3
        print(f"Lowest observed smoothed loss near LR={best_observed_lr:.3e}")
        print(f"Conservative starting point to test: ~{conservative_lr:.3e}")
    print(f"Saved LR finder artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
