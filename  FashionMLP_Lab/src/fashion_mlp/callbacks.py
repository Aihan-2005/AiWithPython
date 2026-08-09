from __future__ import annotations

from pathlib import Path

import keras


class ValTrainRatioCallback(keras.callbacks.Callback):
    """Print val_loss / train_loss after each epoch as a simple overfitting diagnostic."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        if train_loss in (None, 0) or val_loss is None:
            return
        ratio = val_loss / train_loss
        print(f" - val/train loss ratio: {ratio:.3f}")


def build_callbacks(
    run_dir: str | Path,
    patience: int = 7,
    monitor: str = "val_loss",
) -> list[keras.callbacks.Callback]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=run_dir / "best_model.keras",
            monitor=monitor,
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="min",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=run_dir / "tensorboard",
            histogram_freq=1,
        ),
        keras.callbacks.CSVLogger(run_dir / "training_log.csv"),
        keras.callbacks.TerminateOnNaN(),
        ValTrainRatioCallback(),
    ]
