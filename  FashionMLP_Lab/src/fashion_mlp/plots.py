from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .constants import CLASS_NAMES


def plot_history(history, output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hist = history.history
    saved = []

    loss_keys = [k for k in hist if "loss" in k.lower()]
    metric_keys = [k for k in hist if "loss" not in k.lower()]

    if loss_keys:
        fig, ax = plt.subplots(figsize=(9, 5))
        for key in loss_keys:
            ax.plot(hist[key], label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and validation losses")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = output_dir / "learning_curves_loss.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(path)

    if metric_keys:
        fig, ax = plt.subplots(figsize=(9, 5))
        for key in metric_keys:
            ax.plot(hist[key], label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric")
        ax.set_title("Training and validation metrics")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = output_dir / "learning_curves_metrics.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(path)

    return saved


def save_confusion_matrix(y_true, y_pred, path: str | Path, normalize: bool = True) -> Path:
    path = Path(path)
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
    fig, ax = plt.subplots(figsize=(10, 9))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format=".2f" if normalize else "d")
    ax.set_title("Normalized confusion matrix" if normalize else "Confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def save_prediction_grid(x, y_true, probabilities, path: str | Path, n: int = 16) -> Path:
    path = Path(path)
    n = min(n, len(x))
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i in range(n):
        pred = int(np.argmax(probabilities[i]))
        confidence = float(probabilities[i, pred])
        axes[i].imshow(x[i], cmap="gray")
        axes[i].set_title(
            f"true: {CLASS_NAMES[int(y_true[i])]}\npred: {CLASS_NAMES[pred]} ({confidence:.2f})"
        )
        axes[i].axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
