from __future__ import annotations

from dataclasses import dataclass

import keras
import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class FashionData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_valid: np.ndarray
    y_valid: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_fashion_mnist(validation_size: int = 5000, seed: int = 42) -> FashionData:
    """Load Fashion-MNIST, scale pixels to [0, 1], and create a stratified validation set."""
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_full,
        y_train_full,
        test_size=validation_size,
        random_state=seed,
        stratify=y_train_full,
        shuffle=True,
    )

    return FashionData(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        x_test=x_test,
        y_test=y_test,
    )
