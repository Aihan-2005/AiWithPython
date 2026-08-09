from __future__ import annotations

from collections.abc import Sequence

import keras
from keras import layers

from .constants import IMAGE_SHAPE, NUM_CLASSES


def _validate_hidden_units(hidden_units: Sequence[int]) -> tuple[int, ...]:
    units = tuple(int(u) for u in hidden_units)
    if not units:
        raise ValueError("hidden_units must contain at least one hidden layer size")
    if any(u <= 0 for u in units):
        raise ValueError("all hidden layer sizes must be positive")
    return units


def build_sequential_classifier(
    hidden_units: Sequence[int] = (300, 100),
    dropout: float = 0.0,
    num_classes: int = NUM_CLASSES,
) -> keras.Model:
    """Plain stack of Dense/ReLU layers followed by Softmax."""
    hidden_units = _validate_hidden_units(hidden_units)

    model = keras.Sequential(name="fashion_mlp_sequential")
    model.add(keras.Input(shape=IMAGE_SHAPE, name="image"))
    model.add(layers.Flatten(name="flatten"))

    for i, units in enumerate(hidden_units, start=1):
        model.add(layers.Dense(units, activation="relu", name=f"hidden_{i}"))
        if dropout > 0:
            model.add(layers.Dropout(dropout, name=f"dropout_{i}"))

    model.add(layers.Dense(num_classes, activation="softmax", name="class_probs"))
    return model


def build_functional_classifier(
    hidden_units: Sequence[int] = (256, 128),
    dropout: float = 0.0,
    num_classes: int = NUM_CLASSES,
) -> keras.Model:
    """Wide & Deep-style Functional classifier with a skip path from flattened input."""
    hidden_units = _validate_hidden_units(hidden_units)

    inputs = keras.Input(shape=IMAGE_SHAPE, name="image")
    flat = layers.Flatten(name="flatten")(inputs)

    x = flat
    for i, units in enumerate(hidden_units, start=1):
        x = layers.Dense(units, activation="relu", name=f"deep_hidden_{i}")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"deep_dropout_{i}")(x)

    combined = layers.Concatenate(name="wide_deep_concat")([flat, x])
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probs")(combined)
    return keras.Model(inputs=inputs, outputs=outputs, name="fashion_mlp_functional")


def build_functional_aux_classifier(
    hidden_units: Sequence[int] = (256, 128),
    dropout: float = 0.0,
    num_classes: int = NUM_CLASSES,
) -> keras.Model:
    """Functional Wide & Deep classifier with a secondary auxiliary classification head."""
    hidden_units = _validate_hidden_units(hidden_units)

    inputs = keras.Input(shape=IMAGE_SHAPE, name="image")
    flat = layers.Flatten(name="flatten")(inputs)

    x = flat
    for i, units in enumerate(hidden_units, start=1):
        x = layers.Dense(units, activation="relu", name=f"deep_hidden_{i}")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"deep_dropout_{i}")(x)

    combined = layers.Concatenate(name="wide_deep_concat")([flat, x])
    main_output = layers.Dense(num_classes, activation="softmax", name="main_output")(combined)
    aux_output = layers.Dense(num_classes, activation="softmax", name="aux_output")(x)

    return keras.Model(
        inputs=inputs,
        outputs={"main_output": main_output, "aux_output": aux_output},
        name="fashion_mlp_functional_aux",
    )


@keras.saving.register_keras_serializable(package="FashionMLP")
class DynamicMLP(keras.Model):
    """Subclassed MLP whose hidden stack is executed imperatively in call()."""

    def __init__(
        self,
        hidden_units: Sequence[int] = (256, 128),
        dropout: float = 0.0,
        num_classes: int = NUM_CLASSES,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_units = tuple(int(u) for u in hidden_units)
        self.dropout_rate = float(dropout)
        self.num_classes = int(num_classes)

        _validate_hidden_units(self.hidden_units)

        self.flatten_layer = layers.Flatten(name="flatten")
        self.hidden_layers = [
            layers.Dense(units, activation="relu", name=f"hidden_{i}")
            for i, units in enumerate(self.hidden_units, start=1)
        ]
        self.dropout_layers = [
            layers.Dropout(self.dropout_rate, name=f"dropout_{i}")
            for i in range(1, len(self.hidden_units) + 1)
        ]
        self.output_layer = layers.Dense(
            self.num_classes, activation="softmax", name="class_probs"
        )

    def call(self, inputs, training: bool = False):
        x = self.flatten_layer(inputs)
        for dense, dropout in zip(self.hidden_layers, self.dropout_layers):
            x = dense(x)
            if self.dropout_rate > 0:
                x = dropout(x, training=training)
        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_units": list(self.hidden_units),
                "dropout": self.dropout_rate,
                "num_classes": self.num_classes,
            }
        )
        return config


def build_model(
    kind: str,
    hidden_units: Sequence[int],
    dropout: float = 0.0,
    num_classes: int = NUM_CLASSES,
) -> keras.Model:
    kind = kind.lower()
    if kind == "sequential":
        return build_sequential_classifier(hidden_units, dropout, num_classes)
    if kind == "functional":
        return build_functional_classifier(hidden_units, dropout, num_classes)
    if kind == "functional_aux":
        return build_functional_aux_classifier(hidden_units, dropout, num_classes)
    if kind == "subclassed":
        return DynamicMLP(
            hidden_units=hidden_units,
            dropout=dropout,
            num_classes=num_classes,
            name="fashion_mlp_subclassed",
        )
    raise ValueError(
        f"Unknown model kind: {kind}. Choose sequential, functional, functional_aux, or subclassed."
    )
