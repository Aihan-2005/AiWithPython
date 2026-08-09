from __future__ import annotations

import keras


def make_optimizer(name: str, learning_rate: float) -> keras.optimizers.Optimizer:
    name = name.lower()
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    if name == "sgd":
        return keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            nesterov=True,
        )
    raise ValueError(f"Unknown optimizer: {name}. Choose 'adam' or 'sgd'.")


def compile_classifier(
    model: keras.Model,
    model_kind: str,
    optimizer_name: str,
    learning_rate: float,
) -> None:
    optimizer = make_optimizer(optimizer_name, learning_rate)

    if model_kind == "functional_aux":
        model.compile(
            optimizer=optimizer,
            loss={
                "main_output": "sparse_categorical_crossentropy",
                "aux_output": "sparse_categorical_crossentropy",
            },
            loss_weights={"main_output": 0.85, "aux_output": 0.15},
            metrics={
                "main_output": ["accuracy"],
                "aux_output": ["accuracy"],
            },
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )


def targets_for_model(model_kind: str, y):
    if model_kind == "functional_aux":
        return {"main_output": y, "aux_output": y}
    return y


def infer_model_kind(model: keras.Model) -> str:
    output_names = set(getattr(model, "output_names", []) or [])
    if {"main_output", "aux_output"}.issubset(output_names):
        return "functional_aux"
    return "single"


def main_probabilities(predictions):
    """Return main classification probabilities from single- or multi-output predict() results."""
    if isinstance(predictions, dict):
        if "main_output" in predictions:
            return predictions["main_output"]
        return next(iter(predictions.values()))
    if isinstance(predictions, (list, tuple)):
        return predictions[0]
    return predictions
