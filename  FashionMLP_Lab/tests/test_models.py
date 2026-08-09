import numpy as np

from fashion_mlp.models import build_model
from fashion_mlp.training import main_probabilities


def test_sequential_output_shape():
    model = build_model("sequential", (64, 32), 0.1)
    y = model(np.zeros((4, 28, 28), dtype="float32"), training=False)
    assert tuple(y.shape) == (4, 10)


def test_functional_output_shape():
    model = build_model("functional", (64, 32), 0.1)
    y = model(np.zeros((4, 28, 28), dtype="float32"), training=False)
    assert tuple(y.shape) == (4, 10)


def test_subclassed_output_shape():
    model = build_model("subclassed", (64, 32), 0.1)
    y = model(np.zeros((4, 28, 28), dtype="float32"), training=False)
    assert tuple(y.shape) == (4, 10)


def test_aux_model_has_two_heads_and_main_shape():
    model = build_model("functional_aux", (64, 32), 0.1)
    y = model(np.zeros((4, 28, 28), dtype="float32"), training=False)
    assert set(y) == {"main_output", "aux_output"}
    assert tuple(y["main_output"].shape) == (4, 10)
    assert tuple(y["aux_output"].shape) == (4, 10)
    assert tuple(main_probabilities(y).shape) == (4, 10)
