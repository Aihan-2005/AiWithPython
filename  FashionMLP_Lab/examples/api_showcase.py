"""Build all architecture variants and print summaries.

Run after installation:
    python examples/api_showcase.py
"""

import numpy as np

from fashion_mlp.models import build_model


for kind in ["sequential", "functional", "functional_aux", "subclassed"]:
    print("\n" + "=" * 80)
    print(kind.upper())
    model = build_model(kind, hidden_units=(128, 64), dropout=0.1)
    if kind == "subclassed":
        _ = model(np.zeros((1, 28, 28), dtype="float32"), training=False)
    model.summary()
