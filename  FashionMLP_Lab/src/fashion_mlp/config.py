from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TrainConfig:
    model: str = "sequential"
    hidden_units: tuple[int, ...] = (300, 100)
    dropout: float = 0.0
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 50
    patience: int = 7
    validation_size: int = 5000
    seed: int = 42

    def to_dict(self) -> dict:
        data = asdict(self)
        data["hidden_units"] = list(self.hidden_units)
        return data
