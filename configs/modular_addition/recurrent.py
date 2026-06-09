from dataclasses import dataclass
from frozendict import frozendict
from data.modular_addition import VOCAB_SIZE, PAD_TOKEN, MODULUS

@dataclass(frozen=True)
class RecurrentModelConfig:
    vocab_size: int = VOCAB_SIZE
    pad_token: int = PAD_TOKEN
    embed_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 3
    num_classes: int = MODULUS
    bidirectional: bool = False
    dropout: float = 0.1

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 256
    max_epochs: int = 25
    k_folds: int = 5
    optimizer: frozendict = frozendict({
        "lr": 1e-3,
        "weight_decay": 1e-4
    })

@dataclass(frozen=True)
class Config:
    model: RecurrentModelConfig = RecurrentModelConfig()
    train: TrainingConfig = TrainingConfig()

config = Config()
