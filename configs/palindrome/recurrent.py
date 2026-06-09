from dataclasses import dataclass
from frozendict import frozendict
from data.palindrome import VOCAB_SIZE, PAD_TOKEN

@dataclass(frozen=True)
class RecurrentModelConfig:
    vocab_size: int = VOCAB_SIZE
    pad_token: int = PAD_TOKEN
    embed_dim: int = 64
    hidden_dim: int = 64
    num_layers: int = 2
    num_classes: int = 2
    bidirectional: bool = True
    dropout: float = 0.1

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 512
    max_epochs: int = 10
    k_folds: int = 3
    optimizer: frozendict = frozendict({
        "lr": 1e-3,
        "weight_decay": 1e-4
    })

@dataclass(frozen=True)
class Config:
    model: RecurrentModelConfig = RecurrentModelConfig()
    train: TrainingConfig = TrainingConfig()

config = Config()
