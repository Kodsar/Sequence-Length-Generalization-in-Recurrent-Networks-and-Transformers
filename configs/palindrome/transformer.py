from dataclasses import dataclass
from frozendict import frozendict
from data.palindrome import VOCAB_SIZE, PAD_TOKEN

@dataclass(frozen=True)
class TransformerModelConfig:
    vocab_size: int = VOCAB_SIZE
    pad_token: int = PAD_TOKEN
    embed_dim: int = 128
    mlp_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    num_classes: int = 2
    max_len: int = 512   
    dropout: float = 0.1

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64  
    max_epochs: int = 15
    k_folds: int = 5
    optimizer: frozendict = frozendict({
        "lr": 3e-4,
        "weight_decay": 1e-2
    })

@dataclass(frozen=True)
class Config:
    model: TransformerModelConfig = TransformerModelConfig()
    train: TrainingConfig = TrainingConfig()

config = Config()
