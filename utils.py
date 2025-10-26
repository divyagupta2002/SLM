
import os
import yaml
from dataclasses import dataclass
from typing import Optional

# -----------------------------
# Config dataclasses
# -----------------------------
@dataclass
class ModelConfig:
    vocab_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    ff_hidden_dim: int
    dropout: float = 0.1
        
@dataclass
class DataConfig:
    tokenizer_path: str
    data_path: str
    context_size: int
    batch_size: int
    val_split: float
    num_workers: int 
    pin_memory: bool

@dataclass
class TrainingConfig:
    epochs: int
    warmup_steps: int
    learning_rate: float
    grad_clip: float
    grad_accum_steps: int
    logdir: str
    eval_every: int
    save_every: int
    checkpoint_path: str
    snapshot_path: str

@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {}))
        )