from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import Tokenizer
from tokenizers import models, pre_tokenizers, decoders, trainers
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    tokenizer_path: str
    data_path: str
    context_size: int
    batch_size: int
    val_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2


class TextDataset(Dataset):
    def __init__(self, tokens: List[int], context_size: int):
        self.tokens = tokens
        self.context_size = context_size
        
        if len(tokens) < context_size + 1:
            raise ValueError(
                f"Token sequence length ({len(tokens)}) must be at least "
                f"context_size + 1 ({context_size + 1})"
            )
    
    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.context_size)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        context = torch.tensor(
            self.tokens[idx : idx + self.context_size], 
            dtype=torch.long
        )
        target = torch.tensor(
            self.tokens[idx + 1 : idx + self.context_size + 1], 
            dtype=torch.long
        )
        return context, target


def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    path = Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    
    try:
        tokenizer = Tokenizer.from_file(str(path))
        logger.info(f"Tokenizer loaded from {tokenizer_path}")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        raise


def train_tokenizer(
    data_files: List[str],
    tokenizer_path: str,
    vocab_size: int = 4096,
    min_frequency: int = 2
) -> Tokenizer:
    for file_path in data_files:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency)
    tokenizer.train(data_files, trainer)
    
    Path(tokenizer_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(tokenizer_path)
    logger.info(f"Tokenizer trained and saved to {tokenizer_path}")
    
    return tokenizer


def load_and_encode_text(data_path: str, tokenizer: Tokenizer) -> List[int]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading text from {data_path}")
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    encoded = tokenizer.encode(text).ids
    logger.info(f"Encoded {len(text)} characters into {len(encoded)} tokens")
    
    return encoded


def create_dataloaders(
    config: DataConfig,
    tokenizer: Optional[Tokenizer] = None
) -> Tuple[DataLoader, DataLoader]:
    if tokenizer is None:
        tokenizer = load_tokenizer(config.tokenizer_path)
    
    tokens = load_and_encode_text(config.data_path, tokenizer)
    dataset = TextDataset(tokens, config.context_size)
    
    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    
    if train_size <= 0 or val_size <= 0:
        raise ValueError(
            f"Invalid split: train_size={train_size}, val_size={val_size}. "
            f"Dataset has {len(dataset)} samples with {config.val_split} val_split."
        )
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=True
    )
    
    logger.info(
        f"DataLoaders created: {train_size} train samples, "
        f"{val_size} val samples, batch_size={config.batch_size}"
    )
    
    return train_loader, val_loader


def encode_texts(tokenizer: Tokenizer, texts: List[str]) -> List[List[int]]:
    encoded_texts = [tokenizer.encode(text).ids for text in texts]
    logger.info(f"Encoded {len(texts)} texts")
    return encoded_texts


def decode_tokens(tokenizer: Tokenizer, token_ids: List[List[int]]) -> List[str]:
    decoded_texts = [tokenizer.decode(ids) for ids in token_ids]
    logger.info(f"Decoded {len(token_ids)} token sequences")
    return decoded_texts


if __name__ == '__main__':
    # Example usage
    data_files = ['data/100-0.txt']
    tokenizer_path = 'tokenizer/bpe_4096.json'
    
    tokenizer = train_tokenizer(data_files, tokenizer_path)
    logger.info("Tokenizer training complete and saved at " + tokenizer_path)