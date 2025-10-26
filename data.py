import os
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, random_split
from tokenizers import Tokenizer
from tokenizers import models, pre_tokenizers, decoders, trainers
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    if os.path.exists(tokenizer_path):
        try:
            tokenizer = Tokenizer.from_file(tokenizer_path)
            logger.info(f"Tokenizer loaded from {tokenizer_path}")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            raise
    else:
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")


def train_tokenizer(
    data_files: List[str],
    tokenizer_path: str,
    vocab_size: int = 4096,
    min_frequency: int = 2
) -> Tokenizer:
    for file_path in data_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency)
    tokenizer.train(data_files, trainer)
    
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    tokenizer.save(tokenizer_path)
    logger.info(f"Tokenizer trained and saved to {tokenizer_path}")
    
    return tokenizer


def load_and_encode_text(data_path: str, tokenizer: Tokenizer) -> List[int]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    encoded = tokenizer.encode(text).ids
    logger.info(f"Encoded {len(text)} characters into {len(encoded)} tokens")
    
    return encoded


def create_train_val_datasets(
    data_path: str,
    tokenizer_path: str,
    context_size: int,
    val_split: float = 0.1
) -> Tuple[Dataset, Dataset]:
    tokenizer = load_tokenizer(tokenizer_path)
    tokens = load_and_encode_text(data_path, tokenizer)
    dataset = TextDataset(tokens, context_size)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    logger.info(f"Created train dataset of size {train_size} and validation dataset of size {val_size}")
    
    return train_dataset, val_dataset


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