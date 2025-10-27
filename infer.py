import os
import argparse
import logging
from typing import Optional

import torch

from utils import Config, ModelConfig
from model import Model
from data import load_tokenizer, encode_texts, decode_tokens

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_model(config: ModelConfig) -> Model:
    model = Model(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        ff_hidden_dim=config.ff_hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    model = model.to(device)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    return model

def load_checkpoint(model: Model, ckpt_path: str) -> Model:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    # check if the keys starts with _orig_mod
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    epoch = checkpoint.get('epoch', 0)
    logger.info(f"Checkpoint loaded from {ckpt_path} (epoch {epoch})")
    
    return checkpoint

@torch.inference_mode()
def generate_text(
    model: Model,
    prompt: str,
    tokenizer_path: str,
    max_length: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None
) -> str:
    
    model.eval()
    tokenizer = load_tokenizer(tokenizer_path)
    
    encoded = encode_texts(tokenizer, [prompt])[0]
    generated_ids = encoded.copy()
    
    for _ in range(max_length):
        input_tensor = torch.tensor([generated_ids], dtype=torch.long).to(device)
        logits = model(input_tensor)
        next_token_logits = logits[0, -1, :] / temperature
        
        if top_k is not None:
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            next_token_logits = torch.full_like(next_token_logits, float('-inf'))
            next_token_logits[top_k_indices] = top_k_logits
        
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        generated_ids.append(next_token_id)
    
    decoded_text = decode_tokens(tokenizer, [generated_ids])[0]
    return decoded_text


def parse_args():
    parser = argparse.ArgumentParser(description="Train a small language model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--generate-only',
        action='store_true',
        help='Only run generation, skip training'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='Once upon a time',
        help='Prompt for text generation'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=50,
        help='Maximum length for generated text'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config.from_yaml(args.config)
    if not args.ckpt_path:
        raise ValueError("Must provide --ckpt_path checkpoint for infer")
    trained_model = initialize_model(config.model)
    load_checkpoint(trained_model, args.ckpt_path)

    generated_text = generate_text(
        trained_model,
        args.prompt,
        config.data.tokenizer_path,
        max_length=args.max_length
    )
    print(f"\n{'='*80}\nGenerated text:\n{'='*80}\n{generated_text}\n{'='*80}")


if __name__ == "__main__":
    main()
