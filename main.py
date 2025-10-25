import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import yaml
import torch
from torch.utils.tensorboard import SummaryWriter

from model import Model
from dataset import DataConfig, load_tokenizer, encode_texts, decode_tokens, create_dataloaders
from train import TrainingConfig, setup_training, train

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    vocab_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    ff_hidden_dim: Optional[int] = None
    
    def __post_init__(self):
        if self.ff_hidden_dim is None:
            self.ff_hidden_dim = 2 * self.embed_dim
        
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )


@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    training: 'TrainingConfig'
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {}))
        )
    
    def save(self, config_path: str):
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump({
                'model': asdict(self.model),
                'data': asdict(self.data),
                'training': asdict(self.training)
            }, f, default_flow_style=False)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device


def initialize_model(config: ModelConfig, device: torch.device) -> Model:
    model = Model(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        ff_hidden_dim=config.ff_hidden_dim,
        num_layers=config.num_layers
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model initialized with {num_params:,} parameters "
        f"(embed_dim={config.embed_dim}, num_heads={config.num_heads}, "
        f"ff_hidden_dim={config.ff_hidden_dim}, num_layers={config.num_layers})"
    )
    
    model = model.to(device)
    
    return model


def load_checkpoint(
    model: Model,
    ckpt_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> dict:
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    logger.info(f"Checkpoint loaded from {ckpt_path} (epoch {epoch})")
    
    return checkpoint


def save_checkpoint(
    model: Model,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    ckpt_path: str,
    **kwargs
):
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        **kwargs
    }
    
    torch.save(checkpoint, ckpt_path)
    logger.info(f"Checkpoint saved to {ckpt_path} (epoch {epoch})")


@torch.inference_mode()
def generate_text(
    model: Model,
    prompt: str,
    tokenizer_path: str,
    max_length: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    device: Optional[torch.device] = None
) -> str:
    if device is None:
        device = next(model.parameters()).device
    
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


def train_model(config: Config, device: torch.device, resume_from: Optional[str] = None):
    model = initialize_model(config.model, device)
    
    train_loader, val_loader = create_dataloaders(config.data)
    
    loss_fn, optimizer, scheduler, scaler, writer = setup_training(
        model, 
        config.training,
        len(train_loader)
    )
    
    start_epoch = 0
    if resume_from:
        checkpoint = load_checkpoint(model, resume_from, optimizer, device)
        start_epoch = checkpoint.get('epoch', 0)
    
    trained_model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        config=config.training,
        writer=writer,
        start_epoch=start_epoch
    )
    
    return trained_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a small language model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
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
    device = get_device()
    
    if not args.generate_only:
        trained_model = train_model(config, device, resume_from=args.resume)
    else:
        if not args.resume:
            raise ValueError("Must provide --resume checkpoint for generation-only mode")
        trained_model = initialize_model(config.model, device)
        load_checkpoint(trained_model, args.resume, device=device)
    
    generated_text = generate_text(
        trained_model,
        args.prompt,
        config.data.tokenizer_path,
        max_length=args.max_length,
        device=device
    )
    
    logger.info(f"Generated text: {generated_text}")
    print(f"\n{'='*80}\nGenerated text:\n{'='*80}\n{generated_text}\n{'='*80}")


if __name__ == "__main__":
    main()
