# Small Language Model (SLM)

A clean, efficient implementation of a small transformer-based language model with modern best practices.

## Key Improvements

### Dataset (`dataset.py`)
- ✅ **Proper PyTorch Dataset class**: Uses native `torch.utils.data.Dataset` instead of creating tensors upfront
- ✅ **Type hints**: Full type annotations for better IDE support and code clarity
- ✅ **Dataclass configuration**: `DataConfig` dataclass for clean configuration management
- ✅ **Path handling**: Uses `pathlib.Path` for cross-platform file handling
- ✅ **Error handling**: Comprehensive validation and error messages
- ✅ **Memory efficiency**: On-demand tensor creation in `__getitem__` instead of pre-allocating
- ✅ **Reproducible splits**: Fixed random seed for train/val split
- ✅ **Configurable workers**: Proper DataLoader configuration with prefetching

### Main (`main.py`)
- ✅ **CLI arguments**: Uses argparse for flexible command-line interface
- ✅ **Structured configs**: Dataclass-based configuration hierarchy
- ✅ **Device detection**: Automatic detection of CUDA/MPS/CPU with proper logging
- ✅ **Better checkpointing**: Saves optimizer state, epoch info, and metrics
- ✅ **Generation improvements**: Added temperature and top-k sampling
- ✅ **Inference mode**: Uses `@torch.inference_mode()` for faster generation
- ✅ **Separate modes**: Can train, resume, or generate-only
- ✅ **Type safety**: Full type hints throughout

### Training (`train.py`)
- ✅ **Gradient accumulation**: Support for simulating larger batch sizes
- ✅ **Mixed precision training**: AMP support for faster training
- ✅ **Gradient clipping**: Prevents exploding gradients
- ✅ **Learning rate scheduling**: Cosine decay with warmup
- ✅ **Progress bars**: Uses tqdm for better visibility
- ✅ **Best model tracking**: Saves best model based on validation loss
- ✅ **AdamW optimizer**: Better weight decay implementation
- ✅ **Separate evaluate function**: Clean evaluation logic with inference mode

## Usage

### Training
```bash
# Train from scratch
python main.py --config configs.yaml

# Resume from checkpoint
python main.py --config configs.yaml --resume checkpoints/model_latest.pth

# Custom prompt after training
python main.py --config configs.yaml --prompt "In a galaxy far away"
```

### Generation Only
```bash
python main.py --config configs.yaml --generate-only --resume checkpoints/best_model.pth --prompt "Hello world" --max-length 100
```

## Configuration

Edit `configs.yaml` to customize:

- **Model architecture**: vocab_size, embed_dim, num_heads, num_layers
- **Data**: paths, batch_size, context_size, workers
- **Training**: learning rate, epochs, gradient clipping, warmup, AMP

## Project Structure

```
SLM/
├── configs.yaml          # Main configuration file
├── main.py              # Entry point with CLI
├── model.py             # Transformer model architecture
├── dataset.py           # Dataset and data loading
├── train.py             # Training loop and utilities
├── bpe_tokenizer_4096.json
└── data/
    └── 100-0.txt
```

## Requirements

```bash
pip install torch tokenizers pyyaml tqdm tensorboard einops rotary-embedding-torch
```

## Best Practices Implemented

1. **Separation of Concerns**: Clear separation between data, model, training, and main logic
2. **Configuration Management**: Centralized YAML config with dataclass validation
3. **Type Safety**: Comprehensive type hints for better development experience
4. **Error Handling**: Proper validation and informative error messages
5. **Logging**: Structured logging with appropriate levels
6. **Reproducibility**: Fixed seeds and deterministic operations
7. **Efficiency**: Memory-efficient data loading, AMP, gradient accumulation
8. **Monitoring**: TensorBoard integration with proper metric tracking
9. **Checkpointing**: Complete state saving for reproducible training
10. **Modern PyTorch**: Uses latest best practices (inference_mode, AdamW, etc.)
