## SLM (Small Language Model)

A compact Transformer language model with utilities for tokenizer management, single-GPU and multi-GPU training, and text generation. The codebase targets fast experimentation on modest hardware while keeping close to modern PyTorch best practices.

### Highlighted Features
- **Rotary-aware decoder-only Transformer** with tied embeddings and configurable depth/width (`model.py`).
- **Tokenizer tooling** for training or reusing byte-level BPE vocabularies (`data.py`).
- **Training loops** for both single GPU (`train.py`) and Distributed Data Parallel training (`data_parallel_train.py`) with AMP, gradient clipping, and cosine warmup scheduling.
- **Structured configuration** via dataclasses and YAML (`utils.py`, `configs.yaml`).
- **Logging & monitoring** through TensorBoard and `tqdm` progress bars.
- **Inference utilities** for prompt-based generation (`infer.py`).

### Project Layout
```
SLM/
├── configs.yaml           # YAML configuration consumed by Config dataclasses
├── data.py                # Tokenizer helpers and Dataset definition
├── data_parallel_train.py # DDP training entry point
├── infer.py               # Prompt-based text generation script
├── model.py               # Transformer decoder implementation
├── requirements.txt       # Project dependencies
├── train.py               # Single-GPU training loop with evaluation
├── utils.py               # Config dataclasses and YAML loader
└── tokenizer/             # Stores pretrained or newly trained tokenizers
```

### Getting Started
1. **Install dependencies**
     ```bash
     python -m venv slm_env && source slm_env/bin/activate
     pip install -r requirements.txt
     ```
2. **Prepare a tokenizer** (skip if `tokenizer/bpe_4096.json` is already available):
     ```python
     # quick one-off script
     from data import train_tokenizer
     train_tokenizer(["data/100-0.txt"], "tokenizer/bpe_4096.json", vocab_size=4096)
     ```
3. **Edit `configs.yaml`** to point to your dataset, tokenizer, and desired hyperparameters.

### Training Workflows
- **Single GPU / CPU**
    ```bash
    python train.py configs.yaml
    ```
    Checkpoints land in `training.checkpoint_path`; the best model (by validation loss) is updated automatically.

- **Multi-GPU Distributed** (uses NCCL + DDP)
    ```bash
    torchrun --standalone --nproc_per_node=NUM_GPUS data_parallel_train.py configs.yaml
    ```
    Snapshots and TensorBoard logs are controlled by the training config (`save_every`, `eval_every`, `logdir`).

- **Resume Training**
    Set `training.checkpoint_path` (single GPU) or `training.snapshot_path` (DDP) to an existing file. The trainer detects and reloads state automatically.

### Text Generation
```bash
python infer.py \
    --config configs.yaml \
    --ckpt_path checkpoints/best_model.pth \
    --prompt "Once upon a time" \
    --max-length 100
```
Supports temperature scaling and top-k filtering inside `infer.py`.

### Monitoring & Logging
- Launch TensorBoard to inspect training curves:
    ```bash
    tensorboard --logdir runs
    ```
- `tqdm` progress bars show per-device status during training.

### Configuration Reference (`configs.yaml`)
- `model`: vocabulary size, embedding dim, head count, feedforward size, number of decoder blocks.
- `data`: tokenizer path, raw text path, context length, batch size, validation split ratio, dataloader workers.
- `training`: epochs, warmup steps, base learning rate, gradient clipping threshold, gradient accumulation, logging/checkpoint cadence, and output paths.

### License
This project is released under the terms of the `LICENSE` file included in the repository.
