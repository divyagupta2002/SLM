import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    epochs: int
    learning_rate: float
    ckpt_path: str
    logdir: str
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    warmup_steps: int = 0
    use_amp: bool = True
    save_every: int = 1
    log_every: int = 10
    eval_every: int = 1
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_training(
    model: nn.Module,
    config: TrainingConfig,
    steps_per_epoch: int
):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    scheduler = None
    if config.warmup_steps > 0:
        total_steps = config.epochs * steps_per_epoch
        
        def lr_lambda(current_step: int):
            if current_step < config.warmup_steps:
                return float(current_step) / float(config.warmup_steps)
            progress = float(current_step - config.warmup_steps) / float(total_steps - config.warmup_steps)
            return max(0.1, 0.5 * (1.0 + torch.cos(torch.pi * torch.tensor(progress))))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scaler = torch.amp.GradScaler(device = config.device) if config.use_amp else None
    
    Path(config.logdir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(config.logdir)
    
    logger.info(
        f"Training setup: lr={config.learning_rate}, "
        f"grad_clip={config.grad_clip}, "
        f"grad_accum_steps={config.grad_accum_steps}, "
        f"warmup_steps={config.warmup_steps}, "
        f"use_amp={config.use_amp}"
    )
    
    return loss_fn, optimizer, scheduler, scaler, writer


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    config: TrainingConfig,
    writer: SummaryWriter,
    start_epoch: int = 0
):
    global_step = start_epoch * len(train_loader)
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            with torch.amp.autocast(device_type=device.type):
                logits = model(inputs)
                loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss = loss / config.grad_accum_steps

            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                
                scheduler.step()
                
                optimizer.zero_grad()
            
            train_loss += loss.item() * config.grad_accum_steps
            global_step += 1
            
            if global_step % config.log_every == 0:
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Loss/Train', loss.item() * config.grad_accum_steps, global_step)
                writer.add_scalar('Learning_Rate', current_lr, global_step)
                pbar.set_postfix({
                    'loss': f"{loss.item() * config.grad_accum_steps:.4f}",
                    'lr': f"{current_lr:.2e}"
                })
        
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{config.epochs}, Average Training Loss: {avg_train_loss:.4f}")
        
        if (epoch + 1) % config.eval_every == 0:
            avg_val_loss = evaluate(model, val_loader, loss_fn, device)
            logger.info(f"Epoch {epoch+1}/{config.epochs}, Validation Loss: {avg_val_loss:.4f}")
            writer.add_scalar('Loss/Validation', avg_val_loss, global_step)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_ckpt_path = Path(config.ckpt_path).parent / "best_model.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'val_loss': avg_val_loss
                }, best_ckpt_path)
                logger.info(f"New best model saved with val_loss={avg_val_loss:.4f}")
        
        if (epoch + 1) % config.save_every == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'train_loss': avg_train_loss
            }, config.ckpt_path)
            logger.info(f"Checkpoint saved to {config.ckpt_path}")
    
    writer.close()
    return model


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    val_loader,
    loss_fn: nn.Module,
    device: torch.device
) -> float:
    model.eval()
    total_loss = 0.0
    
    for inputs, targets in tqdm(val_loader, desc="Validating", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        
        logits = model(inputs)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


