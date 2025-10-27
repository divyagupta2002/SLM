import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_

import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import Model
from utils import Config, TrainingConfig
from data import create_train_val_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 1. Trainer class
# -----------------------------
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_datset,
        val_dataset,
        batch_size: int,
        training_config: TrainingConfig,
    ):
        self.model = model
        self.model = self.model.to(device)
        self.model = torch.compile(self.model)

        self.warmup_steps = training_config.warmup_steps
        self.lr = training_config.learning_rate
        self.grad_clip = training_config.grad_clip
        self.grad_accum_steps = training_config.grad_accum_steps

        self.epochs_ran = 0
        self.num_epochs = training_config.epochs
        self.batch_size = batch_size
        self.writer = SummaryWriter(log_dir=training_config.logdir)
        self.eval_every = training_config.eval_every
        self.save_every = training_config.save_every
        self.checkpoint_path = training_config.checkpoint_path

        self.loss_fn = torch.nn.CrossEntropyLoss()

        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        if os.path.exists(training_config.checkpoint_path):
            self._load_checkpoint(training_config.checkpoint_path)
            logger.info(f"Resuming training from checkpoint: {training_config.checkpoint_path}")

        self.train_loader, self.ep_steps = self._get_dataloader(train_datset, batch_size)
        self.val_loader, self.val_steps = self._get_dataloader(val_dataset, batch_size, shuffle=False)
        self.total_steps = self.num_epochs * self.ep_steps
        assert self.warmup_steps < self.total_steps, "Warmup steps must be less than total training steps."
        logger.info(f"Training for {self.num_epochs} epochs, starting from epoch {self.epochs_ran} on device {device}")

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epochs_ran = checkpoint.get('epoch', 0)

    def _get_dataloader(self, dataset, batch_size, shuffle=True):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=6,
            pin_memory=True
        )
        steps = len(dataloader)
        return dataloader, steps
    
    def _get_scheduler(self, optimizer):
        warmup_scheduler = LinearLR(optimizer, start_factor=0.000001, end_factor=1.0, total_iters=self.warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=self.total_steps - self.warmup_steps)

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps]
        )
        return lr_scheduler

    def train(self):
        global_step = self.epochs_ran * self.ep_steps
        best_val_loss = float('inf')
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        lr_scheduler = self._get_scheduler(optimizer)
        
        while self.epochs_ran < self.num_epochs:
            self.model.train()
            optimizer.zero_grad()
            pbar = tqdm(self.train_loader, desc=f"Epoch {self.epochs_ran + 1}/{self.num_epochs}", leave=False)
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(device), targets.to(device)
                
                with torch.amp.autocast(device_type=device.type, dtype = torch.bfloat16):
                    logits = self.model(inputs)
                    loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
                    loss = loss / self.grad_accum_steps
                # scaler.scale(loss).backward()
                loss.backward()
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    # scaler.unscale_(optimizer)
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    optimizer.step() #scaler.step(optimizer)
                    lr_scheduler.step()
                    optimizer.zero_grad() #scaler.update()
                global_step += 1
                avg_loss = loss.item() * self.grad_accum_steps
                self.writer.add_scalar('Train/Loss', avg_loss, global_step)
                self.writer.add_scalar('Train/Learning_Rate', lr_scheduler.get_last_lr()[0], global_step)
            self.epochs_ran += 1

            if self.epochs_ran % self.save_every == 0:
                self._save_checkpoint()

            if self.epochs_ran % self.eval_every == 0:
                avg_val_loss = self.evaluate()
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_ckpt_path = self.checkpoint_path.replace('.pt', '_best.pt')
                self._save_checkpoint(best_ckpt_path)  
            
        self.writer.close()
        logger.info("Training complete.")

    @torch.inference_mode()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            with torch.amp.autocast(device_type=device.type, dtype = torch.bfloat16):        
                logits = self.model(inputs)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

            total_loss += loss.item()

        global_step = self.epochs_ran * self.ep_steps
        avg_loss = total_loss / self.val_steps
        self.writer.add_scalar('Val/Loss', avg_loss, global_step)
        logger.info(f"Validation Loss after epoch {self.epochs_ran}: {avg_loss:.4f}")
        return avg_loss

    def _save_checkpoint(self, ckpt_path = None):
        try:
            if ckpt_path is None:
                ckpt_path = self.checkpoint_path
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'epoch': self.epochs_ran
            }
            torch.save(checkpoint, ckpt_path)
            logger.info(f"Checkpoint saved to {ckpt_path} at epoch {self.epochs_ran}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

# -----------------------------
# 4. Main function 
# -----------------------------
def main(config_path: str):
    config = Config.from_yaml(config_path)
    model = Model(
        vocab_size=config.model.vocab_size,
        embed_dim=config.model.embed_dim,
        num_heads=config.model.num_heads,
        ff_hidden_dim=config.model.ff_hidden_dim,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout
    )
    train_dataset, val_dataset = create_train_val_datasets(
        data_path=config.data.data_path,
        tokenizer_path=config.data.tokenizer_path,
        context_size=config.data.context_size,
        val_split=config.data.val_split
    )
    trainer = Trainer(
        model=model,
        train_datset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config.data.batch_size,
        training_config=config.training
    )
    trainer.train()

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train.py <config_path>")
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)
    # to run: python train.py config.yaml

