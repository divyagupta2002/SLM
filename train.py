import os

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging

from utils import Config, TrainingConfig
from model import Model
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

        if os.path.exists(training_config.checkpoint_path):
            self._load_checkpoint(training_config.checkpoint_path)
            logger.info(f"Resuming training from checkpoint: {training_config.checkpoint_path}")

        self.train_loader, self.ep_steps = self._get_dataloader(train_datset, batch_size)
        self.val_loader, self.val_steps = self._get_dataloader(val_dataset, batch_size, shuffle=False)
        self.total_steps = self.num_epochs * self.ep_steps
        logger.info(f"Training for {self.num_epochs} epochs, starting from epoch {self.epochs_ran} on device {device}")

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epochs_ran = checkpoint.get('epoch', 0)

    def _get_dataloader(self, dataset, batch_size, shuffle=True):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=6,
            pin_memory=True
        )
        steps = len(dataloader)
        return dataloader, steps
    
    def lambda_lr(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step) / float(self.warmup_steps)
        progress = float(current_step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.pi * torch.tensor(progress)))

    def train(self):
        global_step = self.epochs_ran * self.ep_steps
        best_val_loss = float('inf')
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lambda_lr)
        
        while self.epochs_ran < self.num_epochs:
            self.model.train()
            optimizer.zero_grad()
            pbar = tqdm(self.train_loader, desc=f"Epoch {self.epochs_ran + 1}/{self.num_epochs}")
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(device), targets.to(device)
                
                with torch.amp.autocast(device_type=device.type, dtype = torch.bfloat16):
                    logits = self.model(inputs)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
                    loss = loss / self.grad_accum_steps
                # scaler.scale(loss).backward()
                loss.backward()
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    # scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    #scaler.step(optimizer)
                    #scaler.update()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                global_step += 1
                self.writer.add_scalar('Train/Loss', loss.item() * self.grad_accum_steps, global_step)
                self.writer.add_scalar('Train/Learning_Rate', lr_scheduler.get_last_lr()[0], global_step)
            self.epochs_ran += 1

            if self.epochs_ran % self.save_every == 0:
                self._save_checkpoint()

            if self.epochs_ran % self.eval_every == 0:
                avg_val_loss = self.evaluate()
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_ckpt_path = os.path.join(
                        os.path.dirname(self.checkpoint_path), 
                        'best_model.pth'
                    )
                self._save_checkpoint(best_ckpt_path)  
            
        self.writer.close()
        logger.info("Training complete.")

    @torch.inference_mode()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        for inputs, targets in tqdm(self.val_loader, desc="Validating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            with torch.amp.autocast(device_type=device.type, dtype = torch.bfloat16):        
                logits = self.model(inputs)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()
        
        avg_loss = total_loss / self.val_steps
        self.writer.add_scalar('Val/Loss', avg_loss, self.epochs_ran * self.ep_steps)
        logger.info(f"Validation Loss after epoch {self.epochs_ran}: {avg_loss:.4f}")
        return avg_loss

    def _save_checkpoint(self, ckpt_path = None):
        if ckpt_path is None:
            ckpt_path = self.checkpoint_path
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': self.epochs_ran
        }
        torch.save(checkpoint, ckpt_path)
        logger.info(f"Checkpoint saved to {ckpt_path} at epoch {self.epochs_ran}")

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
        num_layers=config.model.num_layers
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

