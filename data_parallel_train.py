import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_

from utils import Config
from model import Model
from data import create_train_val_datasets

from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# -----------------------------
# 1. Setup distributed process
# -----------------------------
def setup():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    init_process_group("nccl")

def cleanup():
    destroy_process_group()

# -----------------------------
# 2. Trainer
# -----------------------------

class DistributedTrainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size, gpu_id, training_config):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)

        self.warmup_steps = training_config.warmup_steps
        self.lr = training_config.learning_rate
        self.grad_clip = training_config.grad_clip

        self.epochs_ran = 0
        self.btach_size = batch_size
        self.num_epochs = training_config.epochs
        self.writer = SummaryWriter(log_dir=training_config.logdir) if gpu_id == 0 else None
        self.eval_every = training_config.eval_every
        self.save_every = training_config.save_every
        self.snapshot_path = training_config.snapshot_path

        self.loss_fn = torch.nn.CrossEntropyLoss()

        if os.path.exists(self.snapshot_path):
            self._load_snapshot(self.snapshot_path, map_location=f"cuda:{gpu_id}")
            logger.info(f"Loaded model snapshot from {self.snapshot_path} on GPU {gpu_id} with epoch {self.epochs_ran}")

        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.train_dataloader, self.ep_steps = self._prepare_dataloader(train_dataset, batch_size)
        self.val_dataloader, self.val_steps = self._prepare_dataloader(val_dataset, batch_size, shuffle=False)
        self.total_steps = self.ep_steps * self.num_epochs * dist.get_world_size()
        logger.info(f"Trainer initialized on GPU {gpu_id} with {self.ep_steps} steps/epoch.")

    def _load_snapshot(self, path, map_location):
        snapshot = torch.load(path, map_location=map_location)
        self.model.load_state_dict(snapshot['model_state_dict'])
        self.epochs_ran = snapshot["epochs"]

    def _prepare_dataloader(self, dataset, batch_size, shuffle=True):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            sampler=DistributedSampler(dataset, shuffle=shuffle),
        )
        return dataloader, len(dataloader)

    def lambda_lr(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step) / float(self.warmup_steps)
        progress = float(current_step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.pi * torch.tensor(progress)))
        

    def train(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lambda_lr)
        
        while self.epochs_ran < self.num_epochs:
            self.model.train()
            optimizer.zero_grad()
            progress_bar = tqdm(self.train_dataloader, desc=f"GPU {self.gpu_id} Epoch {self.epochs_ran + 1}/{self.num_epochs}", position=self.gpu_id)
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(self.gpu_id), targets.to(self.gpu_id)

                with torch.amp.autocast(device_type='cuda', dtype = torch.bfloat16):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))

                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                dist.all_reduce(loss.detach(), op=dist.ReduceOp.SUM) # will get called on only after word_size steps, and not on every step

                if self.writer:
                    avg_loss = loss.item() / dist.get_world_size()
                    global_step = (self.epochs_ran*self.ep_steps + progress_bar.n) * dist.get_world_size()
                    self.writer.add_scalar("Train/Loss", avg_loss, global_step)
                    self.writer.add_scalar("Train/LR", lr_scheduler.get_last_lr()[0], global_step)
            self.epochs_ran += 1

            if self.gpu_id == 0 and self.epochs_ran % self.save_every == 0:
                self._save_snapshot()
                logger.info(f"Saved model snapshot at epoch {self.epochs_ran} on GPU {self.gpu_id}")
            
            if self.epochs_ran % self.eval_every == 0:
                self.evaluate()
            


    @torch.inference_mode()
    def evaluate(self):
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.gpu_id)
        progress_bar = tqdm(self.val_dataloader, desc=f"GPU {self.gpu_id} Validation", position=self.gpu_id)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self.gpu_id), targets.to(self.gpu_id)

            with torch.amp.autocast(device_type='cuda', dtype = torch.bfloat16):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            total_loss += loss
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

        if self.writer:    
            avg_loss = total_loss.item() / (self.val_steps * dist.get_world_size())
            global_step = self.epochs_ran * self.ep_steps * dist.get_world_size()
            self.writer.add_scalar("Val/Loss", avg_loss, global_step)
            logger.info(f"Validation Loss at epoch {self.epochs_ran}: {avg_loss}")

    def _save_snapshot(self):
        snapshot = {
            'model_state_dict': self.model.module.state_dict(),
            'epochs': self.epochs_ran
        }
        torch.save(snapshot, self.snapshot_path)

# -----------------------------
# 3. Main function
# -----------------------------

def main(config_path: str):
    setup()
    gpu_id = int(os.environ['LOCAL_RANK'])
    config = Config.from_yaml(config_path)

    # Load or create model
    model = Model(
        vocab_size=config.model.vocab_size,
        embed_dim=config.model.embed_dim,
        num_heads=config.model.num_heads,
        ff_hidden_dim=config.model.ff_hidden_dim,
        num_layers=config.model.num_layers
    )

    # Load datasets
    train_dataset, val_dataset = create_train_val_datasets(
        tokenizer_path=config.data.tokenizer_path,
        data_path=config.data.data_path,
        context_size=config.data.context_size,
        val_split=config.data.val_split
    )

    trainer = DistributedTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config.data.batch_size,
        gpu_id=gpu_id,
        training_config=config.training
    )

    trainer.train()
    cleanup()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train.py <config_path>")
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)