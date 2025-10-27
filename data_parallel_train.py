import os
import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import Model
from utils import Config
from data import create_train_val_datasets

import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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
        self.model = torch.compile(self.model)

        self.warmup_steps = training_config.warmup_steps
        self.lr = training_config.learning_rate
        self.grad_clip = training_config.grad_clip

        self.epochs_ran = 0
        self.num_epochs = training_config.epochs
        self.writer = SummaryWriter(log_dir=training_config.logdir) if gpu_id == 0 else None
        self.eval_every = training_config.eval_every
        self.save_every = training_config.save_every
        self.snapshot_path = training_config.snapshot_path

        self.loss_fn = torch.nn.CrossEntropyLoss()

        os.makedirs(os.path.dirname(self.snapshot_path), exist_ok=True)
        if os.path.exists(self.snapshot_path):
            self._load_snapshot(self.snapshot_path, map_location=f"cuda:{gpu_id}")
            logger.info(f"Loaded model snapshot from {self.snapshot_path} on GPU {gpu_id} with epoch {self.epochs_ran}")


        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.train_dataloader, self.ep_steps = self._prepare_dataloader(train_dataset, batch_size)
        self.val_dataloader, self.val_steps = self._prepare_dataloader(val_dataset, batch_size, shuffle=False)
        self.total_steps = self.num_epochs * self.ep_steps
        assert self.warmup_steps < self.total_steps, "Warmup steps must be less than total training steps."
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
            self.train_dataloader.sampler.set_epoch(self.epochs_ran)
            progress_bar = tqdm(self.train_dataloader, desc=f"GPU {self.gpu_id} Epoch {self.epochs_ran + 1}/{self.num_epochs}", position=self.gpu_id, leave = False)
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

                global_step += 1
                reduced_loss = loss.detach().clone()
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.AVG) # will get called on only after word_size steps, and not on every step
                if self.writer:
                    avg_loss = reduced_loss.item() 
                    self.writer.add_scalar("Train/Loss", avg_loss, global_step)
                    self.writer.add_scalar("Train/LR", lr_scheduler.get_last_lr()[0], global_step)
            self.epochs_ran += 1

            if self.gpu_id == 0 and self.epochs_ran % self.save_every == 0:
                self._save_snapshot()
                logger.info(f"Saved model snapshot at epoch {self.epochs_ran} on GPU {self.gpu_id}")
            
            if self.epochs_ran % self.eval_every == 0:
                avg_val_loss = self.evaluate()
                if self.gpu_id == 0 and avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_path = self.snapshot_path.replace('.pt', '_best.pt')
                    self._save_snapshot(best_val_path)

        if self.writer:
            self.writer.close()
        logger.info(f"Training complete on GPU {self.gpu_id}.")

    @torch.inference_mode()
    def evaluate(self):
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.gpu_id)
        progress_bar = tqdm(self.val_dataloader, desc=f"GPU {self.gpu_id} Validation", position=self.gpu_id, leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self.gpu_id), targets.to(self.gpu_id)

            with torch.amp.autocast(device_type='cuda', dtype = torch.bfloat16):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            total_loss += loss

        dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
        if self.writer:    
            global_step = self.epochs_ran * self.ep_steps
            avg_loss = total_loss.item() / self.val_steps
            self.writer.add_scalar("Val/Loss", avg_loss, global_step)
            logger.info(f"Validation Loss at epoch {self.epochs_ran}: {avg_loss}")
            return avg_loss

    def _save_snapshot(self, snapshot_path=None):
        try:
            if snapshot_path is None:
                snapshot_path = self.snapshot_path
            snapshot = {
                'model_state_dict': self.model.module.state_dict(),
                'epochs': self.epochs_ran
            }
            torch.save(snapshot, snapshot_path)
            logger.info(f"Snapshot saved at {snapshot_path} at epoch {self.epochs_ran}")
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")

# -----------------------------
# 3. Main function
# -----------------------------
def main(config_path: str):
    setup()
    gpu_id = int(os.environ['LOCAL_RANK'])
    config = Config.from_yaml(config_path)

    # initialize model
    model = Model(
        vocab_size=config.model.vocab_size,
        embed_dim=config.model.embed_dim,
        num_heads=config.model.num_heads,
        ff_hidden_dim=config.model.ff_hidden_dim,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout
    )

    # Load datasets
    train_dataset, val_dataset = create_train_val_datasets(
        tokenizer_path=config.data.tokenizer_path,
        data_path=config.data.data_path,
        context_size=config.data.context_size,
        val_split=config.data.val_split
    )

    # Initialize trainer
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