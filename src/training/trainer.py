"""
Training loop for the Transformer translation model.

Handles:
- Epoch-based training with progress bars
- Validation after each epoch
- Early stopping based on validation loss
- Model checkpointing (best + periodic)
- Gradient clipping
- Training metrics logging
- Resume from checkpoint
"""

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from .loss import LabelSmoothingLoss
from .optimizer import get_optimizer, NoamScheduler
from ..utils.helpers import format_number, time_since


class Trainer:
    """Handles the full training loop for the Transformer.

    Args:
        model: The Transformer model.
        config: TranslationConfig instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        logger: Logger instance.
    """

    def __init__(self, model, config, train_loader, val_loader, logger=None):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

        # Determine device
        self.device = next(model.parameters()).device

        # Loss function
        self.criterion = LabelSmoothingLoss(
            smoothing=config.training.label_smoothing,
            padding_idx=0,
        )

        # Optimizer and scheduler
        self.optimizer = get_optimizer(model, config.training.learning_rate)
        self.scheduler = NoamScheduler(
            self.optimizer,
            d_model=config.model.d_model,
            warmup_steps=config.training.warmup_steps,
        )

        # Training state
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_history = []

        # Checkpoint directory
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> dict:
        """Run the full training loop.

        Returns:
            Dictionary with training history.
        """
        self._log(f"Starting training for {self.config.training.epochs} epochs")
        self._log(f"Training batches: {len(self.train_loader)}")
        self._log(f"Validation batches: {len(self.val_loader)}")

        start_time = time.time()

        for epoch in range(self.start_epoch, self.config.training.epochs):
            # Train one epoch
            train_loss = self._train_epoch(epoch)

            # Validate
            val_loss = self._validate(epoch)

            # Learning rate
            current_lr = self.scheduler.current_lr

            # Log epoch summary
            self._log(
                f"Epoch {epoch+1}/{self.config.training.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {time_since(start_time)}"
            )

            # Save history
            self.training_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
            })

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
                self._log(f"  * New best validation loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                self._log(
                    f"  No improvement for {self.patience_counter} epoch(s) "
                    f"(patience: {self.config.training.early_stopping_patience})"
                )

            # Periodic checkpoint
            if (epoch + 1) % self.config.training.checkpoint_every == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Early stopping
            if self.patience_counter >= self.config.training.early_stopping_patience:
                self._log(f"Early stopping triggered at epoch {epoch+1}")
                break

        total_time = time_since(start_time)
        self._log(f"Training complete! Total time: {total_time}")
        self._log(f"Best validation loss: {self.best_val_loss:.4f}")

        return {
            "history": self.training_history,
            "best_val_loss": self.best_val_loss,
            "total_time": total_time,
        }

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0
        total_tokens = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1} [Train]",
            leave=False,
        )

        for batch_idx, batch in enumerate(progress_bar):
            if getattr(self.config.training, "smoke_test", False) and batch_idx >= 5:
                break
            # Move to device
            src = batch["src"].to(self.device)
            tgt_input = batch["tgt_input"].to(self.device)
            tgt_output = batch["tgt_output"].to(self.device)
            src_mask = batch["src_mask"].to(self.device)
            tgt_mask = batch["tgt_mask"].to(self.device)

            # Forward pass
            logits = self.model(src, tgt_input, src_mask, tgt_mask)

            # Compute loss
            loss = self.criterion(logits, tgt_output)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.training.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.grad_clip,
                )

            # Update weights
            self.optimizer.step()
            self.scheduler.step()

            # Track metrics
            num_tokens = (tgt_output != 0).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.current_lr:.2e}",
            })

        return total_loss / max(total_tokens, 1)

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        """Run validation.

        Args:
            epoch: Current epoch number.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        progress_bar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch+1} [Val]",
            leave=False,
        )

        for batch_idx, batch in enumerate(progress_bar):
            if getattr(self.config.training, "smoke_test", False) and batch_idx >= 2:
                break
            src = batch["src"].to(self.device)
            tgt_input = batch["tgt_input"].to(self.device)
            tgt_output = batch["tgt_output"].to(self.device)
            src_mask = batch["src_mask"].to(self.device)
            tgt_mask = batch["tgt_mask"].to(self.device)

            logits = self.model(src, tgt_input, src_mask, tgt_mask)
            loss = self.criterion(logits, tgt_output)

            num_tokens = (tgt_output != 0).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        return total_loss / max(total_tokens, 1)

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save a model checkpoint.

        Args:
            epoch: Current epoch number.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
            "config": {
                "model": vars(self.config.model),
                "training": vars(self.config.training),
                "data": vars(self.config.data),
            },
        }

        if is_best:
            path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, path)
            self._log(f"  Saved best checkpoint: {path}")
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(checkpoint, path)
            self._log(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a checkpoint to resume training.

        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        self._log(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint.get("training_history", [])

        self._log(f"Resumed from epoch {self.start_epoch} "
                  f"(best val loss: {self.best_val_loss:.4f})")

    def _log(self, message: str) -> None:
        """Log a message to the logger or print."""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
