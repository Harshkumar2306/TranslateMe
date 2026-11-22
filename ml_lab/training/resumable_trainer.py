"""
Production Resumable Trainer for Modern 100M+ Transformer
Features:
- Periodic checkpoint saving (latest, best, milestone)
- Graceful Pause feature via touch-file ('pause.flag') or OS signals (SIGINT/SIGTERM)
- Deterministic Resume from checkpoint (restores model, optimizer, scaler, scheduler, RNG)
- Multi-GPU PyTorch AMP (Automatic Mixed Precision FP16)
- Gradient Accumulation for large effective batch sizes
- Label Smoothing Cross-Entropy Loss
"""

import os
import sys
import time
import signal
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from ml_lab.custom_transformer.modern_transformer import ModernSeq2SeqTransformer, ModernTransformerConfig


class ResumableTrainer:
    """
    Advanced training engine with pause, checkpointing, and resume capabilities.
    """
    def __init__(
        self,
        model: ModernSeq2SeqTransformer,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        checkpoint_dir: str = "ml_lab/checkpoints",
        save_every_steps: int = 500,
        pause_flag_file: str = "pause.flag",
        label_smoothing: float = 0.1,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 2,
        use_amp: bool = True,
        device: Optional[torch.device] = None
    ):
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_every_steps = save_every_steps
        self.pause_flag_file = Path(pause_flag_file)
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp and (self.device.type == "cuda")

        # PyTorch AMP FP16 Scaler
        self.scaler = GradScaler(enabled=self.use_amp)

        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=model.pad_idx,
            label_smoothing=label_smoothing
        )

        # State tracking
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self._pause_requested = False

        # Register signal handlers for graceful termination
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        print(f"\n[SIGNAL] Received signal {signum}. Requesting graceful pause & checkpoint...")
        self._pause_requested = True

    def check_pause_flag(self) -> bool:
        """Checks if pause.flag file exists or signal was caught."""
        if self._pause_requested:
            return True
        if self.pause_flag_file.exists():
            print(f"\n[PAUSE FLAG] Found '{self.pause_flag_file}'. Gracefully pausing training...")
            try:
                self.pause_flag_file.unlink()  # Remove flag file
            except Exception:
                pass
            return True
        return False

    def save_checkpoint(
        self,
        filename: str,
        is_best: bool = False,
        extra_meta: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Saves a full resumable checkpoint containing all training states.
        """
        save_path = self.checkpoint_dir / filename
        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "config": self.model.config.__dict__,
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history,
            "rng_state": {
                "cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            },
            "timestamp": time.time(),
            "extra_meta": extra_meta or {}
        }
        torch.save(state, save_path)
        print(f"[CHECKPOINT] Saved training state to: {save_path} (step {self.global_step}, epoch {self.epoch})")

        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(state, best_path)
            print(f"[*] Best checkpoint updated: {best_path} (Val Loss: {self.best_val_loss:.4f})")

        return save_path

    def resume_from_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Restores full model, optimizer, scaler, scheduler, and RNG state.
        """
        if not os.path.exists(checkpoint_path):
            print(f"[RESUME ERROR] Checkpoint not found at: {checkpoint_path}")
            return False

        print(f"[RESUME] Loading training state from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Restore model
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print("  - Model weights restored successfully.")

        # Restore optimizer
        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("  - Optimizer states (momentum & variance) restored.")

        # Restore scaler
        if self.use_amp and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print("  - AMP GradScaler state restored.")

        # Restore scheduler
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("  - LR scheduler restored.")

        # Restore metadata
        self.epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.train_loss_history = checkpoint.get("train_loss_history", [])
        self.val_loss_history = checkpoint.get("val_loss_history", [])

        # Restore RNG states
        if "rng_state" in checkpoint:
            rng = checkpoint["rng_state"]
            if rng.get("cpu") is not None:
                torch.set_rng_state(rng["cpu"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
            print("  - Deterministic RNG states restored.")

        print(f"[RESUME SUCCESS] Ready to continue from Epoch {self.epoch + 1}, Step {self.global_step} (Best Val: {self.best_val_loss:.4f})")
        return True

    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        log_interval: int = 100
    ) -> Dict[str, Any]:
        """
        Trains for one epoch with periodic checkpointing and pause detection.
        Returns dict with status ('completed' or 'paused') and metrics.
        """
        self.model.train()
        epoch_loss = 0.0
        running_loss = 0.0
        total_tokens = 0
        start_time = time.time()
        self.optimizer.zero_grad()

        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            # 1. Check for pause request
            if self.check_pause_flag():
                self.save_checkpoint("checkpoint_latest.pt", extra_meta={"paused": True})
                return {
                    "status": "paused",
                    "epoch": self.epoch,
                    "global_step": self.global_step,
                    "message": "Training gracefully paused by user request."
                }

            # Batch tensors: src, tgt_input, tgt_output
            src = batch["src"].to(self.device, non_blocking=True)
            tgt_in = batch["tgt_in"].to(self.device, non_blocking=True)
            tgt_out = batch["tgt_out"].to(self.device, non_blocking=True)

            # 2. Forward pass with mixed precision AMP
            with autocast(enabled=self.use_amp):
                logits = self.model(src, tgt_in)
                # Reshape for cross entropy: (batch * seq, vocab) vs (batch * seq)
                loss = self.criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
                loss = loss / self.gradient_accumulation_steps

            # 3. Backward pass
            self.scaler.scale(loss).backward()

            batch_tokens = (tgt_out != self.model.pad_idx).sum().item()
            total_tokens += batch_tokens
            running_loss += loss.item() * self.gradient_accumulation_steps
            epoch_loss += loss.item() * self.gradient_accumulation_steps

            # 4. Optimizer step after gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.scheduler:
                    self.scheduler.step()

                self.global_step += 1

                # Periodic step checkpoint
                if self.global_step % self.save_every_steps == 0:
                    self.save_checkpoint("checkpoint_latest.pt")

            # 5. Logging
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
                avg_loss = running_loss / (log_interval if (batch_idx + 1) % log_interval == 0 else (batch_idx % log_interval + 1))
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start_time
                tok_per_sec = total_tokens / max(elapsed, 0.001)

                print(f"Epoch {self.epoch+1:2d} | Batch {batch_idx+1:5d}/{num_batches:5d} | "
                      f"Loss: {avg_loss:.4f} | PPL: {math.exp(min(avg_loss, 20)):.2f} | "
                      f"LR: {lr:.2e} | Speed: {tok_per_sec:.0f} tok/s")

                self.train_loss_history.append(avg_loss)
                running_loss = 0.0

        mean_train_loss = epoch_loss / num_batches

        # 6. Validation
        val_loss = float("inf")
        if val_loader:
            val_loss = self.evaluate(val_loader)
            self.val_loss_history.append(val_loss)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint("checkpoint_latest.pt", is_best=is_best)
            print(f"--- Epoch {self.epoch+1} Summary: Train Loss: {mean_train_loss:.4f} | Val Loss: {val_loss:.4f} | Best Val: {self.best_val_loss:.4f} ---")
        else:
            self.save_checkpoint("checkpoint_latest.pt")

        self.epoch += 1
        return {
            "status": "completed",
            "epoch": self.epoch,
            "train_loss": mean_train_loss,
            "val_loss": val_loss
        }

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        """Computes validation loss across evaluation dataset."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)

        for batch in val_loader:
            src = batch["src"].to(self.device, non_blocking=True)
            tgt_in = batch["tgt_in"].to(self.device, non_blocking=True)
            tgt_out = batch["tgt_out"].to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(src, tgt_in)
                loss = self.criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
                total_loss += loss.item()

        self.model.train()
        return total_loss / max(num_batches, 1)
