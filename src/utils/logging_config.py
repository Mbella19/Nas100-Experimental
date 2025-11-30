"""
Comprehensive logging configuration for training monitoring.

Features:
- Console and file logging
- Structured training logs with metrics
- Memory usage tracking
- Timing information
- Gradient statistics
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import json
import torch
import gc
import numpy as np
from dataclasses import dataclass, field, asdict


def setup_logging(
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    name: str = "trading_system"
) -> logging.Logger:
    """
    Setup comprehensive logging with console and file handlers.

    Args:
        log_dir: Directory for log files (None for console only)
        level: Logging level
        name: Logger name

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    # Detailed format for training
    detailed_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(detailed_format)
    logger.addHandler(console_handler)

    # File handler if log_dir provided
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(
            log_path / f'training_{timestamp}.log'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_format)
        logger.addHandler(file_handler)

        logger.info(f"Logging to: {log_path / f'training_{timestamp}.log'}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


@dataclass
class EpochMetrics:
    """Container for epoch-level metrics."""
    epoch: int
    train_loss: float
    val_loss: float
    train_acc: float = 0.0
    val_acc: float = 0.0
    train_direction_acc: float = 0.0
    val_direction_acc: float = 0.0
    learning_rate: float = 0.0
    epoch_time: float = 0.0
    memory_used_mb: float = 0.0
    grad_norm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BatchMetrics:
    """Container for batch-level metrics."""
    batch: int
    loss: float
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    batch_time: float = 0.0
    samples_per_sec: float = 0.0


class TrainingLogger:
    """
    Comprehensive training logger with metrics tracking.

    Features:
    - Epoch and batch level logging
    - Memory monitoring
    - Gradient statistics
    - Learning rate tracking
    - JSON export for analysis
    """

    def __init__(
        self,
        name: str = "training",
        log_dir: Optional[str] = None,
        log_every_n_batches: int = 50,
        verbose: bool = True
    ):
        """
        Args:
            name: Logger name
            log_dir: Directory for logs and metrics
            log_every_n_batches: Log batch metrics every N batches
            verbose: Print detailed logs to console
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_every_n_batches = log_every_n_batches
        self.verbose = verbose

        # Setup logger
        self.logger = setup_logging(log_dir, name=name)

        # Metrics storage
        self.epoch_history: List[EpochMetrics] = []
        self.batch_history: List[BatchMetrics] = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

        # Timing
        self.epoch_start_time = None
        self.batch_start_time = None
        self.training_start_time = None

    def start_training(self, total_epochs: int, model_params: int):
        """Log training start."""
        self.training_start_time = datetime.now()

        self.logger.info("=" * 70)
        self.logger.info("TRAINING STARTED")
        self.logger.info("=" * 70)
        self.logger.info(f"Total epochs: {total_epochs}")
        self.logger.info(f"Model parameters: {model_params:,}")
        self.logger.info(f"Device: {self._get_device()}")
        self.logger.info(f"Memory available: {self._get_memory_info()}")
        self.logger.info("=" * 70)

    def start_epoch(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        import time
        self.epoch_start_time = time.time()
        self.current_epoch = epoch

        self.logger.info("-" * 70)
        self.logger.info(f"EPOCH {epoch}/{total_epochs}")
        self.logger.info("-" * 70)

    def log_batch(
        self,
        batch: int,
        total_batches: int,
        loss: float,
        grad_norm: float = 0.0,
        lr: float = 0.0,
        extra_metrics: Optional[Dict[str, float]] = None
    ):
        """Log batch metrics."""
        import time

        if self.batch_start_time is not None:
            batch_time = time.time() - self.batch_start_time
        else:
            batch_time = 0.0
        self.batch_start_time = time.time()

        # Store metrics
        metrics = BatchMetrics(
            batch=batch,
            loss=loss,
            grad_norm=grad_norm,
            learning_rate=lr,
            batch_time=batch_time
        )
        self.batch_history.append(metrics)

        # Log periodically
        if batch % self.log_every_n_batches == 0 or batch == total_batches - 1:
            progress = (batch + 1) / total_batches * 100
            msg = (f"  Batch {batch+1:5d}/{total_batches} ({progress:5.1f}%) | "
                   f"Loss: {loss:.6f} | Grad: {grad_norm:.4f} | LR: {lr:.2e}")

            if extra_metrics:
                for k, v in extra_metrics.items():
                    msg += f" | {k}: {v:.4f}"

            self.logger.info(msg)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float = 0.0,
        val_acc: float = 0.0,
        train_direction_acc: float = 0.0,
        val_direction_acc: float = 0.0,
        lr: float = 0.0,
        grad_norm: float = 0.0,
        extra_metrics: Optional[Dict[str, float]] = None
    ):
        """Log epoch summary with all metrics."""
        import time

        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        memory_mb = self._get_memory_usage_mb()

        # Store metrics
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            train_direction_acc=train_direction_acc,
            val_direction_acc=val_direction_acc,
            learning_rate=lr,
            epoch_time=epoch_time,
            memory_used_mb=memory_mb,
            grad_norm=grad_norm
        )
        self.epoch_history.append(metrics)

        # Check for improvement
        improved = val_loss < self.best_val_loss
        if improved:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.epochs_without_improvement = 0
            improvement_str = " ★ NEW BEST"
        else:
            self.epochs_without_improvement += 1
            improvement_str = ""

        # Log summary
        self.logger.info("-" * 70)
        self.logger.info(f"EPOCH {epoch} SUMMARY{improvement_str}")
        self.logger.info("-" * 70)
        self.logger.info(f"  Train Loss:     {train_loss:.6f}")
        self.logger.info(f"  Val Loss:       {val_loss:.6f}")
        self.logger.info(f"  Train Acc:      {train_acc*100:.2f}%")
        self.logger.info(f"  Val Acc:        {val_acc*100:.2f}%")
        self.logger.info(f"  Train Dir Acc:  {train_direction_acc*100:.2f}%")
        self.logger.info(f"  Val Dir Acc:    {val_direction_acc*100:.2f}%")
        self.logger.info(f"  Learning Rate:  {lr:.2e}")
        self.logger.info(f"  Grad Norm:      {grad_norm:.4f}")
        self.logger.info(f"  Epoch Time:     {epoch_time:.1f}s")
        self.logger.info(f"  Memory Used:    {memory_mb:.0f} MB")
        self.logger.info(f"  Best Val Loss:  {self.best_val_loss:.6f}")
        self.logger.info(f"  Patience:       {self.epochs_without_improvement}")

        if extra_metrics:
            self.logger.info("  Extra Metrics:")
            for k, v in extra_metrics.items():
                self.logger.info(f"    {k}: {v:.4f}")

        self.logger.info("-" * 70)

    def log_validation_details(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        sample_size: int = 10
    ):
        """Log sample predictions vs targets for debugging."""
        self.logger.info("  Sample Predictions vs Targets:")
        indices = np.random.choice(len(predictions), min(sample_size, len(predictions)), replace=False)
        for i, idx in enumerate(indices):
            # Convert to float to handle numpy arrays
            pred = float(predictions[idx].item() if hasattr(predictions[idx], 'item') else predictions[idx])
            target = float(targets[idx].item() if hasattr(targets[idx], 'item') else targets[idx])
            error = abs(pred - target)
            direction_match = "✓" if (pred > 0) == (target > 0) else "✗"
            self.logger.info(f"    [{i+1}] Pred: {pred:+.6f} | Target: {target:+.6f} | "
                           f"Error: {error:.6f} | Dir: {direction_match}")

    def log_gradient_stats(self, model: torch.nn.Module):
        """Log gradient statistics for debugging."""
        total_norm = 0.0
        param_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_norms[name] = param_norm

        total_norm = total_norm ** 0.5

        # Log top 5 gradient norms
        sorted_norms = sorted(param_norms.items(), key=lambda x: x[1], reverse=True)[:5]
        self.logger.debug("  Top Gradient Norms:")
        for name, norm in sorted_norms:
            self.logger.debug(f"    {name}: {norm:.6f}")

        return total_norm

    def end_training(self, reason: str = "completed"):
        """Log training end summary."""
        total_time = (datetime.now() - self.training_start_time).total_seconds()

        self.logger.info("=" * 70)
        self.logger.info(f"TRAINING {reason.upper()}")
        self.logger.info("=" * 70)
        self.logger.info(f"Total Time: {total_time/60:.1f} minutes")
        self.logger.info(f"Total Epochs: {len(self.epoch_history)}")
        self.logger.info(f"Best Val Loss: {self.best_val_loss:.6f}")
        self.logger.info(f"Best Val Acc: {self.best_val_acc*100:.2f}%")
        self.logger.info("=" * 70)

        # Save metrics to JSON
        if self.log_dir:
            self._save_metrics()

    def _save_metrics(self):
        """Save all metrics to JSON file."""
        metrics_path = self.log_dir / 'training_metrics.json'

        data = {
            'epoch_history': [m.to_dict() for m in self.epoch_history],
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'total_epochs': len(self.epoch_history)
        }

        with open(metrics_path, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Metrics saved to: {metrics_path}")

    def get_history(self) -> Dict[str, List[float]]:
        """Get training history as dictionary."""
        return {
            'train_loss': [m.train_loss for m in self.epoch_history],
            'val_loss': [m.val_loss for m in self.epoch_history],
            'train_acc': [m.train_acc for m in self.epoch_history],
            'val_acc': [m.val_acc for m in self.epoch_history],
            'train_direction_acc': [m.train_direction_acc for m in self.epoch_history],
            'val_direction_acc': [m.val_direction_acc for m in self.epoch_history],
            'learning_rate': [m.learning_rate for m in self.epoch_history],
            'grad_norm': [m.grad_norm for m in self.epoch_history],
            'memory_mb': [m.memory_used_mb for m in self.epoch_history]
        }

    def _get_device(self) -> str:
        """Get current device."""
        if torch.backends.mps.is_available():
            return "MPS (Apple Silicon)"
        elif torch.cuda.is_available():
            return f"CUDA ({torch.cuda.get_device_name(0)})"
        return "CPU"

    def _get_memory_info(self) -> str:
        """Get memory information."""
        import psutil
        mem = psutil.virtual_memory()
        return f"{mem.available / 1024**3:.1f} GB available / {mem.total / 1024**3:.1f} GB total"

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024**2
