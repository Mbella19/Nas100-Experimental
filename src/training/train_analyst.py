"""
Training script for the Market Analyst (supervised learning).

Trains the Analyst to predict smoothed future returns across
multiple timeframes. After training, the model is frozen for
use with the RL agent.

Memory-optimized for Apple M2 Silicon.

Features:
- Comprehensive logging with TrainingLogger
- Train/Val accuracy and direction accuracy tracking
- Detailed visualizations of training progress
- Memory monitoring and gradient statistics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
import gc
import time

from ..models.analyst import MarketAnalyst, create_analyst
from ..data.features import create_smoothed_target
from ..utils.logging_config import TrainingLogger, setup_logging, get_logger
from ..utils.metrics import (
    MetricsTracker,
    calculate_direction_accuracy,
    calculate_regression_metrics,
    compute_gradient_norm,
    compute_prediction_stats
)
from ..utils.visualization import TrainingVisualizer

logger = get_logger(__name__)


class MultiTimeframeDataset(Dataset):
    """
    Dataset for multi-timeframe analyst training.

    Each sample contains:
    - 15m features with lookback window
    - 1H features with lookback window (subsampled from aligned 15m index)
    - 4H features with lookback window (subsampled from aligned 15m index)
    - Smoothed future return target
    
    FIXED: 1H and 4H lookbacks now correctly subsample from the aligned data
    to get the proper temporal coverage (24 hours of 1H = 24 candles, not 24 indices).
    """

    def __init__(
        self,
        df_15m: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        feature_cols: List[str],
        target: pd.Series,
        lookback_15m: int = 48,
        lookback_1h: int = 24,
        lookback_4h: int = 12
    ):
        """
        Args:
            df_15m: 15-minute DataFrame
            df_1h: 1-hour DataFrame (aligned to 15m index via ffill)
            df_4h: 4-hour DataFrame (aligned to 15m index via ffill)
            feature_cols: Feature columns to use
            target: Smoothed future return target
            lookback_15m: Number of 15m candles (48 = 12 hours)
            lookback_1h: Number of 1H candles to look back (24 = 24 hours)
            lookback_4h: Number of 4H candles to look back (12 = 48 hours)
        """
        self.lookback_15m = lookback_15m
        self.lookback_1h = lookback_1h
        self.lookback_4h = lookback_4h
        
        # Subsampling ratios: how many 15m bars per higher TF bar
        self.subsample_1h = 4   # 4 x 15m = 1H
        self.subsample_4h = 16  # 16 x 15m = 4H

        # Get feature matrices
        self.features_15m = df_15m[feature_cols].values.astype(np.float32)
        self.features_1h = df_1h[feature_cols].values.astype(np.float32)
        self.features_4h = df_4h[feature_cols].values.astype(np.float32)
        self.targets = target.values.astype(np.float32)

        # FIXED: Calculate start index based on actual temporal coverage needed
        # For 1H lookback: need lookback_1h * 4 indices (since data is aligned to 15m)
        # For 4H lookback: need lookback_4h * 16 indices
        self.start_idx = max(
            lookback_15m,
            lookback_1h * self.subsample_1h,
            lookback_4h * self.subsample_4h
        )
        self.valid_mask = ~np.isnan(self.targets[self.start_idx:])
        self.valid_indices = np.where(self.valid_mask)[0] + self.start_idx

        logger.info(f"Dataset created with {len(self.valid_indices)} valid samples")
        logger.info(f"  15m lookback: {lookback_15m} bars = {lookback_15m * 15 / 60:.1f} hours")
        logger.info(f"  1H lookback: {lookback_1h} bars = {lookback_1h} hours (using {lookback_1h * self.subsample_1h} aligned indices)")
        logger.info(f"  4H lookback: {lookback_4h} bars = {lookback_4h * 4} hours (using {lookback_4h * self.subsample_4h} aligned indices)")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actual_idx = self.valid_indices[idx]

        # Get 15m lookback window (direct indexing)
        x_15m = self.features_15m[actual_idx - self.lookback_15m:actual_idx]
        
        # FIXED: Get 1H lookback by subsampling every 4th bar from aligned data
        # This gives us lookback_1h actual 1H candles worth of data
        idx_range_1h = range(actual_idx - self.lookback_1h * self.subsample_1h, actual_idx, self.subsample_1h)
        x_1h = self.features_1h[list(idx_range_1h)]
        
        # FIXED: Get 4H lookback by subsampling every 16th bar from aligned data
        # This gives us lookback_4h actual 4H candles worth of data
        idx_range_4h = range(actual_idx - self.lookback_4h * self.subsample_4h, actual_idx, self.subsample_4h)
        x_4h = self.features_4h[list(idx_range_4h)]

        # Target
        y = self.targets[actual_idx]

        return (
            torch.tensor(x_15m, dtype=torch.float32),
            torch.tensor(x_1h, dtype=torch.float32),
            torch.tensor(x_4h, dtype=torch.float32),
            torch.tensor([y], dtype=torch.float32)
        )


class StandardizedMSELoss(nn.Module):
    """
    Loss function that GUARANTEES gradient signal at mode collapse.
    
    ROOT CAUSE OF ALL PREVIOUS FAILURES:
    - Variance penalty: d(Var)/d(pred_i) = 0 at collapse
    - Absolute diff: d|x|/dx = undefined at x=0  
    - Squared diff: d(x²)/dx = 2x = 0 at x=0
    
    ALL pairwise/variance losses have ZERO gradient when predictions are identical!
    
    SOLUTION: STANDARDIZE predictions before MSE.
    
    pred_std = (pred - pred.mean()) / (pred.std() + eps) * target.std() + target.mean()
    loss = MSE(pred_std, target)
    
    WHY THIS WORKS:
    When pred.std() → 0, the scaling factor target.std()/(pred.std() + eps) → HUGE
    This AMPLIFIES any tiny deviation from constant, creating massive gradient.
    
    The gradient does NOT go to zero at collapse - it goes to INFINITY!
    This is the opposite behavior of all previous losses.
    """
    
    def __init__(self, standardize_weight: float = 0.5):
        """
        Args:
            standardize_weight: Blend between raw MSE (0) and standardized MSE (1)
                               0.5 = equal blend, which balances learning mean vs variance
        """
        super().__init__()
        self.standardize_weight = standardize_weight
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = predictions.squeeze()
        targets = targets.squeeze()
        
        # 1. Raw MSE - learns to predict correct mean
        raw_mse = nn.functional.mse_loss(predictions, targets)
        
        # 2. Standardized MSE - forces correct variance
        pred_mean = predictions.mean()
        pred_std = predictions.std()
        target_mean = targets.mean()
        target_std = targets.std()
        
        # Standardize predictions to match target distribution
        # Add small epsilon to avoid division by zero, but keep it small
        # so that when std is tiny, the scaling is HUGE (strong gradient)
        eps = 1e-6
        
        # Center and scale predictions to match target statistics
        pred_centered = predictions - pred_mean
        pred_standardized = pred_centered / (pred_std + eps) * target_std + target_mean
        
        standardized_mse = nn.functional.mse_loss(pred_standardized, targets)
        
        # 3. Direct variance matching penalty (backup gradient signal)
        # Log ratio has gradient even at small values
        variance_ratio = (pred_std + eps) / (target_std + eps)
        variance_penalty = (torch.log(variance_ratio)) ** 2
        
        # Combined loss
        total_loss = (1 - self.standardize_weight) * raw_mse + \
                     self.standardize_weight * standardized_mse + \
                     0.1 * variance_penalty
        
        return total_loss


class AnalystTrainer:
    """
    Trainer class for the Market Analyst model.

    Features:
    - AdamW optimizer with weight decay
    - DirectionalLoss for robustness and direction accuracy
    - Early stopping
    - Memory-efficient batch processing
    - Checkpoint saving
    - Comprehensive logging and metrics
    - Training visualizations
    """

    def __init__(
        self,
        model: MarketAnalyst,
        device: torch.device,
        learning_rate: float = 1e-3,  # Increased from 1e-4 to escape mode collapse
        weight_decay: float = 1e-5,
        patience: int = 10,
        cache_clear_interval: int = 50,
        log_dir: Optional[str] = None,
        visualize: bool = True
    ):
        """
        Args:
            model: MarketAnalyst model
            device: Torch device
            learning_rate: Learning rate
            weight_decay: AdamW weight decay
            patience: Early stopping patience
            cache_clear_interval: Clear MPS cache every N batches
            log_dir: Directory for logs and visualizations
            visualize: Whether to create visualizations
        """
        self.model = model.to(device)
        self.device = device
        self.patience = patience
        self.cache_clear_interval = cache_clear_interval
        self.visualize = visualize
        self.log_dir = Path(log_dir) if log_dir else None

        # Setup logging
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.training_logger = TrainingLogger(
            name="analyst_training",
            log_dir=str(self.log_dir) if self.log_dir else None,
            log_every_n_batches=50,
            verbose=True
        )

        # Setup visualizer
        if self.visualize and self.log_dir:
            self.visualizer = TrainingVisualizer(save_dir=str(self.log_dir / "plots"))
        else:
            self.visualizer = None

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Loss function: StandardizedMSELoss - GUARANTEED gradient at collapse
        # KEY FIX: Standardizes predictions before MSE
        # When pred.std() → 0, gradient → INFINITY (not zero!)
        # This is the opposite of variance penalty which has gradient → 0
        self.criterion = StandardizedMSELoss(standardize_weight=0.5)

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_direction_accs = []
        self.val_direction_accs = []
        self.learning_rates = []
        self.grad_norms = []
        self.memory_usage = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # Batch-level tracking
        self.batch_losses = []
        self.batch_grad_norms = []

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int
    ) -> Tuple[float, float, float]:
        """
        Train for one epoch with detailed metrics.

        Returns:
            Tuple of (avg_loss, accuracy, direction_accuracy)
        """
        self.model.train()
        total_loss = 0.0
        n_batches = len(train_loader)

        # Metrics tracker for the epoch
        metrics_tracker = MetricsTracker()

        self.training_logger.start_epoch(epoch, total_epochs)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}")
        for batch_idx, (x_15m, x_1h, x_4h, targets) in enumerate(pbar):
            batch_start = time.time()

            # Move to device
            x_15m = x_15m.to(self.device)
            x_1h = x_1h.to(self.device)
            x_4h = x_4h.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            _, predictions = self.model(x_15m, x_1h, x_4h)
            loss = self.criterion(predictions, targets)

            # Backward pass
            loss.backward()

            # Compute gradient norm before clipping
            grad_norm = compute_gradient_norm(self.model)
            self.batch_grad_norms.append(grad_norm)

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Track metrics
            loss_val = loss.item()
            total_loss += loss_val
            self.batch_losses.append(loss_val)

            # Store predictions for accuracy calculation
            metrics_tracker.update(
                predictions.detach().cpu().numpy(),
                targets.detach().cpu().numpy(),
                loss_val
            )

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log batch metrics
            self.training_logger.log_batch(
                batch=batch_idx,
                total_batches=n_batches,
                loss=loss_val,
                grad_norm=grad_norm,
                lr=current_lr
            )

            # Memory cleanup
            if batch_idx % self.cache_clear_interval == 0:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_val:.6f}',
                'grad': f'{grad_norm:.4f}',
                'lr': f'{current_lr:.2e}'
            })

            # Clean up batch tensors
            del x_15m, x_1h, x_4h, targets, predictions, loss

        # Compute epoch metrics
        epoch_metrics = metrics_tracker.compute()
        avg_loss = total_loss / n_batches
        direction_acc = epoch_metrics.get('direction_accuracy', 0.0)

        # For regression, we use R² as "accuracy"
        accuracy = max(0, epoch_metrics.get('r2', 0.0))

        return avg_loss, accuracy, direction_acc

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        Validate the model with detailed metrics.

        Returns:
            Tuple of (avg_loss, accuracy, direction_accuracy, predictions, targets)
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        # Collect all predictions and targets
        all_predictions = []
        all_targets = []

        for x_15m, x_1h, x_4h, targets in val_loader:
            x_15m = x_15m.to(self.device)
            x_1h = x_1h.to(self.device)
            x_4h = x_4h.to(self.device)
            targets = targets.to(self.device)

            _, predictions = self.model(x_15m, x_1h, x_4h)
            loss = self.criterion(predictions, targets)

            total_loss += loss.item()
            n_batches += 1

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            del x_15m, x_1h, x_4h, targets, predictions

        # Concatenate all predictions and targets
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)

        # Calculate metrics
        avg_loss = total_loss / n_batches
        reg_metrics = calculate_regression_metrics(all_predictions, all_targets)
        dir_metrics = calculate_direction_accuracy(all_predictions, all_targets)

        # R² as accuracy (clipped to 0 minimum)
        accuracy = max(0, reg_metrics.r2)
        direction_acc = dir_metrics.accuracy

        return avg_loss, accuracy, direction_acc, all_predictions, all_targets

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Full training loop with early stopping, logging, and visualizations.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            max_epochs: Maximum epochs
            save_path: Path to save best model

        Returns:
            Training history
        """
        # Count model parameters
        total_params = sum(p.numel() for p in self.model.parameters())

        self.training_logger.start_training(max_epochs, total_params)
        logger.info(f"Starting training for up to {max_epochs} epochs")
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        for epoch in range(1, max_epochs + 1):
            # Train
            train_loss, train_acc, train_dir_acc = self.train_epoch(
                train_loader, epoch, max_epochs
            )
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.train_direction_accs.append(train_dir_acc)

            # Validate
            val_loss, val_acc, val_dir_acc, val_preds, val_targets = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            self.val_direction_accs.append(val_dir_acc)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            # Compute average gradient norm for epoch
            recent_grad_norms = self.batch_grad_norms[-len(train_loader):]
            avg_grad_norm = np.mean(recent_grad_norms) if recent_grad_norms else 0.0
            self.grad_norms.append(avg_grad_norm)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Log epoch summary
            self.training_logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                train_direction_acc=train_dir_acc,
                val_direction_acc=val_dir_acc,
                lr=current_lr,
                grad_norm=avg_grad_norm,
                extra_metrics={
                    'val_r2': val_acc,
                    'val_dir_precision_up': calculate_direction_accuracy(val_preds, val_targets).up_precision,
                    'val_dir_recall_up': calculate_direction_accuracy(val_preds, val_targets).up_recall
                }
            )

            # Log prediction statistics for debugging
            pred_stats = compute_prediction_stats(val_preds)
            logger.info(f"  Prediction Stats: mean={pred_stats['mean']:.6f}, std={pred_stats['std']:.6f}, "
                       f"min={pred_stats['min']:.6f}, max={pred_stats['max']:.6f}")
            logger.info(f"  Distribution: {pred_stats['pct_positive']*100:.1f}% pos, "
                       f"{pred_stats['pct_negative']*100:.1f}% neg, "
                       f"{pred_stats['pct_near_zero']*100:.1f}% near zero")

            # Log sample predictions vs targets
            if epoch % 10 == 0 or epoch == 1:
                self.training_logger.log_validation_details(val_preds, val_targets)

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0

                # Save best model
                if save_path:
                    self.save_checkpoint(save_path, epoch, is_best=True)
            else:
                self.epochs_without_improvement += 1

            # Create epoch visualization
            if self.visualizer and epoch % 5 == 0:
                # Get training predictions for visualization
                train_preds, train_tgts = self._get_train_predictions(train_loader)

                metrics = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_r2': train_acc,
                    'val_r2': val_acc,
                    'train_dir_acc': train_dir_acc,
                    'val_dir_acc': val_dir_acc,
                    'learning_rate': current_lr,
                    'grad_norm': avg_grad_norm
                }

                self.visualizer.plot_epoch_summary(
                    epoch=epoch,
                    train_predictions=train_preds,
                    train_targets=train_tgts,
                    val_predictions=val_preds,
                    val_targets=val_targets,
                    metrics=metrics,
                    save_name=f'epoch_{epoch:03d}_summary.png'
                )
                self.visualizer.close_all()

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                self.training_logger.end_training(reason="early_stopped")
                break

            # Memory cleanup
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        else:
            self.training_logger.end_training(reason="completed")

        # Create final visualizations
        if self.visualizer:
            history = self.get_history()
            self.visualizer.plot_training_curves(
                history,
                title="Market Analyst Training History",
                save_name="training_curves.png"
            )

            # Direction confusion matrix
            _, _, _, final_preds, final_targets = self.validate(val_loader)
            self.visualizer.plot_direction_confusion(
                final_preds, final_targets,
                title="Final Direction Classification Performance",
                save_name="direction_confusion.png"
            )

            self.visualizer.plot_predictions_vs_targets(
                final_preds, final_targets,
                title="Final Predictions vs Targets",
                save_name="predictions_vs_targets.png"
            )

            # Learning dynamics
            if len(self.batch_losses) > 0:
                self.visualizer.plot_learning_dynamics(
                    self.batch_losses,
                    self.batch_grad_norms,
                    title="Learning Dynamics",
                    save_name="learning_dynamics.png"
                )

            self.visualizer.close_all()

        return self.get_history()

    @torch.no_grad()
    def _get_train_predictions(
        self,
        train_loader: DataLoader,
        max_batches: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get a sample of training predictions for visualization."""
        self.model.eval()
        predictions = []
        targets = []

        for batch_idx, (x_15m, x_1h, x_4h, tgt) in enumerate(train_loader):
            if batch_idx >= max_batches:
                break

            x_15m = x_15m.to(self.device)
            x_1h = x_1h.to(self.device)
            x_4h = x_4h.to(self.device)

            _, pred = self.model(x_15m, x_1h, x_4h)
            predictions.append(pred.cpu().numpy())
            targets.append(tgt.numpy())

        self.model.train()
        return np.concatenate(predictions), np.concatenate(targets)

    def get_history(self) -> Dict[str, List[float]]:
        """Get training history as dictionary."""
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_acc': self.train_accs,
            'val_acc': self.val_accs,
            'train_direction_acc': self.train_direction_accs,
            'val_direction_acc': self.val_direction_accs,
            'learning_rate': self.learning_rates,
            'grad_norm': self.grad_norms,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': len(self.train_losses)
        }

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        is_best: bool = False
    ):
        """Save model checkpoint."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'train_direction_accs': self.train_direction_accs,
            'val_direction_accs': self.val_direction_accs,
            'best_val_loss': self.best_val_loss,
            'config': {
                'd_model': self.model.d_model,
                'context_dim': self.model.context_dim
            }
        }

        filename = 'best.pt' if is_best else f'epoch_{epoch}.pt'
        torch.save(checkpoint, path / filename)
        logger.info(f"Saved checkpoint to {path / filename}")


def train_analyst(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    feature_cols: List[str],
    save_path: str,
    config: Optional[object] = None,
    device: Optional[torch.device] = None,
    visualize: bool = True
) -> Tuple[MarketAnalyst, Dict]:
    """
    Main function to train the Market Analyst.

    Args:
        df_15m: 15-minute DataFrame with features
        df_1h: 1-hour DataFrame with features
        df_4h: 4-hour DataFrame with features
        feature_cols: Feature columns to use
        save_path: Path to save model
        config: AnalystConfig object
        device: Torch device
        visualize: Whether to create visualizations

    Returns:
        Tuple of (trained model, training history)
    """
    # Default configuration
    if config is None:
        from config.settings import Config
        config = Config().analyst

    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    logger.info(f"Training on device: {device}")

    # Create target
    logger.info("Creating smoothed target...")
    target = create_smoothed_target(
        df_15m,
        future_window=config.future_window if hasattr(config, 'future_window') else 12,
        smooth_window=config.smooth_window if hasattr(config, 'smooth_window') else 12
    )

    # Log target statistics
    valid_target = target.dropna()
    logger.info(f"Target stats: mean={valid_target.mean():.6f}, std={valid_target.std():.6f}, "
               f"min={valid_target.min():.6f}, max={valid_target.max():.6f}")
    logger.info(f"Target distribution: {(valid_target > 0).mean()*100:.1f}% positive, "
               f"{(valid_target < 0).mean()*100:.1f}% negative")

    # Create dataset
    logger.info("Creating dataset...")
    dataset = MultiTimeframeDataset(
        df_15m, df_1h, df_4h,
        feature_cols, target,
        lookback_15m=config.lookback_15m if hasattr(config, 'lookback_15m') else 48,
        lookback_1h=config.lookback_1h if hasattr(config, 'lookback_1h') else 24,
        lookback_4h=config.lookback_4h if hasattr(config, 'lookback_4h') else 12
    )

    # Split into train/validation using CHRONOLOGICAL split (NOT random!)
    # CRITICAL: Random splits cause look-ahead bias in time series data.
    # The model would train on "Tuesday" and test on "Monday", memorizing the future.
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size

    # Use Subset with sequential indices for chronological split
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, len(dataset)))

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    logger.info(f"Train size: {train_size} (indices 0-{train_size-1})")
    logger.info(f"Val size: {val_size} (indices {train_size}-{len(dataset)-1})")
    logger.info("Using CHRONOLOGICAL split (train on past, validate on future)")

    # Create data loaders
    batch_size = config.batch_size if hasattr(config, 'batch_size') else 32

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # MPS doesn't support multiprocessing well
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Create model
    feature_dims = {
        '15m': len(feature_cols),
        '1h': len(feature_cols),
        '4h': len(feature_cols)
    }

    model = create_analyst(feature_dims, config, device)
    logger.info(f"Created MarketAnalyst with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Log model architecture
    logger.info(f"Model config: d_model={model.d_model}, context_dim={model.context_dim}")

    # Create trainer
    trainer = AnalystTrainer(
        model=model,
        device=device,
        learning_rate=config.learning_rate if hasattr(config, 'learning_rate') else 1e-4,
        weight_decay=config.weight_decay if hasattr(config, 'weight_decay') else 1e-5,
        patience=config.patience if hasattr(config, 'patience') else 10,
        log_dir=save_path,
        visualize=visualize
    )

    # Train
    history = trainer.train(
        train_loader,
        val_loader,
        max_epochs=config.max_epochs if hasattr(config, 'max_epochs') else 100,
        save_path=save_path
    )

    # Load best model
    best_path = Path(save_path) / 'best.pt'
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")

    # Final summary
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best validation loss: {history['best_val_loss']:.6f}")
    logger.info(f"Final train direction acc: {history['train_direction_acc'][-1]*100:.2f}%")
    logger.info(f"Final val direction acc: {history['val_direction_acc'][-1]*100:.2f}%")
    logger.info(f"Total epochs trained: {history['epochs_trained']}")
    logger.info("=" * 70)

    return model, history


if __name__ == '__main__':
    # Example usage
    print("Use this module via: python -m src.training.train_analyst")
    print("Or import and call train_analyst() function")
