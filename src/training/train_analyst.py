"""
Training script for the Market Analyst (supervised learning).

Trains the Analyst to classify smoothed future returns across
multiple timeframes (5-class directional buckets). After training,
the model is frozen for use with the RL agent.

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

from src.models.analyst import MarketAnalyst, create_analyst
from src.models.tcn_analyst import TCNAnalyst, create_tcn_analyst
from src.data.loader import load_ohlcv
from src.data.features import (
    engineer_all_features,
    create_smoothed_target,
    create_return_classes,
    create_binary_direction_target,
    add_market_sessions,
    detect_fractals,
    detect_structure_breaks,
    create_auxiliary_targets
)
from src.data.resampler import resample_all_timeframes, align_timeframes, create_multi_timeframe_dataset
from src.data.normalizer import normalize_multi_timeframe
from ..utils.logging_config import TrainingLogger, get_logger
from ..utils.metrics import (
    MetricsTracker,
    calculate_classification_metrics,
    compute_gradient_norm
)
from ..utils.visualization import TrainingVisualizer

logger = get_logger(__name__)


class MultiTimeframeDataset(Dataset):
    """
    Dataset for multi-timeframe analyst training.

    Each sample contains:
    - 5m features with lookback window (base timeframe)
    - 15m features with lookback window (subsampled from aligned 5m index)
    - 45m features with lookback window (subsampled from aligned 5m index)
    - Direction target (binary or multi-class)
    - Optional auxiliary targets (volatility, regime)

    FIXED: 15m and 45m lookbacks now correctly subsample from the aligned data
    to get the proper temporal coverage.
    """

    def __init__(
        self,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_45m: pd.DataFrame,
        feature_cols: List[str],
        target: pd.Series,
        class_labels: pd.Series,
        lookback_5m: int = 48,
        lookback_15m: int = 16,
        lookback_45m: int = 6,
        valid_mask: Optional[pd.Series] = None,
        volatility_target: Optional[pd.Series] = None,
        regime_target: Optional[pd.Series] = None,
        is_binary: bool = False,
        horizon_1h_target: Optional[pd.Series] = None,
        horizon_2h_target: Optional[pd.Series] = None
    ):
        """
        Args:
            df_5m: 5-minute DataFrame (base timeframe)
            df_15m: 15-minute DataFrame (aligned to 5m index via ffill)
            df_45m: 45-minute DataFrame (aligned to 5m index via ffill)
            feature_cols: Feature columns to use
            target: Smoothed future return target (continuous)
            class_labels: Direction labels (0=Down, 1=Up for binary; 0-4 for 5-class)
            lookback_5m: Number of 5m candles (48 = 4 hours)
            lookback_15m: Number of 15m candles to look back (16 = 4 hours)
            lookback_45m: Number of 45m candles to look back (6 = 4.5 hours)
            valid_mask: Optional boolean mask for valid samples (from binary target)
            volatility_target: Optional volatility target for auxiliary loss
            regime_target: Optional regime target for auxiliary loss
            is_binary: Whether this is binary classification
            horizon_1h_target: Optional 1H horizon binary target (for multi-horizon)
            horizon_2h_target: Optional 2H horizon binary target (for multi-horizon)
        """
        self.lookback_5m = lookback_5m
        self.lookback_15m = lookback_15m
        self.lookback_45m = lookback_45m
        self.is_binary = is_binary

        # Subsampling ratios: how many 5m bars per higher TF bar
        self.subsample_15m = 3   # 3 x 5m = 15m
        self.subsample_45m = 9   # 9 x 5m = 45m

        # Get feature matrices
        self.features_5m = df_5m[feature_cols].values.astype(np.float32)
        self.features_15m = df_15m[feature_cols].values.astype(np.float32)
        self.features_45m = df_45m[feature_cols].values.astype(np.float32)
        self.targets = target.values.astype(np.float32)
        self.class_labels = class_labels.values.astype(np.float32)

        # Auxiliary targets (optional)
        self.has_aux = volatility_target is not None and regime_target is not None
        if self.has_aux:
            self.volatility_target = volatility_target.values.astype(np.float32)
            self.regime_target = regime_target.values.astype(np.float32)
        else:
            self.volatility_target = None
            self.regime_target = None

        # Multi-horizon targets (optional, for addressing target mismatch)
        self.has_multi_horizon = horizon_1h_target is not None and horizon_2h_target is not None
        if self.has_multi_horizon:
            self.horizon_1h_target = horizon_1h_target.values.astype(np.float32)
            self.horizon_2h_target = horizon_2h_target.values.astype(np.float32)
        else:
            self.horizon_1h_target = None
            self.horizon_2h_target = None

        # FIXED: Calculate start index based on actual temporal coverage needed
        # For 15m lookback: need lookback_15m * 3 indices (since data is aligned to 5m)
        # For 45m lookback: need lookback_45m * 9 indices
        self.start_idx = max(
            lookback_5m,
            lookback_15m * self.subsample_15m,
            lookback_45m * self.subsample_45m
        )

        # Build valid indices based on:
        # 1. NaN check on class_labels
        # 2. Optional external valid_mask (from binary target filtering)
        base_mask = ~np.isnan(self.class_labels[self.start_idx:])

        if valid_mask is not None:
            # Combine with external mask (for binary filtering)
            external_mask = valid_mask.values[self.start_idx:].astype(bool)
            combined_mask = base_mask & external_mask
        else:
            combined_mask = base_mask

        self.valid_mask = combined_mask
        self.valid_indices = np.where(self.valid_mask)[0] + self.start_idx

        logger.info(f"Dataset created with {len(self.valid_indices)} valid samples")
        logger.info(f"  5m lookback: {lookback_5m} bars = {lookback_5m * 5 / 60:.1f} hours")
        logger.info(f"  15m lookback: {lookback_15m} bars = {lookback_15m * 15 / 60:.1f} hours")
        logger.info(f"  45m lookback: {lookback_45m} bars = {lookback_45m * 45 / 60:.1f} hours")
        if valid_mask is not None:
            n_filtered = base_mask.sum() - combined_mask.sum()
            logger.info(f"  Filtered {n_filtered} neutral/weak samples for binary mode")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Returns:
            Tuple of (x_5m, x_15m, x_45m, direction_target, volatility_target, regime_target,
                      horizon_1h_target, horizon_2h_target)
            If targets not available, they default to 0.0 or NaN
        """
        actual_idx = self.valid_indices[idx]

        # Get 5m lookback window (direct indexing, includes current candle)
        x_5m = self.features_5m[actual_idx - self.lookback_5m + 1:actual_idx + 1]

        # FIXED: Get 15m lookback by subsampling every 3rd bar from aligned data
        # This gives us lookback_15m actual 15m candles worth of data, INCLUDING current candle
        # Note: range() is exclusive at end, so we use actual_idx + 1 to include current
        idx_range_15m = range(
            actual_idx - (self.lookback_15m - 1) * self.subsample_15m,
            actual_idx + 1,
            self.subsample_15m
        )
        x_15m = self.features_15m[list(idx_range_15m)]

        # FIXED: Get 45m lookback by subsampling every 9th bar from aligned data
        # This gives us lookback_45m actual 45m candles worth of data, INCLUDING current candle
        idx_range_45m = range(
            actual_idx - (self.lookback_45m - 1) * self.subsample_45m,
            actual_idx + 1,
            self.subsample_45m
        )
        x_45m = self.features_45m[list(idx_range_45m)]

        # Direction target (use float for binary, long for multi-class)
        y = self.class_labels[actual_idx]
        if self.is_binary:
            y_tensor = torch.tensor(y, dtype=torch.float32)
        else:
            y_tensor = torch.tensor(int(y), dtype=torch.long)

        # Auxiliary targets
        if self.has_aux:
            vol_target = torch.tensor(self.volatility_target[actual_idx], dtype=torch.float32)
            regime_target = torch.tensor(self.regime_target[actual_idx], dtype=torch.float32)
        else:
            vol_target = torch.tensor(0.0, dtype=torch.float32)
            regime_target = torch.tensor(0.0, dtype=torch.float32)

        # Multi-horizon targets
        if self.has_multi_horizon:
            h1_target = torch.tensor(self.horizon_1h_target[actual_idx], dtype=torch.float32)
            h2_target = torch.tensor(self.horizon_2h_target[actual_idx], dtype=torch.float32)
        else:
            h1_target = torch.tensor(float('nan'), dtype=torch.float32)
            h2_target = torch.tensor(float('nan'), dtype=torch.float32)

        return (
            torch.tensor(x_5m, dtype=torch.float32),
            torch.tensor(x_15m, dtype=torch.float32),
            torch.tensor(x_45m, dtype=torch.float32),
            y_tensor,
            vol_target,
            regime_target,
            h1_target,
            h2_target
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Key properties:
    - Down-weights easy examples (high p_t) where model is already confident
    - Focuses learning on hard examples (low p_t) that the model struggles with
    - Perfect for class collapse where model ignores minority classes
    
    In our case: The model predicts Strong Up (34.7%) instead of Weak Up (3.8%)
    because Strong Up is "easier". Focal loss penalizes these easy examples
    and forces attention to the hard Weak Up distinctions.
    
    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(
        self, 
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.1
    ):
        """
        Args:
            weight: Class weights tensor [num_classes]
            gamma: Focusing parameter. Higher = more focus on hard examples.
                   γ=0 is equivalent to CrossEntropyLoss
                   γ=2 is typical for object detection
            reduction: 'none', 'mean', or 'sum'
            label_smoothing: Prevents overconfidence, helps generalization
        """
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model outputs [batch, num_classes]
            targets: Class labels [batch]
        
        Returns:
            Focal loss value
        """
        num_classes = logits.size(-1)
        
        # Apply label smoothing
        # Converts hard labels to soft: [0,0,1,0,0] -> [0.02, 0.02, 0.92, 0.02, 0.02]
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(logits)
                smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        
        # Compute softmax probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Get probability for true class: p_t
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight: (1 - p_t)^gamma
        # When p_t is high (easy example), weight is low
        # When p_t is low (hard example), weight is high
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute cross-entropy (with or without label smoothing)
        if self.label_smoothing > 0:
            # Log softmax for numerical stability
            log_probs = torch.log_softmax(logits, dim=-1)
            ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
        else:
            ce_loss = nn.functional.cross_entropy(
                logits, targets, weight=self.weight, reduction='none'
            )
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights if provided (for label smoothing path)
        if self.weight is not None and self.label_smoothing > 0:
            class_weight = self.weight[targets]
            focal_loss = focal_loss * class_weight
        
        # Reduce
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for binary classification.

    FL(p_t) = -α * (1 - p_t)^γ * log(p_t)

    Key properties:
    - Prevents prediction collapse where model predicts one class dominantly
    - Down-weights easy examples where model is already confident
    - Forces the model to learn hard examples at the decision boundary

    In our case: Model was predicting Down 74% of time (collapse). Focal loss
    down-weights these easy predictions and focuses on the hard Up/Down boundary.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        pos_weight: Optional[torch.Tensor] = None
    ):
        """
        Args:
            alpha: Balance factor for positive class (0.25 = focus on minority)
            gamma: Focusing parameter. γ=2 is typical.
            label_smoothing: Smooth labels [0,1] -> [0.05, 0.95]
            pos_weight: Optional weight for positive class (handles imbalance)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model outputs [batch] or [batch, 1]
            targets: Binary labels [batch] (0 or 1)

        Returns:
            Focal loss value (scalar)
        """
        # Flatten logits if needed
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Apply label smoothing: [0, 1] -> [eps, 1-eps]
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Compute probabilities
        probs = torch.sigmoid(logits)

        # p_t is probability of correct class
        # For target=1: p_t = probs
        # For target=0: p_t = 1 - probs
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weight: alpha for positive, (1-alpha) for negative
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # BCE loss (element-wise) - use pos_weight if provided for class balancing
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none',
            pos_weight=self.pos_weight  # Handles class imbalance if provided
        )

        # Apply focal and alpha weights
        focal_loss = alpha_weight * focal_weight * bce_loss

        return focal_loss.mean()


class BalancedBCELoss(nn.Module):
    """
    BCE loss with CLASS WEIGHTING + DIVERSITY PENALTY to prevent collapse.

    v10 FIX: SymmetricCrossEntropyLoss failed to prevent class collapse because
    it was designed for NEURAL feature collapse (representations becoming identical),
    not CLASS PREDICTION collapse (model predicting 99% one class).

    This loss addresses class collapse directly through two mechanisms:

    1. CLASS WEIGHTING: Dynamic pos_weight based on CLASS LABELS (not predictions)
       - When class imbalance exists, minority class samples get higher weight
       - This creates strong gradients to balance predictions

    2. DIVERSITY PENALTY: Penalizes when PREDICTION distribution deviates from 50%
       - When model predicts 99% Up, penalty = (0.99 - 0.5)^2 = 0.24
       - Creates gradient DIRECTLY on prediction distribution
       - Stronger effect than class weighting alone

    Why this works:
    - At collapse (99% Up predictions), diversity penalty = 0.24 (large)
    - Gradient pushes mean prediction toward 0.5
    - Class weights ensure minority samples have amplified gradients
    - Together, they break the collapse equilibrium in early epochs
    """

    def __init__(self, diversity_weight: float = 1.0, label_smoothing: float = 0.0):
        """
        Args:
            diversity_weight: Weight for diversity penalty (default 1.0)
                             Higher = stronger anti-collapse effect
            label_smoothing: Optional label smoothing (default 0.0)
        """
        super().__init__()
        self.diversity_weight = diversity_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model outputs [batch] or [batch, 1]
            targets: Binary labels [batch] (0 or 1)

        Returns:
            Combined BCE + diversity penalty loss (scalar)
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Apply label smoothing if enabled
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Compute class weights based on BATCH class distribution
        # This is more responsive than dataset-level weights
        n_pos = targets.sum() + 1  # +1 for smoothing
        n_neg = len(targets) - targets.sum() + 1
        pos_weight = (n_neg / n_pos).clamp(0.5, 2.0)  # Clamp to prevent extreme weights

        # Weighted BCE loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=pos_weight.unsqueeze(0) if pos_weight.dim() == 0 else pos_weight
        )

        # DIVERSITY PENALTY: Push mean logit toward 0 (balanced predictions)
        # v12 FIX: Compute on RAW LOGITS, not sigmoid probabilities!
        #
        # Why this matters: When logits are extreme (e.g., all -5):
        #   - sigmoid(-5) = 0.0067
        #   - sigmoid'(-5) = 0.0067 * 0.9933 = 0.0066 (nearly zero!)
        #   - Gradient through sigmoid is killed by saturation
        #
        # By penalizing mean(logits) directly:
        #   - No sigmoid saturation
        #   - Gradient flows directly to model weights
        #   - When mean_logit = 0, sigmoid(0) = 0.5, so predictions are balanced
        #
        logit_mean = logits.mean()
        # Penalty = (mean_logit / 3)^2 - normalized so penalty ~1 at typical collapse
        # At all-negative logits: mean=-3, penalty=1.0
        # At balanced logits: mean=0, penalty=0
        # Divide by 3 to normalize (typical collapsed logit magnitude)
        diversity_penalty = (logit_mean / 3.0) ** 2

        total_loss = bce_loss + self.diversity_weight * diversity_penalty

        return total_loss


class SymmetricCrossEntropyLoss(nn.Module):
    """
    [DEPRECATED in v10 - Use BalancedBCELoss instead]

    Symmetric Cross Entropy with proper soft-label handling in RCE.

    Paper: "Symmetric Cross Entropy for Robust Learning with Noisy Labels" (ICCV'19)

    SCE = alpha * CE(hard_labels) + beta * RCE(soft_labels)

    The Reverse CE term maintains gradients even when model is at collapse state,
    forcing it to recover rather than stay collapsed.

    CRITICAL FIX (v8): The original SCE paper uses soft probability labels, not
    hard binary labels. With hard 0/1 labels, log(0+eps) = -16 which explodes
    the loss to ~4.7 instead of ~0.7. We now use soft labels ONLY in RCE:
    - Hard 0 -> 0.1 (soft)
    - Hard 1 -> 0.9 (soft)

    This prevents log(0) explosion while maintaining anti-collapse gradients.

    NOTE v10: This loss does NOT prevent class prediction collapse. It was designed
    for neural feature collapse. Use BalancedBCELoss instead.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, label_smoothing: float = 0.1):
        """
        Args:
            alpha: Weight for standard cross entropy (default 1.0)
            beta: Weight for reverse cross entropy (default 0.5)
                  Higher beta = stronger anti-collapse effect
            label_smoothing: Epsilon for soft labels in RCE (default 0.1)
                            Hard 0 -> eps, Hard 1 -> (1-eps)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model outputs [batch] or [batch, 1]
            targets: Binary labels [batch] (0 or 1)

        Returns:
            Symmetric cross entropy loss (scalar)
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Compute probabilities with numerical stability
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 1e-7, 1 - 1e-7)

        # Standard Cross Entropy: Uses HARD labels [0, 1] - this is correct
        ce_loss = -(targets * torch.log(probs) +
                   (1 - targets) * torch.log(1 - probs)).mean()

        # Reverse Cross Entropy: Uses SOFT labels to prevent log(0) explosion
        # Convert hard labels: 0 -> eps, 1 -> (1-eps)
        # For eps=0.1: 0 -> 0.1, 1 -> 0.9
        eps = self.label_smoothing
        soft_targets = targets * (1 - eps) + (1 - targets) * eps
        soft_targets = torch.clamp(soft_targets, 1e-4, 1 - 1e-4)

        # RCE with soft labels: log(0.1) = -2.3 instead of log(1e-7) = -16
        rce_loss = -(probs * torch.log(soft_targets) +
                    (1 - probs) * torch.log(1 - soft_targets)).mean()

        return self.alpha * ce_loss + self.beta * rce_loss


class RankingMSELoss(nn.Module):
    """
    Loss that prevents mode collapse using RANKING + NOISE INJECTION.
    
    WHY ALL PREVIOUS LOSSES FAILED:
    MSE with noisy targets INHERENTLY encourages constant predictions!
    The optimal MSE prediction for noisy data is the conditional mean.
    If features aren't predictive, optimal = unconditional mean (constant).
    
    SOLUTION: Two-pronged attack:
    
    1. NOISE INJECTION: Add Gaussian noise to predictions during training.
       This breaks symmetry and forces the model to explore output space.
       The noise is scaled to match target std, so the model MUST learn
       to output values in the correct range to minimize loss.
    
    2. RANKING LOSS: Instead of just matching values, reward correct ORDERING.
       If target_i > target_j, then pred_i should be > pred_j.
       This has gradient even when predictions are constant!
       
       margin_loss = max(0, margin - (pred_i - pred_j)) when target_i > target_j
       
       At collapse (pred_i = pred_j), gradient = -1 (pushes pred_i UP, pred_j DOWN)
    """
    
    def __init__(self, ranking_weight: float = 2.0):
        """
        Args:
            ranking_weight: Weight for ranking loss component
        """
        super().__init__()
        self.ranking_weight = ranking_weight
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = predictions.squeeze()
        targets = targets.squeeze()
        batch_size = predictions.size(0)
        
        # 1. Raw MSE - learns magnitude
        mse_loss = nn.functional.mse_loss(predictions, targets)
        
        # 2. RANKING LOSS - learns relative ordering
        # Pairwise ranking loss
        ranking_loss = torch.tensor(0.0, device=predictions.device)
        if batch_size > 1:
            # Sample random pairs (efficient implementation)
            # Compare i with i+1 (cyclic)
            idx1 = torch.arange(batch_size, device=predictions.device)
            idx2 = (idx1 + 1) % batch_size
            
            target_diff = targets[idx1] - targets[idx2]
            pred_diff = predictions[idx1] - predictions[idx2]
            
            # Only penalize if signs don't match direction
            # If target_diff > 0, we want pred_diff > 0
            # Loss = ReLU(-sign(target_diff) * pred_diff)
            signs = torch.sign(target_diff)
            # Ignore small differences to reduce noise
            mask = (torch.abs(target_diff) > 0.0001).float()
            
            ranking_loss = (torch.nn.functional.relu(-signs * pred_diff) * mask).mean()
        
        # 3. VARIANCE LOSS - Force scale matching (Robust)
        pred_std = predictions.std() + 1e-6
        target_std = targets.std().detach() + 1e-6
        variance_loss = (pred_std - target_std) ** 2

        # 4. MEAN LOSS - Force centering (Anti-bias)
        pred_mean = predictions.mean()
        target_mean = targets.mean().detach()
        mean_loss = (pred_mean - target_mean) ** 2
        
        # Combined loss
        # Variance and Mean losses are critical for avoiding collapse
        total_loss = mse_loss + \
                     self.ranking_weight * ranking_loss + \
                     10.0 * variance_loss + \
                     10.0 * mean_loss
        
        return total_loss


def compute_class_weights(
    labels: np.ndarray,
    num_classes: int,
    device: torch.device
) -> torch.Tensor:
    """
    Compute normalized class weights to counter class imbalance.

    Uses inverse frequency weighting normalized to mean=1.

    Args:
        labels: Training labels
        num_classes: Number of classes
        device: Torch device
    """
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float32)
    total = counts.sum()

    if total == 0:
        return torch.ones(num_classes, device=device, dtype=torch.float32)

    # Standard inverse frequency weights
    weights = np.where(counts > 0, total / (num_classes * counts), 0.0)

    # Normalize to mean=1
    mean_weight = weights.mean() if weights.mean() > 0 else 1.0
    weights = weights / mean_weight

    return torch.tensor(weights, device=device, dtype=torch.float32)


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
        visualize: bool = True,
        num_classes: int = 5,
        class_weights: Optional[torch.Tensor] = None,
        up_classes: Tuple[int, ...] = (3, 4),
        down_classes: Tuple[int, ...] = (0, 1),
        class_names: Optional[List[str]] = None,
        class_meta: Optional[Dict[str, float]] = None,
        use_auxiliary_losses: bool = False,
        aux_volatility_weight: float = 0.3,
        aux_regime_weight: float = 0.2,
        gradient_accumulation_steps: int = 1,
        use_multi_horizon: bool = False,
        horizon_weights: Optional[Dict[str, float]] = None,
        input_noise_std: float = 0.0
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
            num_classes: Number of discrete return classes
            class_weights: Optional class weighting tensor for CrossEntropy
            up_classes: Classes considered bullish for direction metrics
            down_classes: Classes considered bearish for direction metrics
            class_names: Optional human-readable class names (len = num_classes)
            class_meta: Optional metadata about class thresholds/std
            use_auxiliary_losses: Whether to use volatility/regime auxiliary losses
            aux_volatility_weight: Weight for volatility prediction loss (MSE)
            aux_regime_weight: Weight for regime classification loss (BCE)
            gradient_accumulation_steps: Accumulate gradients over N batches for smoother updates
            use_multi_horizon: Whether to use multi-horizon prediction (1H, 2H, 4H)
            horizon_weights: Dict mapping horizon names to loss weights (e.g., {'1h': 0.2, '2h': 0.3, '4h': 1.0})
            input_noise_std: Std dev of Gaussian noise added to inputs during training only.
        """
        self.model = model.to(device)
        self.device = device
        self.input_noise_std = float(input_noise_std)
        self.patience = patience
        self.cache_clear_interval = cache_clear_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.visualize = visualize
        self.log_dir = Path(log_dir) if log_dir else None
        self.num_classes = num_classes
        self.up_classes = up_classes
        self.down_classes = down_classes
        self.class_meta = class_meta or {}

        # Support both 3-class and 5-class schemes
        default_class_names_3 = ["Down", "Neutral", "Up"]
        default_class_names_5 = ["Strong Down", "Weak Down", "Neutral", "Weak Up", "Strong Up"]
        default_class_names = default_class_names_3 if num_classes == 3 else default_class_names_5
        self.class_names = class_names or default_class_names[:num_classes]
        if len(self.class_names) < num_classes:
            missing = [f"Class {i}" for i in range(len(self.class_names), num_classes)]
            self.class_names = self.class_names + missing

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

        # Scheduler: CosineAnnealingLR (NO RESTARTS) with warmup
        # LR restarts destabilize training - at epoch 26, the restart destroyed
        # the balanced recall state the model had achieved (93.7%/7.5% collapse)
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        # Warmup: start at 10% LR for first 20 epochs (FIXED: was 10, too aggressive)
        # Slower warmup prevents gradient collapse in early epochs by letting
        # the model stabilize at low LR before increasing
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=20
        )
        # Cosine decay WITHOUT restarts - smooth decay to eta_min over 80 epochs
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=80,  # Remaining epochs after warmup (100 - 20)
            eta_min=1e-6
        )
        # Sequential: warmup then cosine decay
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[20]  # Match warmup duration (FIXED: was 10)
        )
        self.use_step_scheduler = True  # Step after each epoch, not on plateau

        if class_weights is not None:
            class_weights = class_weights.to(device)

        # Loss function depends on num_classes
        if num_classes == 2:
            # Binary classification: BalancedBCELoss (v10)
            #
            # v10 FIX: SymmetricCrossEntropyLoss (SCE) failed to prevent class collapse.
            # SCE was designed for NEURAL feature collapse, not CLASS PREDICTION collapse.
            # The model collapsed to 99.9% Up predictions by epoch 3 despite SCE.
            #
            # BalancedBCELoss solves this with two mechanisms:
            # 1. Dynamic class weighting: Higher weight for minority class samples
            # 2. Diversity penalty: Directly penalizes imbalanced predictions
            #    - When model predicts 99% Up, penalty = (0.99-0.5)^2 = 0.24
            #    - Creates gradient pushing predictions toward 50/50
            #
            # v11 FIX: diversity_weight=5.0 (was 1.0)
            # v10 had diversity_weight=1.0 which was only 34% of BCE loss - not enough
            # At 98% collapse: penalty = 5.0 * (0.02-0.5)^2 = 1.15 (170% of BCE)
            # This should completely prevent early class collapse
            self.criterion = BalancedBCELoss(
                diversity_weight=5.0,    # v11: 5x stronger to dominate during collapse
                label_smoothing=0.0      # No label smoothing (targets are already clean)
            )
            self.is_binary = True
            logger.info("Using BalancedBCELoss for binary classification (v11: 5x diversity penalty)")
        else:
            # Multi-class: FocalLoss to address class oscillation
            # - gamma=2.0: Down-weights easy examples, focuses on hard Up/Down distinctions
            # - label_smoothing=0.1: Softens noisy boundaries in financial data
            self.criterion = FocalLoss(
                weight=class_weights,
                gamma=2.0,
                label_smoothing=0.1
            )
            self.is_binary = False
            logger.info(f"Using FocalLoss for {num_classes}-class classification")

        # Auxiliary losses for multi-task learning (regularization)
        self.use_aux = use_auxiliary_losses
        self.aux_vol_weight = aux_volatility_weight
        self.aux_regime_weight = aux_regime_weight

        if self.use_aux:
            # Volatility prediction: MSE loss (regression)
            self.aux_vol_criterion = nn.MSELoss()
            # Regime classification: CrossEntropy loss (3-class: Trend, Chop, Volatile)
            self.aux_regime_criterion = nn.CrossEntropyLoss()
            logger.info(f"Using auxiliary losses: vol_weight={aux_volatility_weight}, regime_weight={aux_regime_weight}")

        # Multi-horizon losses (addresses Target Mismatch)
        self.use_multi_horizon = use_multi_horizon
        self.horizon_weights = horizon_weights or {'1h': 0.2, '2h': 0.3, '4h': 1.0}

        if self.use_multi_horizon:
            # Multi-horizon uses BCE for binary direction at each horizon
            self.multi_horizon_criterion = nn.BCEWithLogitsLoss()
            logger.info(f"Using multi-horizon prediction: weights={self.horizon_weights}")

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_direction_accs = []
        self.val_direction_accs = []
        self.train_macro_f1s = []
        self.val_macro_f1s = []
        self.learning_rates = []
        self.grad_norms = []
        self.memory_usage = []
        self.best_val_loss = float('inf')
        self.best_min_recall = 0.0  # Track minimum class recall (rewards balance)
        self.epochs_without_improvement = 0

        # Batch-level tracking
        self.batch_losses = []
        self.batch_grad_norms = []

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int
    ) -> Tuple[float, float, float, float]:
        """
        Train for one epoch with detailed metrics.

        Returns:
            Tuple of (avg_loss, accuracy, direction_accuracy, macro_f1)
        """
        self.model.train()
        # Enable noise injection for FocalLoss (not needed for BCEWithLogitsLoss)
        if hasattr(self.criterion, 'train') and not self.is_binary:
            self.criterion.train()
        total_loss = 0.0
        n_batches = len(train_loader)

        # Metrics tracker for the epoch
        metrics_tracker = MetricsTracker(
            task_type="classification",
            num_classes=self.num_classes,
            up_classes=self.up_classes,
            down_classes=self.down_classes
        )

        self.training_logger.start_epoch(epoch, total_epochs)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}")

        # Gradient accumulation support: effective batch = batch_size * accumulation_steps
        accum_steps = self.gradient_accumulation_steps

        for batch_idx, batch_data in enumerate(pbar):
            batch_start = time.time()

            # Unpack batch data
            # (x_15m, x_1h, x_4h, targets, vol_targets, regime_targets, h1_targets, h2_targets)
            x_15m, x_1h, x_4h, targets, vol_targets, regime_targets, h1_targets, h2_targets = batch_data

            # Move to device
            x_15m = x_15m.to(self.device)
            x_1h = x_1h.to(self.device)
            x_4h = x_4h.to(self.device)
            targets = targets.to(self.device)

            # Input noise regularization (training only)
            if self.input_noise_std > 0:
                x_15m = x_15m + torch.randn_like(x_15m) * self.input_noise_std
                x_1h = x_1h + torch.randn_like(x_1h) * self.input_noise_std
                x_4h = x_4h + torch.randn_like(x_4h) * self.input_noise_std
            
            model_kwargs: dict = {}

            # Forward pass with optional auxiliary and multi-horizon outputs
            if self.use_aux and self.use_multi_horizon:
                vol_targets = vol_targets.to(self.device)
                regime_targets = regime_targets.to(self.device)
                h1_targets = h1_targets.to(self.device)
                h2_targets = h2_targets.to(self.device)
                outputs = self.model(
                    x_15m, x_1h, x_4h,
                    return_aux=True,
                    return_multi_horizon=True
                )
                # Outputs: (context, direction, volatility, regime, horizon_1h, horizon_2h)
                _, logits, vol_pred, regime_pred, h1_logits, h2_logits = outputs
            elif self.use_aux:
                vol_targets = vol_targets.to(self.device)
                regime_targets = regime_targets.to(self.device)
                _, logits, vol_pred, regime_pred = self.model(
                    x_15m, x_1h, x_4h,
                    return_aux=True
                )
                h1_logits, h2_logits = None, None
            elif self.use_multi_horizon:
                h1_targets = h1_targets.to(self.device)
                h2_targets = h2_targets.to(self.device)
                _, logits, h1_logits, h2_logits = self.model(
                    x_15m, x_1h, x_4h,
                    return_multi_horizon=True
                )
                vol_pred, regime_pred = None, None
            else:
                _, logits = self.model(x_15m, x_1h, x_4h, **model_kwargs)
                h1_logits, h2_logits, vol_pred, regime_pred = None, None, None, None

            # Compute main direction loss (4H horizon - primary target)
            if self.is_binary:
                targets_float = targets.float()
                direction_loss = self.criterion(logits.squeeze(-1), targets_float)
            else:
                direction_loss = self.criterion(logits, targets.long())

            # Total loss = direction (4H) + weighted auxiliary + weighted multi-horizon
            loss = direction_loss * self.horizon_weights.get('4h', 1.0)

            if self.use_aux and vol_pred is not None:
                vol_loss = self.aux_vol_criterion(vol_pred, vol_targets)
                regime_loss = self.aux_regime_criterion(regime_pred, regime_targets)
                loss = loss + self.aux_vol_weight * vol_loss + self.aux_regime_weight * regime_loss

            if self.use_multi_horizon and h1_logits is not None:
                # Compute multi-horizon losses (only on valid targets, not NaN)
                h1_valid = ~torch.isnan(h1_targets)
                h2_valid = ~torch.isnan(h2_targets)

                if h1_valid.any():
                    h1_loss = self.multi_horizon_criterion(
                        h1_logits.squeeze(-1)[h1_valid],
                        h1_targets[h1_valid]
                    )
                    loss = loss + self.horizon_weights.get('1h', 0.2) * h1_loss

                if h2_valid.any():
                    h2_loss = self.multi_horizon_criterion(
                        h2_logits.squeeze(-1)[h2_valid],
                        h2_targets[h2_valid]
                    )
                    loss = loss + self.horizon_weights.get('2h', 0.3) * h2_loss

            # Normalize loss by accumulation steps for gradient accumulation
            if accum_steps > 1:
                loss = loss / accum_steps

            # Backward pass (gradients accumulate by default)
            loss.backward()

            # Only step optimizer after accumulating enough gradients
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == n_batches:
                # Compute gradient norm before clipping
                grad_norm = compute_gradient_norm(self.model)
                self.batch_grad_norms.append(grad_norm)

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                grad_norm = 0.0  # Not computed this step
                self.batch_grad_norms.append(grad_norm)  # Ensure length matches batch_losses

            # Track metrics (un-normalize loss for tracking if using gradient accumulation)
            loss_val = loss.item() * accum_steps if accum_steps > 1 else loss.item()
            total_loss += loss_val
            self.batch_losses.append(loss_val)

            # Compute predictions for accuracy
            if self.is_binary:
                # Binary: threshold at 0.5
                pred_classes = (torch.sigmoid(logits.squeeze(-1)) > 0.5).long()
            else:
                pred_classes = torch.argmax(logits, dim=1)

            metrics_tracker.update(
                pred_classes.detach().cpu().numpy(),
                targets.detach().cpu().numpy().astype(int),
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
            del x_15m, x_1h, x_4h, targets, logits, pred_classes, loss

        # Compute epoch metrics
        epoch_metrics = metrics_tracker.compute()
        avg_loss = total_loss / n_batches
        direction_acc = epoch_metrics.get('direction_accuracy', 0.0)

        accuracy = epoch_metrics.get('accuracy', 0.0)
        macro_f1 = epoch_metrics.get('macro_f1', 0.0)

        return avg_loss, accuracy, direction_acc, macro_f1

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
        metrics_tracker = MetricsTracker(
            task_type="classification",
            num_classes=self.num_classes,
            up_classes=self.up_classes,
            down_classes=self.down_classes
        )

        for batch_data in val_loader:
            # Unpack batch data
            x_15m, x_1h, x_4h, targets, vol_targets, regime_targets, h1_targets, h2_targets = batch_data

            x_15m = x_15m.to(self.device)
            x_1h = x_1h.to(self.device)
            x_4h = x_4h.to(self.device)
            targets = targets.to(self.device)

            # Forward pass (use primary 4H target for validation metrics)
            _, logits = self.model(x_15m, x_1h, x_4h)

            # Compute loss (different for binary vs multi-class)
            if self.is_binary:
                targets_float = targets.float()
                loss = self.criterion(logits.squeeze(-1), targets_float)
                # Binary predictions
                pred_classes = (torch.sigmoid(logits.squeeze(-1)) > 0.5).long()
            else:
                loss = self.criterion(logits, targets.long())
                pred_classes = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            n_batches += 1

            all_predictions.append(pred_classes.cpu().numpy())
            all_targets.append(targets.cpu().numpy().astype(int))
            metrics_tracker.update(pred_classes.cpu().numpy(), targets.cpu().numpy().astype(int), loss.item())

            del x_15m, x_1h, x_4h, targets, logits, pred_classes

        # Concatenate all predictions and targets
        all_predictions = np.concatenate(all_predictions) if len(all_predictions) > 0 else np.array([])
        all_targets = np.concatenate(all_targets) if len(all_targets) > 0 else np.array([])

        # Calculate metrics
        avg_loss = total_loss / max(n_batches, 1)
        class_metrics = metrics_tracker.compute()

        accuracy = class_metrics.get('accuracy', 0.0)
        direction_acc = class_metrics.get('direction_accuracy', 0.0)

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
            train_loss, train_acc, train_dir_acc, train_macro_f1 = self.train_epoch(
                train_loader, epoch, max_epochs
            )
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.train_direction_accs.append(train_dir_acc)
            self.train_macro_f1s.append(train_macro_f1)

            # Validate
            val_loss, val_acc, val_dir_acc, val_preds, val_targets = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            self.val_direction_accs.append(val_dir_acc)

            # Detailed validation metrics
            class_metrics = calculate_classification_metrics(
                val_preds,
                val_targets,
                num_classes=self.num_classes,
                up_classes=self.up_classes,
                down_classes=self.down_classes
            )
            self.val_macro_f1s.append(class_metrics.macro_f1)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            # Compute average gradient norm for epoch
            recent_grad_norms = self.batch_grad_norms[-len(train_loader):]
            avg_grad_norm = np.mean(recent_grad_norms) if recent_grad_norms else 0.0
            self.grad_norms.append(avg_grad_norm)

            # Update scheduler
            if self.use_step_scheduler:
                # New cosine annealing scheduler: step per epoch, not on loss
                self.scheduler.step()
            else:
                # Legacy ReduceLROnPlateau: step on val_loss
                self.scheduler.step(val_loss)

            # Check for improvement using MINIMUM RECALL (rewards balance)
            # Min recall = the weaker class's recall - directly penalizes any imbalance
            # Unlike arithmetic mean which can be gamed by excelling at one class
            # NOTE: Must compute BEFORE log_epoch() to sync patience counters
            min_recall = min(class_metrics.up_recall, class_metrics.down_recall)

            # Track best val_loss for logging (but don't use for checkpoint/early stopping)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

            # Determine improvement based on MINIMUM RECALL
            is_best = False
            if min_recall > self.best_min_recall:
                self.best_min_recall = min_recall
                self.epochs_without_improvement = 0
                is_best = True
            else:
                self.epochs_without_improvement += 1

            # Sync training_logger's patience counter with ours (so logged value is correct)
            # The training_logger has its own counter based on val_loss - we override it
            self.training_logger.epochs_without_improvement = self.epochs_without_improvement
            self.training_logger.best_val_loss = self.best_val_loss  # For display

            # Log epoch summary (now with correct patience from min_recall)
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
                    'train_macro_f1': train_macro_f1,
                    'val_macro_f1': class_metrics.macro_f1,
                    'val_up_recall': class_metrics.up_recall,
                    'val_down_recall': class_metrics.down_recall,
                    'val_neutral_recall': class_metrics.neutral_recall,
                    'min_recall': min_recall,
                    'best_min_recall': self.best_min_recall
                },
                skip_patience_update=True  # We manage patience based on min_recall, not val_loss
            )

            # Log checkpoint save (after log_epoch for correct ordering)
            if is_best:
                # Use training_logger to ensure message goes to the log file
                self.training_logger.logger.info(f"  ★ NEW BEST min_recall: {min_recall:.4f} (up={class_metrics.up_recall:.4f}, down={class_metrics.down_recall:.4f})")
                # Save best model
                if save_path:
                    self.save_checkpoint(save_path, epoch, is_best=True)

            # Log class distribution for debugging
            pred_counts = np.bincount(val_preds.astype(int), minlength=self.num_classes)
            tgt_counts = np.bincount(val_targets.astype(int), minlength=self.num_classes)
            total_pred = pred_counts.sum() if pred_counts.sum() > 0 else 1
            total_tgt = tgt_counts.sum() if tgt_counts.sum() > 0 else 1
            self.training_logger.logger.info("  Class distribution (pred | true):")
            for idx, name in enumerate(self.class_names):
                pred_pct = pred_counts[idx] / total_pred * 100
                tgt_pct = tgt_counts[idx] / total_tgt * 100
                self.training_logger.logger.info(f"    {idx} ({name}): pred {pred_pct:5.1f}% | true {tgt_pct:5.1f}%")

            # Log sample predictions vs targets
            if epoch % 10 == 0 or epoch == 1:
                self.training_logger.log_validation_details(
                    val_preds,
                    val_targets,
                    task_type="classification",
                    class_names=self.class_names
                )

            # Create epoch visualization
            if self.visualizer and epoch % 5 == 0:
                # Get training predictions for visualization
                train_preds, train_tgts = self._get_train_predictions(train_loader)

                metrics = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'train_dir_acc': train_dir_acc,
                    'val_dir_acc': val_dir_acc,
                    'train_macro_f1': train_macro_f1,
                    'val_macro_f1': class_metrics.macro_f1,
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
                    save_name=f'epoch_{epoch:03d}_summary.png',
                    task_type="classification",
                    class_names=self.class_names,
                    num_classes=self.num_classes
                )
                self.visualizer.close_all()

            # Early stopping (based on min_recall)
            if self.epochs_without_improvement >= self.patience:
                self.training_logger.logger.info(f"Early stopping at epoch {epoch} (patience={self.patience}, no improvement in min_recall)")
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
                save_name="direction_confusion.png",
                task_type="classification",
                up_classes=self.up_classes,
                down_classes=self.down_classes
            )

            self.visualizer.plot_predictions_vs_targets(
                final_preds, final_targets,
                title="Final Predictions vs Targets",
                save_name="predictions_vs_targets.png",
                task_type="classification",
                class_names=self.class_names
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

        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx >= max_batches:
                break

            # Unpack batch tuple
            x_15m, x_1h, x_4h, tgt, _, _, _, _ = batch_data

            x_15m = x_15m.to(self.device)
            x_1h = x_1h.to(self.device)
            x_4h = x_4h.to(self.device)

            _, logits = self.model(x_15m, x_1h, x_4h)

            # Handle binary vs multi-class predictions
            if self.is_binary:
                pred_classes = (torch.sigmoid(logits.squeeze(-1)) > 0.5).long()
            else:
                pred_classes = torch.argmax(logits, dim=1)

            predictions.append(pred_classes.cpu().numpy())
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
            'train_macro_f1': self.train_macro_f1s,
            'val_macro_f1': self.val_macro_f1s,
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

        # Base config
        model_config = {
            'd_model': self.model.d_model,
            'context_dim': self.model.context_dim,
            'num_classes': self.model.num_classes,
            'dropout': getattr(self.model, 'dropout', 0.3),
            'up_classes': self.up_classes,
            'down_classes': self.down_classes,
            'class_names': self.class_names,
        }

        # Add architecture-specific config
        if hasattr(self.model, 'nhead'):
            # Transformer specific
            model_config.update({
                'nhead': self.model.nhead,
                'num_layers': self.model.num_layers,
                'dim_feedforward': self.model.dim_feedforward,
                'architecture': 'transformer',
            })
        elif hasattr(self.model, 'tcn_num_blocks'):
            # TCN specific
            model_config.update({
                'tcn_num_blocks': self.model.tcn_num_blocks,
                'tcn_kernel_size': self.model.tcn_kernel_size,
                'tcn_num_channels': getattr(self.model, 'tcn_num_channels', None),
                'architecture': 'tcn',
            })

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),  # FIXED: Save scheduler for resume
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'train_direction_accs': self.train_direction_accs,
            'val_direction_accs': self.val_direction_accs,
            'best_val_loss': self.best_val_loss,
            'config': model_config
        }

        filename = 'best.pt' if is_best else f'epoch_{epoch}.pt'
        torch.save(checkpoint, path / filename)
        # Use training_logger to ensure message goes to the log file
        self.training_logger.logger.info(f"  ✓ Saved checkpoint to {path / filename}")


def walk_forward_cross_validate(
    dataset: MultiTimeframeDataset,
    model_factory: callable,
    trainer_factory: callable,
    n_folds: int = 5,
    train_ratio: float = 0.7,
    min_train_samples: int = 5000,
    max_epochs_per_fold: int = 30,
    device: torch.device = None,
    save_path: str = None
) -> Dict:
    """
    Walk-forward cross-validation for time series.

    This addresses Issue 6 (Regime Shift): Instead of a single train/val split,
    we train on multiple time windows and validate on the next period. This tests
    how well the model generalizes to changing market regimes.

    Args:
        dataset: Full MultiTimeframeDataset
        model_factory: Function that returns a new MarketAnalyst model
        trainer_factory: Function that returns a new AnalystTrainer
        n_folds: Number of walk-forward folds
        train_ratio: Ratio of each fold used for training (rest for validation)
        min_train_samples: Minimum training samples per fold
        max_epochs_per_fold: Max epochs to train each fold
        device: Torch device
        save_path: Path to save results

    Returns:
        Dict with fold results and aggregate metrics
    """
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    total_samples = len(dataset)
    fold_size = total_samples // n_folds

    if fold_size < min_train_samples:
        logger.warning(f"Fold size {fold_size} < min_train_samples {min_train_samples}, reducing n_folds")
        n_folds = max(2, total_samples // min_train_samples)
        fold_size = total_samples // n_folds

    logger.info(f"Walk-Forward CV: {n_folds} folds, {fold_size} samples per fold")
    logger.info("=" * 70)

    fold_results = []

    for fold_idx in range(n_folds - 1):  # -1 because we need a val set after each train
        fold_start = fold_idx * fold_size
        fold_end = (fold_idx + 2) * fold_size  # Train on current fold, val on next

        if fold_end > total_samples:
            fold_end = total_samples

        # Split within this window
        train_end = fold_start + int((fold_end - fold_start) * train_ratio)

        train_indices = list(range(fold_start, train_end))
        val_indices = list(range(train_end, fold_end))

        if len(train_indices) < min_train_samples:
            logger.warning(f"Fold {fold_idx + 1}: Skipping, only {len(train_indices)} train samples")
            continue

        logger.info(f"\nFold {fold_idx + 1}/{n_folds - 1}:")
        logger.info(f"  Train: indices {fold_start} to {train_end} ({len(train_indices)} samples)")
        logger.info(f"  Val: indices {train_end} to {fold_end} ({len(val_indices)} samples)")

        # Create subsets
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)

        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=0)

        # Create fresh model and trainer for each fold
        model = model_factory()
        trainer = trainer_factory(model, device)

        # Train this fold (reduced epochs for CV)
        history = trainer.train(
            train_loader,
            val_loader,
            max_epochs=max_epochs_per_fold,
            save_path=None  # Don't save intermediate folds
        )

        # Record fold results
        fold_result = {
            'fold': fold_idx + 1,
            'train_indices': (fold_start, train_end),
            'val_indices': (train_end, fold_end),
            'n_train': len(train_indices),
            'n_val': len(val_indices),
            'best_val_loss': history['best_val_loss'],
            'best_val_acc': max(history['val_acc']) if history['val_acc'] else 0,
            'best_val_dir_acc': max(history['val_direction_acc']) if history['val_direction_acc'] else 0,
            'final_train_acc': history['train_acc'][-1] if history['train_acc'] else 0,
            'epochs_trained': history['epochs_trained']
        }
        fold_results.append(fold_result)

        logger.info(f"  Result: val_loss={fold_result['best_val_loss']:.4f}, "
                   f"val_acc={fold_result['best_val_acc']*100:.1f}%, "
                   f"val_dir_acc={fold_result['best_val_dir_acc']*100:.1f}%")

        # Clean up
        del model, trainer
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Aggregate results
    if fold_results:
        avg_val_loss = np.mean([f['best_val_loss'] for f in fold_results])
        std_val_loss = np.std([f['best_val_loss'] for f in fold_results])
        avg_val_acc = np.mean([f['best_val_acc'] for f in fold_results])
        std_val_acc = np.std([f['best_val_acc'] for f in fold_results])
        avg_val_dir_acc = np.mean([f['best_val_dir_acc'] for f in fold_results])
        std_val_dir_acc = np.std([f['best_val_dir_acc'] for f in fold_results])

        # Check for regime shift: high std indicates inconsistent performance
        regime_shift_warning = std_val_dir_acc > 0.10  # >10% std is concerning

        logger.info("\n" + "=" * 70)
        logger.info("WALK-FORWARD CROSS-VALIDATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Folds completed: {len(fold_results)}")
        logger.info(f"Avg Val Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
        logger.info(f"Avg Val Accuracy: {avg_val_acc*100:.1f}% ± {std_val_acc*100:.1f}%")
        logger.info(f"Avg Val Direction Acc: {avg_val_dir_acc*100:.1f}% ± {std_val_dir_acc*100:.1f}%")

        if regime_shift_warning:
            logger.warning("⚠️  HIGH VARIANCE DETECTED - Possible regime shift sensitivity!")
            logger.warning("   Consider: ensemble models, shorter prediction horizon, or regime-adaptive training")
        else:
            logger.info("✓ Consistent performance across folds - model generalizes well")

        results = {
            'fold_results': fold_results,
            'n_folds': len(fold_results),
            'avg_val_loss': avg_val_loss,
            'std_val_loss': std_val_loss,
            'avg_val_acc': avg_val_acc,
            'std_val_acc': std_val_acc,
            'avg_val_dir_acc': avg_val_dir_acc,
            'std_val_dir_acc': std_val_dir_acc,
            'regime_shift_warning': regime_shift_warning
        }

        # Save results if path provided
        if save_path:
            import json
            results_path = Path(save_path) / 'walk_forward_cv_results.json'
            with open(results_path, 'w') as f:
                # Convert numpy types to native Python for JSON
                json_results = {
                    k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
                    for k, v in results.items() if k != 'fold_results'
                }
                json_results['fold_results'] = fold_results
                json.dump(json_results, f, indent=2)
            logger.info(f"Saved CV results to {results_path}")

        return results
    else:
        logger.error("No folds completed!")
        return {'fold_results': [], 'error': 'No folds completed'}


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
    # Class names depend on num_classes from config
    class_names_3 = [
        "Down (< -0.5σ)",
        "Neutral (-0.5σ to +0.5σ)",
        "Up (> +0.5σ)"
    ]
    class_names_5 = [
        "Strong Down (<-0.5σ)",
        "Weak Down (-0.5σ to -0.1σ)",
        "Neutral (-0.1σ to +0.1σ)",
        "Weak Up (+0.1σ to +0.5σ)",
        "Strong Up (> +0.5σ)"
    ]

    # Default configuration
    if config is None:
        from config.settings import Config
        config = Config().analyst

    num_classes = config.num_classes if hasattr(config, 'num_classes') else 5
    class_names = class_names_3 if num_classes == 3 else class_names_5
    if len(class_names) < num_classes:
        class_names += [f"Class {i}" for i in range(len(class_names), num_classes)]

    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    logger.info(f"Training on device: {device}")

    # Determine if using binary classification
    use_binary = getattr(config, 'use_binary_target', False) or num_classes == 2
    use_aux = getattr(config, 'use_auxiliary_losses', False)

    # Log auxiliary config
    logger.info(f"Auxiliary losses enabled: {use_aux}")

    # Create target based on mode
    logger.info("Creating targets...")
    future_window = getattr(config, 'future_window', 12)
    smooth_window = getattr(config, 'smooth_window', 12)

    if use_binary:
        # Binary classification: Up vs Down, excluding weak/neutral moves
        logger.info("Using BINARY direction target (Up vs Down)")
        min_move_atr = getattr(config, 'min_move_atr_threshold', 0.3)

        class_labels, valid_mask, class_meta = create_binary_direction_target(
            df_15m,
            future_window=future_window,
            smooth_window=smooth_window,
            min_move_atr=min_move_atr
        )

        # Also create continuous target for reference
        target = create_smoothed_target(df_15m, future_window, smooth_window)

        # Update num_classes and class_names for binary
        num_classes = 2
        class_names = ["Down", "Up"]

        logger.info(f"Binary target stats: {class_meta['n_down']} Down, {class_meta['n_up']} Up")
        logger.info(f"  Excluded {class_meta['n_neutral_excluded']} neutral/weak samples ({class_meta['pct_excluded']:.1f}%)")
        logger.info(f"  Class balance: {class_meta['class_balance']*100:.1f}% Up")

    else:
        # Multi-class classification (legacy 3-class or 5-class)
        logger.info(f"Using {num_classes}-class direction target")
        valid_mask = None  # No additional filtering

        target = create_smoothed_target(
            df_15m,
            future_window=future_window,
            smooth_window=smooth_window
        )

        # Log target statistics
        valid_target = target.dropna()
        logger.info(f"Target stats: mean={valid_target.mean():.6f}, std={valid_target.std():.6f}")

        # Convert to classification labels
        thresholds = getattr(config, 'class_std_thresholds', (-0.5, 0.5))
        train_end_idx = int(len(target) * 0.85)
        class_labels, class_meta = create_return_classes(
            target,
            class_std_thresholds=thresholds,
            train_end_idx=train_end_idx
        )

        label_counts = class_labels.value_counts(dropna=True).sort_index()
        total_labels = max(label_counts.sum(), 1)

        logger.info("Class distribution:")
        for idx, count in label_counts.items():
            name = class_names[int(idx)] if int(idx) < len(class_names) else f"Class {int(idx)}"
            pct = count / total_labels * 100
            logger.info(f"  {int(idx)} ({name}): {count} samples ({pct:.1f}%)")

    # Create auxiliary targets if enabled
    volatility_target = None
    regime_target = None
    if use_aux:
        logger.info("Creating auxiliary targets (volatility, regime)...")
        volatility_target, regime_target = create_auxiliary_targets(
            df_15m,
            future_window=future_window
        )
        logger.info(f"  Volatility target: mean={volatility_target.mean():.3f}, std={volatility_target.std():.3f}")
        logger.info(f"  Regime target: {regime_target.mean()*100:.1f}% trending")

    # Create multi-horizon targets if enabled (addresses Target Mismatch)
    use_multi_horizon = getattr(config, 'use_multi_horizon', False)
    horizon_1h_target = None
    horizon_2h_target = None
    if use_multi_horizon and use_binary:
        logger.info("Creating multi-horizon targets (1H, 2H, 4H)...")
        min_move_atr = getattr(config, 'min_move_atr_threshold', 0.3)

        labels_dict, _, meta_dict = create_multi_horizon_targets(
            df_15m,
            horizons={'1h': 4, '2h': 8, '4h': future_window},
            smooth_window=max(2, smooth_window // 2),
            min_move_atr=min_move_atr
        )

        # Extract 1H and 2H targets (4H is the primary target, already created)
        horizon_1h_target = labels_dict['1h']
        horizon_2h_target = labels_dict['2h']

        logger.info(f"  1H target: {meta_dict['1h']['n_down']} Down, {meta_dict['1h']['n_up']} Up")
        logger.info(f"  2H target: {meta_dict['2h']['n_down']} Down, {meta_dict['2h']['n_up']} Up")

    # Add Market Sessions
    logger.info("Adding market session features...")
    df_15m = add_market_sessions(df_15m)
    df_1h = add_market_sessions(df_1h)
    df_4h = add_market_sessions(df_4h)

    # Add Structure Features (BOS/CHoCH)
    logger.info("Adding structure features (BOS/CHoCH)...")
    for df in [df_15m, df_1h, df_4h]:
        f_high, f_low = detect_fractals(df)
        struct_df = detect_structure_breaks(df, f_high, f_low)
        for col in struct_df.columns:
            df[col] = struct_df[col]

    # Update feature columns if not already included
    session_cols = ['session_asian', 'session_london', 'session_ny']
    struct_cols = ['bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish']
    
    for col in session_cols + struct_cols:
        if col not in feature_cols:
            feature_cols.append(col)

    # Create dataset
    logger.info("Creating dataset...")
    dataset = MultiTimeframeDataset(
        df_15m, df_1h, df_4h,
        feature_cols, target, class_labels,
        lookback_5m=getattr(config, 'lookback_5m', 48),
        lookback_15m=getattr(config, 'lookback_15m', 16),
        lookback_45m=getattr(config, 'lookback_45m', 6),
        valid_mask=valid_mask,  # For binary: filter out neutral/weak samples
        volatility_target=volatility_target,  # Auxiliary target
        regime_target=regime_target,  # Auxiliary target
        is_binary=use_binary,  # Affects target tensor dtype
        horizon_1h_target=horizon_1h_target,  # Multi-horizon target (1H)
        horizon_2h_target=horizon_2h_target,  # Multi-horizon target (2H)
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

    # Class weights and split distributions
    valid_label_array = class_labels.values[dataset.valid_indices]
    # Filter out NaN values for label arrays
    valid_label_array = valid_label_array[~np.isnan(valid_label_array)]
    train_label_array = valid_label_array[:train_size].astype(int)
    val_label_array = valid_label_array[train_size:].astype(int)

    if use_binary:
        # Binary mode: pos_weight = n_down / n_up for BCEWithLogitsLoss
        n_down = (train_label_array == 0).sum()
        n_up = (train_label_array == 1).sum()
        pos_weight = n_down / max(n_up, 1)
        class_weights = torch.tensor([pos_weight], device=device, dtype=torch.float32)
        logger.info(f"Binary pos_weight: {pos_weight:.3f} (n_down={n_down}, n_up={n_up})")
    else:
        class_weights = compute_class_weights(train_label_array, num_classes, device)
        logger.info(f"Class weights (normalized): {class_weights.cpu().numpy()}")

    def _log_split_distribution(name: str, labels: np.ndarray):
        counts = np.bincount(labels, minlength=num_classes)
        total = counts.sum() if counts.sum() > 0 else 1
        parts = []
        for idx, count in enumerate(counts):
            class_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
            parts.append(f"{idx} {class_name}: {count} ({count/total*100:.1f}%)")
        logger.info(f"{name} class mix: " + " | ".join(parts))

    _log_split_distribution("Train", train_label_array)
    _log_split_distribution("Val", val_label_array)

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
        '5m': len(feature_cols),
        '15m': len(feature_cols),
        '45m': len(feature_cols)
    }

    # v13: Support both Transformer and TCN architectures
    architecture = getattr(config, 'architecture', 'tcn')

    if architecture == 'tcn':
        model = create_tcn_analyst(feature_dims, config, device)
        arch_name = "TCNAnalyst"
    else:
        model = create_analyst(feature_dims, config, device)
        arch_name = "MarketAnalyst (Transformer)"

    logger.info(f"Created {arch_name} with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Log model architecture
    logger.info(f"Architecture: {architecture}, d_model={model.d_model}, context_dim={model.context_dim}")

    # Create trainer
    # Set up/down classes based on classification mode
    if use_binary:
        up_classes = (1,)   # Binary: 1=Up
        down_classes = (0,)  # Binary: 0=Down
    elif num_classes == 3:
        up_classes = (2,)
        down_classes = (0,)
    else:
        up_classes = (3, 4)
        down_classes = (0, 1)

    # Get auxiliary loss settings from config
    use_aux = getattr(config, 'use_auxiliary_losses', False)
    aux_vol_weight = getattr(config, 'aux_volatility_weight', 0.3)
    aux_regime_weight = getattr(config, 'aux_regime_weight', 0.2)
    grad_accum_steps = getattr(config, 'gradient_accumulation_steps', 1)

    # Get multi-horizon settings from config
    horizon_weights = getattr(config, 'multi_horizon_weights', {'1h': 0.2, '2h': 0.3, '4h': 1.0})

    trainer = AnalystTrainer(
        model=model,
        device=device,
        learning_rate=getattr(config, 'learning_rate', 1e-4),
        weight_decay=getattr(config, 'weight_decay', 1e-5),
        patience=getattr(config, 'patience', 10),
        log_dir=save_path,
        visualize=visualize,
        num_classes=num_classes,
        class_weights=class_weights,
        up_classes=up_classes,
        down_classes=down_classes,
        class_names=class_names,
        class_meta=class_meta,
        use_auxiliary_losses=use_aux,
        aux_volatility_weight=aux_vol_weight,
        aux_regime_weight=aux_regime_weight,
        gradient_accumulation_steps=grad_accum_steps,
        use_multi_horizon=use_multi_horizon,
        horizon_weights=horizon_weights,
        input_noise_std=getattr(config, 'input_noise_std', 0.0)
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
    logger.info(f"Final train accuracy: {history['train_acc'][-1]*100:.2f}%")
    logger.info(f"Final val accuracy: {history['val_acc'][-1]*100:.2f}%")
    logger.info(f"Final train direction acc: {history['train_direction_acc'][-1]*100:.2f}%")
    logger.info(f"Final val direction acc: {history['val_direction_acc'][-1]*100:.2f}%")
    logger.info(f"Total epochs trained: {history['epochs_trained']}")
    logger.info("=" * 70)

    return model, history


if __name__ == '__main__':
    # Run training with full data pipeline
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from config.settings import config
    from src.data.loader import load_ohlcv
    from src.data.resampler import resample_all_timeframes, align_timeframes
    from src.data.features import engineer_all_features, get_feature_columns

    print("=" * 60)
    print(f"Starting Analyst Training (Architecture: {config.analyst.architecture})")
    print("=" * 60)

    # Step 1: Load raw data
    print("\n[1/4] Loading raw 1-minute data...")
    data_path = config.paths.training_data_dir / config.data.raw_file
    df_1m = load_ohlcv(data_path, datetime_format=config.data.datetime_format)
    print(f"  Loaded {len(df_1m):,} bars")

    # Step 2: Resample to native multi-timeframes (5m/15m/45m)
    print("\n[2/4] Resampling to multi-timeframe...")
    resampled = resample_all_timeframes(df_1m, config.data.timeframes)
    df_5m, df_15m, df_45m = resampled['5m'], resampled['15m'], resampled['45m']
    print(f"  5m: {len(df_5m):,} | 15m: {len(df_15m):,} | 45m: {len(df_45m):,}")
    del df_1m  # Free memory

    # Step 3: Engineer features
    print("\n[3/4] Engineering features...")
    feature_config = {
        'pinbar_wick_ratio': config.features.pinbar_wick_ratio,
        'doji_body_ratio': config.features.doji_body_ratio,
        'fractal_window': config.features.fractal_window,
        'sr_lookback': config.features.sr_lookback,
        'sma_period': config.features.sma_period,
        'ema_fast': config.features.ema_fast,
        'ema_slow': config.features.ema_slow,
        'chop_period': config.features.chop_period,
        'adx_period': config.features.adx_period,
        'atr_period': config.features.atr_period
    }
    df_5m = engineer_all_features(df_5m, feature_config)
    df_15m = engineer_all_features(df_15m, feature_config)
    df_45m = engineer_all_features(df_45m, feature_config)

    # Align higher TF features to 5m AFTER engineering
    df_5m, df_15m, df_45m = align_timeframes(df_5m, df_15m, df_45m)

    # Align timeframes - find common valid indices
    valid_5m = ~df_5m.isna().any(axis=1)
    valid_15m = ~df_15m.isna().any(axis=1)
    valid_45m = ~df_45m.isna().any(axis=1)
    common_valid = valid_5m & valid_15m & valid_45m

    df_5m = df_5m[common_valid]
    df_15m = df_15m[common_valid]
    df_45m = df_45m[common_valid]
    
    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in df_5m.columns]
    print(f"  Features: {len(feature_cols)} columns, {len(df_5m):,} aligned rows")
    
    # Step 4: Train Analyst
    print("\n[4/4] Training Market Analyst...")
    save_path = str(config.paths.models_analyst)
    
    # Map 5m/15m/45m into legacy argument names expected by train_analyst()
    model, history = train_analyst(
        df_15m=df_5m,
        df_1h=df_15m,
        df_4h=df_45m,
        feature_cols=feature_cols,
        save_path=save_path,
        config=config.analyst,
        device=config.device,
        visualize=True
    )

    print("=" * 60)
    print("Training Complete!")
    print(f"  Best val loss: {history['best_val_loss']:.6f}")
    print(f"  Epochs trained: {history['epochs_trained']}")
    print("=" * 60)
