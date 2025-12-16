"""
Feature normalization module for the trading system.

Implements StandardScaler (Z-Score) normalization that:
- Fits ONLY on training data (prevents look-ahead bias)
- Transforms all data consistently
- Saves/loads scaler for inference
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)


class FeatureNormalizer:
    """
    Z-Score (StandardScaler) normalization for trading features.

    Critical for neural network training:
    - Transforms features to zero mean and unit variance
    - Prevents large-scale features from dominating gradients
    - Must be fit on training data ONLY to prevent look-ahead bias
    - CLIPS extreme values to prevent ±30+ z-scores from destabilizing training
    """

    def __init__(self, feature_cols: List[str], epsilon: float = 1e-8, clip_zscore: float = 5.0):
        """
        Args:
            feature_cols: List of feature column names to normalize
            epsilon: Small value to prevent division by zero
            clip_zscore: Clip normalized values to [-clip_zscore, +clip_zscore]
                        Default 5.0 prevents extreme outliers (>5 std) from
                        destabilizing gradients. Critical for returns/volatility.
        """
        self.feature_cols = feature_cols
        self.epsilon = epsilon
        self.clip_zscore = clip_zscore

        # Statistics (computed during fit)
        self.means: Optional[Dict[str, float]] = None
        self.stds: Optional[Dict[str, float]] = None
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> 'FeatureNormalizer':
        """
        Compute mean and std from training data.

        IMPORTANT: Only call this on TRAINING data to prevent look-ahead bias.

        Args:
            df: Training DataFrame

        Returns:
            self for chaining
        """
        self.means = {}
        self.stds = {}

        for col in self.feature_cols:
            if col in df.columns:
                self.means[col] = float(df[col].mean())
                std = float(df[col].std())
                # Prevent division by zero for constant features
                self.stds[col] = std if std > self.epsilon else 1.0
            else:
                logger.warning(f"Column '{col}' not found in DataFrame")
                self.means[col] = 0.0
                self.stds[col] = 1.0

        self.is_fitted = True
        logger.info(f"Normalizer fitted on {len(self.feature_cols)} features")

        # Log the scales for debugging
        for col in self.feature_cols[:5]:  # Log first 5
            if col in self.means:
                logger.info(f"  {col}: mean={self.means[col]:.6f}, std={self.stds[col]:.6f}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply normalization to DataFrame with outlier clipping.

        Args:
            df: DataFrame to normalize

        Returns:
            Normalized DataFrame (copy) with values clipped to [-clip_zscore, +clip_zscore]
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform. Call fit() first.")

        df_normalized = df.copy()

        for col in self.feature_cols:
            if col in df.columns and col in self.means:
                normalized = (df[col] - self.means[col]) / self.stds[col]
                # CRITICAL: Clip extreme z-scores to prevent gradient instability
                # Without this, features like 'returns' can reach ±30-60 z-scores
                # during volatile periods, causing the model to learn biased shortcuts
                df_normalized[col] = normalized.clip(
                    -self.clip_zscore, self.clip_zscore
                ).astype(np.float32)

        return df_normalized

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit on data and transform in one step.

        Args:
            df: Training DataFrame

        Returns:
            Normalized DataFrame
        """
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse the normalization.

        Args:
            df: Normalized DataFrame

        Returns:
            Original scale DataFrame
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform.")

        df_original = df.copy()

        for col in self.feature_cols:
            if col in df.columns and col in self.means:
                df_original[col] = (
                    df[col] * self.stds[col] + self.means[col]
                ).astype(np.float32)

        return df_original

    def save(self, path: str | Path):
        """
        Save normalizer to disk.

        Args:
            path: Path to save file (.pkl)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'feature_cols': self.feature_cols,
            'epsilon': self.epsilon,
            'clip_zscore': self.clip_zscore,
            'means': self.means,
            'stds': self.stds,
            'is_fitted': self.is_fitted
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Normalizer saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> 'FeatureNormalizer':
        """
        Load normalizer from disk.

        Args:
            path: Path to saved file

        Returns:
            Loaded FeatureNormalizer
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)

        # Backwards compatibility: older saves may not have clip_zscore
        clip_zscore = state.get('clip_zscore', 5.0)
        normalizer = cls(state['feature_cols'], state['epsilon'], clip_zscore)
        normalizer.means = state['means']
        normalizer.stds = state['stds']
        normalizer.is_fitted = state['is_fitted']

        logger.info(f"Normalizer loaded from {path}")
        return normalizer

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get normalization statistics."""
        if not self.is_fitted:
            return {}
        return {
            col: {'mean': self.means[col], 'std': self.stds[col]}
            for col in self.feature_cols
            if col in self.means
        }


def normalize_multi_timeframe(
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_45m: pd.DataFrame,
    feature_cols: List[str],
    train_end_idx: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, FeatureNormalizer]]:
    """
    Normalize multiple timeframe DataFrames using SEPARATE normalizers per timeframe.

    CRITICAL: Each timeframe MUST have its own normalizer because scale-dependent
    features (ATR, volatility, etc.) are naturally larger on higher timeframes:
    - 5m ATR: ~3 pips
    - 45m ATR: ~15 pips

    Using 5m statistics on 45m data would make 45m ATR look like +4.0 Z-score outliers
    constantly, confusing the neural network.

    Args:
        df_5m: 5-minute DataFrame (base timeframe)
        df_15m: 15-minute DataFrame
        df_45m: 45-minute DataFrame
        feature_cols: Columns to normalize
        train_end_idx: End index of training data for 5m (proportionally adjusted for others)

    Returns:
        Tuple of (normalized_5m, normalized_15m, normalized_45m, normalizers_dict)
    """
    # Determine training portion for each timeframe
    if train_end_idx is None:
        train_end_idx = int(len(df_5m) * 0.85)

    # Calculate proportional train indices for higher timeframes
    train_ratio = train_end_idx / len(df_5m)
    train_end_15m = int(len(df_15m) * train_ratio)
    train_end_45m = int(len(df_45m) * train_ratio)

    logger.info(f"Fitting separate normalizers per timeframe (prevents cross-timeframe scale issues)")
    logger.info(f"  5m:  fit on indices 0-{train_end_idx}")
    logger.info(f"  15m: fit on indices 0-{train_end_15m}")
    logger.info(f"  45m: fit on indices 0-{train_end_45m}")

    # Create and fit SEPARATE normalizers for each timeframe
    normalizer_5m = FeatureNormalizer(feature_cols)
    normalizer_15m = FeatureNormalizer(feature_cols)
    normalizer_45m = FeatureNormalizer(feature_cols)

    normalizer_5m.fit(df_5m.iloc[:train_end_idx])
    normalizer_15m.fit(df_15m.iloc[:train_end_15m])
    normalizer_45m.fit(df_45m.iloc[:train_end_45m])

    # Log ATR stats to verify proper normalization
    if 'atr' in feature_cols:
        logger.info("ATR statistics by timeframe (verifying proper scaling):")
        if 'atr' in normalizer_5m.means:
            logger.info(f"  5m ATR:  mean={normalizer_5m.means['atr']:.6f}, std={normalizer_5m.stds['atr']:.6f}")
        if 'atr' in normalizer_15m.means:
            logger.info(f"  15m ATR: mean={normalizer_15m.means['atr']:.6f}, std={normalizer_15m.stds['atr']:.6f}")
        if 'atr' in normalizer_45m.means:
            logger.info(f"  45m ATR: mean={normalizer_45m.means['atr']:.6f}, std={normalizer_45m.stds['atr']:.6f}")

    # Transform each timeframe with ITS OWN normalizer
    df_5m_norm = normalizer_5m.transform(df_5m)
    df_15m_norm = normalizer_15m.transform(df_15m)
    df_45m_norm = normalizer_45m.transform(df_45m)

    # Return dict of normalizers for saving
    normalizers = {
        '5m': normalizer_5m,
        '15m': normalizer_15m,
        '45m': normalizer_45m
    }

    return df_5m_norm, df_15m_norm, df_45m_norm, normalizers


class RobustNormalizer:
    """
    Robust normalization using median and IQR.

    More resistant to outliers than StandardScaler.
    Uses: (x - median) / IQR
    """

    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols
        self.medians: Optional[Dict[str, float]] = None
        self.iqrs: Optional[Dict[str, float]] = None
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> 'RobustNormalizer':
        """Fit using median and IQR."""
        self.medians = {}
        self.iqrs = {}

        for col in self.feature_cols:
            if col in df.columns:
                self.medians[col] = float(df[col].median())
                q75, q25 = df[col].quantile([0.75, 0.25])
                iqr = float(q75 - q25)
                self.iqrs[col] = iqr if iqr > 1e-8 else 1.0

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply robust normalization."""
        if not self.is_fitted:
            raise RuntimeError("Must fit before transform")

        df_normalized = df.copy()
        for col in self.feature_cols:
            if col in df.columns and col in self.medians:
                df_normalized[col] = (
                    (df[col] - self.medians[col]) / self.iqrs[col]
                ).astype(np.float32)
        return df_normalized

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
