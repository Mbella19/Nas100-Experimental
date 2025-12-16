"""
Market Analyst Model - Complete supervised learning module.

The Analyst produces a dense "Context Vector" that summarizes market state
across all timeframes. It predicts a discrete class for smoothed future
returns (direction-focused) to learn sustained momentum rather than noise.

After training, the Analyst is frozen and its context vector is consumed
by the PPO Sniper Agent.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Union
import gc

from .encoders import TransformerEncoder, LightweightEncoder
from .fusion import AttentionFusion, SimpleFusion, ConcatFusion


class MarketAnalyst(nn.Module):
    """
    Multi-timeframe Market Analyst with Transformer encoders.

    Architecture:
        - 3 TransformerEncoders (15m, 1H, 4H)
        - AttentionFusion layer
        - Trend prediction head (for training)

    Usage:
        Training: Use forward() to get both context and prediction
        Inference: Use get_context() for frozen context vector
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        context_dim: int = 64,
        dropout: float = 0.1,
        use_lightweight: bool = False,
        num_classes: int = 5
    ):
        """
        Args:
            feature_dims: Dict mapping timeframe to input feature dimension
                         e.g., {'15m': 12, '1h': 12, '4h': 12}
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: FFN hidden dimension
            context_dim: Output context vector dimension
            dropout: Dropout rate
            use_lightweight: Use lighter Conv1D encoder instead of Transformer
            num_classes: Number of discrete return classes for classification head
        """
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.context_dim = context_dim
        self.num_classes = num_classes

        # Determine timeframe keys.
        # New system uses {'5m','15m','45m'} aligned to base 5m index.
        # Legacy system uses {'15m','1h','4h'}.
        if any(k in feature_dims for k in ('5m', '45m')):
            self.timeframes: List[str] = ['5m', '15m', '45m']
        else:
            self.timeframes = ['15m', '1h', '4h']

        # Create encoder for each timeframe dynamically
        self.encoders = nn.ModuleDict()
        for tf in self.timeframes:
            input_dim = feature_dims.get(tf, next(iter(feature_dims.values())))
            if use_lightweight:
                self.encoders[tf] = LightweightEncoder(
                    input_dim,
                    hidden_dim=d_model,
                    dropout=dropout
                )
            else:
                self.encoders[tf] = TransformerEncoder(
                    input_dim,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout
                )

        # Backwards-compatible aliases
        # These always map to base/mid/high encoders regardless of key names.
        self.encoder_15m = self.encoders[self.timeframes[0]]
        self.encoder_1h = self.encoders[self.timeframes[1]]
        self.encoder_4h = self.encoders[self.timeframes[2]]

        # Fusion layer
        self.fusion = AttentionFusion(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout
        )

        # Context projection (optional, if input_dim != context_dim)
        context_input_dim = d_model
        if context_input_dim != context_dim:
            self.context_proj = nn.Sequential(
                nn.Linear(context_input_dim, context_dim),
                nn.LayerNorm(context_dim)
            )
        else:
            self.context_proj = nn.Identity()

        # Direction prediction head (main task - for supervised training)
        # For binary: outputs 1 logit (use BCEWithLogitsLoss)
        # For multi-class: outputs num_classes logits
        self.num_classes = num_classes
        if num_classes == 2:
            # Binary classification: single output + sigmoid
            self.direction_head = nn.Sequential(
                nn.Linear(context_dim, context_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(context_dim // 2, 1)  # Single logit for BCE
            )
        else:
            # Multi-class classification
            self.direction_head = nn.Sequential(
                nn.Linear(context_dim, context_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(context_dim // 2, num_classes)
            )

        # Multi-horizon prediction heads
        # These address Target Mismatch by learning at multiple time scales
        # Shorter horizons (1H, 2H) provide easier "stepping stone" targets
        if num_classes == 2:
            # Binary multi-horizon heads (each predicts Up/Down at different horizons)
            self.horizon_1h_head = nn.Sequential(
                nn.Linear(context_dim, context_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(context_dim // 2, 1)
            )
            self.horizon_2h_head = nn.Sequential(
                nn.Linear(context_dim, context_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(context_dim // 2, 1)
            )
            # Note: direction_head is the 4H head (primary target)
        else:
            self.horizon_1h_head = None
            self.horizon_2h_head = None

        # Auxiliary heads for multi-task learning
        # These provide regularization and help learn better representations

        # Volatility prediction: predicts future ATR / current ATR (regression)
        self.volatility_head = nn.Sequential(
            nn.Linear(context_dim, context_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim // 2, 1)
        )

        # Regime prediction: trending (1) vs ranging (0) (binary)
        self.regime_head = nn.Sequential(
            nn.Linear(context_dim, context_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim // 2, 1)
        )

        # Legacy alias for compatibility with existing code
        self.trend_head = self.direction_head

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize linear layer weights for stable training.
        """
        # Initialize all Sequential modules
        modules_to_init = [
            self.context_proj,
            self.direction_head,
            self.volatility_head,
            self.regime_head
        ]

        # Add multi-horizon heads if they exist
        if self.horizon_1h_head is not None:
            modules_to_init.append(self.horizon_1h_head)
        if self.horizon_2h_head is not None:
            modules_to_init.append(self.horizon_2h_head)

        for module in modules_to_init:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            # FIXED: Was zeros_ which creates 50% equilibrium trap
                            # Small positive bias breaks symmetry, prevents the model
                            # from getting stuck at sigmoid(0)=0.5 for all predictions
                            nn.init.constant_(layer.bias, 0.1)

    def _encode_and_fuse(
        self,
        x_15m: torch.Tensor,
        x_1h: torch.Tensor,
        x_4h: torch.Tensor,
        return_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Internal method: encode all timeframes and fuse to context.

        Returns:
            context: Context vector [batch, context_dim]
            weights: Attention weights [batch, 2] (optional)
        """
        # Ensure float32
        x_base = x_15m.float()
        x_mid = x_1h.float()
        x_high = x_4h.float()

        # Encode each timeframe
        enc_base = self.encoder_15m(x_base)  # base timeframe
        enc_mid = self.encoder_1h(x_mid)
        enc_high = self.encoder_4h(x_high)

        # Fuse timeframes
        weights = None
        if return_weights and isinstance(self.fusion, AttentionFusion):
            fused, weights = self.fusion.forward_with_weights(enc_base, enc_mid, enc_high)
        else:
            fused = self.fusion(enc_base, enc_mid, enc_high)  # [batch, d_model]

        # Project to context dimension
        context = self.context_proj(fused)  # [batch, context_dim]

        if return_weights:
            return context, weights
        return context

    def forward(
        self,
        x_15m: torch.Tensor,
        x_1h: torch.Tensor,
        x_4h: torch.Tensor,
        return_aux: bool = False,
        return_multi_horizon: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Full forward pass for training.

        Args:
            x_15m: 15-minute features [batch, seq_len, features]
            x_1h: 1-hour features [batch, seq_len, features]
            x_4h: 4-hour features [batch, seq_len, features]
            return_aux: If True, also return auxiliary predictions (volatility, regime)
            return_multi_horizon: If True, also return multi-horizon predictions (1H, 2H)

        Returns:
            If return_aux=False and return_multi_horizon=False (default):
                Tuple of (context, direction_logits)
            If return_aux=True:
                Tuple of (context, direction_logits, volatility_pred, regime_pred)
            If return_multi_horizon=True:
                Tuple of (context, direction_logits, horizon_1h_logits, horizon_2h_logits)
            If both True:
                Tuple of (context, direction_logits, volatility_pred, regime_pred,
                          horizon_1h_logits, horizon_2h_logits)
        """
        # Encode and fuse to context
        context = self._encode_and_fuse(x_15m, x_1h, x_4h)

        # Main direction prediction (4H horizon)
        direction = self.direction_head(context)  # [batch, 1] for binary, [batch, n] for multi

        if not return_aux and not return_multi_horizon:
            # Backward compatible: return only context and direction
            return context, direction

        # Build return tuple based on what's requested
        result = [context, direction]

        if return_aux:
            # Auxiliary predictions
            volatility = self.volatility_head(context).squeeze(-1)  # [batch]
            regime = self.regime_head(context).squeeze(-1)  # [batch]
            result.extend([volatility, regime])

        if return_multi_horizon and self.horizon_1h_head is not None:
            # Multi-horizon predictions (only for binary mode)
            horizon_1h = self.horizon_1h_head(context)  # [batch, 1]
            horizon_2h = self.horizon_2h_head(context)  # [batch, 1]
            result.extend([horizon_1h, horizon_2h])

        return tuple(result)

    @torch.no_grad()
    def get_context(
        self,
        x_15m: torch.Tensor,
        x_1h: torch.Tensor,
        x_4h: torch.Tensor
    ) -> torch.Tensor:
        """
        Get context vector only (for inference/RL agent).

        This should be used after freezing the model.

        Args:
            x_15m: 15-minute features [batch, seq_len, features]
            x_1h: 1-hour features [batch, seq_len, features]
            x_4h: 4-hour features [batch, seq_len, features]

        Returns:
            Context vector [batch, context_dim]
        """
        context, _ = self.forward(x_15m, x_1h, x_4h)
        return context

    @torch.no_grad()
    def get_probabilities(
        self,
        x_15m: torch.Tensor,
        x_1h: torch.Tensor,
        x_4h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Get context vector AND probabilities (for RL agent).

        Args:
            x_15m: 15-minute features [batch, seq_len, features]
            x_1h: 1-hour features [batch, seq_len, features]
            x_4h: 4-hour features [batch, seq_len, features]

        Returns:
            Tuple of:
                - context: Context vector [batch, context_dim]
                - probs: Probabilities [batch, num_classes]
                - weights: Attention weights [batch, 2]
        """
        # Manually call encode_and_fuse to get weights
        context, weights = self._encode_and_fuse(
            x_15m, x_1h, x_4h,
            return_weights=True
        )
        
        # Get predictions
        direction = self.direction_head(context)
        logits = direction

        if self.num_classes == 2:
            # Binary: logits is [batch, 1], convert to [batch, 2] probs
            p_up = torch.sigmoid(logits)  # [batch, 1]
            p_down = 1 - p_up
            probs = torch.cat([p_down, p_up], dim=-1)  # [batch, 2]
        else:
            # Multi-class: use softmax
            probs = torch.softmax(logits, dim=-1)

        return context, probs, weights

    def freeze(self):
        """
        Freeze all parameters for use with RL agent.

        Call this after training is complete.
        """
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze parameters for fine-tuning."""
        self.train()
        for param in self.parameters():
            param.requires_grad = True

    def get_attention_weights(
        self,
        x_15m: torch.Tensor,
        x_1h: torch.Tensor,
        x_4h: torch.Tensor
    ) -> torch.Tensor:
        """
        Get attention weights for interpretability.

        Returns:
            Attention weights [batch, 2] showing attention to 1H and 4H
        """
        with torch.no_grad():
            enc_base = self.encoder_15m(x_15m)
            enc_mid = self.encoder_1h(x_1h)
            enc_high = self.encoder_4h(x_4h)
            _, weights = self.fusion.forward_with_weights(enc_base, enc_mid, enc_high)
        return weights


def create_analyst(
    feature_dims: Dict[str, int],
    config: Optional[object] = None,
    device: Optional[torch.device] = None
) -> MarketAnalyst:
    """
    Factory function to create a Market Analyst with config.

    Args:
        feature_dims: Feature dimensions per timeframe
        config: AnalystConfig object (optional)
        device: Target device

    Returns:
        MarketAnalyst model
    """
    if config is None:
        # Default configuration
        model = MarketAnalyst(
            feature_dims=feature_dims,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            context_dim=64,
            dropout=0.1,
            num_classes=5
        )
    else:
        model = MarketAnalyst(
            feature_dims=feature_dims,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            context_dim=config.context_dim,
            dropout=config.dropout,
            num_classes=getattr(config, 'num_classes', 5)
        )

    if device is not None:
        model = model.to(device)

    return model


def load_analyst(
    path: str,
    feature_dims: Dict[str, int],
    device: Optional[torch.device] = None,
    freeze: bool = True
) -> MarketAnalyst:
    """
    Load a trained Market Analyst from checkpoint.

    Args:
        path: Path to checkpoint file
        feature_dims: Feature dimensions (must match training)
        device: Target device
        freeze: Whether to freeze the model

    Returns:
        Loaded MarketAnalyst
    """
    # Load checkpoint
    checkpoint = torch.load(path, map_location=device or 'cpu')

    # Check architecture type
    architecture = 'transformer'
    if 'config' in checkpoint:
        architecture = checkpoint['config'].get('architecture', 'transformer')

    # Dispatch to TCN loader if needed
    if architecture == 'tcn':
        from .tcn_analyst import load_tcn_analyst
        # We can pass the path directly as load_tcn_analyst will load it again
        # OR we can modify load_tcn_analyst to accept a loaded checkpoint.
        # For simplicity and to avoid modifying tcn_analyst.py right now, we'll just call it with the path.
        # However, we already loaded the checkpoint here. 
        # Let's just delegate completely.
        del checkpoint
        return load_tcn_analyst(path, feature_dims, device, freeze)

    # Create Transformer model
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        from config.settings import config as global_config
        
        model = MarketAnalyst(
            feature_dims=feature_dims,
            d_model=saved_config.get('d_model', global_config.analyst.d_model),
            nhead=saved_config.get('nhead', global_config.analyst.nhead),
            num_layers=saved_config.get('num_layers', global_config.analyst.num_layers),
            dim_feedforward=saved_config.get('dim_feedforward', global_config.analyst.dim_feedforward),
            context_dim=saved_config.get('context_dim', global_config.analyst.context_dim),
            dropout=saved_config.get('dropout', global_config.analyst.dropout),
            num_classes=saved_config.get('num_classes', global_config.analyst.num_classes),
        )
    else:
        model = create_analyst(feature_dims)

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    if device is not None:
        model = model.to(device)

    if freeze:
        model.freeze()

    # Clear memory
    del checkpoint
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return model
