# Qwen Code Context for AI Trading Bot

## Project Overview

This is a **hybrid trading system for EURUSD** that combines supervised learning (Market Analyst) and reinforcement learning (PPO Agent). The project is optimized for Apple M2 Silicon with 8GB RAM constraints and implements a two-phase hierarchical design:

1. **Phase A - Market Analyst (Supervised Learning)**: Multi-timeframe encoder (15m, 1H, 4H) that produces context vectors for the RL agent, trained on smoothed future returns rather than next-step price prediction.

2. **Phase B - Sniper Agent (Reinforcement Learning)**: PPO policy consuming Analyst context + market state, with actions defined as `MultiDiscrete([3, 4])` for Direction [Flat/Exit, Long, Short] × Size [0.25x, 0.5x, 0.75x, 1.0x].

## System Architecture

### Market Analyst
- **Architecture**: TCN (Temporal Convolutional Network) or Transformer - TCN is more stable for binary classification
- Separate `TemporalEncoder` for each timeframe (15m, 1H, 4H)
- `AttentionFusion` layer combines encodings to produce context vector `[batch, context_dim]`
- Target: Smoothed future return (NOT next-step price): `df['close'].shift(-12).rolling(12).mean() / df['close'] - 1`
- After training, all parameters are frozen and set to `.eval()` mode

### Trading Environment
- **Action Space**: `gym.spaces.MultiDiscrete([3, 4])`
  - Direction: `0`=Flat/Exit, `1`=Long, `2`=Short
  - Size: `0`=0.25x, `1`=0.5x, `2`=0.75x, `3`=1.0x
- **Observation Space**: Contains context vector from Analyst, position state, and normalized market features
- **Reward Function**: Uses continuous PnL delta each step (not exit-only) to prevent "death spiral" during choppy markets

### PPO Agent
- Uses Stable Baselines 3 configuration with MPS fallback for Apple Silicon
- Configured for optimal performance on M2 with limited RAM
- Includes transaction costs, FOMO penalty, and chop avoidance in reward engineering

## Critical Hardware Constraints (Apple M2, 8GB RAM)

Always enforce these rules:
- Device: `device="mps"` (Metal Performance Shaders)
- Precision: `torch.float32` only - **NEVER use float64**
- Batch sizes: 32-64 max
- Clear cache regularly: `torch.mps.empty_cache(); gc.collect()`
- Process data in chunks, never load full dataset to GPU
- Use `torch.no_grad()` during inference
- Delete intermediate tensors immediately

## Tech Stack

- Python 3.10+
- PyTorch 2.0+
- Stable Baselines 3
- Gymnasium
- Pandas, NumPy
- pandas-ta/TA-Lib

## Directory Structure

```
├── requirements.txt
├── CLAUDE.md                 # Main project documentation and guidelines
├── AGENTS.md                 # Repository guidelines and conventions
├── config/
│   └── settings.py           # Hyperparameters & constants
├── data/
│   ├── raw/                  # Raw 1m OHLCV
│   └── processed/            # Multi-timeframe data
├── src/
│   ├── data/                 # loader.py, resampler.py, features.py
│   ├── models/               # analyst.py, encoders.py, fusion.py
│   ├── environments/         # trading_env.py (Gymnasium)
│   ├── agents/               # sniper_agent.py (PPO wrapper)
│   ├── training/             # train_analyst.py, train_agent.py
│   └── evaluation/           # backtest.py, metrics.py
├── scripts/run_pipeline.py   # Main execution
├── notebooks/                # exploration.ipynb
└── models/                   # Saved checkpoints
    ├── analyst/
    └── agent/
```

## Data Requirements

- Source: EURUSD 1-minute OHLCV (5 years, ~2.6M rows)
- Location: `/Users/gervaciusjr/Desktop/AI Trading Bot/Training data/eurusd_m1_5y_part2_no_gaps.csv`
- Multi-timeframe: Resample 1m → 15m, 1H, 4H using forward-fill on complete datetime index
- Gap handling: Create complete `pd.date_range()` index, then `.reindex().ffill()`
- **CRITICAL**: All timeframes are aligned to the 15m index via forward-fill. 1H and 4H data are subsampled from this aligned index (every 4th and 16th bar respectively) to maintain temporal consistency.

## Feature Engineering Categories

1. **Price Action Patterns**: Pinbar (wick > 2× body), Engulfing, Doji (body < 10% range)
2. **Market Structure**: Fractal S/R levels, ATR-normalized distance to S/R
3. **Trend Filters**: SMA(50) distance/ATR, EMA crossovers
4. **Regime Detection**: Choppiness Index (>61.8 = ranging, <38.2 = trending), ADX (>25 = trending)

## Key Implementation Details

### Device & Memory Management
```python
# Device selection
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Always float32
tensor = tensor.to(device=device, dtype=torch.float32)
df = df.astype(np.float32)

# Clear cache every 50 batches
if batch_idx % 50 == 0:
    torch.mps.empty_cache()
    gc.collect()
    del intermediate_tensors
```

### SB3 MPS Fallback
```python
try:
    agent = PPO("MlpPolicy", env, device="mps")
except:
    agent = PPO("MlpPolicy", env, device="cpu")  # SB3 has limited MPS support
```

## Execution Pipeline

**Full pipeline** (`scripts/run_pipeline.py`):
1. Load & clean 1m OHLCV → resample to 15m/1H/4H → align timeframes → engineer features
2. Train Market Analyst on smoothed future returns (85% train, 15% val)
3. Freeze Analyst weights (`param.requires_grad = False`, `.eval()`)
4. Initialize TradingEnv with frozen Analyst → train PPO agent (5M timesteps)
5. Run out-of-sample backtest (final 15%) → compare to buy-and-hold baseline

**Individual modules**:
- Train Analyst only: `python -m src.training.train_analyst`
- Train Agent only: `python -m src.training.train_agent` (requires pre-trained Analyst)
- Run backtest: `python -m src.evaluation.backtest`

## Performance Targets

| Metric | Target |
|--------|--------|
| Sortino Ratio | > 1.5 |
| Total Return | Beat buy-and-hold |
| Max Drawdown | < 20% |
| Win Rate | > 50% |
| Profit Factor | > 1.5 |

## Critical Fixes & Recent Bug Resolutions

**Look-Ahead Bias Prevention**: Feature normalization uses ONLY training data statistics (fit on first 70%, apply to all)
**Reward Function Fix**: Changed from exit-only PnL rewards to **continuous PnL delta** each step to prevent "death spiral"
**Multi-Timeframe Alignment**: All timeframes forward-filled to 15m index, then subsampled (1H=every 4th, 4H=every 16th)
**Normalization**: Z-score normalization applied to ALL features to prevent scale inconsistencies

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| OOM on M2 | Reduce batch size, gradient checkpointing |
| float64 tensors | Enforce float32 everywhere |
| Look-ahead bias | Use ONLY training stats for normalization (all splits) |
| Overfitting | Dropout, early stopping, validation split |
| Passive agent | FOMO penalty, entropy bonus, continuous PnL rewards |
| Over-trading | Transaction cost modeling |
| Choppy markets | Chop avoidance penalty |
| Death spiral in chop | Use continuous PnL delta, not exit-only rewards |
| Timeframe misalignment | Forward-fill all to 15m, then subsample higher TFs |

## Commands

```bash
# Installation
pip install -r requirements.txt

# Full pipeline (data → Analyst → Agent → backtest)
python scripts/run_pipeline.py

# Skip steps with existing models
python scripts/run_pipeline.py --skip-analyst    # Use existing Analyst
python scripts/run_pipeline.py --skip-agent      # Use existing Agent
python scripts/run_pipeline.py --backtest-only   # Only run backtest

# Individual modules
python -m src.training.train_analyst     # Train Analyst only
python -m src.training.train_agent       # Train Agent only (requires trained Analyst)
python -m src.evaluation.backtest        # Run backtest only
```

## Building and Running

1. **Setup**: Create a Python 3.10+ environment and install dependencies: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2. **Data**: Place the raw 1-minute CSV at the configured `training_data_dir` (defaults to `/Users/gervaciusjr/Desktop/Market data/Data`); adjust in `config/settings.py` if needed
3. **Execution**: Run the full workflow: `python scripts/run_pipeline.py`. This will process data, engineer features, train the analyst, freeze it, train the PPO agent, and backtest
4. **Targeted runs**: Use `python -m src.training.train_analyst` to retrain the analyst only, or `python -m src.training.train_agent` to retrain the PPO agent using a saved analyst

## Development Conventions

- Follow PEP8 with 4-space indentation and type hints
- Use snake_case for functions/variables and PascalCase for classes
- Keep modules narrowly scoped (data/, models/, training/, evaluation/)
- Prefer structured logging (`src.utils.logging_config.get_logger`) over print statements
- Name files by role (`train_*.py`, `*_env.py`, `*_agent.py`)

## Testing & Validation

- No dedicated unit test suite yet; rely on integration runs
- After changes, run at least `python scripts/run_pipeline.py` on a trimmed dataset to verify the full loop
- Inspect outputs in `results/` (backtest metrics, plots) and model checkpoints in `models/` for regressions
- Compare against `evaluation.backtest.compare_with_baseline` outputs to ensure the agent still beats buy-and-hold
- When modifying feature engineering or data splits, spot-check `data/processed/*.parquet` to confirm alignment across timeframes
