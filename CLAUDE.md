# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **hybrid trading system for EURUSD** combining supervised learning (Market Analyst) and reinforcement learning (PPO Agent). The project is optimized for Apple M2 Silicon with 8GB RAM constraints.

## System Architecture

**Two-Phase Hierarchical Design:**

1. **Phase A - Market Analyst (Supervised Learning)**
   - Multi-timeframe encoder (15m, 1H, 4H)
   - Produces context vectors for RL agent
   - Trained on smoothed future returns (NOT next-step price)
   - Frozen after pre-training

2. **Phase B - Sniper Agent (Reinforcement Learning)**
   - PPO policy consuming Analyst context + market state
   - Actions: `MultiDiscrete([3, 4])` - Direction [Flat/Exit, Long, Short] × Size [0.25x, 0.5x, 0.75x, 1.0x]
   - Reward engineering: PnL, transaction costs, FOMO penalty, chop avoidance

## Critical Hardware Constraints (Apple M2, 8GB RAM)

**ALWAYS enforce these rules:**

- Device: `device="mps"` (Metal Performance Shaders)
- Precision: `torch.float32` only - **NEVER use float64**
- Batch sizes: 32-64 max
- Clear cache regularly: `torch.mps.empty_cache(); gc.collect()`
- Process data in chunks, never load full dataset to GPU
- Use `torch.no_grad()` during inference
- Delete intermediate tensors immediately

## Tech Stack

Python 3.10+, PyTorch 2.0+, Stable Baselines 3, Gymnasium, Pandas, NumPy, pandas-ta/TA-Lib

## Data Requirements

- Source: EURUSD 1-minute OHLCV (5 years, ~2.6M rows)
- Location: `/Users/gervaciusjr/Desktop/AI Trading Bot/Training data/eurusd_m1_5y_part2_no_gaps.csv`
- Multi-timeframe: Resample 1m → 15m, 1H, 4H using forward-fill on complete datetime index
- Gap handling: Create complete `pd.date_range()` index, then `.reindex().ffill()`
- **CRITICAL**: All timeframes are aligned to the 15m index via forward-fill. 1H and 4H data are subsampled from this aligned index (every 4th and 16th bar respectively) to maintain temporal consistency.

## Feature Engineering Categories

1. **Price Action Patterns**: Pinbar (wick > 2× body), Engulfing, Doji (body < 10% range)
2. **Market Structure**: Fractal S/R levels, ATR-normalized distance to S/R
3. **Trend Filters**: SMA(200) distance/ATR, EMA crossovers
4. **Regime Detection**: Choppiness Index (>61.8 = ranging, <38.2 = trending), ADX (>25 = trending)

## Key Implementation Details

### Market Analyst (src/models/analyst.py)

- **Architecture**: TCN (default) or Transformer - TCN is more stable for binary classification
- Separate `TemporalEncoder` for each timeframe (15m, 1H, 4H)
- `AttentionFusion` layer combines to produce context vector `[batch, context_dim]`
- **Target**: Smoothed future return (NOT next-step price): `df['close'].shift(-12).rolling(12).mean() / df['close'] - 1`
- Training: AdamW optimizer, lr=1e-4 to 3e-4, batch_size=32-64, MSE/Huber loss, early stopping
- After training, **freeze all parameters** and set to `.eval()` mode

### Trading Environment (src/environments/trading_env.py)

**Action Space**: `gym.spaces.MultiDiscrete([3, 4])`
- Direction: `0`=Flat/Exit, `1`=Long, `2`=Short
- Size: `0`=0.25x, `1`=0.5x, `2`=0.75x, `3`=1.0x

**Observation Space**: `gym.spaces.Box` containing:
- Context vector from Analyst (context_dim)
- Position state: [position, entry_price_norm, unrealized_pnl_norm]
- Market features (normalized): [atr, chop, adx, regime, sma_distance]

**Reward Function** (CRITICAL - uses continuous PnL delta, not exit-only):
```python
# Continuous PnL: reward based on CHANGE in unrealized PnL each step
pnl_delta = current_unrealized_pnl - prev_unrealized_pnl
reward = pnl_delta * reward_scaling  # reward_scaling = 1.0 (v15: 1 reward per pip)

# Transaction cost when opening (NOT closing)
if opened_trade:
    reward -= spread_pips * position_size * reward_scaling
    reward += trade_entry_bonus  # v15: +0.5 to encourage exploration

# FOMO: -1.0 if flat during high-momentum moves (|price_move| > 1.5*ATR)
# Chop: disabled (was causing over-penalization)
```

**CRITICAL FIX**: The environment uses **continuous PnL rewards** (delta each step) instead of exit-only rewards to prevent "death spiral" where holding in choppy markets accumulates penalties with no offsetting reward until exit.

### PPO Agent (src/agents/sniper_agent.py)

**Stable Baselines 3 Configuration (v15)**:
```python
PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=256,
    n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
    ent_coef=0.1, vf_coef=0.5, max_grad_norm=0.5, device="mps",
    policy_kwargs={'net_arch': [256, 256]})
```

**MPS Fallback**: If MPS fails, fall back to CPU for SB3 (limited MPS support)

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

## Directory Structure

```
├── requirements.txt
├── config/settings.py           # Hyperparameters & constants
├── data/
│   ├── raw/                     # Raw 1m OHLCV
│   └── processed/               # Multi-timeframe data
├── src/
│   ├── data/                    # loader.py, resampler.py, features.py
│   ├── models/                  # analyst.py, encoders.py, fusion.py
│   ├── environments/            # trading_env.py (Gymnasium)
│   ├── agents/                  # sniper_agent.py (PPO wrapper)
│   ├── training/                # train_analyst.py, train_agent.py
│   └── evaluation/              # backtest.py, metrics.py
├── scripts/run_pipeline.py      # Main execution
├── notebooks/                   # exploration.ipynb
└── models/                      # Saved checkpoints
    ├── analyst/
    └── agent/
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Sortino Ratio | > 1.5 |
| Total Return | Beat buy-and-hold |
| Max Drawdown | < 20% |
| Win Rate | > 50% |
| Profit Factor | > 1.5 |

## Critical Code Patterns

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

## Critical Fixes & Recent Bug Resolutions

**Look-Ahead Bias Prevention** (commits 830e97d, 7c0d6d2):
- Feature normalization uses ONLY training data statistics (fit on first 70%, apply to all)
- Market feature normalization in env uses pre-computed training stats
- Test/eval environments receive training stats via `market_feat_mean` and `market_feat_std` parameters

**Reward Function Fix** (commit ecf6ae8):
- Changed from exit-only PnL rewards to **continuous PnL delta** each step
- Prevents "death spiral" where penalties accumulate without offsetting rewards
- `prev_unrealized_pnl` is reset to 0 when closing positions (critical for multi-trade episodes)

**Multi-Timeframe Alignment** (commit 2c41eec):
- All timeframes forward-filled to 15m index, then subsampled (1H=every 4th, 4H=every 16th)
- `prepare_env_data()` and `create_env_from_dataframes()` correctly subsample aligned data

**Normalization** (commit 53581a8):
- Z-score normalization applied to ALL features (open, high, low, close, indicators)
- Separate normalizers per timeframe (15m, 1H, 4H) saved to `models/analyst/normalizer_{tf}.pkl`
- Prevents scale inconsistencies (ATR ~0.001 vs CHOP 0-100 vs price ~1.05)

**Policy Collapse Fix (v15)**:
- Agent was converging to 99% flat after 4.5M timesteps due to weak reward signal and low entropy
- **Root causes**: `reward_scaling=0.01` (too weak), `ent_coef=0.01` (rapid policy collapse), `fomo_penalty=-0.05` (too small)
- **Fixes applied**:
  - `reward_scaling`: 0.01 → 1.0 (100x increase, 1 reward per pip)
  - `ent_coef`: 0.01 → 0.1 (10x increase, forces exploration)
  - `fomo_penalty`: -0.05 → -1.0 (20x increase, meaningful punishment for inaction)
  - `net_arch`: [64, 64] → [256, 256] (larger policy network)
  - `trade_entry_bonus`: NEW - 0.5 bonus for opening positions (offsets entry cost)
- Must delete old checkpoints and retrain from scratch (policy has converged to degenerate state)

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
| **Policy collapse to flat (v15)** | **ent_coef=0.1, reward_scaling=1.0, trade_entry_bonus=0.5, fomo_penalty=-1.0** |

## Validation Checklist

- Multi-timeframe data aligned without NaN values (check after resampling and feature engineering)
- Normalization: post-normalization feature ranges should be ~[-3, 3] (Z-score)
- Analyst loss decreases, context vectors have variance (not collapsed)
- Environment returns valid (obs, reward, done, info) with correct shapes
- Agent explores diverse actions (not stuck on Flat=0, check action distribution logs)
- Memory usage < 6GB, all tensors float32
- No look-ahead bias: normalization fitted on training data only, applied to all splits
- Reward function: continuous PnL delta each step, NOT just on exit
- `prev_unrealized_pnl` reset to 0 when closing positions

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

# Real-time Visualization Dashboard
cd frontend && npm install               # First time setup
python scripts/start_dashboard.py        # Start dashboard (backend + frontend)
```

## Real-Time Visualization Dashboard

A stunning React + Next.js dashboard for monitoring training in real-time.

**Architecture**:
- Backend: FastAPI + WebSocket (port 8000)
- Frontend: Next.js + Recharts + Framer Motion (port 3000)
- Non-blocking: Training never slows down due to visualization

**Features**:
- **Price Chart**: TradingView-style candlesticks with trade markers, SL/TP lines
- **Neural Network Flow**: Animated attention weights, encoder activations, direction predictions
- **Training Metrics**: Loss curves, episode rewards, equity curve with drawdown
- **Market Gauges**: ATR, Choppiness, ADX, regime indicators
- **Agent Panel**: Position status, value estimates, action probabilities
- **Reward Breakdown**: Waterfall chart showing PnL delta, penalties, bonuses
- **Trade Log**: Live entry/exit tracking with win rate stats
- **System Status**: Memory usage, training speed, connection status

**Setup**:
```bash
# Install Python dependencies
pip install fastapi uvicorn websockets orjson pydantic psutil

# Install frontend dependencies
cd frontend && npm install

# Start dashboard
python scripts/start_dashboard.py
# Opens http://localhost:3000
```

**Enabling Visualization During Training**:
```python
# In config/settings.py, set:
visualization: VisualizationConfig = field(default_factory=lambda: VisualizationConfig(enabled=True))

# Or pass to AgentTrainingLogger:
callback = AgentTrainingLogger(log_dir="models/agent", enable_visualization=True)
```

**Key Files**:
- `visualization/server.py`: FastAPI WebSocket server
- `visualization/data_emitter.py`: Thread-safe training data broadcaster
- `frontend/src/app/page.tsx`: Main dashboard layout
- `frontend/src/components/charts/`: Chart components
- `frontend/src/hooks/useWebSocket.ts`: WebSocket connection hook

## File Organization Specifics

**Data Flow**:
- Raw: `data/raw/eurusd_m1_5y_part2_no_gaps.csv` (external: `/Users/gervaciusjr/Desktop/AI Trading Bot/Training data/`)
- Processed: `data/processed/features_{15m,1h,4h}.parquet` (pre-normalization)
- Normalized: `data/processed/features_{15m,1h,4h}_normalized.parquet` (post-normalization)

**Model Checkpoints**:
- Analyst: `models/analyst/best.pt` (lowest val loss), `models/analyst/normalizer_{15m,1h,4h}.pkl`
- Agent: `models/agent/final_model.zip` (SB3 format), `models/agent/agent_training_metrics.json`

**Logging & Metrics**:
- Training logs: `models/{analyst,agent}/training.log`
- Visualizations: `models/agent/agent_training_summary.png`
- Backtest results: `results/YYYYMMDD_HHMMSS/`

## Key Architecture Details

**Analyst Context Flow**:
1. `TemporalEncoder` (TCN or Transformer) per timeframe → [batch, d_model=32]
2. `AttentionFusion` combines encodings → [batch, d_model]
3. `context_proj` (Linear + LayerNorm) → [batch, context_dim=32]
4. During RL: context vector precomputed for all samples, cached in `TradingEnv._precomputed_contexts`

**Environment Observation Construction** (trading_env.py):
```python
obs = [context_vector (32), analyst_metrics (5), position_state (3), market_features (7), sl_tp_dist (2)]  # Total: 49 dims
# context_vector: frozen analyst embeddings
# analyst_metrics: [p_down, p_up, edge, confidence, uncertainty]
# position_state: [position {-1,0,1}, entry_price_norm, unrealized_pnl_norm]
# market_features (Z-normalized): [atr, chop, adx, regime, sma_distance, bos, choch]
# sl_tp_dist: [distance to SL, distance to TP] normalized by ATR
```

**Reward Scaling Logic (v15)** (config/settings.py):
- `reward_scaling = 1.0` converts ±1 pip trade to ±1.0 reward (direct mapping)
- `trade_entry_bonus = 0.5` offsets entry costs (~0.7 pips) to encourage exploration
- `fomo_penalty = -1.0` makes missing momentum moves costly
- Encourages "Sniper" behavior: selective, high-quality trades after exploration phase

## Debugging & Monitoring

**Agent Training Logs** (src/training/train_agent.py:38-337):
- `AgentTrainingLogger` callback tracks episode rewards, PnL, win rates, action distributions
- Logs every 10 episodes and every 5000 timesteps
- Creates visualizations: reward curves, action distribution, cumulative PnL

**Common Issues**:
1. **Agent stuck on Flat (action=0)**: Check action distribution in logs. If >90% Flat, increase entropy coefficient (`ent_coef`) or reduce penalties
2. **Divergent training**: Check for NaN in observations/rewards. Verify normalization applied correctly
3. **OOM on MPS**: Reduce `batch_size` in config, increase `cache_clear_interval`
4. **SB3 MPS errors**: Fallback to CPU for PPO training (MPS support limited in SB3)

**Verifying No Look-Ahead Bias**:
```python
# Check normalization uses training stats only
print(f"Train end: {train_end_idx}, Total samples: {len(df_15m)}")
# Verify market_feat_mean/std computed from train_market_features[:split_idx]
# Verify test env receives same train stats, NOT test stats
```

## Current Hyperparameters (v15)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `reward_scaling` | 1.0 | 1 reward per pip |
| `ent_coef` | 0.1 | High exploration (was 0.01) |
| `fomo_penalty` | -1.0 | Punish inaction (was -0.05) |
| `trade_entry_bonus` | 0.5 | Encourage trading (NEW) |
| `net_arch` | [256, 256] | Larger network (was [64, 64]) |
| `learning_rate` | 3e-4 | Standard |
| `n_steps` | 2048 | Steps per update |
| `batch_size` | 256 | Minibatch size |
| `total_timesteps` | 20M | Long training run |

## Training Progress Indicators

**Healthy v15 Training @ 1M steps:**
- Action Distribution: ~35% Flat, 15-25% Long, 40-50% Short
- Avg Reward: Improving from -2800 → -700 (should approach 0 by 5M)
- Max Episode Reward: Should see positive episodes (+1000 to +3700)
- Mean Trades: ~100-200/episode (will decrease as agent becomes selective)
- Win Rate: ~40-50% (should improve to >50% after 3M steps)

**Warning Signs:**
- Flat > 90%: Policy collapse - increase `ent_coef` or `trade_entry_bonus`
- Reward not improving after 500K steps: Check reward scaling
- Win rate < 40% after 2M steps: May need architecture changes
- Mean Trades > 300: Over-trading, reduce `trade_entry_bonus`

**Monitoring Commands:**
```bash
# Latest progress
tail -50 models/agent/training_*.log

# Action distribution trend
grep "Action Distribution" models/agent/training_*.log | tail -20

# Check for positive episodes
grep "Recent Avg PnL" models/agent/training_*.log | grep -v "^-"
```
