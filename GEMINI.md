# Hybrid EURUSD Trading Bot (Gemini Context)

## Project Overview
This is a sophisticated **Hybrid AI Trading System** for **EURUSD**, optimized for **Apple Silicon (M2)** hardware with 8GB RAM constraints. It combines supervised deep learning for market analysis with reinforcement learning for trade execution.

**Core Philosophy:**
*   **The Eyes (Market Analyst):** A TCN/Transformer model that "sees" the market structure and outputs a context vector.
*   **The Trigger (Sniper Agent):** A PPO agent that uses the Analyst's context + real-time metrics to execute trades (Long, Short, Flat).

## System Architecture

### 1. Data Pipeline
*   **Input:** 1-minute OHLCV CSV data (EURUSD).
*   **Processing:** Resamples to **15m, 1H, and 4H** timeframes.
*   **Alignment:** Higher timeframes (1H, 4H) are forward-filled to align with the 15m index, ensuring the agent sees a consistent state at every step.
*   **Normalization:** Critical Z-Score normalization fitted *only* on training data to prevent look-ahead bias.

### 2. Market Analyst (Supervised Learning)
*   **Location:** `src/models/analyst.py`
*   **Type:** TCN (Temporal Convolutional Network) or Transformer.
*   **Goal:** Predict future price direction/volatility.
*   **Output:** A dense `context_vector` (32 dims) consumed by the RL Agent.
*   **Status:** Pre-trained and **frozen** during RL training.

### 3. Sniper Agent (Reinforcement Learning)
*   **Location:** `src/agents/sniper_agent.py`
*   **Algorithm:** PPO (Proximal Policy Optimization) via Stable Baselines 3.
*   **Action Space:** `MultiDiscrete` (Direction: [Flat, Long, Short] × Size: [0.25x, 0.5x, 0.75x, 1.0x]).
*   **Reward Function:** Continuous PnL delta (avoids "death spiral"), adjusted by FOMO and Chop penalties.

## Key Commands

### Full Pipeline
Run the complete end-to-end process (Data -> Analyst -> Agent -> Backtest):
```bash
python scripts/run_pipeline.py
```
*Options:* `--skip-analyst`, `--skip-agent`, `--backtest-only`

### Individual Training
Train the Market Analyst only:
```bash
python -m src.training.train_analyst
```

Train the PPO Agent (requires trained Analyst):
```bash
python -m src.training.train_agent
```

Run Backtest:
```bash
python -m src.evaluation.backtest
```

## Configuration
All settings are centralized in `config/settings.py`:
*   **Paths:** Data and model directories.
*   **Data:** Timeframes, lookback windows.
*   **Analyst:** Architecture (TCN/Transformer), hidden dims, learning rate.
*   **Trading:** Spread, slippage, penalties (FOMO/Chop).
*   **Agent:** PPO hyperparameters (learning rate, batch size, entropy).

## Development Conventions

### 1. Hardware Optimization (Apple Silicon)
*   **Device:** Use `mps` (Metal Performance Shaders) where possible, but fallback to `cpu` for operations not supported by MPS (common in SB3).
*   **Precision:** **Strictly `float32`**. Never use `float64` (doubles memory, no speed gain on M2).
*   **Memory:** Aggressive garbage collection and cache clearing (`torch.mps.empty_cache()`) are required due to 8GB RAM limit.

### 2. Code Style
*   **Typing:** Use Python type hints for all function arguments and returns.
*   **Logging:** Use `src.utils.logging_config` instead of `print`.
*   **Safety:** `TradingEnv` implements "hard" stops (SL/TP) that trigger *before* the agent's action to ensure risk management.
