"""
Backtesting engine for the hybrid trading system.

Runs the trained agent on out-of-sample data and compares
performance against a buy-and-hold baseline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import torch
import gc
import logging

from .metrics import (
    TradeRecord,
    calculate_metrics,
    print_metrics_report,
    calculate_max_drawdown
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    equity_curve: np.ndarray
    trades: List[TradeRecord]
    metrics: Dict[str, float]
    actions: np.ndarray
    positions: np.ndarray
    timestamps: np.ndarray = field(default=None)


class Backtester:
    """
    Backtesting engine for evaluating the trading agent.

    Features:
    - Step-by-step simulation
    - Trade recording
    - Equity curve generation
    - Comparison with buy-and-hold
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        pip_value: float = 1.0,       # NAS100: 1 point = 1.0 price movement (was 0.0001 for EURUSD)
        lot_size: float = 1.0,        # NAS100 CFD: $1 per point (was 100000 for EURUSD)
        point_multiplier: float = 1.0,  # PnL: points × pip_value × lot_size × multiplier = $1/point
        # Risk Management (MUST match TradingConfig for 1:1 training parity)
        sl_atr_multiplier: float = 2.0,   # v24: Tighter SL at 2x ATR for better R:R
        tp_atr_multiplier: float = 6.0,   # v23.1: Match config/settings.py TradingConfig (was 12.0)
        use_stop_loss: bool = True,
        use_take_profit: bool = True,
        # Volatility Sizing (Dollar-based risk)
        volatility_sizing: bool = True,
        risk_per_trade: float = 100.0,   # Dollar risk per trade (e.g., $100 per trade)
        # v18: Minimum Hold Time
        min_hold_bars: int = 0,  # v23.1: Match config (was 12) - must hold for N bars before exit
        # v19: Profit-based early exit override
        early_exit_profit_atr: float = 3.0,  # Allow early exit if profit > this ATR multiple
        # v20: Break-even stop loss
        break_even_atr: float = 2.0  # Move SL to break-even when profit reaches this ATR
    ):
        """
        Args:
            initial_balance: Starting account balance
            pip_value: Point value for NAS100 (1.0 = 1 point = 1.0 price movement)
            lot_size: CFD lot size (1.0 for NAS100)
            point_multiplier: PnL multiplier for dollar conversion
            sl_atr_multiplier: Stop Loss multiplier (SL = ATR * multiplier)
            tp_atr_multiplier: Take Profit multiplier (TP = ATR * multiplier)
            use_stop_loss: Enable/disable stop-loss mechanism
            use_take_profit: Enable/disable take-profit mechanism
            risk_per_trade: Dollar risk per trade for volatility sizing
            min_hold_bars: Minimum bars to hold before agent can manually exit/flip
            early_exit_profit_atr: Allow early exit if profit exceeds this ATR multiple
            break_even_atr: Move SL to break-even when profit reaches this ATR multiple
        """
        self.initial_balance = initial_balance
        self.pip_value = pip_value
        self.lot_size = lot_size
        self.point_multiplier = point_multiplier

        # Risk Management
        self.sl_atr_multiplier = sl_atr_multiplier
        self.tp_atr_multiplier = tp_atr_multiplier
        self.use_stop_loss = use_stop_loss
        self.use_take_profit = use_take_profit
        
        # Volatility Sizing (Dollar-based risk)
        self.volatility_sizing = volatility_sizing
        self.risk_per_trade = risk_per_trade
        
        # v18: Minimum Hold Time
        self.min_hold_bars = min_hold_bars
        # v19: Profit-based early exit override
        self.early_exit_profit_atr = early_exit_profit_atr
        # v20: Break-even stop loss
        self.break_even_atr = break_even_atr
        self.break_even_activated = False  # Track if break-even is activated for current trade

        # State
        self.balance = initial_balance
        self.position = 0  # -1, 0, 1
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_atr = 0.0  # v27: Store ATR at entry for fixed SL/TP
        self.entry_time = None
        self.entry_step = 0  # v18: Track entry bar for min hold time
        self.current_step = 0  # v18: Track current bar

        # History
        self.equity_history = []
        self.trades = []
        self.actions_history = []
        self.positions_history = []

    def reset(self):
        """Reset backtester state."""
        self.balance = self.initial_balance
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_atr = 0.0  # v27: Reset entry ATR
        self.entry_time = None
        self.entry_step = 0  # v18: Reset entry step
        self.current_step = 0  # v18: Reset current step
        self.equity_history = [self.initial_balance]
        self.trades = []
        self.actions_history = []
        self.positions_history = []

    def _calculate_pnl_pips(self, exit_price: float) -> float:
        """
        Calculate PnL in pips for current position.
        
        FIXED: Returns raw pips without position_size multiplication.
        Position size is applied in _close_position for dollar conversion.
        """
        if self.position == 0:
            return 0.0

        if self.position == 1:  # Long
            pnl = (exit_price - self.entry_price) / self.pip_value
        else:  # Short
            pnl = (self.entry_price - exit_price) / self.pip_value

        return pnl  # Raw pips, not multiplied by position_size

    def _close_position(
        self,
        exit_price: float,
        exit_time: pd.Timestamp
    ) -> float:
        """
        Close current position and record trade.
        
        FIXED: Position size is now correctly applied only once for dollar conversion.
        """
        if self.position == 0:
            return 0.0

        pnl_pips_raw = self._calculate_pnl_pips(exit_price)  # Raw pips
        pnl_pips_sized = pnl_pips_raw * self.position_size   # Adjusted for position size
        # PnL dollar conversion: points × pip_value × lot_size × multiplier
        # NAS100: points × 0.1 × 1.0 × 10 = $1 per point (user confirmed)
        pnl_dollars = pnl_pips_sized * self.pip_value * self.lot_size * self.point_multiplier

        # Record trade
        trade = TradeRecord(
            entry_time=self.entry_time,
            exit_time=exit_time,
            entry_price=self.entry_price,
            exit_price=exit_price,
            direction=self.position,
            size=self.position_size,
            pnl_pips=pnl_pips_sized,  # Store sized pips for consistency
            pnl_percent=(pnl_dollars / self.balance) * 100
        )
        self.trades.append(trade)

        # Update balance
        self.balance += pnl_dollars

        # Reset position
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_atr = 0.0  # v27: Reset entry ATR
        self.entry_time = None
        self.break_even_activated = False  # v20: Reset break-even flag

        return pnl_pips_sized  # Return sized pips for consistency

    def _open_position(
        self,
        direction: int,
        size: float,
        price: float,
        time: pd.Timestamp,
        spread_pips: float = 1.5,
        entry_atr: float = 0.0  # v27: ATR at entry for fixed SL/TP
    ):
        """Open a new position."""
        self.position = direction
        self.position_size = size
        self.entry_price = price
        self.entry_atr = entry_atr  # v27: Store ATR at entry
        self.entry_time = time
        self.entry_step = self.current_step  # v18: Track entry bar for min hold time

        # Deduct spread cost (NAS100: points × pip_value × lot_size × multiplier)
        spread_cost = spread_pips * self.pip_value * self.lot_size * self.point_multiplier * size
        self.balance -= spread_cost

    def _check_stop_loss_take_profit(
        self,
        high: float,
        low: float,
        close: float,
        time: pd.Timestamp,
        atr: float = 0.001
    ) -> Tuple[float, str]:
        """
        Check and execute stop-loss or take-profit if triggered.

        FIXED: Now uses High/Low to detect intra-bar SL/TP hits, not just Close.
        This prevents false positives where price wicked through SL/TP but closed safe.

        Args:
            high: Current bar high price
            low: Current bar low price
            close: Current bar close price
            time: Current timestamp
            atr: Current ATR for dynamic SL/TP calculation

        Returns:
            Tuple of (pnl_pips, close_reason) if triggered, (0, None) otherwise
        """
        if self.position == 0:
            return 0.0, None

        # v27 FIX: Use ATR stored at entry for FIXED SL/TP levels
        # This ensures risk is known at entry and doesn't widen during volatility spikes
        atr_for_sl = self.entry_atr if self.entry_atr > 0 else atr
        
        # Calculate dynamic SL/TP thresholds in points
        sl_pips_threshold = (atr_for_sl * self.sl_atr_multiplier) / self.pip_value
        tp_pips_threshold = (atr_for_sl * self.tp_atr_multiplier) / self.pip_value

        # Ensure minimum values
        sl_pips_threshold = max(sl_pips_threshold, 5.0)
        tp_pips_threshold = max(tp_pips_threshold, 5.0)

        # v20: BREAK-EVEN STOP LOSS
        # If profit reaches break_even_atr * ATR, move SL to entry price
        if self.break_even_atr > 0 and self.position != 0:
            # Calculate current unrealized PnL
            if self.position == 1:  # Long
                unrealized_pnl = (close - self.entry_price) / self.pip_value * self.position_size
            else:  # Short
                unrealized_pnl = (self.entry_price - close) / self.pip_value * self.position_size
            
            break_even_profit_pips = self.break_even_atr * atr
            if unrealized_pnl >= break_even_profit_pips:
                if not self.break_even_activated:
                    self.break_even_activated = True
                # Override SL to break-even (entry price = 0 pips loss)
                sl_pips_threshold = 0.0  # SL at entry price

        # Calculate SL/TP price levels
        pip_value = self.pip_value
        if self.position == 1:  # Long
            sl_price = self.entry_price - sl_pips_threshold * pip_value
            tp_price = self.entry_price + tp_pips_threshold * pip_value

            # For Long: SL triggered if Low <= SL price, TP triggered if High >= TP price
            # IMPORTANT: Check SL first (worst case) to be conservative
            if self.use_stop_loss and low <= sl_price:
                # Exit at SL level (not at the low - we don't know exact fill)
                exit_price = sl_price
                pnl = self._close_position(exit_price, time)
                return pnl, 'stop_loss'

            if self.use_take_profit and high >= tp_price:
                # Exit at TP level
                exit_price = tp_price
                pnl = self._close_position(exit_price, time)
                return pnl, 'take_profit'

        else:  # Short (position == -1)
            sl_price = self.entry_price + sl_pips_threshold * pip_value
            tp_price = self.entry_price - tp_pips_threshold * pip_value

            # For Short: SL triggered if High >= SL price, TP triggered if Low <= TP price
            # IMPORTANT: Check SL first (worst case) to be conservative
            if self.use_stop_loss and high >= sl_price:
                # Exit at SL level
                exit_price = sl_price
                pnl = self._close_position(exit_price, time)
                return pnl, 'stop_loss'

            if self.use_take_profit and low <= tp_price:
                # Exit at TP level
                exit_price = tp_price
                pnl = self._close_position(exit_price, time)
                return pnl, 'take_profit'

        return 0.0, None

    def step(
        self,
        action: np.ndarray,
        high: float,
        low: float,
        close: float,
        time: pd.Timestamp,
        atr: float = 0.001,
        spread_pips: float = 1.5
    ) -> float:
        """
        Execute one step of the backtest.

        FIXED: Now accepts high/low/close for accurate intra-bar SL/TP detection.

        Args:
            action: [direction, size_idx] where direction is 0=Flat, 1=Long, 2=Short
            high: Current bar high price (for SL/TP checks)
            low: Current bar low price (for SL/TP checks)
            close: Current bar close price (for position opening/closing)
            time: Current timestamp
            atr: Current ATR for volatility sizing
            spread_pips: Spread in pips

        Returns:
            Realized PnL in pips (0 if no trade closed)
        """
        direction = action[0]
        size_idx = action[1]
        base_size_factor = [0.25, 0.5, 0.75, 1.0][size_idx]

        # Dollar-Based Volatility Sizing: Maintain constant dollar risk per trade
        # Size = Risk($) / ($/pip × SL_pips)
        if self.volatility_sizing:
            # Calculate SL distance in pips/points
            sl_pips = (atr * self.sl_atr_multiplier) / self.pip_value
            sl_pips = max(sl_pips, 5.0)

            # NAS100 dollar risk sizing:
            # Risk($) = size(lots) × sl_pips(points) × $/point
            # $/point per 1 lot = pip_value × lot_size × point_multiplier
            dollars_per_pip = self.pip_value * self.lot_size * self.point_multiplier
            risk_amount = self.risk_per_trade * base_size_factor  # Scale by agent's choice
            size = risk_amount / (dollars_per_pip * sl_pips)

            # Clip to reasonable limits to prevent extreme leverage
            size = np.clip(size, 0.1, 50.0)  # Max 50 lots
        else:
            # Fixed sizing (1 lot * factor)
            size = 1.0 * base_size_factor

        pnl = 0.0

        # FIRST: Check stop-loss/take-profit BEFORE agent action
        # This enforces risk management regardless of what the agent wants to do
        # FIXED: Now uses high/low for accurate intra-bar SL/TP detection
        sl_tp_pnl, close_reason = self._check_stop_loss_take_profit(high, low, close, time, atr)
        if sl_tp_pnl != 0.0:
            pnl += sl_tp_pnl

        # v23.1 PARITY FIX: If SL/TP triggered (trade closed), skip agent action this step
        # This matches TradingEnv behavior where agent cannot re-enter same bar
        # Without this fix, backtest allows immediate re-entry after SL/TP which training blocked
        if close_reason is not None:
            # Position was closed by SL/TP - skip agent action (parity with training)
            self.current_step += 1
            # Record equity (mark-to-market, but position is now flat)
            self.equity_history.append(self.balance)
            # Record action and position
            self.actions_history.append(action.copy())
            self.positions_history.append(self.position)
            return pnl

        # v18: MINIMUM HOLD TIME CHECK (parity with TradingEnv)
        # Block manual exits AND position flips before min_hold_bars have passed since entry.
        # SL/TP are NOT affected - they are checked BEFORE this.
        # v19: PROFIT-BASED EARLY EXIT OVERRIDE
        # If profit exceeds early_exit_profit_atr * ATR, allow early exit
        if self.position != 0 and self.min_hold_bars > 0:
            bars_held = self.current_step - self.entry_step
            if bars_held < self.min_hold_bars:
                would_close_or_flip = (
                    direction == 0 or  # Flat/Exit
                    (self.position == 1 and direction == 2) or  # Long→Short flip
                    (self.position == -1 and direction == 1)    # Short→Long flip
                )
                if would_close_or_flip:
                    # v19: Check for profit-based early exit override
                    allow_early_exit = False
                    if self.early_exit_profit_atr > 0:
                        # Calculate current unrealized PnL
                        if self.position == 1:  # Long
                            unrealized_pnl = (close - self.entry_price) / self.pip_value * self.position_size
                        else:  # Short
                            unrealized_pnl = (self.entry_price - close) / self.pip_value * self.position_size
                        
                        profit_threshold = self.early_exit_profit_atr * atr
                        if unrealized_pnl > profit_threshold:
                            allow_early_exit = True
                    
                    if not allow_early_exit:
                        # BLOCK: Force agent to keep current position
                        direction = 1 if self.position == 1 else 2  # Keep Long/Short
                        action[0] = direction

        # Handle agent's action
        if direction == 0:  # Flat/Exit
            if self.position != 0:
                pnl += self._close_position(close, time)

        elif direction == 1:  # Long
            if self.position == -1:  # Close short first
                pnl += self._close_position(close, time)
            if self.position == 0:  # Open long
                self._open_position(1, size, close, time, spread_pips, entry_atr=atr)

        elif direction == 2:  # Short
            if self.position == 1:  # Close long first
                pnl += self._close_position(close, time)
            if self.position == 0:  # Open short
                self._open_position(-1, size, close, time, spread_pips, entry_atr=atr)

        # v18: Increment current step counter
        self.current_step += 1

        # Record equity (mark-to-market using close price)
        # NAS100: points × pip_value × lot_size × multiplier × position_size
        unrealized_pnl = self._calculate_pnl_pips(close) * self.pip_value * self.lot_size * self.point_multiplier * self.position_size
        self.equity_history.append(self.balance + unrealized_pnl)

        # Record action and position
        self.actions_history.append(action.copy())
        self.positions_history.append(self.position)

        return pnl

    def get_results(self, timestamps: Optional[np.ndarray] = None) -> BacktestResult:
        """Get backtest results."""
        equity_curve = np.array(self.equity_history)
        metrics = calculate_metrics(
            equity_curve,
            self.trades,
            self.initial_balance
        )

        return BacktestResult(
            equity_curve=equity_curve,
            trades=self.trades,
            metrics=metrics,
            actions=np.array(self.actions_history),
            positions=np.array(self.positions_history),
            timestamps=timestamps
        )


def run_backtest(
    agent,
    env,
    initial_balance: float = 10000.0,
    deterministic: bool = True,
    start_idx: Optional[int] = None,
    max_steps: Optional[int] = None,
    # Risk Management (MUST match TradingConfig for 1:1 training parity)
    sl_atr_multiplier: float = 2.0,    # v24: Tighter SL at 2x ATR for better R:R
    tp_atr_multiplier: float = 6.0,    # v23.1: Match config/settings.py TradingConfig (was 12.0)
    use_stop_loss: bool = True,
    use_take_profit: bool = True,
    min_action_confidence: float = 0.0,
    spread_pips: float = 3.5,          # v23.1: Match config/settings.py TradingConfig
    # v18: Minimum Hold Time
    min_hold_bars: int = 0,            # v23.1: Match config/settings.py TradingConfig (was 12)
    # v19: Profit-based early exit override
    early_exit_profit_atr: float = 3.0,
    # v20: Break-even stop loss
    break_even_atr: float = 2.0
) -> BacktestResult:
    """
    Run a full backtest with the trained agent.

    CRITICAL FIX: Now backtests on the FULL test set, not just a random 2000-step window!
    - Pass start_idx to begin at the start of test set (not random)
    - Set max_steps to cover entire test period (not default 2000)

    Args:
        agent: Trained SniperAgent
        env: Trading environment (should use test data)
        initial_balance: Starting balance
        deterministic: Use deterministic policy
        start_idx: Starting index (if None, uses env.start_idx for full coverage)
        max_steps: Max steps for episode (if None, uses remaining data length)
        sl_atr_multiplier: Stop Loss multiplier
        tp_atr_multiplier: Take Profit multiplier
        use_stop_loss: Enable/disable stop-loss mechanism
        use_take_profit: Enable/disable take-profit mechanism
        min_action_confidence: Minimum confidence threshold for trades (0.0=disabled)
        spread_pips: Spread cost per trade in pips
        min_hold_bars: Minimum bars to hold before agent can manually exit/flip
        early_exit_profit_atr: Allow early exit if profit exceeds this ATR multiple
        break_even_atr: Move SL to break-even when profit reaches this ATR multiple

    Returns:
        BacktestResult with all metrics and trades
    """
    logger.info("Starting backtest...")

    # Calculate backtest coverage
    start_idx = start_idx if start_idx is not None else 0
    max_steps = max_steps if max_steps is not None else len(env.close_prices) - start_idx

    days_covered = max_steps / 288  # 288 5-min bars per day
    logger.info(f"Backtest coverage: start_idx={start_idx}, max_steps={max_steps} ({days_covered:.1f} days of 5m data)")

    # Log actual parameters being used (may come from env or defaults)
    env_sl = getattr(env, 'sl_atr_multiplier', None)
    env_tp = getattr(env, 'tp_atr_multiplier', None)
    env_spread = getattr(env, 'spread_pips', None)
    logger.info(f"Risk Management: SL={sl_atr_multiplier}x ATR (env={env_sl}), TP={tp_atr_multiplier}x ATR (env={env_tp})")
    logger.info(f"Transaction Cost: spread={spread_pips} pips (env={env_spread})")
    logger.info(f"Position Management: min_hold={min_hold_bars} bars, early_exit={early_exit_profit_atr}x ATR, break_even={break_even_atr}x ATR")
    
    if min_action_confidence > 0.0:
        logger.info(f"Confidence Threshold: {min_action_confidence:.2f}")

    # Read SL/TP from environment if available (ensures 1:1 parity with training)
    # Falls back to function defaults if env doesn't have these attributes
    actual_sl_atr = getattr(env, 'sl_atr_multiplier', sl_atr_multiplier)
    actual_tp_atr = getattr(env, 'tp_atr_multiplier', tp_atr_multiplier)
    actual_spread = getattr(env, 'spread_pips', spread_pips)

    backtester = Backtester(
        initial_balance=initial_balance,
        pip_value=getattr(env, 'pip_value', 1.0),
        sl_atr_multiplier=actual_sl_atr,
        tp_atr_multiplier=actual_tp_atr,
        use_stop_loss=use_stop_loss,
        use_take_profit=use_take_profit,
        risk_per_trade=float(getattr(env, "risk_per_trade", 100.0)),  # Dollar-based sizing
        min_hold_bars=min_hold_bars,  # v18: Pass min_hold_bars
        early_exit_profit_atr=early_exit_profit_atr,  # v19: Profit-based early exit
        break_even_atr=break_even_atr  # v20: Break-even stop loss
    )
    backtester.reset()

    # Temporarily override env.max_steps for full test coverage
    original_max_steps = env.max_steps
    env.max_steps = max_steps

    # Detect if this is a recurrent agent (has LSTM states)
    is_recurrent = hasattr(agent, 'reset_lstm_states')
    if is_recurrent:
        agent.reset_lstm_states()
        logger.info("Using RecurrentPPO agent - LSTM states initialized")

    # Reset with FIXED start_idx to ensure full test coverage (not random!)
    obs, info = env.reset(options={'start_idx': start_idx})
    done = False
    truncated = False
    step = 0
    episode_start = True  # Track episode start for recurrent agents
    equity_timestamps: List[int] = []

    env_timestamps = getattr(env, 'timestamps', None)
    if env_timestamps is None:
        raise ValueError(
            "Backtest requires real timestamps from the data, but `env.timestamps` is None. "
            "Pass `timestamps` when creating the TradingEnv (e.g. from the DataFrame datetime index)."
        )
    env_timestamps_arr = np.asarray(env_timestamps)

    def _timestamp_seconds(index: int) -> int:
        raw = env_timestamps_arr[index]
        if isinstance(raw, (np.datetime64, pd.Timestamp)):
            return int(pd.to_datetime(raw).timestamp())
        if isinstance(raw, (np.integer, int, np.floating, float)):
            return int(raw)
        return int(pd.to_datetime(raw).timestamp())

    # Align the initial equity point with the first bar in the backtest window.
    equity_timestamps.append(_timestamp_seconds(env.current_idx))

    while not done and not truncated:
        # Get action from agent
        if is_recurrent:
            # RecurrentPPO needs episode_start flag for LSTM state management
            action, _ = agent.predict(
                obs,
                deterministic=deterministic,
                episode_start=episode_start,
                min_action_confidence=min_action_confidence
            )
            episode_start = False  # Only first step is episode start
        else:
            # Standard PPO
            action, _ = agent.predict(
                obs,
                deterministic=deterministic,
                min_action_confidence=min_action_confidence
            )

        # Step environment
        obs, reward, done, truncated, info = env.step(action)

        # Get OHLC from environment
        # env.current_idx points to NEXT step after env.step(), so current bar is at current_idx-1
        bar_idx = env.current_idx - 1

        # FIXED: Extract High/Low/Close for accurate intra-bar SL/TP detection
        if hasattr(env, 'ohlc_data') and env.ohlc_data is not None:
            # ohlc_data shape: (n_samples, 4) = [open, high, low, close]
            high = float(env.ohlc_data[bar_idx, 1])
            low = float(env.ohlc_data[bar_idx, 2])
            close = float(env.ohlc_data[bar_idx, 3])
        else:
            # Fallback: use close price for all (legacy behavior)
            close = env.close_prices[bar_idx]
            high = close
            low = close

        ts_sec = _timestamp_seconds(bar_idx)
        time = pd.to_datetime(ts_sec, unit='s')

        # Get ATR from environment
        atr = 0.001
        if hasattr(env, 'market_features') and len(env.market_features.shape) > 1:
            atr = env.market_features[bar_idx, 0]

        # v23.1 PARITY FIX: Use EXECUTED action (after analyst masking), not intended action
        # TradingEnv may have masked the action due to enforce_analyst_alignment=True
        # Without this, backtest would take trades that training blocked
        executed_action = action.copy()
        if 'executed_direction' in info:
            executed_action[0] = info['executed_direction']

        # Step backtester with high/low/close for accurate SL/TP detection
        backtester.step(executed_action, high, low, close, time, atr=atr, spread_pips=actual_spread)
        equity_timestamps.append(ts_sec)

        step += 1

        if step % 1000 == 0:
            logger.info(f"Backtest step {step}, Balance: ${backtester.balance:.2f}")

    # Close any remaining position at the end.
    # Parity with TradingEnv episode-end forced close:
    # TradingEnv closes at `exit_idx = min(current_idx, len(close_prices) - 1)` AFTER incrementing current_idx.
    if backtester.position != 0:
        exit_idx = min(int(env.current_idx), len(env.close_prices) - 1)
        exit_ts = _timestamp_seconds(exit_idx)
        final_price = float(env.close_prices[exit_idx])
        final_time = pd.to_datetime(exit_ts, unit='s')
        backtester._close_position(final_price, final_time)

        # Ensure the equity curve ends at the same bar/time as the forced close.
        if equity_timestamps and equity_timestamps[-1] != exit_ts:
            equity_timestamps.append(exit_ts)
            backtester.equity_history.append(backtester.balance)
        else:
            backtester.equity_history[-1] = backtester.balance

    # Restore original max_steps
    env.max_steps = original_max_steps

    results = backtester.get_results(timestamps=np.asarray(equity_timestamps, dtype=np.int64))
    logger.info(f"Backtest complete. {len(results.trades)} trades, "
                f"Final balance: ${results.equity_curve[-1]:.2f}")

    return results


def calculate_buy_and_hold(
    close_prices: np.ndarray,
    initial_balance: float = 10000.0
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Calculate buy-and-hold baseline performance.

    Args:
        close_prices: Array of close prices
        initial_balance: Starting balance

    Returns:
        Tuple of (equity_curve, metrics)
    """
    # Assume we buy at start and hold
    start_price = close_prices[0]
    position_size = initial_balance / start_price

    equity_curve = position_size * close_prices

    # Simple return calculation
    total_return = ((equity_curve[-1] - initial_balance) / initial_balance) * 100

    max_dd, _, _ = calculate_max_drawdown(equity_curve)

    metrics = {
        'total_return_pct': total_return,
        'max_drawdown_pct': max_dd,
        'final_balance': equity_curve[-1]
    }

    return equity_curve, metrics


def compare_with_baseline(
    agent_results: BacktestResult,
    close_prices: np.ndarray,
    initial_balance: float = 10000.0
) -> Dict:
    """
    Compare agent performance with buy-and-hold baseline.

    Args:
        agent_results: Backtest results from agent
        close_prices: Close prices for the same period
        initial_balance: Starting balance

    Returns:
        Comparison dictionary
    """
    # Calculate buy-and-hold
    bh_equity, bh_metrics = calculate_buy_and_hold(close_prices, initial_balance)

    comparison = {
        'agent': {
            'total_return_pct': agent_results.metrics['total_return_pct'],
            'max_drawdown_pct': agent_results.metrics['max_drawdown_pct'],
            'sharpe_ratio': agent_results.metrics['sharpe_ratio'],
            'sortino_ratio': agent_results.metrics['sortino_ratio'],
            'total_trades': agent_results.metrics['total_trades'],
            'final_balance': agent_results.equity_curve[-1]
        },
        'buy_and_hold': {
            'total_return_pct': bh_metrics['total_return_pct'],
            'max_drawdown_pct': bh_metrics['max_drawdown_pct'],
            'final_balance': bh_metrics['final_balance']
        },
        'outperformance': {
            'return_diff_pct': (
                agent_results.metrics['total_return_pct'] -
                bh_metrics['total_return_pct']
            ),
            'drawdown_diff_pct': (
                bh_metrics['max_drawdown_pct'] -
                agent_results.metrics['max_drawdown_pct']
            ),
            'beats_baseline': (
                agent_results.metrics['total_return_pct'] >
                bh_metrics['total_return_pct']
            )
        }
    }

    return comparison


def print_comparison_report(comparison: Dict):
    """Print formatted comparison report."""
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON: Agent vs Buy-and-Hold")
    print("=" * 70)

    print("\n{:<30} {:>15} {:>15}".format("Metric", "Agent", "Buy & Hold"))
    print("-" * 70)

    agent = comparison['agent']
    bh = comparison['buy_and_hold']

    print("{:<30} {:>14.2f}% {:>14.2f}%".format(
        "Total Return",
        agent['total_return_pct'],
        bh['total_return_pct']
    ))
    print("{:<30} {:>14.2f}% {:>14.2f}%".format(
        "Max Drawdown",
        agent['max_drawdown_pct'],
        bh['max_drawdown_pct']
    ))
    print("{:<30} {:>15.2f} {:>15}".format(
        "Sharpe Ratio",
        agent['sharpe_ratio'],
        "N/A"
    ))
    print("{:<30} {:>15.2f} {:>15}".format(
        "Sortino Ratio",
        agent['sortino_ratio'],
        "N/A"
    ))
    print("{:<30} {:>15} {:>15}".format(
        "Total Trades",
        agent['total_trades'],
        1
    ))
    print("{:<30} ${:>14,.2f} ${:>14,.2f}".format(
        "Final Balance",
        agent['final_balance'],
        bh['final_balance']
    ))

    print("\n" + "-" * 70)
    out = comparison['outperformance']
    status = "✓ BEATS BASELINE" if out['beats_baseline'] else "✗ UNDERPERFORMS"
    print(f"Return Outperformance: {out['return_diff_pct']:+.2f}%  |  {status}")
    print(f"Drawdown Improvement:  {out['drawdown_diff_pct']:+.2f}%")
    print("=" * 70)


def save_backtest_results(
    results: BacktestResult,
    path: str,
    comparison: Optional[Dict] = None
):
    """
    Save backtest results to files.

    Args:
        results: BacktestResult object
        path: Directory path
        comparison: Optional comparison with baseline
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save equity curve
    np.save(path / 'equity_curve.npy', results.equity_curve)

    # Save timestamps (unix seconds) if available
    if results.timestamps is not None:
        np.save(path / 'timestamps.npy', results.timestamps)

    # Save trades as CSV
    if results.trades:
        trades_df = pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'direction': 'Long' if t.direction == 1 else 'Short',
                'size': t.size,
                'pnl_pips': t.pnl_pips,
                'pnl_percent': t.pnl_percent
            }
            for t in results.trades
        ])
        trades_df.to_csv(path / 'trades.csv', index=False)

    # Save metrics as JSON
    import json
    with open(path / 'metrics.json', 'w') as f:
        json.dump(results.metrics, f, indent=2)

    if comparison:
        with open(path / 'comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2, default=str)

    logger.info(f"Results saved to {path}")
