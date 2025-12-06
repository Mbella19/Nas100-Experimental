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
        pip_value: float = 0.0001,
        lot_size: float = 100000.0,  # Standard lot
        # Risk Management
        sl_atr_multiplier: float = 1.5,
        tp_atr_multiplier: float = 3.0,
        use_stop_loss: bool = True,
        use_take_profit: bool = True,
        # Volatility Sizing
        volatility_sizing: bool = True,
        risk_per_trade: float = 100.0  # Dollar risk per trade
    ):
        """
        Args:
            initial_balance: Starting account balance
            pip_value: Pip value for EURUSD
            lot_size: Size of one standard lot
            sl_atr_multiplier: Stop Loss multiplier (SL = ATR * multiplier)
            tp_atr_multiplier: Take Profit multiplier (TP = ATR * multiplier)
            use_stop_loss: Enable/disable stop-loss mechanism
            use_take_profit: Enable/disable take-profit mechanism
        """
        self.initial_balance = initial_balance
        self.pip_value = pip_value
        self.lot_size = lot_size

        # Risk Management
        self.sl_atr_multiplier = sl_atr_multiplier
        self.tp_atr_multiplier = tp_atr_multiplier
        self.use_stop_loss = use_stop_loss
        self.use_take_profit = use_take_profit
        
        # Volatility Sizing
        self.volatility_sizing = volatility_sizing
        self.risk_per_trade = risk_per_trade

        # State
        self.balance = initial_balance
        self.position = 0  # -1, 0, 1
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_time = None

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
        self.entry_time = None
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
        pnl_dollars = pnl_pips_sized * 10  # $10 per pip per lot

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
        self.entry_time = None

        return pnl_pips_sized  # Return sized pips for consistency

    def _open_position(
        self,
        direction: int,
        size: float,
        price: float,
        time: pd.Timestamp,
        spread_pips: float = 1.5
    ):
        """Open a new position."""
        self.position = direction
        self.position_size = size
        self.entry_price = price
        self.entry_time = time

        # Deduct spread cost
        spread_cost = spread_pips * 10 * size  # $10 per pip
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

        # Calculate dynamic SL/TP thresholds in pips
        sl_pips_threshold = (atr * self.sl_atr_multiplier) / 0.0001
        tp_pips_threshold = (atr * self.tp_atr_multiplier) / 0.0001

        # Ensure minimum values
        sl_pips_threshold = max(sl_pips_threshold, 5.0)
        tp_pips_threshold = max(tp_pips_threshold, 5.0)

        # Calculate SL/TP price levels
        pip_value = 0.0001
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

        # Volatility Sizing: Calculate lot size based on risk
        if self.volatility_sizing:
            # Calculate SL distance in pips
            sl_pips = (atr * self.sl_atr_multiplier) / 0.0001
            sl_pips = max(sl_pips, 5.0)

            # Risk Formula: Risk = Lots * 10 * SL_pips
            # Lots = Risk / (10 * SL_pips)
            # We use base_size_factor to scale the risk (e.g. 0.5x risk)

            risk_amount = self.risk_per_trade * base_size_factor
            size = risk_amount / (10 * sl_pips)

            # Cap size to avoid crazy leverage in ultra-low vol
            size = min(size, 50.0)  # Max 50 lots
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
            # Position is now flat after SL/TP, agent can still open new position

        # Handle agent's action
        if direction == 0:  # Flat/Exit
            if self.position != 0:
                pnl += self._close_position(close, time)

        elif direction == 1:  # Long
            if self.position == -1:  # Close short first
                pnl += self._close_position(close, time)
            if self.position == 0:  # Open long
                self._open_position(1, size, close, time, spread_pips)

        elif direction == 2:  # Short
            if self.position == 1:  # Close long first
                pnl += self._close_position(close, time)
            if self.position == 0:  # Open short
                self._open_position(-1, size, close, time, spread_pips)

        # Record equity (mark-to-market using close price)
        unrealized_pnl = self._calculate_pnl_pips(close) * 10 * self.position_size
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
    # Risk Management (defaults from config)
    sl_atr_multiplier: float = 1.5,
    tp_atr_multiplier: float = 3.0,
    use_stop_loss: bool = True,
    use_take_profit: bool = True
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

    Returns:
        BacktestResult with all metrics and trades
    """
    logger.info("Starting backtest...")

    # Calculate backtest coverage
    if start_idx is None:
        start_idx = env.start_idx  # Start from beginning of available data

    if max_steps is None:
        max_steps = env.end_idx - start_idx  # Cover full test set

    logger.info(f"Backtest coverage: start_idx={start_idx}, max_steps={max_steps} "
                f"({max_steps * 15 / 60 / 24:.1f} days of 15m data)")
    logger.info(f"Risk Management: SL={sl_atr_multiplier}x ATR (enabled={use_stop_loss}), "
                f"TP={tp_atr_multiplier}x ATR (enabled={use_take_profit})")

    backtester = Backtester(
        initial_balance=initial_balance,
        sl_atr_multiplier=sl_atr_multiplier,
        tp_atr_multiplier=tp_atr_multiplier,
        use_stop_loss=use_stop_loss,
        use_take_profit=use_take_profit
    )
    backtester.reset()

    # Temporarily override env.max_steps for full test coverage
    original_max_steps = env.max_steps
    env.max_steps = max_steps

    # Reset with FIXED start_idx to ensure full test coverage (not random!)
    obs, info = env.reset(options={'start_idx': start_idx})
    done = False
    truncated = False
    step = 0

    while not done and not truncated:
        # Get action from agent
        action, _ = agent.predict(obs, deterministic=deterministic)

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

        # Create timestamp (or use step number)
        time = pd.Timestamp.now() + pd.Timedelta(minutes=15 * step)

        # Get ATR from environment
        atr = 0.001
        if hasattr(env, 'market_features') and len(env.market_features.shape) > 1:
            atr = env.market_features[bar_idx, 0]

        # Step backtester with high/low/close for accurate SL/TP detection
        backtester.step(action, high, low, close, time, atr=atr)

        step += 1

        if step % 1000 == 0:
            logger.info(f"Backtest step {step}, Balance: ${backtester.balance:.2f}")

    # Close any remaining position at the end
    if backtester.position != 0:
        final_price = env.close_prices[env.current_idx - 1]
        final_time = pd.Timestamp.now() + pd.Timedelta(minutes=15 * step)
        backtester._close_position(final_price, final_time)
        backtester.equity_history[-1] = backtester.balance

    # Restore original max_steps
    env.max_steps = original_max_steps

    results = backtester.get_results()
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
