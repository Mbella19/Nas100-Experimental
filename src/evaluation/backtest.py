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
        lot_size: float = 100000.0  # Standard lot
    ):
        """
        Args:
            initial_balance: Starting account balance
            pip_value: Pip value for EURUSD
            lot_size: Size of one standard lot
        """
        self.initial_balance = initial_balance
        self.pip_value = pip_value
        self.lot_size = lot_size

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
        """Calculate PnL in pips for current position."""
        if self.position == 0:
            return 0.0

        if self.position == 1:  # Long
            pnl = (exit_price - self.entry_price) / self.pip_value
        else:  # Short
            pnl = (self.entry_price - exit_price) / self.pip_value

        return pnl * self.position_size

    def _close_position(
        self,
        exit_price: float,
        exit_time: pd.Timestamp
    ) -> float:
        """Close current position and record trade."""
        if self.position == 0:
            return 0.0

        pnl_pips = self._calculate_pnl_pips(exit_price)
        pnl_dollars = pnl_pips * 10 * self.position_size  # $10 per pip per lot

        # Record trade
        trade = TradeRecord(
            entry_time=self.entry_time,
            exit_time=exit_time,
            entry_price=self.entry_price,
            exit_price=exit_price,
            direction=self.position,
            size=self.position_size,
            pnl_pips=pnl_pips,
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

        return pnl_pips

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

    def step(
        self,
        action: np.ndarray,
        price: float,
        time: pd.Timestamp,
        spread_pips: float = 1.5
    ) -> float:
        """
        Execute one step of the backtest.

        Args:
            action: [direction, size_idx] where direction is 0=Flat, 1=Long, 2=Short
            price: Current close price
            time: Current timestamp
            spread_pips: Spread in pips

        Returns:
            Realized PnL in pips (0 if no trade closed)
        """
        direction = action[0]
        size_idx = action[1]
        size = [0.25, 0.5, 0.75, 1.0][size_idx]

        pnl = 0.0

        # Handle action
        if direction == 0:  # Flat/Exit
            if self.position != 0:
                pnl = self._close_position(price, time)

        elif direction == 1:  # Long
            if self.position == -1:  # Close short first
                pnl = self._close_position(price, time)
            if self.position == 0:  # Open long
                self._open_position(1, size, price, time, spread_pips)

        elif direction == 2:  # Short
            if self.position == 1:  # Close long first
                pnl = self._close_position(price, time)
            if self.position == 0:  # Open short
                self._open_position(-1, size, price, time, spread_pips)

        # Record equity (mark-to-market)
        unrealized_pnl = self._calculate_pnl_pips(price) * 10 * self.position_size
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
    max_steps: Optional[int] = None
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

    backtester = Backtester(initial_balance=initial_balance)
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

        # Get price from environment
        price = env.close_prices[env.current_idx - 1]  # Previous close (current bar)

        # Create timestamp (or use step number)
        time = pd.Timestamp.now() + pd.Timedelta(minutes=15 * step)

        # Step backtester
        backtester.step(action, price, time)

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
