"""
Performance metrics for trading strategy evaluation.

Implements:
- Sortino Ratio (target: > 1.5)
- Sharpe Ratio
- Maximum Drawdown (target: < 20%)
- Win Rate (target: > 50%)
- Profit Factor (target: > 1.5)
- Various trade statistics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TradeRecord:
    """Record of a single trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: int  # 1: Long, -1: Short
    size: float
    pnl_pips: float
    pnl_percent: float


def calculate_returns(equity_curve: np.ndarray) -> np.ndarray:
    """Calculate period returns from equity curve."""
    returns = np.diff(equity_curve) / equity_curve[:-1]
    return returns


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252 * 288  # 5-minute bars
) -> float:
    """
    Calculate annualized Sharpe Ratio.

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year

    Returns:
        Annualized Sharpe Ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    sharpe = np.mean(excess_returns) / np.std(excess_returns)

    # Annualize
    return sharpe * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252 * 288,
    max_value: float = 100.0  # Cap to prevent inf in logs/comparisons
) -> float:
    """
    Calculate annualized Sortino Ratio.

    Uses downside deviation instead of standard deviation.
    Target: > 1.5
    
    FIXED: Returns capped value instead of inf to prevent issues in logging/comparison.

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year
        max_value: Maximum return value (caps inf)

    Returns:
        Annualized Sortino Ratio (capped at max_value)
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = np.minimum(excess_returns, 0)
    downside_std = np.sqrt(np.mean(downside_returns ** 2))

    if downside_std == 0:
        # FIXED: Return capped value instead of inf
        return max_value if np.mean(excess_returns) > 0 else 0.0

    sortino = np.mean(excess_returns) / downside_std

    # Annualize and cap
    result = sortino * np.sqrt(periods_per_year)
    return min(result, max_value)


def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown.

    Target: < 20%

    Args:
        equity_curve: Array of equity values

    Returns:
        Tuple of (max_drawdown_percent, peak_idx, trough_idx)
    """
    if len(equity_curve) == 0:
        return 0.0, 0, 0

    # Running maximum
    running_max = np.maximum.accumulate(equity_curve)

    # Drawdown at each point
    drawdown = (running_max - equity_curve) / running_max

    # Maximum drawdown
    max_dd_idx = np.argmax(drawdown)
    max_dd = drawdown[max_dd_idx]

    # Find peak before max drawdown
    peak_idx = np.argmax(equity_curve[:max_dd_idx + 1])

    return max_dd * 100, peak_idx, max_dd_idx


def calculate_win_rate(trades: List[TradeRecord]) -> float:
    """
    Calculate win rate.

    Target: > 50%

    Args:
        trades: List of trade records

    Returns:
        Win rate as percentage
    """
    if len(trades) == 0:
        return 0.0

    winning_trades = sum(1 for t in trades if t.pnl_pips > 0)
    return (winning_trades / len(trades)) * 100


def calculate_profit_factor(trades: List[TradeRecord]) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Target: > 1.5

    Args:
        trades: List of trade records

    Returns:
        Profit factor
    """
    if len(trades) == 0:
        return 0.0

    gross_profit = sum(t.pnl_pips for t in trades if t.pnl_pips > 0)
    gross_loss = abs(sum(t.pnl_pips for t in trades if t.pnl_pips < 0))

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def calculate_average_trade(trades: List[TradeRecord]) -> Dict[str, float]:
    """
    Calculate average trade statistics.

    Args:
        trades: List of trade records

    Returns:
        Dictionary with average trade stats
    """
    if len(trades) == 0:
        return {
            'avg_pnl': 0.0,
            'avg_winner': 0.0,
            'avg_loser': 0.0,
            'avg_duration_bars': 0.0
        }

    winners = [t for t in trades if t.pnl_pips > 0]
    losers = [t for t in trades if t.pnl_pips < 0]

    return {
        'avg_pnl': np.mean([t.pnl_pips for t in trades]),
        'avg_winner': np.mean([t.pnl_pips for t in winners]) if winners else 0.0,
        'avg_loser': np.mean([t.pnl_pips for t in losers]) if losers else 0.0,
        'avg_duration_bars': np.mean([
            (t.exit_time - t.entry_time).total_seconds() / 300  # 5-min bars
            for t in trades
        ]) if trades else 0.0
    }


def calculate_expectancy(trades: List[TradeRecord]) -> float:
    """
    Calculate expectancy (average profit per trade considering win rate).

    Expectancy = (Win% × Avg Win) - (Loss% × Avg Loss)

    Args:
        trades: List of trade records

    Returns:
        Expectancy in pips
    """
    if len(trades) == 0:
        return 0.0

    win_rate = calculate_win_rate(trades) / 100
    avg_stats = calculate_average_trade(trades)

    expectancy = (win_rate * avg_stats['avg_winner']) - ((1 - win_rate) * abs(avg_stats['avg_loser']))

    return expectancy


def calculate_calmar_ratio(
    total_return: float,
    max_drawdown: float,
    years: float = 1.0
) -> float:
    """
    Calculate Calmar Ratio (annualized return / max drawdown).

    Args:
        total_return: Total return as percentage
        max_drawdown: Maximum drawdown as percentage
        years: Number of years in the period

    Returns:
        Calmar Ratio
    """
    if max_drawdown == 0:
        return float('inf') if total_return > 0 else 0.0

    annualized_return = total_return / years
    return annualized_return / max_drawdown


def calculate_metrics(
    equity_curve: np.ndarray,
    trades: List[TradeRecord],
    initial_balance: float = 10000.0,
    periods_per_year: int = 252 * 288
) -> Dict[str, float]:
    """
    Calculate all performance metrics.

    Args:
        equity_curve: Array of equity values
        trades: List of trade records
        initial_balance: Starting balance
        periods_per_year: Trading periods per year

    Returns:
        Dictionary of all metrics
    """
    returns = calculate_returns(equity_curve)
    max_dd, peak_idx, trough_idx = calculate_max_drawdown(equity_curve)
    avg_stats = calculate_average_trade(trades)

    # Total return
    total_return = ((equity_curve[-1] - initial_balance) / initial_balance) * 100

    # Calculate years for annualization
    years = len(equity_curve) / periods_per_year

    metrics = {
        # Returns
        'total_return_pct': total_return,
        'annualized_return_pct': total_return / years if years > 0 else 0.0,
        # NOTE: `pnl_pips` in TradeRecord excludes entry spread/slippage costs because those are
        # deducted directly from balance when opening positions in the backtester.
        # This cash PnL reflects the true net PnL including those costs.
        'net_pnl_cash': float(equity_curve[-1] - initial_balance),

        # Risk-adjusted returns
        'sharpe_ratio': calculate_sharpe_ratio(returns, periods_per_year=periods_per_year),
        'sortino_ratio': calculate_sortino_ratio(returns, periods_per_year=periods_per_year),
        'calmar_ratio': calculate_calmar_ratio(total_return, max_dd, years),

        # Drawdown
        'max_drawdown_pct': max_dd,

        # Trade statistics
        'total_trades': len(trades),
        'win_rate_pct': calculate_win_rate(trades),
        'profit_factor': calculate_profit_factor(trades),
        'expectancy_pips': calculate_expectancy(trades),

        # Average trade
        'avg_trade_pnl': avg_stats['avg_pnl'],
        'avg_winner': avg_stats['avg_winner'],
        'avg_loser': avg_stats['avg_loser'],
        'avg_trade_duration_bars': avg_stats['avg_duration_bars'],

        # Additional
        'total_pnl_pips': sum(t.pnl_pips for t in trades),
        'gross_profit_pips': sum(t.pnl_pips for t in trades if t.pnl_pips > 0),
        'gross_loss_pips': sum(t.pnl_pips for t in trades if t.pnl_pips < 0),
    }

    return metrics


def meets_performance_targets(metrics: Dict[str, float]) -> Dict[str, bool]:
    """
    Check if metrics meet the specified performance targets.

    Targets:
    - Sortino Ratio > 1.5
    - Max Drawdown < 20%
    - Win Rate > 50%
    - Profit Factor > 1.5

    Args:
        metrics: Calculated metrics

    Returns:
        Dictionary indicating which targets are met
    """
    return {
        'sortino_target_met': metrics.get('sortino_ratio', 0) > 1.5,
        'drawdown_target_met': metrics.get('max_drawdown_pct', 100) < 20,
        'win_rate_target_met': metrics.get('win_rate_pct', 0) > 50,
        'profit_factor_target_met': metrics.get('profit_factor', 0) > 1.5,
    }


def print_metrics_report(metrics: Dict[str, float], title: str = "Performance Report"):
    """Print a formatted metrics report."""
    targets = meets_performance_targets(metrics)

    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print('=' * 60)

    print("\n--- Returns ---")
    print(f"Total Return:        {metrics['total_return_pct']:>10.2f}%")
    print(f"Annualized Return:   {metrics['annualized_return_pct']:>10.2f}%")
    print(f"Gross PnL:           {metrics['total_pnl_pips']:>10.1f} pips  (excludes spread/slippage)")
    print(f"Net PnL:          $  {metrics.get('net_pnl_cash', 0.0):>10.2f}  (includes spread/slippage)")

    print("\n--- Risk-Adjusted ---")
    sortino_status = "✓" if targets['sortino_target_met'] else "✗"
    print(f"Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}  {sortino_status} (target: >1.5)")
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
    print(f"Calmar Ratio:        {metrics['calmar_ratio']:>10.2f}")

    print("\n--- Drawdown ---")
    dd_status = "✓" if targets['drawdown_target_met'] else "✗"
    print(f"Max Drawdown:        {metrics['max_drawdown_pct']:>10.2f}%  {dd_status} (target: <20%)")

    print("\n--- Trade Statistics ---")
    print(f"Total Trades:        {metrics['total_trades']:>10}")
    wr_status = "✓" if targets['win_rate_target_met'] else "✗"
    print(f"Win Rate:            {metrics['win_rate_pct']:>10.1f}%  {wr_status} (target: >50%)")
    pf_status = "✓" if targets['profit_factor_target_met'] else "✗"
    print(f"Profit Factor:       {metrics['profit_factor']:>10.2f}  {pf_status} (target: >1.5)")
    print(f"Expectancy:          {metrics['expectancy_pips']:>10.2f} pips/trade")

    print("\n--- Average Trade ---")
    print(f"Avg PnL:             {metrics['avg_trade_pnl']:>10.2f} pips")
    print(f"Avg Winner:          {metrics['avg_winner']:>10.2f} pips")
    print(f"Avg Loser:           {metrics['avg_loser']:>10.2f} pips")
    print(f"Avg Duration:        {metrics['avg_trade_duration_bars']:>10.1f} bars")

    # Summary
    targets_met = sum(targets.values())
    print(f"\n--- Summary: {targets_met}/4 targets met ---")
    print('=' * 60)
