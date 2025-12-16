
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Set style for professional trading look
plt.style.use('dark_background')
# Use a high-contrast palette
sns.set_palette("bright")

def plot_7x_results():
    base_dir = Path("results/20251213_102342")
    output_file = base_dir / "backtest_7x_risk_detailed.png"
    
    # 1. Load Data
    # ------------
    equity_curve = np.load(base_dir / "equity_curve.npy")
    trades = pd.read_csv(base_dir / "trades.csv")
    with open(base_dir / "metrics.json", "r") as f:
        metrics = json.load(f)

    timestamps_path = base_dir / "timestamps.npy"
    if not timestamps_path.exists():
        raise FileNotFoundError(
            f"Missing `timestamps.npy` in {base_dir}. Re-run backtest saving timestamps."
        )
    timestamps = np.load(timestamps_path)
    times = pd.to_datetime(timestamps, unit="s")

    # 2. Calculate Drawdown Series
    # ----------------------------
    # Running Max
    running_max = np.maximum.accumulate(equity_curve)
    # Drawdown percentage
    drawdown = (equity_curve - running_max) / running_max * 100

    # 3. Setup Plot
    # -------------
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.5])

    # Top Panel: Equity Curve
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(times, equity_curve, color='#00ff00', linewidth=2, label='Portfolio Value')
    
    # Overlay Trades? 
    # Let's mark All-Time Highs with small dots
    # ath_indices = np.where(equity_curve == running_max)[0]
    # ax1.scatter(days[ath_indices], equity_curve[ath_indices], color='white', s=1, alpha=0.3)

    ax1.set_title(f"NAS100 Hybrid Agent: 7x Risk Stress Test (+{metrics['total_return_pct']:.2f}%)", fontsize=24, fontweight='bold', color='white', pad=20)
    ax1.set_ylabel('Account Balance ($)', fontsize=16)
    ax1.grid(True, alpha=0.1)
    
    # Add floating text box with metrics
    textstr = '\n'.join((
        f"Return:      +{metrics['total_return_pct']:.2f}%",
        f"Max DD:      {metrics['max_drawdown_pct']:.2f}%",
        f"Sharpe:      {metrics['sharpe_ratio']:.2f}",
        f"Sortino:     {metrics['sortino_ratio']:.2f}",
        f"Trades:      {len(trades)}",
        f"Win Rate:    {metrics['win_rate_pct']:.1f}%",
        f"Final Bal:   ${equity_curve[-1]:,.2f}"
    ))
    props = dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='#00ff00')
    ax1.text(0.02, 0.95, textstr, transform=ax1.transAxes, fontsize=14,
            verticalalignment='top', bbox=props, fontfamily='monospace', color='#00ff00')

    # Fill area under equity curve
    ax1.fill_between(times, equity_curve, equity_curve.min(), color='#00ff00', alpha=0.1)

    # Middle Panel: Drawdown
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(times, drawdown, color='#ff3333', linewidth=1.5, label='Drawdown')
    ax2.fill_between(times, drawdown, 0, color='#ff3333', alpha=0.3)
    ax2.set_ylabel('Drawdown (%)', fontsize=16)
    ax2.grid(True, alpha=0.1)
    ax2.set_title("Drawdown Profile", fontsize=14, color='#ff3333')

    # Bottom Panel: Trade PnL bars (optional but cool)
    # We need to map trades to steps roughly
    # Since we don't have exact step indices for trades in the CSV, we'll skip aligning them perfectly
    # and instead show a histogram of trade returns
    
    # Alternative: just label the X axis of the bottom plot
    ax2.set_xlabel('Time', fontsize=16)

    # Adjust layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=100, facecolor='black')
    print(f"Chart saved to {output_file.absolute()}")

if __name__ == "__main__":
    plot_7x_results()
