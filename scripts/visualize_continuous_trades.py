
import pandas as pd
import numpy as np
import mplfinance as mpf
import sys
from pathlib import Path

def visualize_continuous_trades(results_dir, data_path, start_trade_idx=50, num_trades=5):
    results_path = Path(results_dir)
    data_path = Path(data_path)
    
    # Load trades
    trades = pd.read_csv(results_path / "trades.csv")
    trades['entry_time'] = pd.to_datetime(trades['entry_time'])
    trades['exit_time'] = pd.to_datetime(trades['exit_time'])
    
    if len(trades) < start_trade_idx + num_trades:
        print(f"Not enough trades. Total: {len(trades)}, Requested start: {start_trade_idx}")
        return

    # Select consecutive trades
    # We want to show a contiguous block of trades
    selected_trades = trades.iloc[start_trade_idx : start_trade_idx + num_trades].copy()
    
    # Define time window with buffer
    start_time = selected_trades['entry_time'].min()
    end_time = selected_trades['exit_time'].max()
    
    # Load market data
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    if 'datetime' in df.columns:
        df = df.set_index('datetime')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    buffer = pd.Timedelta(hours=4)
    mask = (df.index >= start_time - buffer) & (df.index <= end_time + buffer)
    plot_df = df.loc[mask].copy()
    
    if plot_df.empty:
        print("No data found for the selected trade period.")
        return

    print(f"Plotting {len(plot_df)} bars with {num_trades} trades...")

    # Arrays for markers
    long_entries = np.full(len(plot_df), np.nan)
    short_entries = np.full(len(plot_df), np.nan)
    exits = np.full(len(plot_df), np.nan)
    
    lines = []
    line_colors = []
    
    # Track metrics for the title
    wins = 0
    total_pnl = 0.0

    for i, (_, trade) in enumerate(selected_trades.iterrows()):
        try:
            # Find closest indices
            e_idx = plot_df.index.get_indexer([trade['entry_time']], method='nearest')[0]
            x_idx = plot_df.index.get_indexer([trade['exit_time']], method='nearest')[0]
            
            # Determine direction markers
            # CSV 'direction' could be 'Long'/'Short' string or 1/2 int depending on saver
            # `backtest.py` saves as 'Long'/'Short' string.
            is_long = str(trade['direction']).lower() == 'long' or trade['direction'] == 1
            
            if is_long:
                long_entries[e_idx] = trade['entry_price']
            else:
                short_entries[e_idx] = trade['entry_price']
                
            exits[x_idx] = trade['exit_price']
            
            # Line connecting entry to exit
            # Color by Profit (Green) / Loss (Red)
            pnl = trade['pnl_pips']
            total_pnl += pnl
            is_win = pnl > 0
            if is_win: wins += 1
            
            color = 'green' if is_win else 'red'
            lines.append([
                (trade['entry_time'], trade['entry_price']),
                (trade['exit_time'], trade['exit_price'])
            ])
            line_colors.append(color)

        except Exception as e:
            print(f"Skipping trade {i}: {e}")

    # Create addplots
    # Long Entry = Green Triangle Up
    # Short Entry = Red Triangle Down
    # Exit = Black X
    ap = []
    
    if not np.isnan(long_entries).all():
        ap.append(mpf.make_addplot(long_entries, type='scatter', markersize=120, marker='^', color='green', label='Long Entry'))
    
    if not np.isnan(short_entries).all():
        ap.append(mpf.make_addplot(short_entries, type='scatter', markersize=120, marker='v', color='red', label='Short Entry'))
        
    if not np.isnan(exits).all():
         ap.append(mpf.make_addplot(exits, type='scatter', markersize=80, marker='x', color='black', label='Exit'))

    # Save plot
    output_file = results_path / "continuous_trades_view_refined.png"
    
    title = (f"Trades {start_trade_idx}-{start_trade_idx+num_trades} | "
             f"Green ^ = Long, Red v = Short | "
             f"Wins: {wins}/{num_trades} | Total PnL: {total_pnl:.1f} pips")

    mpf.plot(
        plot_df,
        type='candle',
        style='charles',
        alines=dict(alines=lines, colors=line_colors, linewidths=2, alpha=0.8),
        addplot=ap,
        title=title,
        ylabel='Price',
        volume=False,
        savefig=str(output_file),
        figsize=(18, 9),
        show_nontrading=False 
    )
    
    print(f"Chart saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_continuous_trades.py <results_dir> <data_path> [start_idx] [num_trades]")
        sys.exit(1)
        
    start_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    num_trades = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    visualize_continuous_trades(sys.argv[1], sys.argv[2], start_idx, num_trades)
