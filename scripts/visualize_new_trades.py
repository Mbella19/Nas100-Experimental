import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def visualize_trades(results_dir, data_path, num_winners=5, num_losers=5, offset=3):
    results_path = Path(results_dir)
    data_path = Path(data_path)
    
    # Load trades
    trades = pd.read_csv(results_path / "trades.csv")
    trades['entry_time'] = pd.to_datetime(trades['entry_time'])
    trades['exit_time'] = pd.to_datetime(trades['exit_time'])
    
    # Load data
    df = pd.read_parquet(data_path)
    if 'datetime' in df.columns:
        df = df.set_index('datetime')
    df.index = pd.to_datetime(df.index)
    
    # Selection: Skip the ones we already saw (offset) and get the next batch
    winners = trades[trades['pnl_pips'] > 0].sort_values('pnl_pips', ascending=False).iloc[offset:offset+num_winners]
    losers = trades[trades['pnl_pips'] < 0].sort_values('pnl_pips', ascending=True).iloc[offset:offset+num_losers]
    
    selected_trades = pd.concat([winners, losers])
    
    output_dir = results_path / "trade_plots_new"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for i, (_, trade) in enumerate(selected_trades.iterrows()):
        # Define window (50 bars before entry, 20 bars after exit)
        # Use index-based window to be precise
        entry_idx = df.index.get_indexer([trade['entry_time']], method='nearest')[0]
        exit_idx = df.index.get_indexer([trade['exit_time']], method='nearest')[0]
        
        start_idx = max(0, entry_idx - 40)
        end_idx = min(len(df), exit_idx + 40)
        
        plot_df = df.iloc[start_idx:end_idx].copy()
        
        # Add trade markers
        plot_df['marker'] = np.nan
        # Find index within the slice
        m_entry_idx = plot_df.index.get_indexer([trade['entry_time']], method='nearest')[0]
        m_exit_idx = plot_df.index.get_indexer([trade['exit_time']], method='nearest')[0]
        
        plot_df.iloc[m_entry_idx, plot_df.columns.get_loc('marker')] = trade['entry_price']
        plot_df.iloc[m_exit_idx, plot_df.columns.get_loc('marker')] = trade['exit_price']
        
        # Color based on direction and result
        trade_type = "WINNER" if trade['pnl_pips'] > 0 else "LOSER"
        color = 'green' if trade['pnl_pips'] > 0 else 'red'
        direction = trade['direction']
        
        # Create addplot for markers
        ap = [
            mpf.make_addplot(plot_df['marker'], type='scatter', markersize=100, marker='^' if direction == 'Long' else 'v', color=color)
        ]
        
        # Title
        title = f"New Trade {i+1+offset}: {trade_type} ({direction}) | PnL: {trade['pnl_pips']:.1f} pips"
        
        fname = output_dir / f"new_trade_{i+1+offset}_{trade_type.lower()}.png"
        
        # Plot with show_nontrading=False to hide weekend gaps
        mpf.plot(plot_df, type='candle', style='charles', addplot=ap,
                 title=title, ylabel='Price', savefig=fname,
                 show_nontrading=False, # Hides the time gaps!
                 vlines=dict(vlines=[trade['entry_time'], trade['exit_time']], 
                             colors=['blue', 'orange'], alpha=0.5, linestyle='--'))
        
        print(f"Saved: {fname}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_new_trades.py <results_dir> <data_path>")
        sys.exit(1)
    visualize_trades(sys.argv[1], sys.argv[2])
