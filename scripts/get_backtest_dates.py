
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import load_ohlcv
from config.settings import Config

def get_dates():
    config = Config()
    data_path = config.paths.data_raw / config.data.raw_file
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return

    print("Loading data...")
    df_1m = load_ohlcv(data_path, datetime_format=config.data.datetime_format)
    
    # Logic from run_pipeline.py
    # 1. Resample to 5m (base timeframe)
    print(f"Columns: {df_1m.columns}")
    resample_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }
    if 'Volume' in df_1m.columns:
        resample_dict['Volume'] = 'sum'
    elif 'volume' in df_1m.columns:
        resample_dict['volume'] = 'sum'
        
    df_5m = df_1m.resample('5min').agg(resample_dict).dropna()
    
    # 2. prepare_env_data creates windows, effectively trimming the start
    lookback_5m = 48
    lookback_15m = 16
    lookback_45m = 6
    subsample_15m = 3
    subsample_45m = 9
    
    start_idx = max(
        lookback_5m,
        (lookback_15m - 1) * subsample_15m + 1,
        (lookback_45m - 1) * subsample_45m + 1,
    )
    
    # Remaining samples after windowing
    n_samples = len(df_5m) - start_idx
    
    # 3. Test split is last 15% of THESE samples
    test_start_offset = int(0.85 * n_samples)
    
    # Calculate actual indices in df_5m
    # Test Start Index = start_idx + test_start_offset
    actual_test_start_idx = start_idx + test_start_offset
    actual_test_end_idx = len(df_5m) - 1
    
    # Training Split (First 85%)
    # Starts after lookback trimming
    actual_train_start_idx = start_idx
    actual_train_end_idx = actual_test_start_idx - 1
    
    train_start_date = df_5m.index[actual_train_start_idx]
    train_end_date = df_5m.index[actual_train_end_idx]
    train_duration = train_end_date - train_start_date

    test_start_date = df_5m.index[actual_test_start_idx]
    test_end_date = df_5m.index[actual_test_end_idx]
    test_duration = test_end_date - test_start_date
    
    print("\n" + "="*50)
    print("DATA SPLIT ANALYSIS")
    print("="*50)
    print(f"Total 5m Bars:       {len(df_5m):,}")
    print(f"Valid Samples:       {n_samples:,} (after lookback trimming)")
    print("-" * 50)
    print("TRAINING SET (First 85%)")
    print(f"Samples:             {test_start_offset:,} bars")
    print(f"Start Date:          {train_start_date}")
    print(f"End Date:            {train_end_date}")
    print(f"Duration:            {train_duration}")
    print("-" * 50)
    print("TEST SET (Last 15%)")
    print(f"Samples:             {n_samples - test_start_offset:,} bars")
    print(f"Start Date:          {test_start_date}")
    print(f"End Date:            {test_end_date}")
    print(f"Duration:            {test_duration}")
    print("="*50)

if __name__ == "__main__":
    get_dates()
