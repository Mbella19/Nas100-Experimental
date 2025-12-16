import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from config.settings import Config
from src.data.loader import load_ohlcv
from src.data.resampler import resample_ohlcv

def verify_gap_removal():
    config = Config()
    raw_path = config.paths.data_raw / config.data.raw_file
    
    print(f"Loading raw data from {raw_path}...")
    # Load first 100,000 rows to be fast, but enough to cover a weekend
    # actually load_ohlcv might load everything, let's just load it
    # If it's too big, we might want to just read a chunk with pandas directly
    
    # Let's try reading just a chunk to be safe and fast
    try:
        df_raw = pd.read_csv(raw_path, nrows=50000)
        print(f"Columns found: {list(df_raw.columns)}")
        
        # Normalize column names
        df_raw.columns = [c.lower().strip() for c in df_raw.columns]
        
        # Try to find datetime column
        date_col = None
        for col in ['datetime', 'date', 'time', 'timestamp']:
            if col in df_raw.columns:
                date_col = col
                break
        
        if date_col is None:
            print(f"Could not find datetime column in {df_raw.columns}")
            return
            
        print(f"Using datetime column: {date_col}")
        df_raw.rename(columns={date_col: 'datetime'}, inplace=True)
        
        df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
        df_raw.set_index('datetime', inplace=True)
        
        print(f"Loaded {len(df_raw)} rows.")
        
    except Exception as e:
        print(f"Error loading raw data: {e}")
        return

    print("Resampling to 15m (dropping empty bins)...")
    df_15m = resample_ohlcv(df_raw, '15min')
    
    print(f"Resampled size: {len(df_15m)}")
    
    # Check for flat lines
    closes = df_15m['close'].values
    diffs = np.diff(closes)
    zeros = (diffs == 0).astype(int)
    
    # Find longest run of zeros
    max_run = 0
    current_run = 0
    for z in zeros:
        if z == 1:
            current_run += 1
        else:
            max_run = max(max_run, current_run)
            current_run = 0
    max_run = max(max_run, current_run)
    
    print(f"Longest sequence of identical prices: {max_run} bars")
    
    # Check for time gaps
    time_diffs = df_15m.index.to_series().diff().dropna()
    # Expected diff is 15 minutes
    expected_diff = pd.Timedelta(minutes=15)
    
    gaps = time_diffs[time_diffs > expected_diff]
    
    print(f"Number of time gaps > 15m: {len(gaps)}")
    if len(gaps) > 0:
        print("Sample gaps:")
        print(gaps.head())
        
    # Assertions
    if max_run > 20: # Allow some small flat periods (night trading), but not 192 (weekend)
        print("FAIL: Found long flat sequences! Gap removal might not be working.")
    else:
        print("PASS: No long flat sequences found.")
        
    if len(gaps) > 0:
        print("PASS: Found time gaps (expected behavior for weekend removal).")
    else:
        print("WARNING: No time gaps found. (Maybe the sample didn't include a weekend?)")

if __name__ == "__main__":
    verify_gap_removal()
