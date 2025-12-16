"""
Data loading module for OHLC data.

Handles CSV loading with validation and memory-efficient processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Generator
import logging

logger = logging.getLogger(__name__)

# Required columns for OHLC data (with aliases)
REQUIRED_COLUMNS = ['open', 'high', 'low', 'close']
COLUMN_ALIASES = {
    'timestamp': 'datetime',
    'time': 'datetime',
    'date': 'datetime',
}

DATETIME_COLUMNS = ('datetime', 'timestamp', 'time', 'date')


def _select_ohlc_usecols(path: Path) -> List[str]:
    """
    Select only the datetime + OHLC columns to load.

    This intentionally ignores any other columns present in the CSV.
    """
    header = pd.read_csv(path, nrows=0)
    cols = list(header.columns)
    if not cols:
        raise ValueError(f"No columns found in CSV: {path}")

    lower_map = {str(c).strip().lower(): c for c in cols}

    # Prefer explicit datetime-like columns; fall back to first column.
    dt_col = None
    for candidate in DATETIME_COLUMNS:
        if candidate in lower_map:
            dt_col = lower_map[candidate]
            break
    if dt_col is None:
        dt_col = cols[0]

    # Required OHLC columns (case-insensitive)
    missing = [c for c in REQUIRED_COLUMNS if c not in lower_map]
    if missing:
        raise ValueError(f"Missing required columns: {set(missing)}")

    ohlc_cols = [lower_map[c] for c in REQUIRED_COLUMNS]
    usecols = [dt_col, *ohlc_cols]

    # Deduplicate while preserving order
    return list(dict.fromkeys(usecols))


def validate_ohlcv(df: pd.DataFrame) -> bool:
    """
    Validate OHLCV DataFrame has required columns and valid data.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns.str.lower())
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for NaN in critical columns
    critical_cols = ['open', 'high', 'low', 'close']
    for col in critical_cols:
        col_lower = col.lower()
        if col_lower in df.columns:
            nan_count = df[col_lower].isna().sum()
            if nan_count > 0:
                logger.warning(f"Column '{col}' has {nan_count} NaN values")

    # Validate OHLC relationships
    if 'high' in df.columns and 'low' in df.columns:
        invalid_hl = (df['high'] < df['low']).sum()
        if invalid_hl > 0:
            logger.warning(f"Found {invalid_hl} rows where high < low")

    return True


def load_ohlcv(
    path: str | Path,
    datetime_format: str = "%Y-%m-%d %H:%M:%S",
    chunk_size: Optional[int] = None,
    validate: bool = True
) -> pd.DataFrame:
    """
    Load OHLC data from CSV file.

    Args:
        path: Path to CSV file
        datetime_format: Format string for datetime parsing
        chunk_size: If provided, process in chunks for memory efficiency
        validate: Whether to validate the data

    Returns:
        DataFrame with datetime index and float32 OHLC columns
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info(f"Loading OHLCV data from {path}")

    usecols = _select_ohlc_usecols(path)

    if chunk_size:
        # Memory-efficient chunked loading
        df = _load_chunked(path, datetime_format, chunk_size, usecols=usecols)
    else:
        # Standard loading
        df = pd.read_csv(path, usecols=usecols)

    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()

    # Apply column aliases (timestamp/time/date -> datetime)
    df.rename(columns=COLUMN_ALIASES, inplace=True)

    # Parse datetime and set as index
    def _parse_datetime(series: pd.Series) -> pd.Series:
        """Parse datetime with optional format."""
        fmt = datetime_format
        if fmt is None:
            return pd.to_datetime(series, utc=True, errors='coerce')
        fmt_str = str(fmt).lower()
        if fmt_str in ("iso8601", "auto", "auto-detect", "autodetect"):
            return pd.to_datetime(series, utc=True, errors='coerce')
        return pd.to_datetime(series, format=fmt, utc=True, errors='coerce')

    if 'datetime' in df.columns:
        # Handle timezone-aware timestamps
        df['datetime'] = _parse_datetime(df['datetime'])
        df['datetime'] = df['datetime'].dt.tz_localize(None)  # Remove timezone for simplicity
        df.set_index('datetime', inplace=True)
    elif df.index.name == 'datetime' or isinstance(df.index, pd.DatetimeIndex):
        pass  # Already has datetime index
    else:
        # Try first column as datetime
        first_col = df.columns[0]
        df[first_col] = _parse_datetime(df[first_col])
        df[first_col] = df[first_col].dt.tz_localize(None)
        df.set_index(first_col, inplace=True)
        df.index.name = 'datetime'

    # Sort by datetime
    df.sort_index(inplace=True)

    # Convert to float32 for memory efficiency (CRITICAL for M2)
    numeric_cols = ['open', 'high', 'low', 'close']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)

    # Validate if requested
    if validate:
        validate_ohlcv(df.reset_index())

    logger.info(f"Loaded {len(df):,} rows from {df.index.min()} to {df.index.max()}")

    return df


def _load_chunked(
    path: Path,
    datetime_format: str,
    chunk_size: int,
    usecols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Load CSV in chunks and concatenate."""
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunk_size, usecols=usecols):
        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


def load_ohlcv_generator(
    path: str | Path,
    chunk_size: int = 100_000
) -> Generator[pd.DataFrame, None, None]:
    """
    Generator for memory-efficient streaming of large OHLC files.

    Yields:
        DataFrame chunks
    """
    path = Path(path)
    usecols = _select_ohlc_usecols(path)
    for chunk in pd.read_csv(path, chunksize=chunk_size, usecols=usecols):
        chunk.columns = chunk.columns.str.lower()
        if 'datetime' in chunk.columns:
            chunk['datetime'] = pd.to_datetime(chunk['datetime'])
            chunk.set_index('datetime', inplace=True)

        # Convert to float32
        for col in ['open', 'high', 'low', 'close']:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype(np.float32)

        yield chunk


def get_data_info(path: str | Path) -> dict:
    """
    Get information about data file without loading it entirely.

    Returns:
        Dictionary with file info
    """
    path = Path(path)
    info = {
        'path': str(path),
        'size_mb': path.stat().st_size / (1024 * 1024),
    }

    # Read first and last few rows
    sample_head = pd.read_csv(path, nrows=5)
    info['columns'] = list(sample_head.columns)

    # Count total rows (memory-efficient)
    with open(path, 'r') as f:
        info['total_rows'] = sum(1 for _ in f) - 1  # Subtract header

    return info
