"""
Data Preprocessing & Feature Engineering Module
Handles: date parsing, missing values, lag features, rolling stats, calendar features
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_and_clean(filepath: str) -> pd.DataFrame:
    """Load CSV, parse dates (multiple formats), clean numeric sales column."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    # Clean numeric Total column (remove spaces, commas)
    df['Total'] = (
        df['Total']
        .astype(str)
        .str.replace(r'[\s,]', '', regex=True)
        .pipe(pd.to_numeric, errors='coerce')
    )

    # Parse dates — dataset has mixed formats (M/D/YYYY and DD-MM-YYYY)
    def parse_date(s):
        for fmt in ('%m/%d/%Y', '%d-%m-%Y', '%Y-%m-%d', '%d/%m/%Y'):
            try:
                return pd.to_datetime(s, format=fmt)
            except Exception:
                pass
        return pd.NaT

    df['Date'] = df['Date'].astype(str).apply(parse_date)

    # Drop rows where date or sales couldn't be parsed
    df.dropna(subset=['Date', 'Total'], inplace=True)

    # Sort
    df.sort_values(['State', 'Date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[Preprocessing] Loaded {len(df)} rows | "
          f"{df['State'].nunique()} states | "
          f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all required features per assignment:
      - Lag features: t-1, t-7, t-30 (mapped to prior observations per state)
      - Rolling mean / std (window=3 observations)
      - Day of week, month, holiday flag
    """
    df = df.copy()

    # Calendar features
    df['month']       = df['Date'].dt.month
    df['quarter']     = df['Date'].dt.quarter
    df['year']        = df['Date'].dt.year
    df['dayofweek']   = df['Date'].dt.dayofweek   # 0=Mon
    df['weekofyear']  = df['Date'].dt.isocalendar().week.astype(int)

    # Simple US holiday flag (major holidays by month/day)
    US_HOLIDAYS = {(1,1),(7,4),(12,25),(11,11),(12,31),(1,17),(2,21)}
    df['holiday_flag'] = df['Date'].apply(
        lambda d: 1 if (d.month, d.day) in US_HOLIDAYS else 0
    )

    # Lag & rolling features per state
    df.sort_values(['State', 'Date'], inplace=True)
    for lag in [1, 2, 3]:   # lag-1, lag-2, lag-3 (previous observations)
        df[f'lag_{lag}'] = df.groupby('State')['Total'].shift(lag)

    df['rolling_mean_3'] = (
        df.groupby('State')['Total']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df['rolling_std_3'] = (
        df.groupby('State')['Total']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).std().fillna(0))
    )

    # Trend feature: observation index within each state
    df['obs_index'] = df.groupby('State').cumcount()

    df.reset_index(drop=True, inplace=True)
    return df


def train_val_split(state_df: pd.DataFrame,
                    val_periods: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-series aware split: last `val_periods` observations → validation.
    No data leakage.
    """
    state_df = state_df.sort_values('Date').reset_index(drop=True)
    train = state_df.iloc[:-val_periods]
    val   = state_df.iloc[-val_periods:]
    return train, val


def get_state_series(df: pd.DataFrame, state: str) -> pd.DataFrame:
    """Return time-sorted DataFrame for a single state."""
    return (df[df['State'] == state]
              .sort_values('Date')
              .reset_index(drop=True))
