"""
resampler.py — OHLCV resampling utilities.

Two functions for the two execution modes:

  resample_ohlcv(df, timeframe)
      Converts a 1m OHLCV DataFrame to a higher timeframe.
      Used by Mode 1 strategies — strategy runs entirely on resampled bars.

  align_signals_to_1m(signals_htf, df_1m)
      Takes signals generated on higher TF bars and aligns them safely
      back to the 1m index for bar-by-bar execution.
      Used by Mode 2 strategies — 1m execution with higher TF indicators.

LOOKAHEAD SAFEGUARDS:
  - resample_ohlcv: each bar is labelled at its OPEN time (left edge).
    A 60m bar starting at 09:00 closes at 09:59 and is labelled 09:00.
    The signal can only be acted on at 10:00 (the next bar open).
  - align_signals_to_1m: signals are shifted forward by one HTF bar
    before forward-filling, ensuring the forming bar never contributes
    to a live signal.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Supported timeframes
# ---------------------------------------------------------------------------

TIMEFRAME_MAP: dict[str, str] = {
    "1m":    "1min",
    "5m":    "5min",
    "15m":   "15min",
    "30m":   "30min",
    "60m":   "60min",
    "120m":  "120min",
    "240m":  "240min",
    "daily": "1D",
    "1D":    "1D",
}


def _resolve_tf(timeframe: str) -> str:
    """Convert user-friendly TF string to pandas offset alias."""
    tf = TIMEFRAME_MAP.get(timeframe.lower())
    if tf is None:
        raise ValueError(
            f"Unknown timeframe '{timeframe}'. "
            f"Supported: {list(TIMEFRAME_MAP.keys())}"
        )
    return tf


# ---------------------------------------------------------------------------
# Mode 1 — Resample to higher TF
# ---------------------------------------------------------------------------

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample a 1m OHLCV DataFrame to a higher timeframe.

    Bar labelling: LEFT edge (bar open time).
    A 60m bar from 09:00–09:59 is labelled 09:00.

    OHLCV aggregation:
      Open   = first
      High   = max
      Low    = min
      Close  = last
      Volume = sum

    Incomplete bars at the end of the data are dropped — we only
    want fully closed bars.

    Args:
        df:         1m OHLCV DataFrame with DatetimeIndex.
        timeframe:  Target timeframe string e.g. "60m", "daily".

    Returns:
        Resampled DataFrame with DatetimeIndex (bar open times).
    """
    if timeframe in ("1m", "1min"):
        return df.copy()

    tf = _resolve_tf(timeframe)

    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"

    resampled = (
        df.resample(tf, label="left", closed="left")
        .agg(agg)
        .dropna(subset=["open", "close"])  # drop empty bars
    )

    # Drop the last bar if it may be incomplete
    # (i.e. the raw data ends before the bar's expected close)
    if len(resampled) > 1:
        resampled = resampled.iloc[:-1]

    return resampled


# ---------------------------------------------------------------------------
# Mode 2 — Align HTF signals back to 1m index
# ---------------------------------------------------------------------------

def align_signals_to_1m(
    signals_htf: pd.Series,
    df_1m: pd.DataFrame,
) -> pd.Series:
    """
    Align higher-timeframe signals safely onto the 1m bar index.

    The critical step: signals are shifted forward by ONE HTF bar before
    forward-filling. This ensures that a signal generated on the close of
    bar N only becomes active on the 1m bars AFTER that bar closes —
    i.e. from the open of bar N+1 onwards.

    Without this shift, the signal would be active on 1m bars within the
    same HTF bar that generated it, which is lookahead bias.

    Args:
        signals_htf:  Signal Series indexed at HTF bar open times (+1/-1/0).
        df_1m:        The original 1m DataFrame.

    Returns:
        Signal Series aligned to df_1m.index, safe for bar-by-bar execution.
    """
    # Shift forward by 1 HTF bar so signal is only active AFTER bar closes
    signals_shifted = signals_htf.shift(1)

    # Reindex to 1m, forward-fill within each HTF bar period
    signals_1m = (
        signals_shifted
        .reindex(df_1m.index, method="ffill")
        .fillna(0)
        .astype(int)
    )

    return signals_1m


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def describe_resample(df_1m: pd.DataFrame, timeframe: str) -> None:
    """Print a summary of what the resample will produce — useful for validation."""
    if timeframe in ("1m", "1min"):
        print(f"  Timeframe: 1m (no resampling) — {len(df_1m):,} bars")
        return

    resampled = resample_ohlcv(df_1m, timeframe)
    print(f"  Timeframe : {timeframe}")
    print(f"  1m bars   : {len(df_1m):,}")
    print(f"  {timeframe} bars  : {len(resampled):,}")
    print(f"  Date range: {resampled.index[0]} → {resampled.index[-1]}")
    print(f"  First bar : O={resampled['open'].iloc[0]}  "
          f"H={resampled['high'].iloc[0]}  "
          f"L={resampled['low'].iloc[0]}  "
          f"C={resampled['close'].iloc[0]}")
