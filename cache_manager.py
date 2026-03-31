"""
cache_manager.py — Pre-resampled OHLCV cache system.

Eliminates repeated resampling of raw 1m data on every backtest run.
Once built, a cache file is reused across all runs on that asset/timeframe.
Cache is invalidated automatically when the raw file is newer.

Cache files live in:
    data/cache/<SYMBOL>_<TF>m.cache   e.g. data/cache/FGBL_120m.cache

Cache index lives at:
    data/cache_index.json

Usage:
    from cache_manager import load_cached, cache_is_fresh, build_cache

    # In backtest.py — transparent cache load
    df = load_cached(symbol, timeframe, raw_path, date_format)

    # In cache_build.py — explicit build
    build_cache(symbol, timeframe, raw_path, date_format)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from resampler import resample_ohlcv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_DIR   = Path(__file__).parent / "data"
_CACHE_DIR  = _DATA_DIR / "cache"
_INDEX_PATH = _DATA_DIR / "cache_index.json"


def _cache_path(symbol: str, timeframe: int) -> Path:
    return _CACHE_DIR / f"{symbol}_{timeframe}m.cache"


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def _load_index() -> dict:
    if _INDEX_PATH.exists():
        try:
            with open(_INDEX_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_index(index: dict) -> None:
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _INDEX_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(index, f, indent=2)
    tmp.replace(_INDEX_PATH)


def _raw_mtime(raw_path: str) -> float:
    """Return modification time of raw file, or 0 if missing."""
    try:
        return Path(raw_path).stat().st_mtime
    except FileNotFoundError:
        return 0.0


def get_split_date(symbol: str) -> str | None:
    """
    Return the train/test split date for a symbol from the cache index.
    Split is calculated at cache build time as the midpoint row of the
    full 1m dataset — consistent and automatic per asset.
    Returns ISO timestamp string or None if not yet cached.
    """
    index = _load_index()
    return index.get(symbol, {}).get("split_date")


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Freshness check
# ---------------------------------------------------------------------------

def cache_is_fresh(symbol: str, timeframe: int, raw_path: str) -> bool:
    """
    Return True if a valid, up-to-date cache exists for symbol/timeframe.
    Cache is stale if:
      - Cache file doesn't exist
      - Index entry missing
      - Raw file is newer than when cache was built
    """
    cp = _cache_path(symbol, timeframe)
    if not cp.exists():
        return False

    index = _load_index()
    entry = index.get(symbol, {}).get("timeframes", {}).get(str(timeframe))
    if not entry:
        return False

    raw_mtime  = _raw_mtime(raw_path)
    built_mtime = entry.get("raw_mtime_at_build", 0)

    return raw_mtime <= built_mtime


# ---------------------------------------------------------------------------
# Raw data loader (shared with backtest.py)
# ---------------------------------------------------------------------------

def load_raw(path: str, date_format: str) -> pd.DataFrame:
    """Load raw 1m CSV into standardised DataFrame."""
    import numpy as np

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    for candidate in ("time", "datetime", "timestamp", "date"):
        if candidate in df.columns:
            df.rename(columns={candidate: "date"}, inplace=True)
            break

    if "date" not in df.columns:
        raise ValueError(f"Cannot find datetime column in {path}. Found: {list(df.columns)}")

    try:
        df["date"] = pd.to_datetime(df["date"], format=date_format)
    except ValueError:
        df["date"] = pd.to_datetime(df["date"])

    required = {"open", "high", "low", "close"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    if "volume" not in df.columns:
        df["volume"] = np.nan

    keep = ["date", "open", "high", "low", "close", "volume", "vwap"]
    df   = df[[c for c in keep if c in df.columns]]
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _write_cache(df: pd.DataFrame, symbol: str, timeframe: int) -> None:
    """Write resampled DataFrame to cache file."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cp = _cache_path(symbol, timeframe)
    tmp = cp.with_suffix(".tmp")
    df.to_csv(tmp)
    tmp.replace(cp)


def _read_cache(symbol: str, timeframe: int) -> pd.DataFrame:
    """Read cache file into DataFrame."""
    cp = _cache_path(symbol, timeframe)
    df = pd.read_csv(cp, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Build (full rebuild)
# ---------------------------------------------------------------------------

def build_cache(
    symbol: str,
    timeframe: int,
    raw_path: str,
    date_format: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build (or rebuild) the cache for symbol/timeframe from raw 1m data.
    Returns the resampled DataFrame.
    """
    if verbose:
        print(f"  Building cache: {symbol} {timeframe}m ... ", end="", flush=True)

    df_1m = load_raw(raw_path, date_format)

    if timeframe == 1:
        df_out = df_1m
    else:
        df_out = resample_ohlcv(df_1m, f"{timeframe}m")

    _write_cache(df_out, symbol, timeframe)

    # Calculate split date as midpoint of 1m rows (done once, same for all timeframes)
    split_idx  = len(df_1m) // 2
    split_date = str(df_1m.index[split_idx])

    # Update index
    index = _load_index()
    if symbol not in index:
        index[symbol] = {"timeframes": {}}
    if "timeframes" not in index[symbol]:
        index[symbol]["timeframes"] = {}

    # Store split_date at asset level (same regardless of timeframe)
    index[symbol]["split_date"] = split_date

    index[symbol]["timeframes"][str(timeframe)] = {
        "cache_rows":         len(df_out),
        "cache_last_ts":      str(df_out.index[-1]),
        "raw_rows":           len(df_1m),
        "raw_last_ts":        str(df_1m.index[-1]),
        "raw_mtime_at_build": _raw_mtime(raw_path),
        "built_at":           _now_str(),
    }
    _save_index(index)

    if verbose:
        print(f"         Split date: {split_date} (row {split_idx:,} of {len(df_1m):,})")

    if verbose:
        print(f"done  ({len(df_out):,} bars, {df_out.index[0].date()} → {df_out.index[-1].date()})")

    return df_out


# ---------------------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------------------

def update_cache(
    symbol: str,
    timeframe: int,
    raw_path: str,
    date_format: str,
    verbose: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Incrementally append new rows to an existing cache.
    Drops the last cached bar (may be incomplete) and re-derives from that
    point using the updated raw data.

    Returns updated DataFrame, or None if cache didn't exist (use build_cache).
    """
    cp = _cache_path(symbol, timeframe)
    if not cp.exists():
        if verbose:
            print(f"  {symbol} {timeframe}m: no cache found — run build_cache first.")
        return None

    index = _load_index()
    entry = index.get(symbol, {}).get("timeframes", {}).get(str(timeframe), {})
    last_cached_ts = entry.get("cache_last_ts")

    if not last_cached_ts:
        if verbose:
            print(f"  {symbol} {timeframe}m: index missing — rebuilding.")
        return build_cache(symbol, timeframe, raw_path, date_format, verbose)

    if verbose:
        print(f"  Updating cache: {symbol} {timeframe}m ... ", end="", flush=True)

    df_1m = load_raw(raw_path, date_format)

    # Find the start of the last cached bar's period so we can re-derive it
    last_ts = pd.Timestamp(last_cached_ts)

    # For resampling, roll back by timeframe minutes to get the bar start
    bar_start = last_ts - pd.Timedelta(minutes=timeframe - 1)
    new_1m    = df_1m[df_1m.index >= bar_start]

    if len(new_1m) == 0:
        if verbose:
            print("already up to date.")
        return _read_cache(symbol, timeframe)

    # Resample the tail
    if timeframe == 1:
        new_bars = new_1m
    else:
        new_bars = resample_ohlcv(new_1m, f"{timeframe}m")

    # Load existing cache and drop last bar (stale), append new
    existing = _read_cache(symbol, timeframe)
    existing = existing[existing.index < last_ts]  # drop last bar
    updated  = pd.concat([existing, new_bars])
    updated  = updated[~updated.index.duplicated(keep="last")]
    updated.sort_index(inplace=True)

    _write_cache(updated, symbol, timeframe)

    # Update index
    new_rows = len(updated) - len(existing)
    index[symbol]["timeframes"][str(timeframe)] = {
        "cache_rows":         len(updated),
        "cache_last_ts":      str(updated.index[-1]),
        "raw_rows":           len(df_1m),
        "raw_last_ts":        str(df_1m.index[-1]),
        "raw_mtime_at_build": _raw_mtime(raw_path),
        "built_at":           _now_str(),
    }
    _save_index(index)

    if verbose:
        print(f"done  (+{new_rows} bars → {len(updated):,} total, up to {updated.index[-1].date()})")

    return updated


# ---------------------------------------------------------------------------
# Main load function — used by backtest.py
# ---------------------------------------------------------------------------

def load_cached(
    symbol: str,
    timeframe: int,
    raw_path: str,
    date_format: str,
    auto_build: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Load resampled data for symbol/timeframe, using cache if fresh.

    If cache is missing or stale:
      - If auto_build=True: rebuild cache automatically (first run only)
      - If auto_build=False: fall back to raw resample (old behaviour)

    Args:
        symbol:      Asset symbol e.g. 'FGBL'
        timeframe:   Integer minutes e.g. 120
        raw_path:    Path to full raw 1m CSV
        date_format: strptime format string
        auto_build:  Whether to build cache automatically if missing/stale
        verbose:     Print cache status

    Returns:
        Resampled DataFrame (full history, no date filtering).
    """
    if cache_is_fresh(symbol, timeframe, raw_path):
        if verbose:
            print(f"  [{symbol} {timeframe}m] Loading from cache.")
        return _read_cache(symbol, timeframe)

    if auto_build:
        if verbose or True:  # always announce first-time builds
            status = "stale" if _cache_path(symbol, timeframe).exists() else "missing"
            print(f"  [{symbol} {timeframe}m] Cache {status} — building now.")
        return build_cache(symbol, timeframe, raw_path, date_format, verbose=True)

    # Fallback: resample on the fly (old behaviour)
    if verbose:
        print(f"  [{symbol} {timeframe}m] Cache not used — resampling from raw.")
    df_1m = load_raw(raw_path, date_format)
    if timeframe == 1:
        return df_1m
    return resample_ohlcv(df_1m, f"{timeframe}m")


# ---------------------------------------------------------------------------
# Strategy-specific resample cache
# ---------------------------------------------------------------------------

def build_resampled_cache(
    symbol: str,
    timeframe: int,
    df_1m: pd.DataFrame,
    verbose: bool = True,
    offset_min: int = 15,
) -> pd.DataFrame:
    """
    Build and cache a resampled OHLCV+VWAP dataframe from 1m data.
    Uses the same resample parameters as FGBLMomentum.

    The cache key includes the offset so different offsets get separate files.
    """
    import numpy as np

    cache_key = f"{symbol}_{timeframe}m_off{offset_min}"
    cp = _CACHE_DIR / f"{cache_key}.cache"

    if verbose:
        print(f"  Building {timeframe}m cache: {symbol} (offset={offset_min}min) ... ",
              end="", flush=True)

    sig = df_1m.resample(
        f"{timeframe}min",
        label="left",
        closed="left",
        origin="start_day",
        offset=f"{offset_min}min",
    ).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum" if "volume" in df_1m.columns else "first",
    }).dropna(subset=["open", "close"])

    if "vwap" in df_1m.columns:
        sig["vwap"] = df_1m["vwap"].resample(
            f"{timeframe}min",
            label="left",
            closed="left",
            origin="start_day",
            offset=f"{offset_min}min",
        ).last()

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cp.with_suffix(".tmp")
    sig.to_csv(tmp)
    tmp.replace(cp)

    # Update index
    index = _load_index()
    if symbol not in index:
        index[symbol] = {"timeframes": {}}
    tf_key = f"{timeframe}_off{offset_min}"
    if "timeframes" not in index[symbol]:
        index[symbol]["timeframes"] = {}
    index[symbol]["timeframes"][tf_key] = {
        "cache_rows":    len(sig),
        "cache_last_ts": str(sig.index[-1]),
        "built_at":      _now_str(),
        "offset_min":    offset_min,
    }
    _save_index(index)

    if verbose:
        print(f"done  ({len(sig):,} bars)")

    return sig


def load_resampled_cache(
    symbol: str,
    timeframe: int,
    df_1m: pd.DataFrame,
    offset_min: int = 15,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Load resampled data from cache, building it automatically if missing.
    This is the function called by grid_search preload.

    On first call: resamples df_1m, saves cache, returns result.
    On subsequent calls: loads cache file directly — very fast.
    """
    import numpy as np

    cache_key = f"{symbol}_{timeframe}m_off{offset_min}"
    cp = _CACHE_DIR / f"{cache_key}.cache"

    # Check if cache is still valid — use 1m cache last_ts as staleness signal
    index = _load_index()
    tf_key = f"{timeframe}_off{offset_min}"
    cached_entry = index.get(symbol, {}).get("timeframes", {}).get(tf_key)
    raw_entry    = index.get(symbol, {}).get("timeframes", {}).get("1")

    cache_fresh = (
        cp.exists()
        and cached_entry is not None
        and (
            raw_entry is None  # no 1m cache to compare against
            or cached_entry.get("built_at", "") >= raw_entry.get("built_at", "")
        )
    )

    if cache_fresh:
        if verbose:
            print(f"  [{symbol} {timeframe}m] Loading resampled from cache.")
        df = pd.read_csv(cp, index_col=0, parse_dates=True)
        df.index.name = "date"
        return df

    # Cache missing or stale — build it
    return build_resampled_cache(symbol, timeframe, df_1m,
                                  verbose=True, offset_min=offset_min)


# ---------------------------------------------------------------------------
# Status report
# ---------------------------------------------------------------------------

def cache_status(datasets: Optional[dict] = None) -> None:
    """
    Print a status table for all cached assets/timeframes.
    If datasets dict provided, also shows uncached assets.
    """
    index = _load_index()

    print()
    print(f"  {'Asset':<8} {'TF':>4}  {'Rows':>8}  {'Last timestamp':<22}  {'Status'}")
    print(f"  {'-'*8} {'-'*4}  {'-'*8}  {'-'*22}  {'-'*12}")

    if not index and not datasets:
        print("  No cache entries found. Run cache_build.py to build.")
        return

    shown = set()
    for symbol, sym_data in sorted(index.items()):
        for tf_str, entry in sorted(sym_data.get("timeframes", {}).items(),
                                    key=lambda x: int(x[0])):
            raw_path = None
            if datasets and symbol in datasets:
                raw_path = datasets[symbol].get("full") or datasets[symbol].get("train")

            # Check freshness
            cp = _cache_path(symbol, int(tf_str))
            if not cp.exists():
                status = "✗ file missing"
            elif raw_path and _raw_mtime(raw_path) > entry.get("raw_mtime_at_build", 0):
                status = "⚠ stale"
            else:
                status = "✓ fresh"

            last_ts = entry.get("cache_last_ts", "—")[:19]
            rows    = f"{entry.get('cache_rows', 0):,}"
            print(f"  {symbol:<8} {tf_str+'m':>4}  {rows:>8}  {last_ts:<22}  {status}")
            shown.add((symbol, tf_str))

    # Show datasets with no cache at all
    if datasets:
        for symbol, ds in sorted(datasets.items()):
            if symbol not in index:
                raw_path = ds.get("full") or ds.get("train", "")
                exists   = Path(raw_path).exists() if raw_path else False
                note     = "raw exists" if exists else "no raw file"
                print(f"  {symbol:<8} {'—':>4}  {'—':>8}  {'—':<22}  ✗ not built ({note})")
    print()
