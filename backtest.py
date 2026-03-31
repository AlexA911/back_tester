"""
backtest.py — Backtest orchestrator.

Coordinates: data loading → strategy signal generation → engine simulation → metrics.

Supports two strategy interfaces:
  Legacy:   strategy.generate_signals() returns pd.Series of +1/-1/0
  Extended: strategy.generate_signals() returns dict with
                "signals"     pd.Series
                "exit_params" pd.DataFrame (entry_price, stop_price,
                                             target_price, time_exit_bars)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy import Strategy
from engine import simulate_trades, simulate_trades_multilot, simulate_trades_fgbl, simulate_trades_generic, calculate_metrics
from resampler import resample_ohlcv
from cache_manager import load_cached, load_raw, get_split_date


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str, date_format: str) -> pd.DataFrame:
    """
    Load a CSV into a standardised 1m OHLCV DataFrame.
    Delegates to cache_manager.load_raw for consistency.
    """
    return load_raw(path, date_format)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_backtest(
    dataset: dict,
    strategy: Strategy,
    split: str = "train",
    trade_size: int = 1,
    commission_per_side: float = 2.0,
) -> dict[str, Any]:
    """
    Run a backtest for a single dataset/strategy combination.

    Args:
        dataset:             Entry from DATASETS dict.
        strategy:            Instantiated Strategy object.
        split:               "train", "test", or "full".
        trade_size:          Number of contracts per trade.
        commission_per_side: USD commission per contract per side.

    Returns:
        Dict with metrics, trades DataFrame, and run metadata.
    """
    symbol     = dataset["symbol"]
    tick_size  = dataset.get("tick_size")
    tick_value = dataset.get("tick_value")

    if not tick_size or not tick_value:
        return {"symbol": symbol, "split": split,
                "error": f"Missing tick specs for {symbol}"}

    date_format = dataset["date_format"]
    period      = dataset.get("period")       # (from_date, to_date) or None

    # --- Determine data source ---
    # Prefer full-history cache; fall back to split-specific CSVs
    full_path = dataset.get("full")
    raw_path  = dataset.get(split) or full_path

    if not raw_path:
        return {"symbol": symbol, "split": split,
                "error": f"No data file defined for {symbol}"}
    if not Path(raw_path).exists():
        return {"symbol": symbol, "split": split,
                "error": f"File not found: {raw_path}"}

    # --- Load 1m data ---
    # If full CSV exists: use cache (auto-build on first run, fast thereafter).
    # build_cache writes split_date synchronously, so filter is always correct.
    # If no full CSV: fall back to split-specific CSV (original behaviour).
    if full_path and Path(full_path).exists():
        df_1m = load_cached(
            symbol=symbol,
            timeframe=1,
            raw_path=full_path,
            date_format=date_format,
            auto_build=True,
            verbose=False,
        )
        # Apply filter — split_date guaranteed in index after load_cached returns
        if period is not None:
            from_date, to_date = period
            if from_date:
                df_1m = df_1m[df_1m.index >= pd.Timestamp(from_date)]
            if to_date:
                df_1m = df_1m[df_1m.index <= pd.Timestamp(to_date)]
        elif split in ("train", "test"):
            split_date = dataset.get("split_date") or get_split_date(symbol)
            if split_date:
                if split == "train":
                    df_1m = df_1m[df_1m.index < pd.Timestamp(split_date)]
                else:
                    df_1m = df_1m[df_1m.index >= pd.Timestamp(split_date)]
    else:
        # No full CSV — load split-specific CSV directly (original behaviour)
        if period is not None:
            load_path = raw_path
            df_1m = load_raw(load_path, date_format)
            from_date, to_date = period
            if from_date:
                df_1m = df_1m[df_1m.index >= pd.Timestamp(from_date)]
            if to_date:
                df_1m = df_1m[df_1m.index <= pd.Timestamp(to_date)]
        else:
            df_1m = load_raw(raw_path, date_format)

    if len(df_1m) == 0:
        return {"symbol": symbol, "split": split,
                "error": "No data in selected period"}

    # Inject tick_size into strategy params so it can use it
    # for breakout/entry/stop calculations
    strategy.params["tick_size"] = tick_size

    # Generate signals — handle both legacy and extended interface
    raw = strategy.generate_signals(df_1m)

    if isinstance(raw, dict):
        signals     = raw["signals"]
        exit_params = raw.get("exit_params")
    else:
        # Legacy — plain Series, no stop/target
        signals     = raw
        exit_params = None

    # Determine execution DataFrame
    tf = strategy.timeframe
    # Strategies that manage their own resampling (e.g. FGBLMomentum) always
    # execute on 1m bars — their signals are already mapped to 1m timestamps
    if getattr(strategy, "manages_own_lots", False):
        exec_df = df_1m
    elif strategy.resample_mode == "bars" and tf != "1m":
        exec_df = resample_ohlcv(df_1m, tf)
    else:
        exec_df = df_1m

    # Align signals to execution df
    signals = signals.reindex(exec_df.index).fillna(0).astype(int)
    if exit_params is not None:
        exit_params = exit_params.reindex(exec_df.index)

    # Determine routing — FGBLMomentum manages its own 3-lot structure
    manages_own_lots = getattr(strategy, "manages_own_lots", False)
    multi_lot = strategy.params.get("multi_lot", False)

    # Route to generic engine for strategies that implement get_exit_decision()
    if getattr(strategy, "uses_generic_engine", False):
        if isinstance(raw, dict):
            signals_series = raw.get("signals", pd.Series(0, index=exec_df.index))
            # Strategy may supply its own pre-built execution df (e.g. 5m bars)
            strat_exec_df  = raw.get("exec_df")
            if strat_exec_df is not None:
                exec_df = strat_exec_df
        else:
            signals_series = raw if raw is not None else pd.Series(0, index=exec_df.index)
        n_lots = getattr(strategy, "n_lots", 3)
        lot_configs = [
            {"size": strategy.params.get(f"lot{k+1}_size", 1)}
            for k in range(n_lots)
        ]
        extra_cols = raw.get("extra_cols") if isinstance(raw, dict) else None
        trades = simulate_trades_generic(
            df=exec_df,
            signals=signals_series,
            strategy=strategy,
            tick_size=tick_size,
            tick_value=tick_value,
            commission_per_side=commission_per_side,
            lot_configs=lot_configs,
            extra_cols=extra_cols,
        )
        metrics = calculate_metrics(trades, exec_df)
        metrics["trades"]           = trades
        metrics["bars"]             = len(exec_df)
        metrics["bars_1m"]          = len(df_1m)
        metrics["date_range"]       = (str(exec_df.index[0]), str(exec_df.index[-1]))
        metrics["skipped_entries"]  = 0
        metrics["symbol"]           = symbol
        metrics["split"]            = split
        metrics["total_commission"] = trades["commission"].sum() if not trades.empty else 0
        return metrics

    if manages_own_lots and exit_params is not None:
        from engine import simulate_trades_fgbl
        # Build lot_configs from strategy params so user grid choices take effect
        p = strategy.params
        fgbl_lot_configs = []
        for k in range(1, 4):
            size = int(p.get(f"lot{k}_size", 1 if k <= 3 else 0))
            if size <= 0:
                break
            fgbl_lot_configs.append({
                "size":          size,
                "tp_points":     float(p.get(f"lot{k}_tp_points", 0.0)),
                "stop_mode":     str(p.get(f"lot{k}_stop_mode", "none")),
                "exit_modes":    str(p.get(f"lot{k}_exit_modes", "fixed")),
                "is_runner":     float(p.get(f"lot{k}_tp_points", 0.0)) <= 0,
                "atr_stop_mult": float(p.get(f"lot{k}_atr_stop_mult", 0.0)),
            })
        if not fgbl_lot_configs:
            # Use strategy default_lot_config if available
            dlc = getattr(strategy, "default_lot_config", None)
            dn  = getattr(strategy, "default_n_lots", 3)
            if dlc:
                fgbl_lot_configs = [{"size": 1, **lc} for lc in dlc[:dn]]
        trades = simulate_trades_fgbl(
            df=exec_df,
            signals=signals,
            exit_params=exit_params,
            tick_size=tick_size,
            tick_value=tick_value,
            commission_per_side=commission_per_side,
            lot_configs=fgbl_lot_configs if fgbl_lot_configs else None,
        )
    elif multi_lot and exit_params is not None:
        # Validate trade_size matches sum of lot sizes
        p = strategy.params
        lot_configs = []
        for k in range(1, 4):
            size = int(p.get(f"lot{k}_size", 0))
            if size <= 0:
                continue
            tp = float(p.get(f"lot{k}_tp_points", 0.0))
            lot_configs.append({
                "size":       size,
                "tp_points":  tp,
                "stop_mode":  str(p.get(f"lot{k}_stop_mode", "none")),
                "exit_modes": str(p.get(f"lot{k}_exit_modes", "fixed")),
            })

        if not lot_configs:
            return {"symbol": symbol, "split": split,
                    "error": "multi_lot=True but all lot sizes are 0"}

        total_size = sum(lc["size"] for lc in lot_configs)
        if trade_size != total_size:
            return {"symbol": symbol, "split": split,
                    "error": f"trade_size={trade_size} does not match sum of lot sizes={total_size}"}

        trades = simulate_trades_multilot(
            df=exec_df,
            signals=signals,
            exit_params=exit_params,
            tick_size=tick_size,
            tick_value=tick_value,
            commission_per_side=commission_per_side,
            lot_configs=lot_configs,
            runner_ema_fast=int(p.get("runner_ema_fast", 9)),
            runner_ema_slow=int(p.get("runner_ema_slow", 21)),
            runner_activation_atr=float(p.get("runner_activation_atr", 0.0)),
        )
    else:
        trades = simulate_trades(
            df=exec_df,
            signals=signals,
            exit_params=exit_params,
            tick_size=tick_size,
            tick_value=tick_value,
            trade_size=trade_size,
            commission_per_side=commission_per_side,
        )
    metrics = calculate_metrics(trades, exec_df)
    # Carry through skip count from FGBL engine if available
    if hasattr(trades, "attrs") and "skipped_entries" in trades.attrs:
        metrics["skipped_entries"] = trades.attrs["skipped_entries"]

    return {
        "symbol":     symbol,
        "split":      split,
        "strategy":   strategy.name,
        "timeframe":  tf,
        "params":     strategy.params,
        "bars":       len(exec_df),
        "bars_1m":    len(df_1m),
        "date_range": (str(exec_df.index[0]), str(exec_df.index[-1])),
        **metrics,
    }
