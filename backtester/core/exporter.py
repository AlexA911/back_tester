# Directory refactor
"""
exporter.py — Export run results to CSV.

Auto-saves after every run into results/<run_timestamp>/:
  - trades_<SYMBOL>.csv     one file per asset with full trade log
  - summary.csv             one row per asset with aggregate metrics
  - run_config.txt          full record of every parameter that was run

Trade log columns:
  entry_time, exit_time, direction, entry_price, exit_price,
  raw_pnl, net_pnl, duration_bars, cumulative_pnl
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from backtester.paths import RESULTS_DIR
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------



def _make_run_dir(timestamp: str) -> Path:
    run_dir = RESULTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Trade log builder
# ---------------------------------------------------------------------------

def _build_trade_log(result: dict) -> pd.DataFrame | None:
    """Extract and format the trade log from a single asset result."""
    if "error" in result or "trades" not in result:
        return None

    trades = result["trades"].copy()
    if trades.empty:
        return None

    # Base columns always included
    base_cols = [
        "entry_time", "exit_time", "direction",
        "entry_price", "exit_price", "exit_reason",
        "raw_pnl", "net_pnl", "duration_bars",
    ]

    # Dynamically add per-lot columns if present (multi-lot mode)
    lot_cols = []
    for k in range(1, 4):
        for suffix in ["exit_price", "exit_reason", "raw_pnl"]:
            col = f"lot{k}_{suffix}"
            if col in trades.columns:
                lot_cols.append(col)

    cols = base_cols + lot_cols
    cols = [c for c in cols if c in trades.columns]
    trades = trades[cols].copy()

    # Add cumulative PnL
    trades["cumulative_pnl"] = trades["net_pnl"].cumsum()

    # Round prices and PnL for readability
    round_cols = ["entry_price", "exit_price", "raw_pnl", "net_pnl", "cumulative_pnl"]
    round_cols += [c for c in lot_cols if "pnl" in c or "price" in c]
    for col in round_cols:
        if col in trades.columns:
            trades[col] = trades[col].round(2)

    return trades.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(results: dict, strategy_name: str,
                   split: str, timeframe: str) -> pd.DataFrame:
    """Build one-row-per-asset summary DataFrame."""
    rows = []
    for symbol, result in results.items():
        if "error" in result:
            rows.append({
                "symbol":   symbol,
                "error":    result["error"],
            })
            continue

        rows.append({
            "symbol":          symbol,
            "strategy":        strategy_name,
            "timeframe":       timeframe,
            "split":           split,
            "date_from":       result["date_range"][0][:10],
            "date_to":         result["date_range"][1][:10],
            "bars":            result["bars"],
            "bars_1m":         result.get("bars_1m", result["bars"]),
            "num_trades":      result["num_trades"],
            "total_pnl":       result["total_pnl"],
            "win_rate":        round(result["win_rate"] * 100, 2),
            "profit_factor":   result["profit_factor"],
            "avg_win":         result["avg_win"],
            "avg_loss":        result["avg_loss"],
            "avg_trade":       result["avg_trade"],
            "sharpe":          result["sharpe"],
            "calmar":          result["calmar"],
            "max_drawdown":    result["max_drawdown"],
            "total_commission": result["total_commission"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Config writer
# ---------------------------------------------------------------------------

def _write_config(run_dir: Path, config: dict) -> None:
    """Write a human-readable record of everything that was run."""
    lines = [
        "=" * 60,
        "  BACKTEST RUN CONFIGURATION",
        "=" * 60,
        f"  Saved at     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    for key, val in config.items():
        if isinstance(val, dict):
            lines.append(f"  {key}:")
            for k, v in val.items():
                lines.append(f"    {k:<18} = {v}")
        elif isinstance(val, list):
            lines.append(f"  {key:<20} = {val}")
        else:
            lines.append(f"  {key:<20} = {val}")

    lines += ["", "=" * 60]

    with open(run_dir / "run_config.txt", "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_results(
    results: dict[str, Any],
    config: dict,
    timestamp: str | None = None,
) -> Path:
    """
    Export all results to CSV files under results/<timestamp>/.

    Args:
        results:   Dict of symbol → result dict from run_backtest.
        config:    Dict describing the full run configuration.
        timestamp: Optional override — defaults to current time.

    Returns:
        Path to the run directory.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    strategy_name = config.get("strategy", "unknown")
    split         = config.get("split", "unknown")
    timeframe     = config.get("timeframe", "1m")
    assets        = config.get("assets", [])
    assets_str    = "_".join(sorted(assets)) if assets else "unknown"

    # Folder: YYYY-MM-DD_HH-MM_Strategy_Assets_Split
    # timestamp is "YYYYMMDD_HHMMSS" — reformat for readability
    try:
        from datetime import datetime as _dt
        _ts = _dt.strptime(timestamp, "%Y%m%d_%H%M%S")
        folder_ts = _ts.strftime("%Y-%m-%d_%H-%M")
    except ValueError:
        folder_ts = timestamp
    folder_name = f"{folder_ts}_{strategy_name}_{assets_str}_{split}"
    run_dir     = RESULTS_DIR / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Trade logs ---
    for symbol, result in results.items():
        trade_log = _build_trade_log(result)
        if trade_log is not None:
            path = run_dir / f"trades_{symbol}.csv"
            trade_log.to_csv(path, index=False)

    # --- Summary ---
    summary = _build_summary(results, strategy_name, split, timeframe)
    summary.to_csv(run_dir / "summary.csv", index=False)

    # --- Run config ---
    _write_config(run_dir, config)

    # --- Console confirmation ---
    valid  = sum(1 for v in results.values() if "error" not in v)
    errors = sum(1 for v in results.values() if "error" in v)
    print(f"  Exported to  : {run_dir}")
    print(f"  Trade logs   : {valid} file(s)")
    print(f"  Summary      : summary.csv")
    print(f"  Config       : run_config.txt")
    if errors:
        print(f"  Skipped      : {errors} asset(s) with errors")
    print()

    return run_dir
