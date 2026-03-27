"""
grid_search.py — Exhaustive parameter grid search.

Produces two views:
  1. Shared params — best single param set ranked by aggregate profit factor
     across all selected assets combined.
  2. Per-asset best — best params found independently for each asset.

Usage:
    from grid_search import run_grid_search
    results = run_grid_search(
        param_grid={"fast": [5, 10, 20], "slow": [30, 50, 100], "mode": ["both"]},
        strategy_class=EMACrossover,
        assets=["ES", "NQ", "GC"],
        split="train",
        trade_size=1,
        commission=2.0,
        rank_by="profit_factor",
    )
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import time
from functools import partial
from typing import Any, Type

import numpy as np
import pandas as pd

from datasets import DATASETS
from backtest import run_backtest, load_data
from strategy import Strategy


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------

def expand_grid(param_grid: dict[str, list]) -> list[dict]:
    """
    Expand a param grid dict into a list of all combinations.

    Example:
        {"fast": [5, 10], "slow": [30, 50]} →
        [{"fast": 5, "slow": 30}, {"fast": 5, "slow": 50},
         {"fast": 10, "slow": 30}, {"fast": 10, "slow": 50}]
    """
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# ---------------------------------------------------------------------------
# Single combination runner
# ---------------------------------------------------------------------------

def _run_combo(
    params: dict,
    strategy_class: Type[Strategy],
    assets: list[str],
    split: str,
    trade_size: int,
    commission: float,
    cached_data: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """
    Run one param combination across all assets.
    Returns per-asset results and aggregated metrics.
    Routes through run_backtest to support multi-lot mode.
    """
    per_asset = {}

    for key in assets:
        dataset = DATASETS[key]

        # Derive effective trade_size for multi-lot
        multi_lot = params.get("multi_lot", False)
        if multi_lot:
            effective_size = sum(
                int(params.get(f"lot{k}_size", 0))
                for k in range(1, 4)
                if int(params.get(f"lot{k}_size", 0)) > 0
            )
            if effective_size <= 0:
                effective_size = trade_size
        else:
            effective_size = trade_size

        result = run_backtest(
            dataset=dataset,
            strategy=strategy_class(params=params),
            split=split,
            trade_size=effective_size,
            commission_per_side=commission,
        )

        # Flatten to metrics dict (drop heavy objects for grid ranking)
        metrics = {k: v for k, v in result.items()
                   if k not in ("equity_curve", "drawdown_series", "trades")}
        # Keep trades for export of best result
        metrics["trades"]     = result.get("trades")
        metrics["date_range"] = result.get("date_range",
                                (str(""), str("")))
        metrics["bars"]       = result.get("bars", 0)
        metrics["bars_1m"]    = result.get("bars_1m", 0)
        per_asset[key] = metrics

    # Aggregate across all valid assets
    valid = {k: v for k, v in per_asset.items() if "error" not in v and v}
    if not valid:
        return {
            "params":    params,
            "per_asset": per_asset,
            "aggregate": {"error": "No valid results"},
        }

    try:
        all_pnl = pd.concat(
            [v["trades"]["net_pnl"] for v in valid.values() if v.get("trades") is not None],
            ignore_index=True
        )
        total_pnl    = sum(v.get("total_pnl", 0)  for v in valid.values())
        total_trades = sum(v.get("num_trades", 0) for v in valid.values())
        winners      = all_pnl[all_pnl > 0].sum()
        losers       = all_pnl[all_pnl <= 0].sum()
        agg_pf       = (winners / abs(losers)) if losers != 0 else np.inf
        agg_wr       = (all_pnl > 0).sum() / len(all_pnl) if len(all_pnl) else 0
        agg_sharpe   = (
            all_pnl.mean() / all_pnl.std() * np.sqrt(len(all_pnl))
            if all_pnl.std() > 0 else 0
        )
        agg_dd     = sum(v.get("max_drawdown", 0) for v in valid.values())
        agg_calmar = (total_pnl / abs(agg_dd)) if agg_dd != 0 else np.inf
        agg_result = {
            "total_pnl":     round(total_pnl, 2),
            "num_trades":    total_trades,
            "profit_factor": round(float(agg_pf), 4),
            "win_rate":      round(float(agg_wr), 4),
            "sharpe":        round(float(agg_sharpe), 4),
            "max_drawdown":  round(agg_dd, 2),
            "calmar":        round(float(agg_calmar), 4),
            "assets_valid":  len(valid),
            "assets_total":  len(per_asset),
        }
    except Exception as e:
        agg_result = {"error": f"Aggregate failed: {e}"}

    return {
        "params":    params,
        "per_asset": per_asset,
        "aggregate": agg_result,
    }


# ---------------------------------------------------------------------------
# Main grid search
# ---------------------------------------------------------------------------

def run_grid_search(
    param_grid: dict[str, list],
    strategy_class: Type[Strategy],
    assets: list[str],
    split: str = "train",
    trade_size: int = 1,
    commission: float = 2.0,
    rank_by: str = "profit_factor",
    min_trades: int = 30,
) -> dict[str, Any]:
    """
    Run an exhaustive grid search over all parameter combinations.

    Args:
        param_grid:      Dict of param name → list of values to test.
        strategy_class:  The Strategy subclass to instantiate per combo.
        assets:          List of asset keys from DATASETS.
        split:           "train", "test", or "full".
        trade_size:      Contracts per trade.
        commission:      USD per contract per side.
        rank_by:         Metric to rank combinations by (aggregate view).
        min_trades:      Minimum trades required to include a result.

    Returns:
        Dict with:
            "all_results"    — every combination tested
            "shared_ranking" — combinations ranked by aggregate rank_by
            "per_asset_best" — best params per asset independently
            "summary"        — run metadata
    """
    combos = expand_grid(param_grid)
    total  = len(combos)

    print(f"\n  Grid search: {total} combination(s) × {len(assets)} asset(s) "
          f"= {total * len(assets):,} backtests")
    print(f"  Ranking by: {rank_by} | Min trades: {min_trades}")
    print()

    # Pre-load all data once — avoid re-reading CSVs on every combo
    print("  Loading data...", end=" ", flush=True)
    cached_data = {}
    skipped = []
    for key in assets:
        dataset = DATASETS.get(key, {})
        path    = dataset.get(split)
        tick_size  = dataset.get("tick_size")
        tick_value = dataset.get("tick_value")

        if not path or not tick_size or not tick_value:
            skipped.append(key)
            continue
        try:
            cached_data[key] = load_data(path, dataset["date_format"])
        except Exception as e:
            print(f"\n  WARNING: Could not load {key}: {e}")
            skipped.append(key)

    runnable = [a for a in assets if a in cached_data]
    print(f"loaded {len(runnable)} asset(s)."
          + (f" Skipped: {skipped}" if skipped else ""))

    # Run all combinations — use multiprocessing if >1 combo, else single-process
    n_cores = max(1, os.cpu_count() or 1)
    use_mp  = total > 1 and n_cores > 1

    _worker = partial(
        _run_combo,
        strategy_class=strategy_class,
        assets=runnable,
        split=split,
        trade_size=trade_size,
        commission=commission,
        cached_data=cached_data,
    )

    all_results = []
    t0 = time.time()

    if use_mp:
        n_workers = min(n_cores, total)
        print(f"  Using {n_workers} CPU cores (of {n_cores} available)")
        completed = 0
        with mp.Pool(processes=n_workers) as pool:
            for result in pool.imap_unordered(_worker, combos):
                all_results.append(result)
                completed += 1
                elapsed = time.time() - t0
                rate    = completed / elapsed
                eta     = (total - completed) / rate if rate > 0 else 0
                print(f"  [{completed:>4}/{total}]  elapsed={elapsed:.0f}s  eta={eta:.0f}s", end="\r", flush=True)
    else:
        for i, params in enumerate(combos, start=1):
            result = _worker(params)
            all_results.append(result)
            if i % 10 == 0 or i == total:
                elapsed = time.time() - t0
                rate    = i / elapsed
                eta     = (total - i) / rate if rate > 0 else 0
                print(f"  [{i:>4}/{total}]  elapsed={elapsed:.0f}s  eta={eta:.0f}s", end="\r")

    print(f"  [{total}/{total}]  done in {time.time() - t0:.1f}s          ")

    # ------------------------------------------------------------------
    # View 1: Shared ranking — aggregate metric across all assets
    # ------------------------------------------------------------------
    shared = []
    for r in all_results:
        agg = r["aggregate"]
        if "error" in agg:
            continue
        if agg["num_trades"] < min_trades:
            continue
        shared.append({
            **r["params"],
            "total_pnl":     agg["total_pnl"],
            "num_trades":    agg["num_trades"],
            "profit_factor": agg["profit_factor"],
            "win_rate":      agg["win_rate"],
            "sharpe":        agg["sharpe"],
            "max_drawdown":  agg["max_drawdown"],
            "calmar":        agg["calmar"],
            "assets_valid":  agg["assets_valid"],
        })

    shared_df = pd.DataFrame(shared)
    if shared_df.empty:
        # Diagnostic — show what went wrong
        print(f"\n  WARNING: No valid results to rank. Reasons:")
        for r in all_results:
            agg = r["aggregate"]
            params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items()
                                   if k not in ("timeframe",))[:80]
            if "error" in agg:
                print(f"    Error: {agg['error']}  [{params_str}]")
            else:
                n = agg.get("num_trades", 0)
                print(f"    {n} trades (min={min_trades})  [{params_str}]")
        # Return empty results
        shared_df = pd.DataFrame(columns=list(list(all_results[0]["params"].keys()) +
                                              ["total_pnl","num_trades","profit_factor",
                                               "win_rate","sharpe","max_drawdown","calmar","assets_valid"])
                                  if all_results else [])
    else:
        shared_df = (
            shared_df
            .sort_values(rank_by, ascending=False)
            .reset_index(drop=True)
        )
        shared_df.index += 1  # rank from 1

    # ------------------------------------------------------------------
    # View 2: Per-asset best — best params for each asset independently
    # ------------------------------------------------------------------
    per_asset_best = {}
    for key in runnable:
        rows = []
        for r in all_results:
            asset_result = r["per_asset"].get(key, {})
            if "error" in asset_result or not asset_result:
                continue
            if asset_result.get("num_trades", 0) < min_trades:
                continue
            rows.append({
                **r["params"],
                "total_pnl":     asset_result["total_pnl"],
                "num_trades":    asset_result["num_trades"],
                "profit_factor": asset_result["profit_factor"],
                "win_rate":      asset_result["win_rate"],
                "sharpe":        asset_result["sharpe"],
                "max_drawdown":  asset_result["max_drawdown"],
                "calmar":        asset_result["calmar"],
            })

        if rows:
            df = pd.DataFrame(rows).sort_values(rank_by, ascending=False).reset_index(drop=True)
            df.index += 1
            per_asset_best[key] = df

    return {
        "all_results":    all_results,
        "shared_ranking": shared_df,
        "per_asset_best": per_asset_best,
        "summary": {
            "strategy":    strategy_class.__name__,
            "param_grid":  param_grid,
            "combos":      total,
            "assets":      runnable,
            "split":       split,
            "rank_by":     rank_by,
            "min_trades":  min_trades,
            "trade_size":  trade_size,
            "commission":  commission,
        },
    }
