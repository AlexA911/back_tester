"""
validator.py — Manual trade validation tool.

Prints the first N trades for a selected asset with:
  - OHLCV bar context around entry and exit
  - Setup conditions at signal bar (flag high/low, range, ATR, filters)
  - PnL arithmetic check
  - Exit reason

Strategy-aware fill check:
  - MomentumIgnition: entry fill = flag_boundary ± 1 tick (not bar open)
  - Legacy strategies: entry fill = bar open
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from datasets import DATASETS
from backtest import load_data
from resampler import resample_ohlcv


# ---------------------------------------------------------------------------
# ATR helper (mirrors momentum_ignition.py)
# ---------------------------------------------------------------------------

def _wilder_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Setup debug info builder
# ---------------------------------------------------------------------------

def _build_setup_debug(
    df_1m: pd.DataFrame,
    signal_time: pd.Timestamp,
    params: dict,
) -> dict | None:
    """
    Recompute setup conditions at the signal bar so we can display them.
    Returns None if strategy is not MomentumIgnition or params missing.
    """
    try:
        fb        = params.get("flag_bars", 8)
        atr_1m_p  = params.get("atr_period_1m", 14)
        atr_15m_p = params.get("atr_period_15m", 14)
        sma_15m   = params.get("atr_sma_period_15m", 50)
        comp_mult = params.get("compression_atr_mult", 1.5)
        tick_size = params.get("tick_size", 0.25)

        atr_1m     = _wilder_atr(df_1m, atr_1m_p)
        flag_high  = df_1m["high"].rolling(fb).max().shift(1)
        flag_low   = df_1m["low"].rolling(fb).min().shift(1)
        flag_range = flag_high - flag_low

        # 15m ATR aligned to 1m (completed bars only)
        df_15m = df_1m.resample("15min", label="right", closed="right").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).dropna(subset=["open", "close"])
        atr_15m         = _wilder_atr(df_15m, atr_15m_p)
        # right/right resampling + merge_asof backward (matches strategy)
        idx_name  = df_1m.index.name or "index"
        df_15m_ts = df_15m.reset_index().rename(columns={df_15m.index.name or "index": "ts"})
        df_15m_ts["atr_15m_val"] = atr_15m.values
        df_15m_ts["sma_15m_val"] = atr_15m.rolling(sma_15m, min_periods=sma_15m).mean().values
        df_1m_ts  = df_1m.reset_index().rename(columns={idx_name: "ts"})
        merged_dbg = pd.merge_asof(
            df_1m_ts[["ts"]].sort_values("ts"),
            df_15m_ts[["ts", "atr_15m_val", "sma_15m_val"]].sort_values("ts"),
            on="ts", direction="backward"
        ).set_index("ts")
        atr_15m_shifted = merged_dbg["atr_15m_val"].reindex(df_1m.index)
        atr_15m_sma     = merged_dbg["sma_15m_val"].reindex(df_1m.index)

        # For resampled timeframes, entry_time may not be in df_1m.
        # Find the last 1m bar at or before entry_time.
        pos = df_1m.index.searchsorted(signal_time, side="right") - 1
        if pos <= 0:
            return None
        # The signal bar is the bar whose close triggered the breakout —
        # one bar before the entry bar in the execution timeframe.
        # In 1m that's idx-1; for higher TFs we step back one more 1m bar.
        sig_idx = pos - 1
        if sig_idx < 0:
            return None
        sig_time = df_1m.index[sig_idx]

        fh    = flag_high.iloc[sig_idx]
        fl    = flag_low.iloc[sig_idx]
        fr    = flag_range.iloc[sig_idx]
        a1m   = atr_1m.iloc[sig_idx]
        a15m     = atr_15m_shifted.iloc[sig_idx]
        a15m_sma = atr_15m_sma.iloc[sig_idx]

        comp_threshold  = comp_mult * a1m
        compression_ok  = fr <= comp_threshold
        regime_ok       = a15m > a15m_sma

        return {
            "signal_bar":       sig_time,
            "flag_high":        round(fh, 4)  if not pd.isna(fh)  else None,
            "flag_low":         round(fl, 4)  if not pd.isna(fl)  else None,
            "flag_range":       round(fr, 4)  if not pd.isna(fr)  else None,
            "atr_1m":           round(a1m, 4) if not pd.isna(a1m) else None,
            "comp_threshold":   round(comp_threshold, 4),
            "compression_ok":   compression_ok,
            "atr_15m_current":  round(a15m, 4)     if not pd.isna(a15m)     else None,
            "atr_15m_sma50":    round(a15m_sma, 4) if not pd.isna(a15m_sma) else None,
            "regime_ok":        regime_ok,
            "tick_size":        tick_size,
            "breakout_long":    round(fh + 2 * tick_size, 4) if fh else None,
            "breakout_short":   round(fl - 2 * tick_size, 4) if fl else None,
        }

    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print()
    print("=" * 75)
    print(f"  {title}")
    print("=" * 75)


def _bar_table(df: pd.DataFrame, label: str, highlight_idx=None) -> None:
    print(f"\n  {label}")
    print(f"  {'Time':<22} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10}  {'Note'}")
    print("  " + "-" * 72)
    for ts, row in df.iterrows():
        note = "◄ FILL" if (highlight_idx is not None and ts == highlight_idx) else ""
        print(
            f"  {str(ts)[:19]:<22} "
            f"{row['open']:>10.4f} "
            f"{row['high']:>10.4f} "
            f"{row['low']:>10.4f} "
            f"{row['close']:>10.4f}  "
            f"{note}"
        )


def _tick(ok: bool) -> str:
    return "✓" if ok else "✗"


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

def validate_trades(
    symbol: str,
    result: dict[str, Any],
    dataset: dict,
    split: str,
    timeframe: str,
    strategy_name: str = "",
    strategy_params: dict | None = None,
    n_trades: int = 10,
    bars_context: int = 3,
) -> None:
    if "error" in result:
        print(f"\n  Cannot validate {symbol}: {result['error']}")
        return

    trades = result.get("trades")
    if trades is None or trades.empty:
        print(f"\n  No trades to validate for {symbol}.")
        return

    path = dataset.get(split)
    if not path or not Path(path).exists():
        print(f"\n  Cannot load data for {symbol}: path not found.")
        return

    df_1m = load_data(path, dataset["date_format"])
    exec_df = resample_ohlcv(df_1m, timeframe) if timeframe != "1m" else df_1m

    tick_size   = dataset["tick_size"]
    tick_value  = dataset["tick_value"]
    point_value = tick_value / tick_size

    is_momentum = "MomentumIgnition" in strategy_name
    params      = strategy_params or {}

    _section(f"TRADE VALIDATION — {symbol}  |  {timeframe}  |  {split}  |  {strategy_name}")
    print(f"  Showing first {n_trades} of {len(trades)} trades")
    print(f"  tick_size={tick_size}  tick_value=${tick_value}  "
          f"point_value=${point_value:.2f}/pt")

    sample = trades.head(n_trades)

    for i, (_, trade) in enumerate(sample.iterrows(), start=1):
        entry_time  = trade["entry_time"]
        exit_time   = trade["exit_time"]
        # direction column may not exist in generic engine trades (uses position column)
        if "direction" in trade.index:
            direction = trade["direction"]
        elif "position" in trade.index:
            direction = "LONG" if trade["position"] > 0 else "SHORT"
        else:
            direction = "LONG" if trade.get("net_pnl", 0) >= 0 else "SHORT"
        entry_px    = trade["entry_price"]
        net_pnl     = trade["net_pnl"]
        raw_pnl     = trade["raw_pnl"]
        dur         = trade["duration_bars"]
        exit_reason = trade.get("exit_reason", "unknown")

        # Detect multi-lot — no single exit_price column
        is_multi_lot = "lot1_exit_price" in trade.index and "exit_price" not in trade.index
        if is_multi_lot:
            # Use last active lot's exit price for display
            exit_px   = trade.get("lot1_exit_price")
            stop_px   = trade.get("stop_price")
            target_px = trade.get("target_price")
        else:
            exit_px   = trade["exit_price"]
            stop_px   = trade.get("stop_price")
            target_px = trade.get("target_price")

        print()
        print(f"  {'─' * 73}")
        print(f"  Trade {i:>3}  |  {direction.upper():<5}  |  "
              f"Entry: {entry_time}  →  Exit: {exit_time}  |  Reason: {exit_reason.upper()}")
        print(f"  {'─' * 73}")

        # --- Setup debug (MomentumIgnition only) ---
        if is_momentum:
            debug = _build_setup_debug(df_1m, entry_time, params)
            if debug and "error" not in debug:
                print(f"\n  Setup conditions at signal bar ({debug['signal_bar']}):")
                print(f"    Flag high        : {debug['flag_high']}")
                print(f"    Flag low         : {debug['flag_low']}")
                print(f"    Flag range       : {debug['flag_range']}  "
                      f"(threshold: {debug['comp_threshold']}  "
                      f"{_tick(debug['compression_ok'])} compression)")
                print(f"    ATR(1m,14)       : {debug['atr_1m']}")
                print(f"    Breakout long    ≥ {debug['breakout_long']}  "
                      f"(flag_high + 2 ticks)")
                print(f"    Breakout short   ≤ {debug['breakout_short']}  "
                      f"(flag_low  - 2 ticks)")
                print(f"    ATR(15m) current : {debug['atr_15m_current']}")
                print(f"    ATR(15m) SMA(50) : {debug['atr_15m_sma50']}  "
                      f"{_tick(debug['regime_ok'])} regime")
                stop_buf = params.get("stop_buf_atr", 0.1)
                stop_side = "low" if direction == "long" else "high"
                print(f"    Stop price       : {stop_px}  "
                      f"(floor/ceil(flag_{stop_side} ± {stop_buf} × ATR(1m)))")
                print(f"    Target price     : {target_px}  "
                      f"(entry ± {params.get('target_points', 3.5)} pts)")
            elif debug and "error" in debug:
                print(f"\n  Setup debug error: {debug['error']}")

        # --- Entry bar context ---
        try:
            entry_loc = exec_df.index.get_loc(entry_time)
            ctx_start = max(0, entry_loc - bars_context)
            ctx_end   = min(len(exec_df), entry_loc + bars_context + 1)
            _bar_table(exec_df.iloc[ctx_start:ctx_end],
                       "Entry context:", highlight_idx=entry_time)
        except KeyError:
            print(f"\n  Entry context: {entry_time} not found in exec_df")

        # --- Exit bar context ---
        try:
            exit_loc  = exec_df.index.get_loc(exit_time)
            ctx_start = max(0, exit_loc - bars_context)
            ctx_end   = min(len(exec_df), exit_loc + bars_context + 1)
            _bar_table(exec_df.iloc[ctx_start:ctx_end],
                       "Exit context:", highlight_idx=exit_time)
        except KeyError:
            print(f"\n  Exit context: {exit_time} not found in exec_df")

        # --- PnL check ---
        sign      = 1 if direction == "long" else -1

        if is_multi_lot:
            # Show per-lot breakdown, skip single-lot calc
            calc_raw  = raw_pnl   # trust the stored value
            pnl_match = True      # can't verify without lot sizes here
        else:
            calc_raw  = (exit_px - entry_px) * sign * point_value
            pnl_match = abs(calc_raw - raw_pnl) < 0.01

        # Fill check — MomentumIgnition fills at flag boundary, not bar open
        if is_momentum and stop_px is not None:
            debug = _build_setup_debug(df_1m, entry_time, params)
            if debug and "error" not in debug:
                expected_fill = (debug["flag_high"] + tick_size
                                 if direction == "long"
                                 else debug["flag_low"] - tick_size)
                fill_match = abs(entry_px - expected_fill) < tick_size * 0.5 \
                             if expected_fill else False
                fill_note  = (f"{'✓' if fill_match else '✗'} vs flag_"
                              f"{'high' if direction == 'long' else 'low'} "
                              f"± 1 tick (expected {expected_fill})")
            else:
                fill_note = "(could not verify)"
        else:
            bar_open   = exec_df.loc[entry_time, "open"] \
                         if entry_time in exec_df.index else None
            if bar_open is not None:
                # Limit order logic: long fills if bar opened at or below limit,
                # short fills if bar opened at or above limit
                is_long_trade = direction == "long"
                if is_long_trade:
                    fill_match = bar_open <= entry_px + 1e-6
                else:
                    fill_match = bar_open >= entry_px - 1e-6
                fill_note = f"{'✓' if fill_match else '✗'} vs bar open ({bar_open})"
            else:
                fill_match = False
                fill_note  = "(could not verify)"

        # Round prices to tick precision for display
        _dp = max(0, -int(np.floor(np.log10(tick_size)))) if tick_size < 1 else 2
        _r  = lambda x: round(float(x), _dp) if x is not None and not pd.isna(x) else x
        print(f"\n  PnL check:")
        print(f"    Entry fill     : {_r(entry_px)}  {fill_note}")
        print(f"    Stop           : {_r(stop_px)}")
        print(f"    Target         : {_r(target_px)}")
        print(f"    Exit fill      : {_r(exit_px)}  ({exit_reason})")
        if exit_px is not None:
            print(f"    Price move     : {exit_px - entry_px:+.4f} pts  ({direction})")

        if is_multi_lot:
            # Show per-lot detail
            k = 1
            while f"lot{k}_exit_price" in trade.index:
                lep = trade.get(f"lot{k}_exit_price")
                ler = trade.get(f"lot{k}_exit_reason", "?")
                lpnl = trade.get(f"lot{k}_raw_pnl", "?")
                print(f"    Lot {k}           : exit={_r(lep)}  reason={ler}  raw_pnl=${lpnl:.2f}")
                k += 1
            print(f"    Raw PnL stored : ${raw_pnl:.2f}  (multi-lot combined)")
        else:
            print(f"    Raw PnL calc   : ({_r(exit_px)} - {_r(entry_px)}) × {sign} × "
                  f"{point_value:.2f} = ${calc_raw:.2f}")
            print(f"    Raw PnL stored : ${raw_pnl:.2f}  {_tick(pnl_match)}")

        print(f"    Commission     : ${trade['commission']:.2f}")
        print(f"    Net PnL        : ${net_pnl:.2f}")
        print(f"    Duration       : {dur} bars")

    print()
    print(f"  {'─' * 73}")
    print(f"  Validation complete — {n_trades} trades checked for {symbol}")
    print()


# ---------------------------------------------------------------------------
# Interactive prompt
# ---------------------------------------------------------------------------

def run_validation(
    results: dict[str, Any],
    split: str,
    timeframe: str,
    strategy_name: str = "",
    strategy_params: dict | None = None,
) -> None:
    valid_assets = [k for k, v in results.items()
                    if "error" not in v
                    and v.get("trades") is not None
                    and not v["trades"].empty]

    if not valid_assets:
        print("  No valid results to validate.")
        return

    print()
    print("=" * 55)
    print("  TRADE VALIDATION")
    print("=" * 55)
    print("  Inspect setup conditions and bar data per trade.")
    print()
    for i, key in enumerate(valid_assets, start=1):
        n = results[key]["num_trades"]
        print(f"  {i:<4} {key:<14} ({n} trades)")
    print()

    choice = input("  Select asset number (or Enter to skip): ").strip()
    if not choice:
        print("  Skipping validation.")
        return

    try:
        asset = valid_assets[int(choice) - 1]
    except (ValueError, IndexError):
        print("  Invalid selection — skipping validation.")
        return

    raw_n   = input("  How many trades to inspect? [10]: ").strip()
    n_trades = int(raw_n) if raw_n.isdigit() else 10

    raw_ctx = input("  Bars of context around each fill? [3]: ").strip()
    bars_context = int(raw_ctx) if raw_ctx.isdigit() else 3

    validate_trades(
        symbol=asset,
        result=results[asset],
        dataset=DATASETS[asset],
        split=split,
        timeframe=timeframe,
        strategy_name=strategy_name,
        strategy_params=strategy_params,
        n_trades=n_trades,
        bars_context=bars_context,
    )
