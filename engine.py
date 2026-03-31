"""
engine.py — Core trade simulation engine with intra-bar stop/target checking.

Strategy interface:
    Strategy.generate_signals(df_1m) returns either:
      - A pd.Series of +1/-1/0  (legacy, signal-exit only)
      - A dict with keys:
            "signals"      pd.Series of +1/-1/0
            "exit_params"  pd.DataFrame with columns:
                               entry_price     (float)  strategy-defined fill
                               stop_price      (float)
                               target_price    (float)
                               time_exit_bars  (int)
                               entry_type      (str, optional)
                                   "stop"  — gap fills at bar open (default)
                                             use for breakout/momentum strategies
                                   "limit" — gap skips trade entirely
                                             use for mean reversion strategies

Entry gap handling (exit_params mode only):
    Uses entry_gap_ticks from exit_params (optional, default 0 = no tolerance).

    Long stop order:
        bar_open < stop_price                          → skip (through stop)
        bar_open > intended_entry + gap_ticks×tick     → skip (excess slippage)
        bar_open > intended_entry                      → fill at bar_open
        else                                           → fill at intended_entry

    Short stop order:
        bar_open > stop_price                          → skip (through stop)
        bar_open < intended_entry - gap_ticks×tick     → skip (excess slippage)
        bar_open < intended_entry                      → fill at bar_open
        else                                           → fill at intended_entry

    "limit" entry type skips if bar_open is beyond intended_entry (any amount).

Exit modes:
    With exit_params:  stop → target → time only. Signal is entry-only.
                       No signal flip exit. Trade held until one of the
                       three explicit exits fires.
    Without exit_params (legacy): signal flip exits still apply.

Exit priority (exit_params mode):
    1. Stop breach   — low <= stop (long) or high >= stop (short)
    2. Target breach — high >= target (long) or low <= target (short)
    3. Time limit    — bars held >= time_exit_bars → exit at bar open

If stop AND target both breached on same bar, stop takes priority.

Exit reasons: "stop", "target", "time", "end_of_data"
Legacy only:  "signal"
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def _ceil_tick(price: float, tick_size: float) -> float:
    """Round price UP to nearest tick."""
    return round(np.ceil(round(price / tick_size, 8)) * tick_size, 10)


def _floor_tick(price: float, tick_size: float) -> float:
    """Round price DOWN to nearest tick."""
    return round(np.floor(round(price / tick_size, 8)) * tick_size, 10)


def simulate_trades(
    df: pd.DataFrame,
    signals: pd.Series,
    exit_params: pd.DataFrame | None,
    tick_size: float,
    tick_value: float,
    trade_size: int,
    commission_per_side: float,
) -> pd.DataFrame:
    """
    Bar-by-bar trade simulation.

    With exit_params:    stop/target/time exits only — no signal flip.
    Without exit_params: legacy signal-flip exit behaviour.
    """
    point_value     = tick_value / tick_size
    has_exit_params = exit_params is not None

    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    times  = df.index
    sig    = signals.values

    if has_exit_params:
        stops            = exit_params["stop_price"].values
        targets          = exit_params["target_price"].values
        time_limits      = exit_params["time_exit_bars"].values
        entry_prices_arr = exit_params["entry_price"].values if "entry_price" in exit_params.columns else None
        entry_types_arr  = exit_params["entry_type"].values if "entry_type" in exit_params.columns else None
        entry_gap_arr      = exit_params["entry_gap_ticks"].values if "entry_gap_ticks" in exit_params.columns else None
    else:
        stops = targets = time_limits = None
        entry_prices_arr = entry_types_arr = entry_gap_arr = None


    trades = []

    position     = 0
    entry_price  = None
    entry_time   = None
    entry_bar    = None
    stop_price   = None
    target_price = None
    time_limit   = None

    def _reset():
        nonlocal position, entry_price, entry_time, entry_bar
        nonlocal stop_price, target_price, time_limit
        position     = 0
        entry_price  = None
        entry_time   = None
        entry_bar    = None
        stop_price   = None
        target_price = None
        time_limit   = None

    def _close_trade(exit_px, exit_time, exit_bar, reason):
        raw_pnl    = (exit_px - entry_price) * position * point_value * trade_size
        commission = commission_per_side * 2 * trade_size
        trades.append({
            "entry_time":    entry_time,
            "exit_time":     exit_time,
            "direction":     "long" if position == 1 else "short",
            "entry_price":   entry_price,
            "exit_price":    exit_px,
            "stop_price":    stop_price,
            "target_price":  target_price,
            "raw_pnl":       raw_pnl,
            "commission":    commission,
            "net_pnl":       raw_pnl - commission,
            "duration_bars": exit_bar - entry_bar,
            "exit_reason":   reason,
        })

    for i in range(1, len(df)):
        bar_time = times[i]
        bar_open = opens[i]
        bar_high = highs[i]
        bar_low  = lows[i]

        # ------------------------------------------------------------
        # IN A TRADE
        # ------------------------------------------------------------
        if position != 0:

            if has_exit_params:
                # --- Stop/target/time exits only ---
                stop_hit   = False
                target_hit = False

                if stop_price is not None:
                    stop_hit   = (bar_low  <= stop_price) if position ==  1 \
                                 else (bar_high >= stop_price)
                if target_price is not None:
                    target_hit = (bar_high >= target_price) if position ==  1 \
                                 else (bar_low  <= target_price)

                if stop_hit:
                    _close_trade(stop_price, bar_time, i, "stop")
                    _reset()
                    continue  # no new entry same bar after stop

                if target_hit:
                    _close_trade(target_price, bar_time, i, "target")
                    _reset()
                    continue  # no new entry same bar after target

                bars_held = i - entry_bar
                if time_limit is not None and bars_held >= time_limit:
                    _close_trade(bar_open, bar_time, i, "time")
                    _reset()
                    # Allow new entry on same bar after time exit
                    # (time exit is orderly, not a forced flush)

            else:
                # --- Legacy: signal flip exit ---
                prev_signal = sig[i - 1]
                if prev_signal != position:
                    _close_trade(bar_open, bar_time, i, "signal")
                    _reset()
                    # Fall through to check for new entry

        # ------------------------------------------------------------
        # FLAT — check for new entry
        # ------------------------------------------------------------
        if position == 0:
            prev_signal = sig[i - 1]
            if prev_signal != 0:

                if has_exit_params:
                    ep           = exit_params.iloc[i - 1]
                    intended_entry = float(entry_prices_arr[i - 1])                                      if entry_prices_arr is not None                                      else bar_open
                    _stop_price   = float(ep["stop_price"])
                    _target_price = float(ep["target_price"])
                    _time_limit   = int(ep["time_exit_bars"])
                    _entry_type   = str(entry_types_arr[i - 1])                                     if entry_types_arr is not None                                     else "stop"

                    # --- Gap handling ---
                    _gap_ticks = float(entry_gap_arr[i - 1])                                  if entry_gap_arr is not None else 0.0
                    _gap_limit = _gap_ticks * tick_size

                    if prev_signal == 1:  # long
                        if bar_open < _stop_price:
                            # Gapped through stop — skip
                            pass
                        elif _entry_type == "limit" and bar_open > intended_entry:
                            # Limit order — skip if already above
                            pass
                        elif bar_open > intended_entry + _gap_limit:
                            # Gapped beyond tolerance — skip
                            pass
                        else:
                            # Fill at bar_open if gapped within tolerance, else intended
                            actual_entry = bar_open if bar_open > intended_entry else intended_entry
                            position     = prev_signal
                            entry_bar    = i
                            entry_time   = bar_time
                            entry_price  = actual_entry
                            stop_price   = _stop_price
                            target_price = _target_price
                            time_limit   = _time_limit

                    else:  # short
                        if bar_open > _stop_price:
                            # Gapped through stop — skip
                            pass
                        elif _entry_type == "limit" and bar_open < intended_entry:
                            # Limit order — skip if already below
                            pass
                        elif bar_open < intended_entry - _gap_limit:
                            # Gapped beyond tolerance — skip
                            pass
                        else:
                            # Fill at bar_open if gapped within tolerance, else intended
                            actual_entry = bar_open if bar_open < intended_entry else intended_entry
                            position     = prev_signal
                            entry_bar    = i
                            entry_time   = bar_time
                            entry_price  = actual_entry
                            stop_price   = _stop_price
                            target_price = _target_price
                            time_limit   = _time_limit

                else:
                    # Legacy — always fill at bar open
                    position     = prev_signal
                    entry_bar    = i
                    entry_time   = bar_time
                    entry_price  = bar_open
                    stop_price   = None
                    target_price = None
                    time_limit   = None

    # ------------------------------------------------------------
    # Close open position at end of data
    # ------------------------------------------------------------
    if position != 0 and entry_price is not None:
        last_px    = df["close"].iloc[-1]
        raw_pnl    = (last_px - entry_price) * position * point_value * trade_size
        commission = commission_per_side * 2 * trade_size
        trades.append({
            "entry_time":    entry_time,
            "exit_time":     df.index[-1],
            "direction":     "long" if position == 1 else "short",
            "entry_price":   entry_price,
            "exit_price":    last_px,
            "stop_price":    stop_price,
            "target_price":  target_price,
            "raw_pnl":       raw_pnl,
            "commission":    commission,
            "net_pnl":       raw_pnl - commission,
            "duration_bars": len(df) - entry_bar,
            "exit_reason":   "end_of_data",
        })

    return pd.DataFrame(trades)



# ---------------------------------------------------------------------------
# Multi-lot simulation
# ---------------------------------------------------------------------------

def simulate_trades_multilot(
    df: pd.DataFrame,
    signals: pd.Series,
    exit_params: pd.DataFrame,
    tick_size: float,
    tick_value: float,
    commission_per_side: float,
    lot_configs: list[dict],   # list of {size, tp_points, stop_mode, exit_modes,
                               #           activation_pts, activation_atr_mult}
    runner_ema_fast: int,
    runner_ema_slow: int,
    runner_activation_atr: float = 0.0,
) -> pd.DataFrame:
    """
    Multi-lot bar-by-bar simulation.

    Each lot in lot_configs is managed independently after entry:
      - Lot with tp_points > 0: fixed target exit
      - Lot with tp_points = 0: runner — exits on first of exit_modes triggers

    Runner activation threshold (applies to all runner lots):
      Before activation: stop stays at original level, no runner exit modes checked.
      After activation:  stop moves per stop_mode, runner exit modes become active.

      Activation triggered when open profit >= threshold:
        runner_activation_atr  > 0  → N × ATR(1m) of open profit required
        0                          → activates immediately (original behaviour)

    stop_mode (per runner lot, applied on activation):
      "breakeven"     — move stop to entry
      "entry_plus_1"  — move stop to entry + 1 tick
      "trail_tp1"     — move stop to TP1 price (if TP1 was hit)
      "none"          — stop stays at original level

    exit_modes (comma-separated string, first trigger wins):
      "ema_flip"   — fast EMA crosses below/above slow EMA
      "time_exit"  — bars held >= time_exit_bars
      "fixed"      — use tp_points as fixed target

    Runner EMAs are computed once upfront on the execution df.
    """
    point_value = tick_value / tick_size

    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    times  = df.index
    sig    = signals.values

    # Pre-compute runner EMAs
    close_s = df["close"]
    ema_fast = close_s.ewm(span=runner_ema_fast, adjust=False).mean().values
    ema_slow = close_s.ewm(span=runner_ema_slow, adjust=False).mean().values

    # Pre-compute ATR(1m) for activation threshold
    _h = df["high"]; _l = df["low"]; _c = df["close"]
    _pc = _c.shift(1)
    _tr = pd.concat([_h - _l, (_h - _pc).abs(), (_l - _pc).abs()], axis=1).max(axis=1)
    atr_1m = _tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values

    stops            = exit_params["stop_price"].values
    targets          = exit_params["target_price"].values
    time_limits      = exit_params["time_exit_bars"].values
    entry_prices_arr = exit_params["entry_price"].values
    entry_gap_arr    = exit_params["entry_gap_ticks"].values if "entry_gap_ticks" in exit_params.columns else None

    trades = []

    # State — one entry shared, per-lot exit tracking
    position    = 0
    entry_price = None
    entry_time  = None
    entry_bar   = None
    stop_price  = None    # original stop — shared across lots
    time_limit  = None
    tp1_price   = None    # price at which TP1 was hit (for trail_tp1 stop mode)

    # Per-lot state: target, active flag, stop (may differ after TP1)
    n_lots = len(lot_configs)
    lot_active      = [False] * n_lots
    lot_stop        = [None]  * n_lots   # per-lot stop (may be moved)
    lot_target      = [None]  * n_lots
    lot_exit_price  = [None]  * n_lots
    lot_exit_reason = [None]  * n_lots
    lot_activated   = [False] * n_lots   # runner activation state per lot

    # Track whether TP1 (first fixed lot) has been hit
    tp1_hit = False

    def _reset():
        nonlocal position, entry_price, entry_time, entry_bar
        nonlocal stop_price, time_limit, tp1_hit, tp1_price
        position    = 0
        entry_price = None
        entry_time  = None
        entry_bar   = None
        stop_price  = None
        time_limit  = None
        tp1_hit     = False
        tp1_price   = None
        for k in range(n_lots):
            lot_active[k]      = False
            lot_stop[k]        = None
            lot_target[k]      = None
            lot_exit_price[k]  = None
            lot_exit_reason[k] = None
            lot_activated[k]   = False

    def _flush_trade(exit_bar, exit_time):
        """Record a trade row when all lots are done (or forced close)."""
        total_raw = 0.0
        total_comm = 0.0
        row = {
            "entry_time":  entry_time,
            "exit_time":   exit_time,
            "direction":   "long" if position == 1 else "short",
            "entry_price": entry_price,
            "duration_bars": exit_bar - entry_bar,
        }
        for k, lc in enumerate(lot_configs):
            size = lc["size"]
            ep   = lot_exit_price[k] if lot_exit_price[k] is not None else (df["close"].iloc[exit_bar] if exit_bar < len(df) else df["close"].iloc[-1])
            reason = lot_exit_reason[k] if lot_exit_reason[k] else "end_of_data"
            raw  = (ep - entry_price) * position * point_value * size
            comm = commission_per_side * 2 * size
            total_raw  += raw
            total_comm += comm
            row[f"lot{k+1}_size"]        = size
            row[f"lot{k+1}_exit_price"]  = round(ep, 6)
            row[f"lot{k+1}_exit_reason"] = reason
            row[f"lot{k+1}_raw_pnl"]     = round(raw, 2)
        row["raw_pnl"]   = round(total_raw, 2)
        row["commission"] = round(total_comm, 2)
        row["net_pnl"]   = round(total_raw - total_comm, 2)
        row["exit_reason"] = lot_exit_reason[0] if lot_exit_reason[0] else "end_of_data"
        trades.append(row)

    for i in range(1, len(df)):
        bar_time = times[i]
        bar_open = opens[i]
        bar_high = highs[i]
        bar_low  = lows[i]

        # ----------------------------------------------------------------
        # IN A TRADE — process each active lot
        # ----------------------------------------------------------------
        if position != 0:
            # Current open profit (use bar high/low best case for activation check)
            best_price   = bar_high if position == 1 else bar_low
            open_profit  = (best_price - entry_price) * position

            # Compute activation threshold for this bar
            _act_threshold = runner_activation_atr * atr_1m[i] if runner_activation_atr > 0 else 0.0

            # Check if first fixed lot (TP1) gets hit this bar
            # TP1 = first lot with tp_points > 0 that's still active
            tp1_idx = next((k for k, lc in enumerate(lot_configs)
                           if lc.get("tp_points", 0) > 0 and lot_active[k]), None)

            if not tp1_hit and tp1_idx is not None:
                t1 = lot_target[tp1_idx]
                if t1 is not None:
                    tp1_triggered = (bar_high >= t1) if position == 1 else (bar_low <= t1)
                    if tp1_triggered:
                        tp1_price = t1
                        tp1_hit = True
                        lot_exit_price[tp1_idx]  = t1
                        lot_exit_reason[tp1_idx] = "target"
                        lot_active[tp1_idx]      = False

            # Check activation for each runner lot and apply stop_mode on activation
            for k, lc in enumerate(lot_configs):
                if not lot_active[k] or lot_activated[k]:
                    continue
                is_runner = lc.get("tp_points", 0) <= 0
                if not is_runner:
                    lot_activated[k] = True  # fixed lots always "activated"
                    continue
                # Activate if open profit >= threshold OR if TP1 already hit
                if open_profit >= _act_threshold or tp1_hit:
                    lot_activated[k] = True
                    mode = lc.get("stop_mode", "breakeven")
                    if mode == "breakeven":
                        lot_stop[k] = entry_price
                    elif mode == "entry_plus_1":
                        lot_stop[k] = (entry_price + tick_size) if position == 1                                       else (entry_price - tick_size)
                    elif mode == "trail_tp1":
                        # Move stop to TP1 price if TP1 was hit, else breakeven
                        lot_stop[k] = tp1_price if tp1_price is not None else entry_price
                    # "none" — leave stop unchanged

            # Process remaining active lots
            for k, lc in enumerate(lot_configs):
                if not lot_active[k]:
                    continue

                is_runner = lc.get("tp_points", 0) <= 0
                modes = [m.strip() for m in lc.get("exit_modes", "fixed").split(",")]
                exited = False

                # Check stop first (highest priority — always active)
                ls = lot_stop[k]
                if ls is not None:
                    stop_hit = (bar_low <= ls) if position == 1 else (bar_high >= ls)
                    if stop_hit:
                        lot_exit_price[k]  = ls
                        lot_exit_reason[k] = "stop"
                        lot_active[k]      = False
                        exited = True

                if not exited:
                    for mode in modes:
                        if mode == "fixed":
                            t = lot_target[k]
                            if t is not None:
                                hit = (bar_high >= t) if position == 1 else (bar_low <= t)
                                if hit:
                                    lot_exit_price[k]  = t
                                    lot_exit_reason[k] = "target"
                                    lot_active[k]      = False
                                    exited = True
                                    break

                        elif mode == "ema_flip":
                            # Only check if runner is activated
                            if not lot_activated[k]:
                                continue
                            if i > 0:
                                if position == 1 and ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]:
                                    lot_exit_price[k]  = bar_open
                                    lot_exit_reason[k] = "ema_flip"
                                    lot_active[k]      = False
                                    exited = True
                                    break
                                elif position == -1 and ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]:
                                    lot_exit_price[k]  = bar_open
                                    lot_exit_reason[k] = "ema_flip"
                                    lot_active[k]      = False
                                    exited = True
                                    break

                        elif mode == "time_exit":
                            # Only check if runner is activated
                            if not lot_activated[k]:
                                continue
                            bars_held = i - entry_bar
                            if time_limit is not None and bars_held >= time_limit:
                                lot_exit_price[k]  = bar_open
                                lot_exit_reason[k] = "time"
                                lot_active[k]      = False
                                exited = True
                                break

            # Check if all lots now done
            all_done = not any(lot_active)
            if all_done:
                _flush_trade(i, bar_time)
                _reset()
                continue

        # ----------------------------------------------------------------
        # FLAT — check for new entry
        # ----------------------------------------------------------------
        if position == 0:
            prev_signal = sig[i - 1]
            if prev_signal != 0:
                ep            = exit_params.iloc[i - 1]
                intended_entry = float(entry_prices_arr[i - 1])
                _stop_price   = float(ep["stop_price"])
                _time_limit   = int(ep["time_exit_bars"])

                _gap_ticks = float(entry_gap_arr[i - 1]) if entry_gap_arr is not None else 0.0
                _gap_limit = _gap_ticks * tick_size

                # Gap fill logic
                actual_entry = None
                if prev_signal == 1:
                    if bar_open < _stop_price:
                        pass
                    elif bar_open > intended_entry + _gap_limit:
                        pass
                    else:
                        actual_entry = bar_open if bar_open > intended_entry else intended_entry
                else:
                    if bar_open > _stop_price:
                        pass
                    elif bar_open < intended_entry - _gap_limit:
                        pass
                    else:
                        actual_entry = bar_open if bar_open < intended_entry else intended_entry

                if actual_entry is not None:
                    position    = prev_signal
                    entry_bar   = i
                    entry_time  = bar_time
                    entry_price = actual_entry
                    stop_price  = _stop_price
                    time_limit  = _time_limit
                    tp1_hit     = False

                    # Initialise per-lot state
                    for k, lc in enumerate(lot_configs):
                        lot_active[k]      = True
                        lot_stop[k]        = _stop_price
                        lot_exit_price[k]  = None
                        lot_exit_reason[k] = None

                        tp = lc.get("tp_points", 0)
                        if tp and tp > 0:
                            if prev_signal == 1:
                                lot_target[k] = _ceil_tick(actual_entry + tp, tick_size)
                            else:
                                lot_target[k] = _floor_tick(actual_entry - tp, tick_size)
                        else:
                            lot_target[k] = None  # runner — no fixed target

    # ----------------------------------------------------------------
    # Force-close any open lots at end of data
    # ----------------------------------------------------------------
    if position != 0:
        last_close = df["close"].iloc[-1]
        last_time  = df.index[-1]
        for k in range(n_lots):
            if lot_active[k]:
                lot_exit_price[k]  = last_close
                lot_exit_reason[k] = "end_of_data"
                lot_active[k]      = False
        _flush_trade(len(df) - 1, last_time)

    return pd.DataFrame(trades)



def simulate_trades_fgbl(
    df: pd.DataFrame,
    signals: pd.Series,
    exit_params: pd.DataFrame,
    tick_size: float,
    tick_value: float,
    commission_per_side: float,
    lot_configs: list[dict] | None = None,
) -> pd.DataFrame:
    """
    3-lot simulation for FGBLMomentum strategy.

    Lot structure:
      Base lot:   fixed ATR target, EMA-21 stop (live after stop_live_bars), base_timeout
      Runner 1:   EMA-21 stop pre-BaseTP, then BE/EMA trail post-BaseTP, r1_timeout
      Runner 2:   EMA-21 stop until hook (hook_atr_mult × ATR), then chandelier trail, r2_timeout

    All stops use close-based triggers (not intra-bar high/low) matching original logic.
    Gap handling: if bar opens through stop, fill at bar open.
    """
    point_value = tick_value / tick_size

    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    times  = df.index
    sig    = signals.values
    n      = len(df)

    # Extract exit_params arrays
    ep_entry     = exit_params["entry_price"].values
    ep_stop      = exit_params["stop_price"].values       # EMA-21 at entry
    ep_target    = exit_params["target_price"].values     # base lot target
    ep_time      = exit_params["time_exit_bars"].values   # base timeout
    ep_gap       = exit_params["entry_gap_ticks"].values if "entry_gap_ticks" in exit_params.columns else None
    ep_stop_live = exit_params["stop_live_bars"].values   if "stop_live_bars" in exit_params.columns else None
    ep_r1_timeout= exit_params["r1_timeout_bars"].values  if "r1_timeout_bars" in exit_params.columns else None
    ep_r2_timeout= exit_params["r2_timeout_bars"].values  if "r2_timeout_bars" in exit_params.columns else None
    ep_hook_atr  = exit_params["hook_atr_mult"].values    if "hook_atr_mult" in exit_params.columns else None
    ep_chand_atr = exit_params["chandelier_atr_mult"].values if "chandelier_atr_mult" in exit_params.columns else None
    ep_atr       = exit_params["atr_at_entry"].values     if "atr_at_entry" in exit_params.columns else None
    ep_ema_mid   = exit_params["ema_mid_1m"].values       if "ema_mid_1m" in exit_params.columns else None

    trades = []
    skipped_entries = [0]  # mutable counter for gap-skip tracking

    # Position state
    position     = 0
    entry_bar    = None
    entry_time   = None
    entry_price  = None
    trade_atr    = None   # ATR at entry (used for targets/trails)
    trade_ema    = None   # EMA-21 at entry
    stop_live_i  = None
    base_timeout_i = None
    r1_timeout_i = None
    r2_timeout_i = None
    hook_level   = None   # runner2 activation price
    chandelier_atr_mult = None

    # Per-lot exit state
    base_exit    = None   # (exit_time, exit_price, reason)
    r1_exit      = None
    r2_exit      = None

    # Runner state flags
    base_tp_hit  = False
    base_tp_i    = None
    r2_hook_active = False
    r2_best      = None   # best price seen since hook

    def _reset():
        nonlocal position, entry_bar, entry_time, entry_price
        nonlocal trade_atr, trade_ema, stop_live_i
        nonlocal base_timeout_i, r1_timeout_i, r2_timeout_i
        nonlocal hook_level, chandelier_atr_mult
        nonlocal base_exit, r1_exit, r2_exit
        nonlocal base_tp_hit, base_tp_i, r2_hook_active, r2_best
        nonlocal _l1_hard_stop, _l2_hard_stop, _l3_hard_stop
        position = 0; entry_bar = None; entry_time = None; entry_price = None
        trade_atr = None; trade_ema = None; stop_live_i = None
        base_timeout_i = None; r1_timeout_i = None; r2_timeout_i = None
        hook_level = None; chandelier_atr_mult = None
        base_exit = None; r1_exit = None; r2_exit = None
        base_tp_hit = False; base_tp_i = None
        r2_hook_active = False; r2_best = None
        _l1_hard_stop = None; _l2_hard_stop = None; _l3_hard_stop = None

    def _flush():
        """Record completed trade."""
        total_raw = (base_exit[1] + r1_exit[1] + r2_exit[1] - 3 * entry_price) * position * point_value
        commission = commission_per_side * 2 * 3  # 3 lots
        # Duration: use stored bar indices (fast, no get_loc needed)
        _exit_bar = max(base_exit[3], r1_exit[3], r2_exit[3])
        trades.append({
            "entry_time":          entry_time,
            "exit_time":           max(base_exit[0], r1_exit[0], r2_exit[0]),
            "direction":           "long" if position == 1 else "short",
            "entry_price":         entry_price,
            "lot1_exit_price":     base_exit[1],
            "lot1_exit_reason":    base_exit[2],
            "lot1_raw_pnl":        round((base_exit[1] - entry_price) * position * point_value, 2),
            "lot2_exit_price":     r1_exit[1],
            "lot2_exit_reason":    r1_exit[2],
            "lot2_raw_pnl":        round((r1_exit[1] - entry_price) * position * point_value, 2),
            "lot3_exit_price":     r2_exit[1],
            "lot3_exit_reason":    r2_exit[2],
            "lot3_raw_pnl":        round((r2_exit[1] - entry_price) * position * point_value, 2),
            "raw_pnl":             round(total_raw, 2),
            "commission":          round(commission, 2),
            "net_pnl":             round(total_raw - commission, 2),
            "duration_bars":       _exit_bar - entry_bar,
            "exit_reason":         base_exit[2],
        })

    # Maximum bars any trade can stay open — hard safety guard
    _r1_to = int(exit_params["r1_timeout_bars"].max()) if "r1_timeout_bars" in exit_params.columns else 3840
    _r2_to = int(exit_params["r2_timeout_bars"].max()) if "r2_timeout_bars" in exit_params.columns else 2400
    _bt_to = int(exit_params["time_exit_bars"].max())
    MAX_TRADE_BARS = max(_r1_to, _r2_to, _bt_to) + 240

    for i in range(1, n):
        bar_time = times[i]
        bar_open = opens[i]
        bar_high = highs[i]
        bar_low  = lows[i]
        bar_close = closes[i]

        # Safety guard — force close if trade exceeds max duration
        if position != 0 and entry_bar is not None and (i - entry_bar) > MAX_TRADE_BARS:
            if base_exit is None: base_exit = (bar_time, bar_close, "ForceClose", i)
            if r1_exit   is None: r1_exit   = (bar_time, bar_close, "ForceClose", i)
            if r2_exit   is None: r2_exit   = (bar_time, bar_close, "ForceClose", i)

        # Current EMA-21 for this bar (dynamic, forward-filled from signal bars)
        ema_stop = float(ep_ema_mid[i]) if ep_ema_mid is not None and not np.isnan(ep_ema_mid[i]) \
                   else (trade_ema if trade_ema is not None else np.nan)

        # ----------------------------------------------------------------
        # IN A TRADE
        # ----------------------------------------------------------------
        if position != 0:
            stop_active = (i >= stop_live_i)
            is_long     = (position == 1)

            # Effective stop = tighter of EMA-21 and hard ATR stop (if set and stop active)
            def _eff_stop(ema_s, hard_s, long):
                if hard_s is None or not stop_active:
                    return ema_s
                return max(hard_s, ema_s) if long else min(hard_s, ema_s)

            eff_stop_l1 = _eff_stop(ema_stop, _l1_hard_stop, is_long)
            eff_stop_l2 = _eff_stop(ema_stop, _l2_hard_stop, is_long)
            eff_stop_l3 = _eff_stop(ema_stop, _l3_hard_stop, is_long)

            # ---- Base lot ----
            if base_exit is None:
                _es1 = eff_stop_l1
                base_tgt = _base_tgt_override if _base_tgt_override is not None \
                           else (float(ep_target[entry_bar - 1]) if entry_bar > 0 else np.nan)

                stop_hit = stop_active and (
                    (is_long  and bar_close <= _es1) or
                    (not is_long and bar_close >= _es1)
                )
                tp_hit = (
                    (is_long  and bar_high >= base_tgt) or
                    (not is_long and bar_low  <= base_tgt)
                )
                gap_stop = stop_active and (
                    (is_long  and bar_open < _es1) or
                    (not is_long and bar_open > _es1)
                )

                if stop_hit and tp_hit:
                    # Ambiguous — stop-first bias
                    exit_px = bar_open if gap_stop else _es1
                    base_exit = (bar_time, exit_px, "StopAmbiguous", i)
                elif stop_hit:
                    exit_px = bar_open if gap_stop else _es1
                    base_exit = (bar_time, exit_px, "Stop", i)
                elif tp_hit:
                    base_exit = (bar_time, base_tgt, "BaseTP", i)
                    base_tp_hit = True
                    base_tp_i   = i
                elif i >= base_timeout_i:
                    base_exit = (bar_time, bar_close, "BaseTimeout", i)

            # ---- Runner 1 ----
            if r1_exit is None:
                _es2 = eff_stop_l2
                if not base_tp_hit:
                    # Pre-BaseTP: same EMA stop logic
                    stop_active_r1 = (i >= stop_live_i)
                    stop_hit = stop_active_r1 and (
                        (is_long  and bar_close <= _es2) or
                        (not is_long and bar_close >= _es2)
                    )
                    gap = stop_active_r1 and (
                        (is_long  and bar_open < _es2) or
                        (not is_long and bar_open > _es2)
                    )
                    if stop_hit:
                        exit_px = bar_open if gap else ema_stop
                        r1_exit = (bar_time, exit_px, "Runner1PreTPStop", i)
                    elif i >= r1_timeout_i:
                        r1_exit = (bar_time, bar_close, "Runner1Timeout", i)
                else:
                    # Post-BaseTP (activate from next bar)
                    if i > base_tp_i:
                        if is_long:
                            be_hit  = bar_low  <= entry_price
                            ema_hit = bar_close <= _es2
                            gap_be  = bar_open  < entry_price
                            gap_ema = bar_open  < _es2
                            if be_hit and ema_hit:
                                stop_level = max(entry_price, _es2)
                                gap = bar_open < stop_level
                                exit_px = bar_open if gap else stop_level
                                r1_exit = (bar_time, exit_px, "Runner1Stop", i)
                            elif be_hit:
                                r1_exit = (bar_time, bar_open if gap_be else entry_price, "Runner1Stop", i)
                            elif ema_hit:
                                r1_exit = (bar_time, bar_open if gap_ema else _es2, "Runner1Stop", i)
                            elif i >= r1_timeout_i:
                                r1_exit = (bar_time, bar_close, "Runner1Timeout", i)
                        else:
                            be_hit  = bar_high >= entry_price
                            ema_hit = bar_close >= _es2
                            gap_be  = bar_open  > entry_price
                            gap_ema = bar_open  > _es2
                            if be_hit and ema_hit:
                                stop_level = min(entry_price, _es2)
                                gap = bar_open > stop_level
                                exit_px = bar_open if gap else stop_level
                                r1_exit = (bar_time, exit_px, "Runner1Stop", i)
                            elif be_hit:
                                r1_exit = (bar_time, bar_open if gap_be else entry_price, "Runner1Stop", i)
                            elif ema_hit:
                                r1_exit = (bar_time, bar_open if gap_ema else _es2, "Runner1Stop", i)
                            elif i >= r1_timeout_i:
                                r1_exit = (bar_time, bar_close, "Runner1Timeout", i)

            # ---- Runner 2 (chandelier) ----
            if r2_exit is None:
                _es3 = eff_stop_l3
                if not r2_hook_active:
                    # Check if hook level reached (only relevant for chandelier mode)
                    _use_chandelier = "chandelier" in _r2_exit_modes
                    if _use_chandelier and (
                        (is_long  and bar_high >= hook_level) or
                        (not is_long and bar_low  <= hook_level)
                    ):
                        r2_hook_active = True
                        r2_best = bar_high if is_long else bar_low
                    else:
                        # Pre-hook or non-chandelier: EMA stop (with ATR hard stop)
                        stop_active_r2 = (i >= stop_live_i)
                        stop_hit = stop_active_r2 and (
                            (is_long  and bar_close <= _es3) or
                            (not is_long and bar_close >= _es3)
                        )
                        gap = stop_active_r2 and (
                            (is_long  and bar_open < _es3) or
                            (not is_long and bar_open > _es3)
                        )

                        # ema_flip exit for Runner2
                        # ema_flip exit for Runner2 — use EMA-21 stop cross
                        if not stop_hit and "ema_flip" in _r2_exit_modes:
                            ema_cross = (
                                (is_long  and bar_close <= ema_stop) or
                                (not is_long and bar_close >= ema_stop)
                            )
                            if ema_cross:
                                gap_ef = (is_long and bar_open < ema_stop) or (not is_long and bar_open > ema_stop)
                                exit_px_ef = bar_open if gap_ef else ema_stop
                                r2_exit = (bar_time, exit_px_ef, "Runner2EmaFlip", i)
                        if r2_exit is None and stop_hit:
                            exit_px = bar_open if gap else _es3
                            r2_exit = (bar_time, exit_px, "Runner2PreHookStop", i)
                        elif r2_exit is None and i >= r2_timeout_i:
                            r2_exit = (bar_time, bar_close, "Runner2Timeout", i)

                if r2_hook_active and r2_exit is None:
                    # Update best price
                    r2_best = max(r2_best, bar_high) if is_long else min(r2_best, bar_low)
                    # Chandelier stop
                    chand_stop = (r2_best - chandelier_atr_mult * trade_atr) if is_long \
                                 else (r2_best + chandelier_atr_mult * trade_atr)
                    stop_hit = (
                        (is_long  and bar_low  <= chand_stop) or
                        (not is_long and bar_high >= chand_stop)
                    )
                    gap = (
                        (is_long  and bar_open < chand_stop) or
                        (not is_long and bar_open > chand_stop)
                    )
                    if stop_hit:
                        exit_px = bar_open if gap else chand_stop
                        r2_exit = (bar_time, exit_px, "Runner2ChandStop", i)
                    elif i >= r2_timeout_i:
                        r2_exit = (bar_time, bar_close, "Runner2Timeout", i)

            # ---- All lots done? ----
            if base_exit is not None and r1_exit is not None and r2_exit is not None:
                _flush()
                _reset()
                continue

        # ----------------------------------------------------------------
        # FLAT — check for entry
        # ----------------------------------------------------------------
        if position == 0:
            prev_sig = sig[i - 1]
            if prev_sig != 0:
                ep  = exit_params.iloc[i - 1]
                intended_entry = float(ep_entry[i - 1])
                gap_limit = 4 * tick_size  # max 4 ticks gap allowed

                # Skip if bar opens too far from intended entry
                if prev_sig == 1 and bar_open > intended_entry + gap_limit:
                    skipped_entries[0] += 1
                    continue
                if prev_sig == -1 and bar_open < intended_entry - gap_limit:
                    skipped_entries[0] += 1
                    continue

                # Fill at intended price (signal_close ± 1 tick)
                position    = prev_sig
                entry_bar   = i
                entry_time  = bar_time
                entry_price = intended_entry
                trade_atr   = float(ep_atr[i - 1]) if ep_atr is not None and not np.isnan(ep_atr[i - 1]) else 1.0
                trade_ema   = float(ep_ema_mid[i - 1]) if ep_ema_mid is not None else float(ep_stop[i - 1])
                chandelier_atr_mult = float(ep_chand_atr[i - 1]) if ep_chand_atr is not None else 0.25

                sl  = int(ep_stop_live[i - 1])  if ep_stop_live  is not None else 120
                bt  = int(ep_time[i - 1])
                r1t = int(ep_r1_timeout[i - 1]) if ep_r1_timeout is not None else 3840
                r2t = int(ep_r2_timeout[i - 1]) if ep_r2_timeout is not None else 2400
                hm  = float(ep_hook_atr[i - 1]) if ep_hook_atr   is not None else 0.4

                # Per-lot target and ATR stop overrides from lot_configs
                # lot1 = Base, lot2 = Runner1, lot3 = Runner2
                if lot_configs and len(lot_configs) >= 1:
                    lc1 = lot_configs[0]
                    lc2 = lot_configs[1] if len(lot_configs) >= 2 else None
                    lc3 = lot_configs[2] if len(lot_configs) >= 3 else None

                    # Hard ATR stops: entry ± atr_stop_mult × ATR (0 = off)
                    def _hard_stop(lc, ep, atr, pos):
                        m = lc.get("atr_stop_mult", 0.0) if lc else 0.0
                        if not m:
                            return None
                        return (ep - m * atr) if pos == 1 else (ep + m * atr)

                    _l1_hard_stop = _hard_stop(lc1, intended_entry, trade_atr, prev_sig)
                    _l2_hard_stop = _hard_stop(lc2, intended_entry, trade_atr, prev_sig)
                    _l3_hard_stop = _hard_stop(lc3, intended_entry, trade_atr, prev_sig)
                    # Base lot target: lot1_tp_points is treated as ATR multiple.
                    # If > 0 it overrides base_tp_atr; if 0 fall back to exit_params target_price.
                    _l1_tp = lc1.get("tp_points", 0.0)
                    if _l1_tp and _l1_tp > 0:
                        _base_tgt_override = (
                            _ceil_tick(entry_price + _l1_tp * trade_atr, tick_size) if position == 1
                            else _floor_tick(entry_price - _l1_tp * trade_atr, tick_size)
                        )
                    else:
                        _base_tgt_override = None  # use exit_params target_price (set by base_tp_atr)
                    # Runner1 lot: if fixed exit mode and tp_points > 0, it has a target
                    _r1_is_runner  = lc2 is None or lc2.get("tp_points", 0.0) <= 0
                    _r1_exit_modes = lc2.get("exit_modes", "ema_flip") if lc2 else "ema_flip"
                    _r1_stop_mode  = lc2.get("stop_mode", "breakeven") if lc2 else "breakeven"
                    # Runner2 lot: chandelier or ema_flip
                    _r2_exit_modes = lc3.get("exit_modes", "chandelier") if lc3 else "chandelier"
                    _r2_stop_mode  = lc3.get("stop_mode", "none") if lc3 else "none"
                    _r2_tp         = lc3.get("tp_points", 0.0) if lc3 else 0.0
                else:
                    _base_tgt_override = None
                    _r1_is_runner = True
                    _r1_exit_modes = "ema_flip"
                    _r1_stop_mode  = "breakeven"
                    _r2_exit_modes = "chandelier"
                    _r2_stop_mode  = "none"
                    _r2_tp         = 0.0

                stop_live_i    = i + sl
                base_timeout_i = min(n - 1, i + bt)
                r1_timeout_i   = min(n - 1, i + r1t)
                r2_timeout_i   = min(n - 1, i + r2t)
                hook_level     = entry_price + hm * trade_atr if position == 1 \
                                 else entry_price - hm * trade_atr

                base_tp_hit    = False
                base_tp_i      = None
                r2_hook_active = False
                r2_best        = None
                # hard stops already set above from lot_configs

    # ----------------------------------------------------------------
    # Force-close at end of data
    # ----------------------------------------------------------------
    if position != 0:
        last_t = times[-1]
        last_c = closes[-1]
        if base_exit is None:
            base_exit = (last_t, last_c, "EndOfData", i)
        if r1_exit is None:
            r1_exit   = (last_t, last_c, "EndOfData", i)
        if r2_exit is None:
            r2_exit   = (last_t, last_c, "EndOfData", i)
        _flush()

    result_df = pd.DataFrame(trades)
    result_df.attrs["skipped_entries"] = skipped_entries[0]
    return result_df



    if trades.empty:
        return {"error": "No trades generated"}

    pnl     = trades["net_pnl"]
    winners = pnl[pnl > 0]
    losers  = pnl[pnl <= 0]

    total_pnl        = pnl.sum()
    num_trades       = len(trades)
    win_rate         = len(winners) / num_trades
    avg_win          = winners.mean() if not winners.empty else 0
    avg_loss         = losers.mean()  if not losers.empty  else 0
    profit_factor    = (winners.sum() / abs(losers.sum())) if losers.sum() != 0 else np.inf
    total_commission = trades["commission"].sum()

    equity       = pnl.cumsum()
    drawdown     = equity - equity.cummax()
    max_drawdown = drawdown.min()

    sharpe = (pnl.mean() / pnl.std() * np.sqrt(num_trades)) if pnl.std() > 0 else 0
    calmar = (total_pnl / abs(max_drawdown)) if max_drawdown != 0 else np.inf

    exit_counts = trades["exit_reason"].value_counts().to_dict() \
                  if "exit_reason" in trades.columns else {}

    # ------------------------------------------------------------------
    # Drawdown period analysis
    # A "drawdown period" = contiguous run where equity is below its
    # running peak. Duration = number of trades in that underwater run.
    # Max losing streak = longest consecutive run of losing trades.
    # ------------------------------------------------------------------
    in_dd         = drawdown < 0
    dd_durations  = []
    dd_depths     = []
    current_start = None

    for idx_pos in range(len(equity)):
        if in_dd.iloc[idx_pos]:
            if current_start is None:
                current_start = idx_pos
        else:
            if current_start is not None:
                dd_durations.append(idx_pos - current_start)
                dd_depths.append(float(drawdown.iloc[current_start:idx_pos].min()))
                current_start = None

    if current_start is not None:  # still in drawdown at end of data
        dd_durations.append(len(equity) - current_start)
        dd_depths.append(float(drawdown.iloc[current_start:].min()))

    n_dd_periods    = len(dd_durations)
    avg_dd_depth    = float(np.mean(dd_depths))    if dd_depths    else 0.0
    avg_dd_duration = float(np.mean(dd_durations)) if dd_durations else 0.0
    max_dd_duration = int(max(dd_durations))        if dd_durations else 0
    current_dd      = float(drawdown.iloc[-1])
    recovery_factor = round(total_pnl / abs(max_drawdown), 4) if max_drawdown != 0 else np.inf

    # Consecutive losing streak (trades, not equity-based)
    # Sort by entry_time to ensure chronological order
    _pnl_sorted = trades.sort_values("entry_time")["net_pnl"].values if "entry_time" in trades.columns else pnl.values
    lose_streak = cur_streak = 0
    for p in _pnl_sorted:
        if p < 0:
            cur_streak += 1
            lose_streak = max(lose_streak, cur_streak)
        else:
            cur_streak = 0

    # Underwater days — calendar days below equity peak
    # Build daily equity series from exit times
    max_underwater_days = 0
    total_underwater_days = 0
    if "exit_time" in trades.columns:
        _t = trades.sort_values("exit_time").copy()
        _t["exit_date"] = pd.to_datetime(_t["exit_time"]).dt.normalize()
        # Last cumulative PnL per exit date
        _daily = _t.groupby("exit_date")["net_pnl"].sum().cumsum()
        # Fill all calendar days between first and last trade
        _daily = _daily.reindex(
            pd.date_range(_daily.index[0], _daily.index[-1], freq="D")
        ).ffill()
        _peak = _daily.cummax()
        _uw   = _daily < _peak
        # Count contiguous underwater calendar day runs
        _cur = 0
        for uw in _uw:
            if uw:
                _cur += 1
                max_underwater_days   = max(max_underwater_days, _cur)
                total_underwater_days += 1
            else:
                _cur = 0

    return {
        "total_pnl":        round(total_pnl, 2),
        "num_trades":       num_trades,
        "win_rate":         round(win_rate, 4),
        "avg_win":          round(avg_win, 2),
        "avg_loss":         round(avg_loss, 2),
        "profit_factor":    round(profit_factor, 4),
        "avg_trade":        round(pnl.mean(), 2),
        "max_drawdown":     round(max_drawdown, 2),
        "sharpe":           round(sharpe, 4),
        "calmar":           round(calmar, 4),
        "total_commission": round(total_commission, 2),
        "exit_reasons":     exit_counts,
        "equity_curve":     equity,
        "drawdown_series":  drawdown,
        "trades":           trades,
        # Drawdown analytics
        "dd_n_periods":     n_dd_periods,
        "dd_avg_depth":     round(avg_dd_depth, 2),
        "dd_avg_duration":  round(avg_dd_duration, 1),
        "dd_max_duration":  max_dd_duration,
        "dd_max_consec_loss":      lose_streak,
        "dd_max_underwater_days":  max_underwater_days,
        "dd_total_underwater_days": total_underwater_days,
        "dd_current":              round(current_dd, 2),
        "recovery_factor":         recovery_factor,
    }


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Generic multi-lot engine — works with any strategy implementing
# get_exit_decision(position_state, bar_data) -> list[LotExitDecision]
# ---------------------------------------------------------------------------

def simulate_trades_generic(
    df: pd.DataFrame,
    signals: pd.Series,
    strategy,
    tick_size: float,
    tick_value: float,
    commission_per_side: float,
    lot_configs: list[dict] | None = None,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Generic multi-lot simulation engine.

    On each bar:
      1. If flat: check for entry signal
      2. If in position: call strategy.get_exit_decision(position_state, bar_data)
         to get exit instructions, execute them mechanically

    The strategy owns all exit logic — stop prices, target prices, trailing
    stops, trend-flip checks, timeouts. The engine owns order execution and
    PnL calculation only.

    Args:
        df:                   OHLCV DataFrame (execution timeframe)
        signals:              Entry signals (+1 long, -1 short, 0 flat)
                              aligned to df.index. entry_price pre-computed
                              by strategy in exit_params or derived from bar open.
        strategy:             Strategy instance with get_exit_decision() method
        tick_size:            Minimum price increment
        tick_value:           $ value per tick
        commission_per_side:  Commission per lot per side in $
        lot_configs:          List of dicts with 'size' key per lot.
                              Default: [{"size": 1}] × n_lots from strategy.

    Returns:
        trades DataFrame with one row per closed position (combined lots),
        matching the schema expected by calculate_metrics() and the exporter.
    """
    from exit_model import PositionState, BarData, LotExitDecision

    point_value = tick_value / tick_size

    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    times  = df.index
    sig    = signals.reindex(df.index).fillna(0).values
    n      = len(df)

    # Determine lot structure
    n_lots = getattr(strategy, "n_lots", 3)
    if lot_configs is None:
        lot_configs = [{"size": 1}] * n_lots

    trades = []

    # Position state
    in_position     = False
    position_side   = 0
    entry_bar       = None
    entry_time      = None
    entry_price     = None
    lots_open       = None
    lot_entry_px    = None
    high_since      = None
    low_since       = None
    user_state      = {}

    # Per-lot exit records (filled as lots close)
    lot_exits       = None   # list of (exit_time, exit_price, reason) per lot

    for i in range(1, n):
        # Build extra dict from designated columns (HTF indicators etc.)
        extra_data = {}
        if extra_cols:
            for col in extra_cols:
                if col in df.columns:
                    val = df[col].iloc[i]
                    extra_data[col] = bool(val) if df[col].dtype == bool else float(val) if pd.notna(val) else 0.0

        bar = BarData(
            index     = i,
            timestamp = times[i],
            open      = opens[i],
            high      = highs[i],
            low       = lows[i],
            close     = closes[i],
            extra     = extra_data,
        )

        # ------------------------------------------------------------------
        # Manage open position
        # ------------------------------------------------------------------
        if in_position:
            # Update tracking
            high_since = max(high_since, bar.high)
            low_since  = min(low_since,  bar.low)
            bars_held  = i - entry_bar

            pos_state = PositionState(
                side             = position_side,
                entry_price      = entry_price,
                entry_bar        = entry_bar,
                entry_time       = entry_time,
                bars_held        = bars_held,
                lots_open        = list(lots_open),
                lot_entry_prices = list(lot_entry_px),
                high_since_entry = high_since,
                low_since_entry  = low_since,
                user_state       = user_state,
            )

            # Ask strategy what to do
            decisions: list[LotExitDecision] = strategy.get_exit_decision(pos_state, bar)

            # Sort by priority (stop=0 first, then targets, then runners)
            decisions = sorted(decisions, key=lambda d: d.priority)

            for decision in decisions:
                k = decision.lot_index
                if k < len(lots_open) and lots_open[k]:
                    lots_open[k]   = False
                    lot_exits[k]   = (bar.timestamp, decision.exit_price, decision.reason)

                    # If stop (priority=0) exits any lot, close ALL remaining lots at same price
                    if decision.priority == 0:
                        for j in range(len(lots_open)):
                            if lots_open[j]:
                                lots_open[j] = False
                                lot_exits[j] = (bar.timestamp, decision.exit_price, "stop_cascade")
                        break  # all lots closed

            # Check if fully flat
            if not any(lots_open):
                # Build trade record
                raw_pnl = 0.0
                last_exit_time = entry_time

                row = {
                    "entry_time":  entry_time,
                    "entry_price": entry_price,
                    "position":    position_side,
                    "direction":   "LONG" if position_side > 0 else "SHORT",
                }

                for k, (ex_time, ex_price, ex_reason) in enumerate(lot_exits):
                    size      = lot_configs[k]["size"] if k < len(lot_configs) else 1
                    lot_pnl   = (ex_price - entry_price) * position_side * point_value * size
                    raw_pnl  += lot_pnl
                    row[f"lot{k+1}_exit_price"]  = ex_price
                    row[f"lot{k+1}_exit_reason"] = ex_reason
                    row[f"lot{k+1}_raw_pnl"]     = round(lot_pnl, 2)
                    row[f"lot{k+1}_size"]         = size
                    if ex_time > last_exit_time:
                        last_exit_time = ex_time

                total_lots = sum(lc["size"] for lc in lot_configs)
                commission = commission_per_side * 2 * total_lots
                net_pnl    = raw_pnl - commission

                row["exit_time"]   = last_exit_time
                row["exit_reason"] = lot_exits[0][2] if lot_exits else "unknown"
                row["raw_pnl"]     = round(raw_pnl, 2)
                row["commission"]  = round(commission, 2)
                row["net_pnl"]     = round(net_pnl, 2)
                row["duration_bars"] = i - entry_bar

                trades.append(row)

                # Reset
                in_position   = False
                position_side = 0
                entry_bar     = None
                entry_time    = None
                entry_price   = None
                lots_open     = None
                lot_entry_px  = None
                high_since    = None
                low_since     = None
                user_state    = {}
                lot_exits     = None

        # ------------------------------------------------------------------
        # New entry if flat
        # ------------------------------------------------------------------
        if not in_position and sig[i] != 0:
            position_side = int(sig[i])
            entry_bar     = i
            entry_time    = times[i]
            entry_price   = opens[i]   # fill at bar open (next bar after signal)
            lots_open     = [True] * n_lots
            lot_entry_px  = [entry_price] * n_lots
            high_since    = highs[i]
            low_since     = lows[i]
            user_state    = {}
            lot_exits     = [(None, None, None)] * n_lots
            in_position   = True

    return pd.DataFrame(trades) if trades else pd.DataFrame()


def calculate_metrics(trades: pd.DataFrame, df: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {"error": "No trades generated"}

    pnl     = trades["net_pnl"]
    winners = pnl[pnl > 0]
    losers  = pnl[pnl <= 0]

    total_pnl        = pnl.sum()
    num_trades       = len(trades)
    win_rate         = len(winners) / num_trades
    avg_win          = winners.mean() if not winners.empty else 0
    avg_loss         = losers.mean()  if not losers.empty  else 0
    profit_factor    = (winners.sum() / abs(losers.sum())) if losers.sum() != 0 else np.inf
    total_commission = trades["commission"].sum()

    equity       = pnl.cumsum()
    drawdown     = equity - equity.cummax()
    max_drawdown = drawdown.min()

    sharpe = (pnl.mean() / pnl.std() * np.sqrt(num_trades)) if pnl.std() > 0 else 0
    calmar = (total_pnl / abs(max_drawdown)) if max_drawdown != 0 else np.inf

    exit_counts = trades["exit_reason"].value_counts().to_dict() \
                  if "exit_reason" in trades.columns else {}

    # Drawdown period analysis
    in_dd         = drawdown < 0
    dd_durations  = []
    dd_depths     = []
    current_start = None

    for idx_pos in range(len(equity)):
        if in_dd.iloc[idx_pos]:
            if current_start is None:
                current_start = idx_pos
        else:
            if current_start is not None:
                dd_durations.append(idx_pos - current_start)
                dd_depths.append(float(drawdown.iloc[current_start:idx_pos].min()))
                current_start = None

    if current_start is not None:
        dd_durations.append(len(equity) - current_start)
        dd_depths.append(float(drawdown.iloc[current_start:].min()))

    n_dd_periods    = len(dd_durations)
    avg_dd_depth    = float(np.mean(dd_depths))    if dd_depths    else 0.0
    avg_dd_duration = float(np.mean(dd_durations)) if dd_durations else 0.0
    max_dd_duration = int(max(dd_durations))        if dd_durations else 0
    current_dd      = float(drawdown.iloc[-1])
    recovery_factor = round(total_pnl / abs(max_drawdown), 4) if max_drawdown != 0 else np.inf

    # Consecutive losing streak
    _pnl_sorted = trades.sort_values("entry_time")["net_pnl"].values if "entry_time" in trades.columns else pnl.values
    lose_streak = cur_streak = 0
    for p in _pnl_sorted:
        if p < 0:
            cur_streak += 1
            lose_streak = max(lose_streak, cur_streak)
        else:
            cur_streak = 0

    # Underwater days
    max_underwater_days = 0
    total_underwater_days = 0
    if "exit_time" in trades.columns:
        _t = trades.sort_values("exit_time").copy()
        _t["exit_date"] = pd.to_datetime(_t["exit_time"]).dt.normalize()
        _daily = _t.groupby("exit_date")["net_pnl"].sum().cumsum()
        _daily = _daily.reindex(
            pd.date_range(_daily.index[0], _daily.index[-1], freq="D")
        ).ffill()
        _peak = _daily.cummax()
        _uw   = _daily < _peak
        _cur = 0
        for uw in _uw:
            if uw:
                _cur += 1
                max_underwater_days   = max(max_underwater_days, _cur)
                total_underwater_days += 1
            else:
                _cur = 0

    return {
        "total_pnl":        round(total_pnl, 2),
        "num_trades":       num_trades,
        "win_rate":         round(win_rate, 4),
        "avg_win":          round(avg_win, 2),
        "avg_loss":         round(avg_loss, 2),
        "profit_factor":    round(profit_factor, 4),
        "avg_trade":        round(pnl.mean(), 2),
        "max_drawdown":     round(max_drawdown, 2),
        "sharpe":           round(sharpe, 4),
        "calmar":           round(calmar, 4),
        "total_commission": round(total_commission, 2),
        "exit_reasons":     exit_counts,
        "equity_curve":     equity,
        "drawdown_series":  drawdown,
        "trades":           trades,
        "dd_n_periods":     n_dd_periods,
        "dd_avg_depth":     round(avg_dd_depth, 2),
        "dd_avg_duration":  round(avg_dd_duration, 1),
        "dd_max_duration":  max_dd_duration,
        "dd_max_consec_loss":      lose_streak,
        "dd_max_underwater_days":  max_underwater_days,
        "dd_total_underwater_days": total_underwater_days,
        "dd_current":              round(current_dd, 2),
        "recovery_factor":         recovery_factor,
    }
