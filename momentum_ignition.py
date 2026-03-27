"""
momentum_ignition.py — Flag compression breakout strategy.

Logic (bar by bar on 1m data):
  1. Compute ATR(1m, 14, Wilder) on 1m bars
  2. Resample to 15m (right/right), compute ATR(15m, 14, Wilder) and SMA(50)
  3. Align 15m regime to 1m using merge_asof backward (no lookahead)
  4. Each bar: look at last flag_bars completed bars → flag high/low (shift 1)
  5. Compression filter: flag_range <= compression_atr_mult × ATR(1m)
  6. Extra breakout condition: prev_close must be inside flag (not already broken)
  7. Breakout: close >= flag_high + 2 ticks → long
               close <= flag_low  - 2 ticks → short
  8. Entry:  ceil_to_tick(flag_high) + 1 tick (long)
             floor_to_tick(flag_low) - 1 tick (short)
  9. Stop:   floor_to_tick(flag_low  - 0.1 × ATR(1m))  (long)
             ceil_to_tick(flag_high + 0.1 × ATR(1m))   (short)
  10. Target: ceil_to_tick(entry + target_points)  (long)
              floor_to_tick(entry - target_points) (short)
  11. Time exit: time_exit_bars bars from entry

All prices tick-rounded. All params configurable for grid search.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy import Strategy


# ---------------------------------------------------------------------------
# Tick rounding helpers
# ---------------------------------------------------------------------------

def _floor_tick(x: float, ts: float) -> float:
    return float(np.floor(x / ts) * ts)

def _ceil_tick(x: float, ts: float) -> float:
    return float(np.ceil(x / ts) * ts)

def _floor_tick_series(s: pd.Series, ts: float) -> pd.Series:
    return (np.floor(s.values / ts) * ts)

def _ceil_tick_series(s: pd.Series, ts: float) -> pd.Series:
    return (np.ceil(s.values / ts) * ts)


# ---------------------------------------------------------------------------
# ATR (Wilder)
# ---------------------------------------------------------------------------

def _wilder_atr(df: pd.DataFrame, period: int,
                high_col="high", low_col="low", close_col="close") -> pd.Series:
    h  = df[high_col]
    l  = df[low_col]
    c  = df[close_col]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# 15m regime — right/right resampling + merge_asof backward
# ---------------------------------------------------------------------------

def _build_15m_regime(df_1m: pd.DataFrame,
                      atr_period: int,
                      sma_period: int) -> pd.Series:
    """
    Compute ATR(15m) > SMA(ATR(15m), 50) and align to 1m index.

    right/right: each 15m bar is stamped at its close time.
    merge_asof backward: at 1m bar T, use last 15m bar that closed before T.
    No lookahead.
    """
    df_15m = df_1m.resample("15min", label="right", closed="right").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["open", "close"])

    df_15m["atr_15m"] = _wilder_atr(df_15m, atr_period)
    df_15m["sma_15m"] = df_15m["atr_15m"].rolling(sma_period, min_periods=sma_period).mean()
    df_15m["regime"]  = (df_15m["atr_15m"] > df_15m["sma_15m"]).fillna(False)

    idx_name = df_1m.index.name or "index"
    df_1m_ts  = df_1m.reset_index().rename(columns={idx_name: "ts"})
    df_15m_ts = df_15m.reset_index().rename(columns={df_15m.index.name or "index": "ts"})

    merged = pd.merge_asof(
        df_1m_ts[["ts"]].sort_values("ts"),
        df_15m_ts[["ts", "regime"]].sort_values("ts"),
        on="ts",
        direction="backward",
    )

    return merged.set_index("ts")["regime"].reindex(df_1m.index).fillna(False)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class MomentumIgnition(Strategy):
    """
    Flag compression breakout — captures the quiet → fast transition.

    Params:
        flag_bars            (int):   Lookback bars for flag. Default 8.
        compression_atr_mult (float): Flag range <= this × ATR(1m). Default 1.5.
        atr_period_1m        (int):   ATR period 1m bars. Default 14.
        atr_period_15m       (int):   ATR period 15m bars. Default 14.
        atr_sma_period_15m   (int):   SMA period for regime filter. Default 50.
        stop_buf_atr         (float): Stop offset = this × ATR(1m). Default 0.1.
        min_stop_ticks       (int):   Minimum stop distance in ticks. Default 4.
                                      Stop = max(stop_buf_atr × ATR, min_stop_ticks × tick_size).
        target_points        (float): Fixed target in points. Default 3.5.
        time_exit_bars       (int):   Max bars to hold. Default 60.
        direction            (str):   'both'/'long_only'/'short_only'. Default 'both'.
        timeframe            (str):   Must be '1m'. Default '1m'.
    """

    resample_mode = "bars"

    def _defaults(self) -> dict:
        return {
            "flag_bars":            8,
            "compression_atr_mult": 1.5,
            "atr_period_1m":        14,
            "atr_period_15m":       14,
            "atr_sma_period_15m":   50,
            "stop_buf_atr":         0.1,
            "min_stop_ticks":       4,
            "target_points":        3.5,
            "time_exit_bars":       60,
            "entry_gap_ticks":      4,
            "direction":            "both",
            # --- Multi-lot ---
            "multi_lot":            False,
            "lot1_size":            1,
            "lot1_tp_points":       2.0,
            "lot2_size":            1,
            "lot2_tp_points":       0.0,    # 0 = runner
            "lot2_stop_mode":       "breakeven",
            "lot2_exit_modes":      "ema_flip,time_exit",
            "lot3_size":            0,      # 0 = disabled
            "lot3_tp_points":       0.0,
            "lot3_stop_mode":       "breakeven",
            "lot3_exit_modes":      "ema_flip,time_exit",
            "runner_ema_fast":      9,
            "runner_ema_slow":      21,
            "runner_time_bars":     120,
            "runner_activation_atr": 0.0,   # 0 = immediate; N = require N×ATR open profit
            "timeframe":            "1m",
        }

    def generate_signals(self, df_1m: pd.DataFrame) -> dict:
        p  = self.params
        ts = p.get("tick_size", 0.25)

        # ----------------------------------------------------------------
        # 1. ATR(1m, Wilder)
        # ----------------------------------------------------------------
        atr_1m = _wilder_atr(df_1m, p["atr_period_1m"])

        # ----------------------------------------------------------------
        # 2. 15m regime — right/right + merge_asof backward
        # ----------------------------------------------------------------
        regime_ok = _build_15m_regime(
            df_1m,
            atr_period=p["atr_period_15m"],
            sma_period=p["atr_sma_period_15m"],
        )

        # ----------------------------------------------------------------
        # 3. Flag boundaries — last flag_bars completed bars (shift 1)
        # ----------------------------------------------------------------
        fb         = p["flag_bars"]
        flag_high  = df_1m["high"].rolling(fb, min_periods=fb).max().shift(1)
        flag_low   = df_1m["low"].rolling(fb, min_periods=fb).min().shift(1)
        flag_range = flag_high - flag_low

        # ----------------------------------------------------------------
        # 4. Compression filter
        # ----------------------------------------------------------------
        compressed = (flag_range <= p["compression_atr_mult"] * atr_1m).fillna(False)

        # ----------------------------------------------------------------
        # 5. Breakout conditions
        #    prev_close inside flag — prevents re-triggering on already
        #    broken flags
        # ----------------------------------------------------------------
        prev_close = df_1m["close"].shift(1)
        cur_close  = df_1m["close"]

        long_breakout = (
            compressed &
            regime_ok &
            (prev_close <= flag_high) &
            (cur_close  >= flag_high + 2 * ts)
        )
        short_breakout = (
            compressed &
            regime_ok &
            (prev_close >= flag_low) &
            (cur_close  <= flag_low - 2 * ts)
        )

        direction = p["direction"]
        if direction == "long_only":
            short_breakout = pd.Series(False, index=df_1m.index)
        elif direction == "short_only":
            long_breakout  = pd.Series(False, index=df_1m.index)

        # ----------------------------------------------------------------
        # 6. Entry, stop, target — all tick-rounded
        # ----------------------------------------------------------------
        signals       = pd.Series(0,     index=df_1m.index, dtype=int)
        entry_prices  = pd.Series(np.nan, index=df_1m.index)
        stop_prices   = pd.Series(np.nan, index=df_1m.index)
        target_prices = pd.Series(np.nan, index=df_1m.index)
        time_exits    = pd.Series(p["time_exit_bars"], index=df_1m.index, dtype=int)

        # --- Long ---
        lm = long_breakout.fillna(False)
        entry_long   = _ceil_tick_series(flag_high, ts) + ts
        _stop_buf    = np.maximum(p["stop_buf_atr"] * atr_1m, p["min_stop_ticks"] * ts)
        stop_long    = _floor_tick_series(flag_low  - _stop_buf, ts)
        target_long  = _ceil_tick_series(pd.Series(entry_long, index=df_1m.index) + p["target_points"], ts)

        signals[lm]       = 1
        entry_prices[lm]  = pd.Series(entry_long,  index=df_1m.index)[lm]
        stop_prices[lm]   = pd.Series(stop_long,   index=df_1m.index)[lm]
        target_prices[lm] = pd.Series(target_long, index=df_1m.index)[lm]

        # --- Short ---
        sm = short_breakout.fillna(False)
        entry_short  = _floor_tick_series(flag_low, ts) - ts
        stop_short   = _ceil_tick_series(flag_high  + _stop_buf, ts)
        target_short = _floor_tick_series(pd.Series(entry_short, index=df_1m.index) - p["target_points"], ts)

        signals[sm]       = -1
        entry_prices[sm]  = pd.Series(entry_short,  index=df_1m.index)[sm]
        stop_prices[sm]   = pd.Series(stop_short,   index=df_1m.index)[sm]
        target_prices[sm] = pd.Series(target_short, index=df_1m.index)[sm]

        # ----------------------------------------------------------------
        # 7. Warmup
        # ----------------------------------------------------------------
        warmup_15m = (p["atr_period_15m"] + p["atr_sma_period_15m"]) * 15
        warmup_1m  = max(p["atr_period_1m"], fb)
        warmup     = max(warmup_1m, warmup_15m)
        signals.iloc[:warmup] = 0

        exit_params = pd.DataFrame({
            "entry_price":    entry_prices,
            "stop_price":     stop_prices,
            "target_price":   target_prices,
            "time_exit_bars": time_exits,
            "entry_type":       "stop",
            "entry_gap_ticks":  float(p["entry_gap_ticks"]),
            # Multi-lot config columns — engine reads these if multi_lot=True
            "lot1_size":        int(p["lot1_size"]),
            "lot1_tp_points":   float(p["lot1_tp_points"]),
            "lot2_size":        int(p["lot2_size"]),
            "lot2_tp_points":   float(p["lot2_tp_points"]),
            "lot2_stop_mode":   str(p["lot2_stop_mode"]),
            "lot2_exit_modes":  str(p["lot2_exit_modes"]),
            "lot3_size":        int(p["lot3_size"]),
            "lot3_tp_points":   float(p["lot3_tp_points"]),
            "lot3_stop_mode":   str(p["lot3_stop_mode"]),
            "lot3_exit_modes":  str(p["lot3_exit_modes"]),
        })

        return {
            "signals":     signals,
            "exit_params": exit_params,
        }

    def _generate_on_tf(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("MomentumIgnition overrides generate_signals directly.")
