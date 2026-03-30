"""
fgbl_momentum.py — FGBL EMA-stack momentum strategy with 3-lot structure.

Signal generation (120m bars, left/left, offset 15min):
  - ATR rising + ATR > TR_mean(50) regime filter
  - EMA stack: 8 > 21 > 50 (long) / 8 < 21 < 50 (short)
  - EMA8 rising (long) / falling (short)
  - VWAP displacement: close > VWAP + 0.2×ATR (long) / close < VWAP - 0.2×ATR (short)
  - RSI > rsi_long_thresh (long) / RSI < rsi_short_thresh (short)

Entry:
  - Next 120m bar open (left/left: entry bar starts at signal bar close time)
  - Entry price: signal_bar_close ± 1 tick

3-lot structure (all lots enter simultaneously):
  - Base lot:    exits at entry ± base_tp_atr × ATR, EMA-21 stop (live after stop_live_bars),
                 base_timeout_bars timeout
  - Runner 1:    same stop pre-BaseTP, then BE/EMA-21 trail post-BaseTP,
                 r1_timeout_bars timeout
  - Runner 2:    EMA-21 stop until hook (price moves hook_atr × ATR),
                 then chandelier trail at best ± chandelier_atr × ATR,
                 r2_timeout_bars timeout

All timeouts and stops expressed in 1m bars.
VWAP column required in data (session VWAP, reset each day).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from strategy import Strategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _floor_tick(x: float, ts: float) -> float:
    return float(np.floor(round(x / ts, 8)) * ts)

def _ceil_tick(x: float, ts: float) -> float:
    return float(np.ceil(round(x / ts, 8)) * ts)

def _wilder_atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class FGBLMomentum(Strategy):
    """
    EMA-stack momentum strategy for FGBL (and similar instruments).

    Requires VWAP column in the 1m data.

    Params:
        signal_minutes       (int):   Signal bar timeframe in minutes. Default 120.
        anchor_offset_minutes(int):   Bar boundary offset from midnight. Default 15.
        atr_period           (int):   ATR/EMA periods on signal bars. Default 14.
        tr_mean_period       (int):   Rolling mean period for TR regime filter. Default 50.
        ema_fast             (int):   Fast EMA period. Default 8.
        ema_mid              (int):   Mid EMA period. Default 21.
        ema_slow             (int):   Slow EMA period. Default 50.
        vwap_atr_mult        (float): VWAP displacement threshold as ATR multiple. Default 0.2.
        rsi_long_thresh      (float): RSI threshold for long signals. Default 55.
        rsi_short_thresh     (float): RSI threshold for short signals. Default 45.
        base_tp_atr          (float): Base lot target = entry ± base_tp_atr × ATR. Default 2.0.
        stop_live_bars       (int):   1m bars after entry before stop activates. Default 120.
        base_timeout_bars    (int):   1m bars before base lot times out. Default 480.
        r1_timeout_bars      (int):   1m bars before runner1 times out. Default 3840.
        r2_timeout_bars      (int):   1m bars before runner2 times out. Default 2400.
        hook_atr_mult        (float): Runner2 hook = entry ± hook_atr_mult × ATR. Default 0.4.
        chandelier_atr_mult  (float): Chandelier trail = best ∓ chandelier_atr_mult × ATR. Default 0.25.
        atr_regime_filter    (float): Skip signals where ATR > this value. 0.0 = off. Default 0.0.
        cot_regime_filter    (str):   Use ZN slope signal as regime filter.
                                      'off'       = disabled (default)
                                      'gate'      = skip signals when ZN signal == 0 (flat zone)
                                      'direction' = only take longs when ZN==+1, shorts when ZN==-1
                                      'both'      = gate + direction combined
        direction            (str):   'both'/'long_only'/'short_only'. Default 'both'.
        timeframe            (str):   Must be '1m'. Default '1m'.
    """

    resample_mode    = "bars"
    manages_own_lots = True   # use simulate_trades_fgbl engine
    default_n_lots   = 3      # pre-fill multi-lot prompt with 3 lots
    # Pre-fill for multi-lot prompt — matches original 3-lot structure
    default_lot_config = [
        {"tp_points": 2.0,  "exit_modes": "fixed",                "stop_mode": "none",      "is_runner": False},
        {"tp_points": 0.0,  "exit_modes": "ema_flip",             "stop_mode": "breakeven", "is_runner": True},
        {"tp_points": 0.0,  "exit_modes": "chandelier",           "stop_mode": "none",      "is_runner": True},
    ]
    description      = (
        "EMA-stack momentum on 120m signal bars with 3-lot management.\n"
        "  Regime: ATR rising + ATR > TR_mean(50) + EMA 8/21/50 stack aligned.\n"
        "  Entry:  VWAP displacement > 0.2×ATR + RSI confirmation.\n"
        "  Lots:   Base (ATR target) + Runner1 (BE/EMA trail) + Runner2 (chandelier trail).\n"
        "  Stop:   EMA-21, live after 120 bars. Instruments: FGBL and similar bonds."
    )

    def _defaults(self) -> dict:
        return {
            "signal_minutes":        120,
            "anchor_offset_minutes": 15,
            "atr_period":            14,
            "tr_mean_period":        50,
            "ema_fast":              8,
            "ema_mid":               21,
            "ema_slow":              50,
            "vwap_atr_mult":         0.2,
            "rsi_long_thresh":       55.0,
            "rsi_short_thresh":      45.0,
            "base_tp_atr":           2.0,
            "stop_live_bars":        120,
            "base_timeout_bars":     480,
            "r1_timeout_bars":       3840,
            "r2_timeout_bars":       2400,
            "hook_atr_mult":         0.4,
            "chandelier_atr_mult":   0.25,
            "atr_regime_filter":     0.0,
            "cot_regime_filter":     "off",
            "direction":             "both",
            "timeframe":             "1m",
        }

    def generate_signals(self, df_1m: pd.DataFrame) -> dict:
        p  = self.params
        ts = p.get("tick_size", 0.01)

        # Load COT regime signal if filter is active
        _cot_filter = str(p.get("cot_regime_filter", "off")).lower()
        _cot_signal = None
        if _cot_filter != "off":
            import os
            _sig_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data", "ZN_refined_signal.csv"
            )
            try:
                _cot_df = pd.read_csv(_sig_path, parse_dates=["Timestamp"])
                _cot_df = _cot_df.set_index("Timestamp")["signal"]
                _cot_signal = _cot_df
            except FileNotFoundError:
                import warnings
                warnings.warn(
                    f"cot_regime_filter is '{_cot_filter}' but signal file not found: "
                    f"{_sig_path}. Run update_signal.py first. Filter disabled.",
                    UserWarning
                )
                _cot_filter = "off"

        if "vwap" not in df_1m.columns:
            raise ValueError(
                "FGBLMomentum requires a 'vwap' column in the data. "
                "Check your dataset has VWAP data."
            )

        # ----------------------------------------------------------------
        # 1. Build signal bars — use pre-resampled cache if available
        # ----------------------------------------------------------------
        sig_min    = int(p["signal_minutes"])
        offset_min = int(p["anchor_offset_minutes"])

        # Check if pre-resampled data was injected via params (from grid_search cache)
        _presampled = p.get("_presampled_sig")
        if _presampled is not None:
            sig = _presampled.copy()
        else:
            m1_idx = df_1m.copy()
            m1_idx.index = pd.to_datetime(m1_idx.index)

            sig = m1_idx.resample(
                f"{sig_min}min",
                label="left",
                closed="left",
                origin="start_day",
                offset=f"{offset_min}min",
            ).agg({
                "open":   "first",
                "high":   "max",
                "low":    "min",
                "close":  "last",
                "volume": "sum",
            }).dropna(subset=["open", "close"])

            # VWAP: last value in each signal bar (session VWAP)
            sig["vwap"] = m1_idx["vwap"].resample(
                f"{sig_min}min",
                label="left",
                closed="left",
                origin="start_day",
                offset=f"{offset_min}min",
            ).last()

        # ----------------------------------------------------------------
        # 2. Indicators on signal bars
        # ----------------------------------------------------------------
        atr_p  = int(p["atr_period"])
        tr_p   = int(p["tr_mean_period"])
        ef, em, es = int(p["ema_fast"]), int(p["ema_mid"]), int(p["ema_slow"])

        prev_close    = sig["close"].shift(1)
        tr            = pd.concat([
            (sig["high"] - sig["low"]).abs(),
            (sig["high"] - prev_close).abs(),
            (sig["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr           = tr.ewm(alpha=1.0/atr_p, adjust=False, min_periods=atr_p).mean()
        tr_mean       = tr.rolling(tr_p, min_periods=tr_p).mean()

        delta         = sig["close"].diff()
        avg_gain      = delta.clip(lower=0).ewm(alpha=1.0/atr_p, adjust=False, min_periods=atr_p).mean()
        avg_loss      = (-delta.clip(upper=0)).ewm(alpha=1.0/atr_p, adjust=False, min_periods=atr_p).mean()
        rsi           = 100 - 100 / (1 + avg_gain / (avg_loss.replace(0, np.nan)))

        ema_f         = sig["close"].ewm(span=ef, adjust=False).mean()
        ema_m         = sig["close"].ewm(span=em, adjust=False).mean()
        ema_s         = sig["close"].ewm(span=es, adjust=False).mean()

        atr_rising    = atr > atr.shift(1)
        regime_ok     = atr_rising & (atr > tr_mean)

        vwap_mult     = float(p["vwap_atr_mult"])
        rsi_long      = float(p["rsi_long_thresh"])
        rsi_short     = float(p["rsi_short_thresh"])

        long_sig = (
            regime_ok &
            (ema_f > ema_m) & (ema_m > ema_s) &
            (ema_f > ema_f.shift(1)) &
            ((sig["close"] - sig["vwap"]) > vwap_mult * atr) &
            (rsi > rsi_long)
        )
        short_sig = (
            regime_ok &
            (ema_f < ema_m) & (ema_m < ema_s) &
            (ema_f < ema_f.shift(1)) &
            ((sig["vwap"] - sig["close"]) > vwap_mult * atr) &
            (rsi < rsi_short)
        )

        # ATR regime filter — skip signals in high-volatility bars
        atr_regime = float(p.get("atr_regime_filter", 0.0))
        if atr_regime > 0:
            regime_ok_atr = atr <= atr_regime
            long_sig  = long_sig  & regime_ok_atr
            short_sig = short_sig & regime_ok_atr

        # COT regime filter — align signal bars with ZN weekly signal
        if _cot_filter != "off" and _cot_signal is not None:
            # Forward-fill ZN signal to 120m signal bar timestamps
            # Each signal bar's date is used to look up the ZN signal
            sig_dates  = sig.index.normalize()
            zn_aligned = pd.Series(index=sig.index, dtype=float)
            for dt in sig_dates.unique():
                # Find most recent ZN signal on or before this date
                past = _cot_signal[_cot_signal.index.normalize() <= dt]
                zn_val = past.iloc[-1] if len(past) > 0 else 0.0
                zn_aligned[sig_dates == dt] = zn_val

            if _cot_filter in ("gate", "both"):
                # Block all signals when ZN is in flat zone (signal == 0)
                zn_active = zn_aligned != 0
                long_sig  = long_sig  & zn_active
                short_sig = short_sig & zn_active

            if _cot_filter in ("direction", "both"):
                # Only take longs when ZN bullish (+1), shorts when ZN bearish (-1)
                long_sig  = long_sig  & (zn_aligned == 1)
                short_sig = short_sig & (zn_aligned == -1)

        direction = p["direction"]
        if direction == "long_only":
            short_sig = pd.Series(False, index=sig.index)
        elif direction == "short_only":
            long_sig  = pd.Series(False, index=sig.index)

        # ----------------------------------------------------------------
        # 3. Map signal bar values back to 1m bars
        #    Entry bar = first 1m bar of the NEXT signal period
        #    (left/left: signal bar at T covers T → T+120m,
        #     entry is at T+120m which is the start of the next signal bar)
        # ----------------------------------------------------------------
        # Build lookup: signal_bar_start → indicators needed for exit_params
        sig_data = pd.DataFrame({
            "long_sig":    long_sig,
            "short_sig":   short_sig,
            "atr":         atr,
            "ema_mid":     ema_m,         # EMA-21 stop
            "close":       sig["close"],
        }, index=sig.index)

        # For each 1m bar, find which signal bar it belongs to
        # then check if that signal bar had a signal — entry is the
        # first 1m bar of the period AFTER the signal
        def _floor_sig(ts_val):
            day_start = ts_val.normalize() + pd.Timedelta(minutes=offset_min)
            period    = pd.Timedelta(minutes=sig_min)
            if ts_val < day_start:
                day_start -= pd.Timedelta(days=1)
            n = math.floor((ts_val - day_start) / period)
            return day_start + n * period

        # Build 1m signals by matching entry times
        # entry_time = sig_bar_start + sig_min minutes
        active_sigs = sig_data[(sig_data["long_sig"] | sig_data["short_sig"])].copy()
        active_sigs["entry_bar_time"] = active_sigs.index + pd.Timedelta(minutes=sig_min)

        # Create signal series on 1m index
        signals       = pd.Series(0,     index=df_1m.index, dtype=int)
        entry_prices  = pd.Series(np.nan, index=df_1m.index)
        stop_prices   = pd.Series(np.nan, index=df_1m.index)
        base_targets  = pd.Series(np.nan, index=df_1m.index)
        atr_at_entry  = pd.Series(np.nan, index=df_1m.index)
        ema_mid_entry = pd.Series(np.nan, index=df_1m.index)
        time_exits    = pd.Series(int(p["base_timeout_bars"]), index=df_1m.index, dtype=int)

        _1m_index = df_1m.index

        for sig_start, row in active_sigs.iterrows():
            entry_t = row["entry_bar_time"]
            if entry_t not in _1m_index:
                continue

            # Engine checks sig[i-1] to enter on bar i.
            # Place signal on bar BEFORE entry_t so engine fills on entry_t.
            entry_pos = _1m_index.get_loc(entry_t)
            if entry_pos == 0:
                continue
            signal_t = _1m_index[entry_pos - 1]

            is_long   = bool(row["long_sig"])
            sig_atr   = float(row["atr"])
            sig_ema   = float(row["ema_mid"])
            signal_bar_close = float(row["close"])  # close of completed signal bar (bar before entry)

            # Entry price = signal bar close ± 1 tick
            ep = _ceil_tick(signal_bar_close + ts, ts) if is_long else _floor_tick(signal_bar_close - ts, ts)

            # Base target = entry ± base_tp_atr × ATR
            bt = _ceil_tick(ep + float(p["base_tp_atr"]) * sig_atr, ts) if is_long \
                 else _floor_tick(ep - float(p["base_tp_atr"]) * sig_atr, ts)

            signals[signal_t]       = 1 if is_long else -1
            entry_prices[signal_t]  = ep
            stop_prices[signal_t]   = sig_ema
            base_targets[signal_t]  = bt
            atr_at_entry[signal_t]  = sig_atr
            ema_mid_entry[signal_t] = sig_ema
            time_exits[signal_t]    = int(p["base_timeout_bars"])

        # ----------------------------------------------------------------
        # 4. Build EMA-21 series on 1m bars for dynamic stop tracking
        #    Original script uses EMA-21 of the PREVIOUS signal bar as stop
        #    (ema21_vals[idx - 1]) — forward-filled across all 1m bars in period
        # ----------------------------------------------------------------
        ema_mid_1m = pd.Series(np.nan, index=df_1m.index)
        sig_index_list = list(sig.index)
        for i_sig, sig_start in enumerate(sig_index_list):
            sig_end = sig_start + pd.Timedelta(minutes=sig_min)
            mask    = (df_1m.index >= sig_start) & (df_1m.index < sig_end)
            # Use previous signal bar's EMA-21 (idx-1), matching original
            if i_sig > 0:
                prev_ema = ema_m.iloc[i_sig - 1]
            else:
                prev_ema = np.nan
            ema_mid_1m[mask] = prev_ema

        # ----------------------------------------------------------------
        # 5. Build exit_params DataFrame
        # ----------------------------------------------------------------
        exit_params = pd.DataFrame({
            "entry_price":           entry_prices,
            "stop_price":            stop_prices,
            "target_price":          base_targets,
            "time_exit_bars":        time_exits,
            "entry_type":            "limit",      # fill at intended price only
            "entry_gap_ticks":       0.0,
            # Multi-lot config
            "multi_lot":             True,
            "lot1_size":             1,
            "lot1_tp_points":        0.0,          # unused — base target set via target_price
            "lot1_exit_modes":       "fixed",
            "lot1_stop_mode":        "none",
            "lot2_size":             1,
            "lot2_tp_points":        0.0,
            "lot2_exit_modes":       "ema_flip",   # runner1: BE/EMA trail post-BaseTP
            "lot2_stop_mode":        "breakeven",
            "lot3_size":             1,
            "lot3_tp_points":        0.0,
            "lot3_exit_modes":       "chandelier",  # runner2: chandelier trail
            "lot3_stop_mode":        "none",
            # Strategy-specific params for engine
            "stop_live_bars":        int(p["stop_live_bars"]),
            "r1_timeout_bars":       int(p["r1_timeout_bars"]),
            "r2_timeout_bars":       int(p["r2_timeout_bars"]),
            "hook_atr_mult":         float(p["hook_atr_mult"]),
            "chandelier_atr_mult":   float(p["chandelier_atr_mult"]),
            "atr_at_entry":          atr_at_entry,
            "ema_mid_1m":            ema_mid_1m,
        })

        return {
            "signals":     signals,
            "exit_params": exit_params,
        }

    def _generate_on_tf(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("FGBLMomentum overrides generate_signals directly.")
