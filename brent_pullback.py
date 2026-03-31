"""
brent_pullback.py — Brent Early Trend Pullback Strategy

Faithful port of the original v1.5 (1-lot) and v2.4 (3-lot) systems.

v1.5: Single lot, stop=1xATR, target=2xATR, runner exits on 15m trend flip.
v2.4: Three lots, tp1=1xATR, tp2=2xATR, lot3 runner on 15m trend flip + BE.

Signal generation mirrors the original build_execution_frame / build_signals
exactly — including the ffill merge (no index shift) and 5m trend confirmation.

Entry fill: close of signal bar + entry_slip (matching original behaviour).
Note: this fills at signal bar close which is a minor fill approximation —
      the bar whose close triggers the signal is also the fill bar.
      This matches the original script and can be reviewed separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from strategy import Strategy
from exit_model import LotExitDecision, PositionState, BarData

# Signal constants — match original exactly
FAST              = 9
SLOW              = 21
ATR_N             = 14
ATR_LOOKBACK      = 200
ATR_PCTL          = 0.30
PULLBACK_ATR_FRAC = 0.35
PULLBACK_LOOKBACK = 12
ENTRY_START_UTC   = "07:00"
ENTRY_END_UTC     = "18:00"
ENTRY_SLIP_TICKS  = 1
STOP_SLIP_TICKS   = 2
TARGET_SLIP_TICKS = 1
FLIP_SLIP_TICKS   = 1


class BrentPullback(Strategy):
    """
    Brent Early Trend Pullback — faithful port of v1.5/v2.4.

    variant="v1.5": single lot, target=2xATR
    variant="v2.4": three lots, tp1=1xATR, tp2=2xATR, lot3 runner (TODO)

    Default: v1.5 (single lot) for initial validation.
    """

    resample_mode = "bars"
    manages_own_lots = False
    n_lots = 1   # v1.5 default; v2.4 will use 3

    description = (
        "Brent Early Trend Pullback — port of original v1.5/v2.4 systems.\n"
        "  Regime: 15m ATR > 30th pct of last 200 bars.\n"
        "  Entry:  5m close reclaims EMA9 after pullback >= 0.35xATR.\n"
        "  Confirm: 5m + 10m + 15m trend all aligned.\n"
        "  v1.5: 1 lot, stop=1xATR, target=2xATR, runner=15m flip.\n"
        "  v2.4: 3 lots, tp1=1xATR, tp2=2xATR, lot3=15m flip + BE."
    )

    @property
    def uses_generic_engine(self) -> bool:
        return True

    def _defaults(self) -> dict:
        return {
            "timeframe":    "5m",
            "variant":      "v1.5",   # "v1.5" or "v2.4"
            "direction":    "both",
            "cot_regime_filter": "off",
            # Slippage (in ticks)
            "entry_slip_ticks":  ENTRY_SLIP_TICKS,
            "stop_slip_ticks":   STOP_SLIP_TICKS,
            "target_slip_ticks": TARGET_SLIP_TICKS,
            "flip_slip_ticks":   FLIP_SLIP_TICKS,
            # Signal params
            "ema_fast":          FAST,
            "ema_slow":          SLOW,
            "atr_len":           ATR_N,
            "atr_lookback":      ATR_LOOKBACK,
            "atr_pctl":          ATR_PCTL,
            "pullback_atr_frac": PULLBACK_ATR_FRAC,
            "pullback_lookback": PULLBACK_LOOKBACK,
            "entry_start_utc":   ENTRY_START_UTC,
            "entry_end_utc":     ENTRY_END_UTC,
        }

    # ------------------------------------------------------------------
    # Helpers — match original exactly
    # ------------------------------------------------------------------
    @staticmethod
    def _resample(df1m: pd.DataFrame, rule: str) -> pd.DataFrame:
        return pd.DataFrame({
            "open":   df1m["open"].resample(rule).first(),
            "high":   df1m["high"].resample(rule).max(),
            "low":    df1m["low"].resample(rule).min(),
            "close":  df1m["close"].resample(rule).last(),
            "volume": df1m["volume"].resample(rule).sum()
                      if "volume" in df1m.columns
                      else df1m["close"].resample(rule).count(),
        }).dropna(subset=["open", "high", "low", "close"])

    @staticmethod
    def _add_trend(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
        df = df.copy()
        df["ema_fast"]       = df["close"].ewm(span=fast, adjust=False).mean()
        df["ema_slow"]       = df["close"].ewm(span=slow, adjust=False).mean()
        df["ema_fast_slope"] = df["ema_fast"].diff()
        df["trend_state"]    = 0
        df.loc[(df["ema_fast"] > df["ema_slow"]) &
               (df["ema_fast_slope"] > 0), "trend_state"] = 1
        df.loc[(df["ema_fast"] < df["ema_slow"]) &
               (df["ema_fast_slope"] < 0), "trend_state"] = -1
        return df

    @staticmethod
    def _add_atr_regime(df: pd.DataFrame, n: int, lookback: int, pctl: float) -> pd.DataFrame:
        df = df.copy()
        prev_c = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_c).abs(),
            (df["low"]  - prev_c).abs(),
        ], axis=1).max(axis=1)
        df["atr"]   = tr.ewm(alpha=1 / n, adjust=False).mean()
        df["atr_p"] = df["atr"].rolling(lookback).quantile(pctl)
        return df

    @staticmethod
    def _infer_tick(close: pd.Series) -> float:
        d = close.diff().abs()
        d = d[(d > 0) & np.isfinite(d)]
        return float(d.min()) if not d.empty else 0.01

    # ------------------------------------------------------------------
    # Build execution frame — mirrors original build_execution_frame
    # Uses ffill reindex (no index shift) to match original exactly
    # ------------------------------------------------------------------
    def _build_exec_frame(self, df1m: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        fast = p["ema_fast"]
        slow = p["ema_slow"]

        df5  = self._add_trend(self._resample(df1m, "5min"),  fast, slow)
        df10 = self._add_trend(self._resample(df1m, "10min"), fast, slow)
        df15 = self._add_trend(self._resample(df1m, "15min"), fast, slow)
        df15 = self._add_atr_regime(df15, p["atr_len"], p["atr_lookback"], p["atr_pctl"])

        # Merge HTF → 5m using ffill (original behaviour, no index shift)
        m = df5.copy()
        m["trend10"] = df10["trend_state"].reindex(m.index, method="ffill")
        tmp = df15[["trend_state", "atr", "atr_p"]].rename(
            columns={"trend_state": "trend15"})
        m[["trend15", "atr", "atr_p"]] = tmp.reindex(m.index, method="ffill")
        return m

    # ------------------------------------------------------------------
    # Signal generation — mirrors original build_signals exactly
    # ------------------------------------------------------------------
    def _build_signals(self, m: pd.DataFrame, tick: float):
        p = self.params
        close    = m["close"].to_numpy(float)
        high     = m["high"].to_numpy(float)
        low      = m["low"].to_numpy(float)
        ema_fast = m["ema_fast"].to_numpy(float)
        trend5   = m["trend_state"].to_numpy(float)
        trend10  = m["trend10"].to_numpy(float)
        trend15  = m["trend15"].to_numpy(float)
        atr      = m["atr"].to_numpy(float)
        atr_p    = m["atr_p"].to_numpy(float)

        # Entry time window
        sh, sm = [int(x) for x in p["entry_start_utc"].split(":")]
        eh, em = [int(x) for x in p["entry_end_utc"].split(":")]
        s_min  = sh * 60 + sm
        e_min  = eh * 60 + em
        ts_min = m.index.hour * 60 + m.index.minute
        entry_mask = (ts_min >= s_min) & (ts_min <= e_min)

        regime_ok = (np.isfinite(atr) & np.isfinite(atr_p) & (atr > atr_p))

        # Reclaim trigger (close crosses EMA9)
        reclaim_long  = np.zeros(len(m), dtype=bool)
        reclaim_short = np.zeros(len(m), dtype=bool)
        reclaim_long[1:]  = (close[:-1] <= ema_fast[:-1]) & (close[1:] > ema_fast[1:])
        reclaim_short[1:] = (close[:-1] >= ema_fast[:-1]) & (close[1:] < ema_fast[1:])

        # Pullback detection — original loop
        pb_long  = np.zeros(len(m), dtype=bool)
        pb_short = np.zeros(len(m), dtype=bool)
        frac = p["pullback_atr_frac"]
        lb   = p["pullback_lookback"]
        for i in range(len(m)):
            if not (np.isfinite(ema_fast[i]) and np.isfinite(atr[i])):
                continue
            j0 = max(0, i - lb + 1)
            if np.any(low[j0:i+1]  <= ema_fast[i] - frac * atr[i]):
                pb_long[i]  = True
            if np.any(high[j0:i+1] >= ema_fast[i] + frac * atr[i]):
                pb_short[i] = True

        direction = p.get("direction", "both")
        allow_l = direction in ("both", "long_only")
        allow_s = direction in ("both", "short_only")

        entry_long = (
            entry_mask & (trend5 == 1) & (trend10 == 1) & (trend15 == 1) &
            regime_ok & pb_long & reclaim_long
        ) & allow_l

        entry_short = (
            entry_mask & (trend5 == -1) & (trend10 == -1) & (trend15 == -1) &
            regime_ok & pb_short & reclaim_short
        ) & allow_s

        return entry_long, entry_short

    # ------------------------------------------------------------------
    # generate_signals — override base class
    # ------------------------------------------------------------------
    def _generate_on_tf(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=df.index, dtype=int)

    def generate_signals(self, df_1m: pd.DataFrame) -> dict:
        p    = self.params
        tick = p.get("tick_size", self._infer_tick(df_1m["close"]))

        m              = self._build_exec_frame(df_1m)
        entry_long, entry_short = self._build_signals(m, tick)

        # Fill at next bar open (Option A) — engine handles this by default
        slip_s = tick * p["stop_slip_ticks"]
        slip_t = tick * p["target_slip_ticks"]
        slip_f = tick * p["flip_slip_ticks"]

        signals = pd.Series(0, index=m.index, dtype=int)
        signals[entry_long]  =  1
        signals[entry_short] = -1

        # Embed HTF data + slippage into exec_df for get_exit_decision
        # Shift ATR back by 1 so that on fill bar i+1, bar.extra["_atr"]
        # contains the ATR from signal bar i (used to set stop/target correctly)
        m["_atr"]       = m["atr"].shift(1)
        m["_trend15"]   = m["trend15"]   # trend15 on fill bar is correct (current)
        m["_slip_stop"] = slip_s
        m["_slip_tgt"]  = slip_t
        m["_slip_flip"] = slip_f

        self._extra_cols = ["_atr", "_trend15", "_slip_stop",
                            "_slip_tgt", "_slip_flip"]
        self._exec_df    = m

        # --- Fill price: open of the first 1m bar after the 5m signal bar ---
        # Signal fires at close of 5m bar at time T (bar label = bar open time).
        # The 5m bar covers T to T+5min. The first tradeable 1m bar opens at T+5min.
        # We look up df_1m.loc[T+5min, "open"] and store as _entry_px.
        # The engine uses _entry_px when present instead of the exec_df bar open.
        td5 = pd.Timedelta(minutes=5)
        entry_prices = pd.Series(np.nan, index=m.index, dtype=float)
        for idx_ts in m.index[signals != 0]:
            fill_ts = idx_ts + td5
            if fill_ts in df_1m.index:
                raw_open = float(df_1m.at[fill_ts, "open"])
                side     = int(signals.at[idx_ts])
                slip_e   = tick * p["entry_slip_ticks"]
                entry_prices.at[idx_ts] = raw_open + slip_e if side == 1 else raw_open - slip_e

        # Shift _entry_px forward by 1 bar to align with the shifted signals.
        # Signal at bar i → fill at bar i+1. _entry_px must be at bar i+1
        # so the engine reads it from extra_data when processing the fill bar.
        m["_entry_px"] = entry_prices.shift(1)
        self._extra_cols = ["_atr", "_trend15", "_slip_stop",
                            "_slip_tgt", "_slip_flip", "_entry_px"]

        # Shift signals by 1 so the engine fires at open of next bar
        signals = signals.shift(1).fillna(0).astype(int)

        return {
            "signals":    signals,
            "exec_df":    m,
            "extra_cols": self._extra_cols,
        }

    # ------------------------------------------------------------------
    # Exit decisions — called by generic engine per bar
    # ------------------------------------------------------------------
    def get_exit_decision(
        self,
        position: PositionState,
        bar: BarData,
    ) -> list[LotExitDecision]:

        side = position.side
        ep   = position.user_state

        # Read bar data
        atr15    = bar.extra.get("_atr",       0.0)
        trend15  = bar.extra.get("_trend15",   0.0)
        slip_s   = bar.extra.get("_slip_stop", 0.02)
        slip_t   = bar.extra.get("_slip_tgt",  0.01)
        slip_f   = bar.extra.get("_slip_flip", 0.01)

        # Initialise stop/target on first bar
        if "stop_px" not in ep:
            # Use pre-computed entry price from exec_df if available
            entry = position.entry_price
            if side == 1:
                ep["stop_px"]   = entry - 1.0 * atr15
                ep["target_px"] = entry + 2.0 * atr15  # v1.5: 2xATR
            else:
                ep["stop_px"]   = entry + 1.0 * atr15
                ep["target_px"] = entry - 2.0 * atr15

        stop_px   = ep["stop_px"]
        target_px = ep["target_px"]

        # --- Stop (priority=0, cascades all lots) ---
        stop_hit = (side == 1 and bar.low <= stop_px) or \
                   (side == -1 and bar.high >= stop_px)
        if stop_hit:
            exit_px = (stop_px - slip_s) if side == 1 else (stop_px + slip_s)
            return [LotExitDecision(
                lot_index=0, exit_price=exit_px, reason="stop", priority=0
            )]

        # --- Target (priority=1) ---
        if position.lots_open[0]:
            tgt_hit = (side == 1 and bar.high >= target_px) or \
                      (side == -1 and bar.low  <= target_px)
            if tgt_hit:
                exit_px = (target_px - slip_t) if side == 1 else (target_px + slip_t)
                return [LotExitDecision(
                    lot_index=0, exit_price=exit_px, reason="target", priority=1
                )]

        # --- 15m trend flip (priority=2) ---
        if position.lots_open[0]:
            flip = (side == 1 and trend15 != 1) or \
                   (side == -1 and trend15 != -1)
            if flip and np.isfinite(trend15):
                exit_px = (bar.close - slip_f) if side == 1 else (bar.close + slip_f)
                return [LotExitDecision(
                    lot_index=0, exit_price=exit_px, reason="trend_flip_15m", priority=2
                )]

        return []
