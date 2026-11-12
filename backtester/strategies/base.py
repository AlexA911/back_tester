# Directory refactor
"""
strategy.py — Strategy base class and implementations.

Each strategy declares:
  - params:        dict of strategy parameters including 'timeframe'
  - resample_mode: "bars"      — strategy runs on resampled bars (Mode 1)
                   "indicators" — 1m execution, HTF indicators (Mode 2)

The engine calls generate_signals(df_1m) in both cases.
Resampling is handled internally by the strategy so the engine
interface stays identical regardless of timeframe.

ADDING A NEW STRATEGY:
  1. Subclass Strategy
  2. Set resample_mode = "bars" or "indicators"
  3. Implement _generate_on_tf(df) — receives the correctly resampled df
  4. Do NOT override generate_signals() — it handles resampling safely
"""

from __future__ import annotations

import pandas as pd
from abc import ABC, abstractmethod

from backtester.core.resampler import resample_ohlcv, align_signals_to_1m


class Strategy(ABC):

    # Subclasses set this to "bars" or "indicators"
    resample_mode: str = "bars"

    def __init__(self, params: dict | None = None):
        defaults = self._defaults()
        self.params = {**defaults, **(params or {})}

    def _defaults(self) -> dict:
        """Override in subclass to provide default params including timeframe."""
        return {"timeframe": "1m"}

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def timeframe(self) -> str:
        return self.params.get("timeframe", "1m")

    def generate_signals(self, df_1m: pd.DataFrame) -> pd.Series:
        """
        Public interface — called by the engine with 1m data.

        Handles resampling internally based on resample_mode:
          "bars"       — resamples df to target TF, generates signals on
                         resampled bars, returns signals on resampled index.
                         Engine executes on resampled bars.
          "indicators" — generates signals on resampled bars, then aligns
                         them safely back to 1m index for 1m execution.
        """
        tf = self.timeframe

        if tf == "1m":
            # No resampling needed
            return self._generate_on_tf(df_1m)

        if self.resample_mode == "bars":
            # Mode 1: strategy runs entirely on resampled bars
            df_htf = resample_ohlcv(df_1m, tf)
            return self._generate_on_tf(df_htf)

        elif self.resample_mode == "indicators":
            # Mode 2: indicators on HTF, execution on 1m
            df_htf       = resample_ohlcv(df_1m, tf)
            signals_htf  = self._generate_on_tf(df_htf)
            signals_1m   = align_signals_to_1m(signals_htf, df_1m)
            return signals_1m

        else:
            raise ValueError(
                f"Unknown resample_mode '{self.resample_mode}'. "
                f"Use 'bars' or 'indicators'."
            )

    @abstractmethod
    def _generate_on_tf(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signals on the (already resampled) DataFrame.
        Return a Series of +1 / -1 / 0 aligned to df.index.
        """
        ...

    def get_exit_decision(self, position, bar) -> list:
        """
        Called by the generic engine on every bar while in a position.

        Args:
            position: PositionState — current position state
            bar:      BarData — current bar OHLCV + extra HTF data

        Returns:
            List of LotExitDecision objects. Empty list = hold all lots.
            Stop exits should have priority=0 so they cascade to all lots.
            Target/runner exits should have priority=1+.

        Default implementation returns [] (hold) — override in strategies
        that use the generic engine. Strategies using simulate_trades_fgbl
        do NOT need to implement this.
        """
        return []

    @property
    def uses_generic_engine(self) -> bool:
        """
        Return True to route through simulate_trades_generic.
        Default False = use legacy simulate_trades_fgbl.
        Override in strategies that implement get_exit_decision().
        """
        return False


# ---------------------------------------------------------------------------
# EMA Crossover
# ---------------------------------------------------------------------------

class EMACrossover(Strategy):
    """
    Dual EMA crossover strategy.

    Params:
        fast       (int): Fast EMA period. Default 10.
        slow       (int): Slow EMA period. Default 30.
        direction  (str): 'long_only', 'short_only', or 'both'. Default 'both'.
        timeframe  (str): Bar timeframe e.g. '1m', '60m', 'daily'. Default '1m'.

    resample_mode = "bars":
        Signals and execution both happen on the resampled TF bars.
        Use when you want a clean higher-TF backtest.

    Set resample_mode = "indicators" if you want 1m fills with HTF signals.
    """

    resample_mode = "bars"

    def _defaults(self) -> dict:
        return {
            "fast":      10,
            "slow":      30,
            "direction": "both",
            "timeframe": "1m",
        }

    def _generate_on_tf(self, df: pd.DataFrame) -> pd.Series:
        fast = df["close"].ewm(span=self.params["fast"], adjust=False).mean()
        slow = df["close"].ewm(span=self.params["slow"], adjust=False).mean()

        signals = pd.Series(0, index=df.index, dtype=int)

        direction = self.params["direction"]
        if direction in ("both", "long_only"):
            signals[fast > slow] = 1
        if direction in ("both", "short_only"):
            signals[fast < slow] = -1

        # Zero out warmup bars
        warmup = self.params["slow"]
        signals.iloc[:warmup] = 0

        return signals
