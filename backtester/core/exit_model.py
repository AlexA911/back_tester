# Directory refactor
"""
exit_model.py — Shared exit decision types for the generic engine.

Strategies implement get_exit_decision() returning one or more LotExitDecision
objects. The generic engine executes them mechanically.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class LotExitDecision:
    """
    Instruction to exit a specific lot at a specific price.

    Attributes:
        lot_index:   0-based lot index (0=first, 1=second, etc.)
        exit_price:  Exact fill price (already slippage-adjusted by strategy)
        reason:      Human-readable exit reason string (e.g. "tp1", "trend_flip", "stop")
        priority:    Lower = checked first. Stop=0, targets=1, runners=2.
                     When multiple exits fire on the same bar, lowest priority wins.
    """
    lot_index:  int
    exit_price: float
    reason:     str
    priority:   int = 1


@dataclass
class PositionState:
    """
    Current state of an open position — passed to get_exit_decision() each bar.

    Attributes:
        side:        +1 for long, -1 for short
        entry_price: Fill price at entry
        entry_bar:   Bar index of entry
        entry_time:  Timestamp of entry
        bars_held:   Number of bars since entry (current bar - entry_bar)
        lots_open:   List of booleans, one per lot. True = lot still open.
        lot_entry_prices: Entry price per lot (same as entry_price unless scaled in)
        high_since_entry: Best high seen since entry (for trailing stops)
        low_since_entry:  Best low seen since entry
        user_state:  Dict for strategy to store arbitrary state across bars
                     (e.g. whether runner has activated, best trailing price, etc.)
    """
    side:              int
    entry_price:       float
    entry_bar:         int
    entry_time:        object
    bars_held:         int
    lots_open:         list[bool]
    lot_entry_prices:  list[float]
    high_since_entry:  float
    low_since_entry:   float
    user_state:        dict = field(default_factory=dict)


@dataclass
class BarData:
    """
    OHLCV data for the current bar — passed to get_exit_decision().
    Strategies use this to check stops, targets, trend flips etc.
    """
    index:      int
    timestamp:  object
    open:       float
    high:       float
    low:        float
    close:      float
    volume:     float = 0.0
    extra:      dict  = field(default_factory=dict)  # HTF indicators etc.
