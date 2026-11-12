# Directory refactor
"""
cache_status.py — Inspect the state of the backtest data cache.

Usage:
    python -m backtester.tools.cache_status
"""

from __future__ import annotations



from backtester.config.datasets import DATASETS
from backtester.core.cache_manager import cache_status

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  BACKTEST CACHE STATUS")
    print("=" * 60)
    cache_status(DATASETS)
    print("  To build:  python -m backtester.tools.cache_build")
    print("  To update: python -m backtester.tools.cache_update")
    print()
