"""
cache_status.py — Inspect the state of the backtest data cache.

Usage:
    python cache_status.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datasets import DATASETS
from cache_manager import cache_status

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  BACKTEST CACHE STATUS")
    print("=" * 60)
    cache_status(DATASETS)
    print("  To build:  python cache_build.py")
    print("  To update: python cache_update.py")
    print()
