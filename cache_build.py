"""
cache_build.py — Build (or rebuild) the resampled data cache.

Run this once per asset to pre-bake resampled OHLCV files.
Subsequent backtests will load from cache instead of resampling on the fly,
dramatically reducing backtest runtime (from ~30s to ~1s per run).

Usage:
    # Build all assets for 120m (most common)
    python cache_build.py

    # Build specific assets
    python cache_build.py --assets FGBL FBTP ZN

    # Build multiple timeframes
    python cache_build.py --assets FGBL --timeframes 1 60 120 240

    # Force rebuild even if cache is fresh
    python cache_build.py --force

    # Show current cache status only
    python cache_build.py --status
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from datasets import DATASETS
from cache_manager import build_cache, cache_is_fresh, cache_status


DEFAULT_TIMEFRAMES = [1, 120]


def main():
    parser = argparse.ArgumentParser(
        description="Build the backtest data cache for faster backtesting."
    )
    parser.add_argument(
        "--assets", nargs="+", metavar="SYMBOL",
        help="Asset symbols to build (default: all assets with a 'full' CSV)"
    )
    parser.add_argument(
        "--timeframes", nargs="+", type=int, default=DEFAULT_TIMEFRAMES,
        metavar="TF",
        help=f"Timeframes in minutes to build (default: {DEFAULT_TIMEFRAMES})"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force rebuild even if cache is already fresh"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show cache status and exit without building"
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  BACKTEST CACHE BUILDER")
    print("=" * 60)

    if args.status:
        cache_status(DATASETS)
        return

    # Determine which assets to process
    if args.assets:
        symbols = [s.upper() for s in args.assets]
        invalid = [s for s in symbols if s not in DATASETS]
        if invalid:
            print(f"\n  ERROR: Unknown asset(s): {invalid}")
            print(f"  Available: {sorted(DATASETS.keys())}")
            sys.exit(1)
    else:
        # Default: all assets that have a full CSV defined
        symbols = [
            k for k, v in sorted(DATASETS.items())
            if v.get("full") and Path(v["full"]).exists()
        ]
        if not symbols:
            print("\n  No assets found with a 'full' CSV path.")
            print("  Add 'full' paths to datasets.py or specify --assets explicitly.")
            sys.exit(1)

    print(f"\n  Assets     : {', '.join(symbols)}")
    print(f"  Timeframes : {args.timeframes} minutes")
    print(f"  Force      : {'yes' if args.force else 'no (skip if fresh)'}")
    print()

    built = 0
    skipped = 0
    errors = 0

    for symbol in symbols:
        dataset = DATASETS[symbol]
        full_path = dataset.get("full")
        date_format = dataset.get("date_format", "%d/%m/%Y %H:%M")

        if not full_path:
            print(f"  {symbol}: no 'full' path defined — skipping")
            skipped += 1
            continue

        if not Path(full_path).exists():
            print(f"  {symbol}: file not found: {full_path} — skipping")
            skipped += 1
            continue

        for tf in args.timeframes:
            if not args.force and cache_is_fresh(symbol, tf, full_path):
                print(f"  {symbol} {tf}m: cache is fresh — skipping (use --force to rebuild)")
                skipped += 1
                continue

            try:
                build_cache(symbol, tf, full_path, date_format, verbose=True)
                built += 1
            except Exception as e:
                print(f"  {symbol} {tf}m: ERROR — {e}")
                errors += 1

    print()
    print("=" * 60)
    print(f"  Done.  Built: {built}  Skipped: {skipped}  Errors: {errors}")
    print("=" * 60)
    print()

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
