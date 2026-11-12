# Directory refactor
"""
cache_update.py — Incrementally update the data cache with new raw data.

Run this after your colleague provides updated CSVs.
Only reprocesses the new tail — does not rebuild the entire cache.

Workflow:
    1. Receive updated CSVs from colleague
    2. Replace files in data/raw/ (or data/ if using the existing structure)
    3. Run: python -m backtester.tools.cache_update --assets FGBL FBTP

Usage:
    # Update all cached assets
    python -m backtester.tools.cache_update

    # Update specific assets
    python -m backtester.tools.cache_update --assets FGBL FBTP R

    # Update specific timeframes only
    python -m backtester.tools.cache_update --assets FGBL --timeframes 120

    # Force full rebuild instead of incremental
    python -m backtester.tools.cache_update --rebuild
"""

from __future__ import annotations

import argparse
from pathlib import Path


from backtester.config.datasets import DATASETS
from backtester.core.cache_manager import (
    update_cache, build_cache, cache_is_fresh,
    cache_status, _load_index, _cache_path
)


DEFAULT_TIMEFRAMES = [1, 120]


def main():
    parser = argparse.ArgumentParser(
        description="Incrementally update the backtest data cache with new raw data."
    )
    parser.add_argument(
        "--assets", nargs="+", metavar="SYMBOL",
        help="Asset symbols to update (default: all assets with existing cache)"
    )
    parser.add_argument(
        "--timeframes", nargs="+", type=int, default=DEFAULT_TIMEFRAMES,
        metavar="TF",
        help=f"Timeframes in minutes to update (default: {DEFAULT_TIMEFRAMES})"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force full rebuild instead of incremental update"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show cache status and exit"
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  BACKTEST CACHE UPDATER")
    print("=" * 60)

    if args.status:
        cache_status(DATASETS)
        return

    # Determine which assets to process
    index = _load_index()

    if args.assets:
        symbols = [s.upper() for s in args.assets]
        invalid = [s for s in symbols if s not in DATASETS]
        if invalid:
            print(f"\n  ERROR: Unknown asset(s): {invalid}")
            sys.exit(1)
    else:
        # Default: all assets that have any cached timeframe
        symbols = [k for k in sorted(DATASETS.keys()) if k in index]
        if not symbols:
            print("\n  No cached assets found. Run cache_build.py first.")
            sys.exit(0)

    print(f"\n  Assets     : {', '.join(symbols)}")
    print(f"  Timeframes : {args.timeframes} minutes")
    print(f"  Mode       : {'full rebuild' if args.rebuild else 'incremental'}")
    print()

    updated = 0
    skipped = 0
    errors  = 0

    for symbol in symbols:
        dataset     = DATASETS.get(symbol, {})
        full_path   = dataset.get("full")
        date_format = dataset.get("date_format", "%d/%m/%Y %H:%M")

        if not full_path:
            print(f"  {symbol}: no 'full' path — skipping")
            skipped += 1
            continue

        if not Path(full_path).exists():
            print(f"  {symbol}: file not found: {full_path} — skipping")
            skipped += 1
            continue

        for tf in args.timeframes:
            # Check if cache exists at all
            cp = _cache_path(symbol, tf)
            if not cp.exists():
                print(f"  {symbol} {tf}m: no cache — building from scratch.")
                try:
                    build_cache(symbol, tf, full_path, date_format, verbose=True)
                    updated += 1
                except Exception as e:
                    print(f"  {symbol} {tf}m: ERROR — {e}")
                    errors += 1
                continue

            # Already fresh?
            if not args.rebuild and cache_is_fresh(symbol, tf, full_path):
                print(f"  {symbol} {tf}m: already up to date — skipping")
                skipped += 1
                continue

            try:
                if args.rebuild:
                    build_cache(symbol, tf, full_path, date_format, verbose=True)
                else:
                    result = update_cache(symbol, tf, full_path, date_format, verbose=True)
                    if result is None:
                        # Fallback to full build
                        build_cache(symbol, tf, full_path, date_format, verbose=True)
                updated += 1
            except Exception as e:
                print(f"  {symbol} {tf}m: ERROR — {e}")
                errors += 1

    print()
    print("=" * 60)
    print(f"  Done.  Updated: {updated}  Skipped: {skipped}  Errors: {errors}")
    print("=" * 60)

    # Show final status
    print()
    cache_status(DATASETS)


if __name__ == "__main__":
    main()
