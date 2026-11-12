# Directory refactor
"""
Quick diagnostic — run on ES train data and report why trades are being skipped.
Run from the project root: python -m backtester.tools.diagnose_gaps
"""
import numpy as np
from backtester.config.datasets import DATASETS
from backtester.core.backtest import load_data, run_backtest
from backtester.strategies.momentum_ignition import MomentumIgnition
from backtester.core import engine


def main():

    # ── config ──────────────────────────────────────────────────────────────────
    SYMBOL       = "ES"
    SPLIT        = "train"
    GAP_TICKS    = 4      # change to test different tolerances
    # ────────────────────────────────────────────────────────────────────────────

    skip_counts = {
        "long_through_stop":  0,
        "long_excess_gap":    0,
        "long_filled":        0,
        "short_through_stop": 0,
        "short_excess_gap":   0,
        "short_filled":       0,
    }
    gap_sizes_long  = []
    gap_sizes_short = []

    _orig_simulate = engine.simulate_trades

    def _instrumented(df, signals, exit_params, tick_size, tick_value, trade_size, commission_per_side):
        opens = df["open"].values
        sig   = signals.values

        if exit_params is not None and "entry_price" in exit_params.columns:
            entry_prices_arr = exit_params["entry_price"].values
            stops_arr        = exit_params["stop_price"].values
            gap_arr          = exit_params["entry_gap_ticks"].values \
                               if "entry_gap_ticks" in exit_params.columns else None

            for i in range(1, len(sig)):
                prev_signal = sig[i - 1]
                if prev_signal == 0:
                    continue
                bar_open       = opens[i]
                intended_entry = float(entry_prices_arr[i - 1])
                stop_price     = float(stops_arr[i - 1])
                gap_ticks      = float(gap_arr[i - 1]) if gap_arr is not None else 0.0
                gap_limit      = gap_ticks * tick_size

                if prev_signal == 1:
                    gap_sizes_long.append(bar_open - intended_entry)
                    if bar_open < stop_price:
                        skip_counts["long_through_stop"] += 1
                    elif bar_open > intended_entry + gap_limit:
                        skip_counts["long_excess_gap"] += 1
                    else:
                        skip_counts["long_filled"] += 1
                else:
                    gap_sizes_short.append(intended_entry - bar_open)
                    if bar_open > stop_price:
                        skip_counts["short_through_stop"] += 1
                    elif bar_open < intended_entry - gap_limit:
                        skip_counts["short_excess_gap"] += 1
                    else:
                        skip_counts["short_filled"] += 1

        return _orig_simulate(df, signals, exit_params, tick_size, tick_value,
                              trade_size, commission_per_side)

    engine.simulate_trades = _instrumented

    # Load data
    dataset   = DATASETS[SYMBOL]
    path      = dataset[SPLIT]
    df        = load_data(path, dataset["date_format"])
    tick_size = dataset["tick_size"]

    strategy = MomentumIgnition(params={"entry_gap_ticks": GAP_TICKS})
    raw      = strategy.generate_signals(df)
    signals     = raw["signals"]
    exit_params = raw["exit_params"]

    trades = engine.simulate_trades(
        df=df, signals=signals, exit_params=exit_params,
        tick_size=tick_size, tick_value=dataset["tick_value"],
        trade_size=1, commission_per_side=2.0
    )

    total = sum(skip_counts.values())
    print(f"\n{'='*55}")
    print(f"  Gap diagnostic — {SYMBOL} {SPLIT} | entry_gap_ticks={GAP_TICKS}")
    print(f"{'='*55}")
    print(f"  Total signals fired : {total:,}")
    print(f"\n  Long signals:")
    print(f"    Through stop   : {skip_counts['long_through_stop']:>5}  ({skip_counts['long_through_stop']/max(1,total)*100:.1f}%)")
    print(f"    Excess gap     : {skip_counts['long_excess_gap']:>5}  ({skip_counts['long_excess_gap']/max(1,total)*100:.1f}%)")
    print(f"    Filled         : {skip_counts['long_filled']:>5}  ({skip_counts['long_filled']/max(1,total)*100:.1f}%)")
    print(f"\n  Short signals:")
    print(f"    Through stop   : {skip_counts['short_through_stop']:>5}  ({skip_counts['short_through_stop']/max(1,total)*100:.1f}%)")
    print(f"    Excess gap     : {skip_counts['short_excess_gap']:>5}  ({skip_counts['short_excess_gap']/max(1,total)*100:.1f}%)")
    print(f"    Filled         : {skip_counts['short_filled']:>5}  ({skip_counts['short_filled']/max(1,total)*100:.1f}%)")

    for label, gaps in [("Long  (bar_open - intended)", gap_sizes_long),
                        ("Short (intended - bar_open)", gap_sizes_short)]:
        if not gaps:
            continue
        gs = np.array(gaps)
        print(f"\n  {label}:")
        print(f"    min={gs.min():.2f}  p25={np.percentile(gs,25):.2f}  "
              f"median={np.median(gs):.2f}  p75={np.percentile(gs,75):.2f}  max={gs.max():.2f}")
        print(f"    Fill rate by tolerance:")
        for t in [0, 1, 2, 4, 8, 16, 32]:
            pct = (gs <= t * tick_size).mean() * 100
            marker = " ◄" if t == GAP_TICKS else ""
            print(f"      <= {t:>2} ticks : {pct:5.1f}%{marker}")

    print(f"\n  Actual trades filled : {len(trades):,}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
