"""Check losing streak calculation against trade log CSV."""
import sys, os, glob
import pandas as pd
import numpy as np

# Find most recent trade log
results_dir = r"results"
folders = sorted(glob.glob(os.path.join(results_dir, "*")))
if not folders:
    print("No results folders found — run from backtest directory")
    sys.exit(1)

latest = folders[-1]
logs = glob.glob(os.path.join(latest, "trades_*.csv"))
if not logs:
    print(f"No trade log in {latest}")
    sys.exit(1)

df = pd.read_csv(logs[0])
df = df.sort_values("entry_time").reset_index(drop=True)
print(f"Loaded {len(df)} trades from {logs[0]}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 net_pnl values: {df['net_pnl'].head().tolist()}")
print(f"net_pnl dtype: {df['net_pnl'].dtype}")
print(f"Negative count : {(df['net_pnl'] < 0).sum()}")
print(f"Zero count     : {(df['net_pnl'] == 0).sum()}")
print(f"Positive count : {(df['net_pnl'] > 0).sum()}")

# Manual streak calculation
lose_streak = cur_streak = 0
streaks = []
for p in df['net_pnl']:
    if p < 0:
        cur_streak += 1
        lose_streak = max(lose_streak, cur_streak)
    else:
        if cur_streak > 0:
            streaks.append(cur_streak)
        cur_streak = 0
if cur_streak > 0:
    streaks.append(cur_streak)

print(f"\nMax consecutive losses : {lose_streak}")
print(f"All losing streaks (len>3): {sorted([s for s in streaks if s > 3], reverse=True)[:20]}")

# Check win rate
wr = (df['net_pnl'] > 0).mean()
print(f"\nWin rate from CSV: {wr:.1%}")
print(f"Expected max streak at {wr:.0%} WR over {len(df)} trades: "
      f"~{int(np.log(len(df)) / np.log(1/(1-wr)))}")
