<!-- Directory refactor -->
# Backtester

Run strategy backtests, search parameter grids, and export trade logs.

## Layout

```text
backtester/
    cli.py             Interactive menus and run configuration
    core/              Simulation, caching, resampling, search, and exports
    strategies/        Base strategy, EMA crossover, and trading strategies
    config/            Dataset definitions, sectors, and optional named periods
    tools/             Cache maintenance and diagnostic commands
    paths.py           Shared project paths
data/                  Existing cache index, caches, and signal files
results/               Exported runs and saved best parameters
tests/                 Import, path, and backtest smoke checks
run.py                 Interactive entry point
requirements.txt       Runtime dependencies
```

## Setup and run

From the project root, with your virtual environment activated:

```powershell
python -m pip install -r requirements.txt
python run.py
```

You can also launch with `python -m backtester`.

Set `BASE_DIR` in `backtester/config/datasets.py` to your market data directory.
The existing machine-specific setting is preserved. Dataset definitions and
sector groupings live in that same configuration directory. Interactive run
defaults live near the top of `backtester/cli.py`.

Optional named periods belong in `backtester/config/named_periods.json`.
The COT signal file belongs at `data/ZN_refined_signal.csv`.
Cache and result paths are anchored to this project, regardless of the current
working directory. Existing data and results have been kept in place.

## Maintenance tools

Run tools as modules from the project root:

```powershell
python -m backtester.tools.cache_status
python -m backtester.tools.cache_build --assets FGBL --timeframes 1 120
python -m backtester.tools.cache_update --assets FGBL
python -m backtester.tools.diagnose_gaps
python -m backtester.tools.diagnose_streaks
```

Cache build and update commands support `--help`. Diagnostics require market
data or exported trade logs; gap diagnostic settings are at the start of its
`main()` function.

For Python integrations, use package imports, for example:

```python
from backtester.core.backtest import run_backtest
from backtester.strategies.momentum_ignition import MomentumIgnition
```

## Checks

```powershell
python -m unittest discover -s tests -v
```

Checks use synthetic data and do not require your market data files.
