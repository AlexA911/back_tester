# Directory refactor
"""Smoke checks for package imports and project paths after reorganising."""
import importlib
import os
from pathlib import Path
import pickle
import unittest
from uuid import uuid4

import numpy as np
import pandas as pd

from backtester.core.backtest import run_backtest
from backtester.paths import PROJECT_ROOT
from backtester.strategies.base import EMACrossover


class StructureTests(unittest.TestCase):
    def test_modules_import_without_running_commands(self):
        for path in (PROJECT_ROOT / "backtester").rglob("*.py"):
            if path.name == "__main__.py":
                continue
            module = ".".join(path.relative_to(PROJECT_ROOT).with_suffix("").parts)
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_paths_stay_in_project_from_another_working_directory(self):
        previous = Path.cwd()
        try:
            os.chdir(PROJECT_ROOT / "tests")
            from backtester.core import best_params, cache_manager, exporter
            from backtester.cli import _load_named_periods

            self.assertEqual(cache_manager._DATA_DIR, PROJECT_ROOT / "data")
            self.assertEqual(exporter.RESULTS_DIR, PROJECT_ROOT / "results")
            self.assertEqual(best_params.BEST_PARAMS_DIR,
                             PROJECT_ROOT / "results" / "best_params")
            self.assertIsInstance(_load_named_periods(), list)
        finally:
            os.chdir(previous)

    def test_synthetic_backtest_with_pickled_strategy(self):
        # Windows grid workers serialise strategies using their module paths.
        strategy = pickle.loads(pickle.dumps(EMACrossover({"fast": 3, "slow": 8})))
        path = PROJECT_ROOT / "tests" / f"synthetic_{uuid4().hex}.csv"
        try:
            close = 100 + np.sin(np.arange(240) / 8) * 5
            pd.DataFrame({
                "date": pd.date_range("2026-01-01", periods=len(close), freq="min"),
                "open": close, "high": close + 1, "low": close - 1,
                "close": close, "volume": 100,
            }).to_csv(path, index=False)
            result = run_backtest({
                "symbol": "SYNTHETIC", "train": str(path),
                "date_format": "%Y-%m-%d %H:%M:%S",
                "tick_size": 0.25, "tick_value": 1.0,
            }, strategy)
        finally:
            path.unlink(missing_ok=True)
        self.assertNotIn("error", result)
        self.assertGreater(result["num_trades"], 0)
        self.assertTrue(np.isfinite(result["total_pnl"]))


if __name__ == "__main__":
    unittest.main()
