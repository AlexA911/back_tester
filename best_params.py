"""
best_params.py — Persistent storage of best grid search parameters.

Files are saved under:
  results/best_params/<StrategyName>_<ASSET1>_<ASSET2>_<split>.json

Each file stores the best shared param set from the most recent grid run
on that strategy/asset combination/split, along with key metrics and
the timestamp of the run it came from.

Usage:
    from best_params import save_best_params, load_best_params, list_best_params
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """Handles numpy int/float types that standard json can't serialize."""
    def default(self, obj):
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

BEST_PARAMS_DIR = Path("results") / "best_params"


def _key(strategy_name: str, assets: list[str], split: str) -> str:
    """Canonical filename key for a strategy/asset/split combination."""
    assets_str = "_".join(sorted(assets))
    return f"{strategy_name}_{assets_str}_{split}"


def _filepath(strategy_name: str, assets: list[str], split: str) -> Path:
    return BEST_PARAMS_DIR / f"{_key(strategy_name, assets, split)}.json"


def save_best_params(
    strategy_name: str,
    assets: list[str],
    split: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    timestamp: str,
) -> None:
    BEST_PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    path = _filepath(strategy_name, assets, split)

    new_pf = float(metrics.get("profit_factor", 0))

    # Only overwrite if new result is better than existing best
    existing = load_best_params(strategy_name, assets, split)
    if existing is not None:
        existing_pf = existing.get("metrics", {}).get("profit_factor", 0)
        if new_pf <= existing_pf:
            return  # Keep the better result

    data = {
        "strategy":     strategy_name,
        "assets":       sorted(assets),
        "split":        split,
        "saved_at":     timestamp,
        "params":       params,
        "metrics": {
            "profit_factor": round(float(metrics.get("profit_factor", 0)), 4),
            "num_trades":    int(metrics.get("num_trades", 0)),
            "total_pnl":     round(float(metrics.get("total_pnl", 0)), 2),
            "win_rate":      round(float(metrics.get("win_rate", 0)), 4),
            "sharpe":        round(float(metrics.get("sharpe", 0)), 4),
            "max_drawdown":  round(float(metrics.get("max_drawdown", 0)), 2),
        },
    }

    # Write to a temp file first, then rename — prevents corrupt files on crash
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)
    tmp.replace(path)


def load_best_params(
    strategy_name: str,
    assets: list[str],
    split: str,
) -> dict[str, Any] | None:
    path = _filepath(strategy_name, assets, split)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Corrupt or unreadable file — delete and start fresh
        path.unlink(missing_ok=True)
        return None


def list_best_params(strategy_name: str | None = None) -> list[dict[str, Any]]:
    """
    List all saved best param files, optionally filtered by strategy.
    Returns list of data dicts sorted by saved_at descending.
    """
    BEST_PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for path in BEST_PARAMS_DIR.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
            if strategy_name is None or data.get("strategy") == strategy_name:
                results.append(data)
        except Exception:
            continue
    return sorted(results, key=lambda d: d.get("saved_at", ""), reverse=True)
