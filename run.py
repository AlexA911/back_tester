"""
run.py — Fully interactive backtest entry point.

Flow:
  1. Select assets by sector
  2. Select data split (train / test / full)
  3. Select timeframe
  4. Select mode (single / grid)
  5. If grid → param range prompts
  6. Results printed
  7. Auto-export to results/<timestamp>/
  8. Optional trade validation — inspect first N trades with bar context
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from datasets import DATASETS
from sectors import SECTORS
from backtest import run_backtest
from grid_search import run_grid_search
from strategy import EMACrossover
from momentum_ignition import MomentumIgnition
from fgbl_momentum import FGBLMomentum
from resampler import TIMEFRAME_MAP
from exporter import export_results
from validator import run_validation
from best_params import save_best_params, load_best_params, list_best_params


# ============================================================================
# FIXED PARAMS — change STRATEGY_CLASS to switch strategy
# ============================================================================

STRATEGY_CLASS = MomentumIgnition   # switch to FGBLMomentum or EMACrossover to change strategy
TRADE_SIZE     = 1
COMMISSION     = 2.0
RANK_BY        = "profit_factor"
MIN_TRADES     = 30
TOP_N          = 20

SINGLE_PARAMS  = {}   # empty = use strategy defaults

# ============================================================================


def _dedupe(lst: list[str]) -> list[str]:
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]


def _section(title: str) -> None:
    print()
    print("=" * 55)
    print(f"  {title}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# 1. Asset selection
# ---------------------------------------------------------------------------

def select_assets() -> list[str]:
    sector_names = list(SECTORS.keys())

    _section("ASSET SELECTION")
    print(f"  {'#':<4} {'Sector':<24} Assets")
    print("-" * 55)
    for i, (sector, assets) in enumerate(SECTORS.items(), start=1):
        available = [a for a in assets if a in DATASETS]
        print(f"  {i:<4} {sector:<24} {', '.join(available)}")
    print("-" * 55)
    print(f"  {'A':<4} {'ALL SECTORS':<24} Run full universe")
    print(f"  {'C':<4} {'CUSTOM':<24} Type asset codes manually")
    print("=" * 55)
    print()

    choice = input("  Enter sector numbers (e.g. 1,3,5), A for all, or C for custom: ").strip()

    if choice.upper() == "A":
        selected = [a for assets in SECTORS.values() for a in assets if a in DATASETS]
        print(f"\n  → All {len(selected)} available assets selected.")
        return _dedupe(selected)

    if choice.upper() == "C":
        raw     = input("  Enter asset codes (e.g. ES,NQ,GC): ").strip()
        codes   = [c.strip().upper() for c in raw.split(",") if c.strip()]
        valid   = [c for c in codes if c in DATASETS]
        invalid = [c for c in codes if c not in DATASETS]
        if invalid:
            print(f"\n  WARNING: Unknown assets ignored: {invalid}")
        print(f"\n  → {len(valid)} asset(s) selected: {valid}")
        return _dedupe(valid)

    selected = []
    try:
        indices = [int(x.strip()) for x in choice.split(",") if x.strip()]
    except ValueError:
        print("  Invalid input — no assets selected.")
        return []

    for idx in indices:
        if 1 <= idx <= len(sector_names):
            sector = sector_names[idx - 1]
            assets = [a for a in SECTORS[sector] if a in DATASETS]
            selected.extend(assets)
            print(f"  → {sector}: {assets}")
        else:
            print(f"  WARNING: Sector {idx} out of range — skipped.")

    result = _dedupe(selected)
    print(f"\n  → {len(result)} asset(s) selected total.")
    return result


# ---------------------------------------------------------------------------
# 2. Split selection
# ---------------------------------------------------------------------------

def select_split() -> str:
    _section("DATA SPLIT")
    options = ["train", "test", "both"]
    print("  1   train")
    print("  2   test")
    print("  3   both  (grid on train → top 3 validated on test)")
    print()
    choice = input("  Select split [1]: ").strip() or "1"
    split  = options[int(choice) - 1] if choice in ("1", "2", "3") else "train"
    print(f"\n  → Split: {split}")
    return split


# ---------------------------------------------------------------------------
# 3. Timeframe selection
# ---------------------------------------------------------------------------

def select_timeframe() -> str:
    _section("TIMEFRAME")
    timeframes = ["1m", "5m", "15m", "30m", "60m", "120m", "240m", "daily"]
    for i, tf in enumerate(timeframes, start=1):
        note = "no resample" if tf == "1m" else f"resample 1m → {tf}"
        print(f"  {i}   {tf:<8} {note}")
    print()

    choice = input("  Select timeframe [1]: ").strip() or "1"
    try:
        tf = timeframes[int(choice) - 1]
    except (ValueError, IndexError):
        print("  Invalid — defaulting to 1m.")
        tf = "1m"

    print(f"\n  → Timeframe: {tf}  |  Resample mode: {STRATEGY_CLASS.resample_mode}")
    return tf


# ---------------------------------------------------------------------------
# 4. Mode selection
# ---------------------------------------------------------------------------

def select_mode() -> str:
    _section("RUN MODE")
    print("  1   Single run  — one param set, full per-asset results")
    print("  2   Grid search — exhaustive param sweep, shared + per-asset best")
    print()
    choice = input("  Select mode [1]: ").strip() or "1"
    mode   = "single" if choice == "1" else "grid"
    print(f"\n  → Mode: {mode}")
    return mode



# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

# Registry — add new strategies here
STRATEGY_REGISTRY = {
    "1": ("MomentumIgnition", "MomentumIgnition"),
    "2": ("FGBLMomentum",     "FGBLMomentum"),
}

def select_strategy():
    """Prompt user to select a strategy and return the class."""
    # Build registry with loaded classes so we can show descriptions
    from momentum_ignition import MomentumIgnition as _MI
    from fgbl_momentum import FGBLMomentum as _FGBL
    _registry = [
        ("MomentumIgnition", _MI),
        ("FGBLMomentum",     _FGBL),
    ]

    _section("STRATEGY")
    for i, (name, cls) in enumerate(_registry, 1):
        desc = getattr(cls, "description", "")
        print(f"  {i}   {name}")
        if desc:
            for line in desc.strip().split("\n"):
                print(f"        {line.strip()}")
        print()

    choice = input(f"  Select strategy [1]: ").strip() or "1"
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(_registry):
            name, cls = _registry[idx]
        else:
            name, cls = _registry[0]
    except ValueError:
        name, cls = _registry[0]

    print(f"  → Strategy: {name}")
    return cls


# ---------------------------------------------------------------------------
# 5. Grid param prompts
# ---------------------------------------------------------------------------

def _safe_float_range(p_min, p_max, p_step, fallback_step):
    """Build a float range, guarding against step=0 and min>max."""
    if p_min >= p_max - 1e-9:    # min==max or min>max: single value
        return [round(p_min, 10)]
    p_step = max(abs(p_step), 1e-9) if p_step else fallback_step
    vals = []
    v = p_min
    while v <= p_max + 1e-9:
        vals.append(round(v, 10))
        v += p_step
        if len(vals) > 10000:
            break
    return vals if vals else [p_min]


def _safe_int_range(p_min, p_max, p_step):
    """Build an int range, guarding against step=0 and min>max."""
    if p_min >= p_max:           # min==max or min>max: single value
        return [p_min]
    p_step = max(abs(p_step), 1) if p_step else 1
    return list(range(p_min, p_max + 1, p_step))


def _ml_defaults_for_n(n_lots: int, prior_params: dict | None = None) -> dict:
    """Build default multi-lot param dict for n lots (all size=1).
    Uses prior_params (best saved) first, then default_lot_config, then strategy defaults.
    """
    d   = STRATEGY_CLASS().params
    dlc = getattr(STRATEGY_CLASS, "default_lot_config", None)
    p   = prior_params or {}
    out = {"multi_lot": True}
    for k in range(1, n_lots + 1):
        idx = k - 1
        cfg = dlc[idx] if (dlc and idx < len(dlc)) else {}
        out[f"lot{k}_size"]          = 1
        out[f"lot{k}_tp_points"]     = p.get(f"lot{k}_tp_points",     cfg.get("tp_points",  d.get(f"lot{k}_tp_points",  0.0)))
        out[f"lot{k}_stop_mode"]     = p.get(f"lot{k}_stop_mode",     cfg.get("stop_mode",  d.get(f"lot{k}_stop_mode",  "breakeven")))
        out[f"lot{k}_exit_modes"]    = p.get(f"lot{k}_exit_modes",    cfg.get("exit_modes", d.get(f"lot{k}_exit_modes",  "ema_flip,time_exit")))
        out[f"lot{k}_atr_stop_mult"] = p.get(f"lot{k}_atr_stop_mult", cfg.get("atr_stop_mult", 0.0))
    # disable unused lots
    for k in range(n_lots + 1, 4):
        out[f"lot{k}_size"] = 0
    out["runner_ema_fast"]        = p.get("runner_ema_fast",       d.get("runner_ema_fast", 9))
    out["runner_ema_slow"]        = p.get("runner_ema_slow",       d.get("runner_ema_slow", 21))
    out["runner_time_bars"]       = p.get("runner_time_bars",      d.get("runner_time_bars", 120))
    out["runner_activation_atr"]  = p.get("runner_activation_atr", d.get("runner_activation_atr", 0.0))
    return out

def _select_multi_lot_single(prior_params: dict | None = None) -> tuple[dict, int]:
    """Interactive multi-lot config for single-run mode. Returns (params_dict, trade_size)."""
    print("  -- Multi-lot --")
    # Show default lot summary for strategies with fixed lot structures
    _dlc = getattr(STRATEGY_CLASS, "default_lot_config", None)
    _dn  = getattr(STRATEGY_CLASS, "default_n_lots", None)
    if _dlc and _dn:
        # Build summary from effective params (prior best if available, else class defaults)
        _p = prior_params or {}
        def _lot_label(i, lc):
            k = i + 1
            em = _p.get(f"lot{k}_exit_modes", lc.get("exit_modes", "fixed"))
            tp = _p.get(f"lot{k}_tp_points",  lc.get("tp_points",  0.0))
            is_runner = float(tp) <= 0
            return f"Lot{k}={'runner:'+str(em) if is_runner else 'fixed:'+str(tp)+'xATR'}"
        _lot_sum = ", ".join(_lot_label(i, _dlc[i]) for i in range(_dn))
        print(f"  Default: {_dn} lots  ({_lot_sum})")
        if input("  Change multi-lot config? [y/N]: ").strip().lower() not in ("y", "yes"):
            print()
            return _ml_defaults_for_n(_dn, prior_params=prior_params), _dn
        print()
    use_ml = input("  Enable multi-lot? [y/N]: ").strip().lower() in ("y", "yes")
    if not use_ml:
        print()
        return {"multi_lot": False}, TRADE_SIZE

    _default_n = getattr(STRATEGY_CLASS, "default_n_lots", 2)
    raw_n = input(f"  How many lots? [{_default_n}]: ").strip()
    n_lots = int(raw_n) if raw_n.isdigit() and int(raw_n) >= 1 else _default_n
    print()

    defs = STRATEGY_CLASS().params
    d    = _ml_defaults_for_n(n_lots, prior_params=prior_params)
    has_runner = False

    for k in range(1, n_lots + 1):
        print(f"  Lot {k}:")
        is_runner = input(f"    Runner? [y/N]: ").strip().lower() in ("y", "yes")
        d[f"lot{k}_size"] = 1

        if is_runner:
            has_runner = True
            em_def = d[f"lot{k}_exit_modes"]
            _valid_modes = {"ema_flip", "time_exit", "fixed", "chandelier"}
            while True:
                raw_em = input(f"    exit_modes [{em_def}]: ").strip()
                em_val = raw_em if raw_em else em_def
                _modes = {m.strip() for m in em_val.split(",")}
                if _modes <= _valid_modes:
                    d[f"lot{k}_exit_modes"] = em_val
                    break
                print(f"      Invalid modes: {_modes - _valid_modes}. Valid: {sorted(_valid_modes)}")
            d[f"lot{k}_tp_points"]  = 0.0
            act_def = defs.get("runner_activation_atr", 0.0)
            raw_act = input(f"    activation_atr (N x ATR, 0=immediate) [{act_def}]: ").strip()
            d["runner_activation_atr"] = float(raw_act) if raw_act else act_def
        else:
            tp_def = d[f"lot{k}_tp_points"]
            raw_tp = input(f"    tp_points [{tp_def}]: ").strip()
            d[f"lot{k}_tp_points"] = float(raw_tp) if raw_tp else tp_def

        if k > 1:
            sm_def = d[f"lot{k}_stop_mode"]
            raw_sm = input(f"    stop_mode [{sm_def}]: ").strip()
            d[f"lot{k}_stop_mode"] = raw_sm if raw_sm else sm_def

        # ATR hard stop — tighter of EMA-21 or entry ± atr_stop_mult × ATR
        atr_stop_def = d.get(f"lot{k}_atr_stop_mult", 0.0)
        raw_atr = input(f"    atr_stop_mult (0=off) [{atr_stop_def}]: ").strip()
        d[f"lot{k}_atr_stop_mult"] = float(raw_atr) if raw_atr else atr_stop_def
        print()

    if has_runner:
        print("  -- Runner EMA params --")
        raw = input(f"  runner_ema_fast [{d['runner_ema_fast']}]: ").strip()
        d["runner_ema_fast"] = int(raw) if raw.isdigit() else d["runner_ema_fast"]
        raw = input(f"  runner_ema_slow [{d['runner_ema_slow']}]: ").strip()
        d["runner_ema_slow"] = int(raw) if raw.isdigit() else d["runner_ema_slow"]
        raw = input(f"  runner_time_bars [{d['runner_time_bars']}]: ").strip()
        d["runner_time_bars"] = int(raw) if raw.isdigit() else d["runner_time_bars"]
        print()
    else:
        d["runner_activation_atr"] = defs.get("runner_activation_atr", 0.0)

    return d, n_lots


def _select_multi_lot_grid(tick_size: float, price_params: set, prior_params: dict | None = None) -> dict:
    """Interactive multi-lot config for grid-search mode. Returns param_grid fragment."""
    print()
    print("  -- Multi-lot --")
    # Show default lot summary for strategies with fixed lot structures
    _dlc = getattr(STRATEGY_CLASS, "default_lot_config", None)
    _dn  = getattr(STRATEGY_CLASS, "default_n_lots", None)
    defs = {**STRATEGY_CLASS().params, **(prior_params or {})}
    if _dlc and _dn:
        # Build summary from effective params (prior best if available, else class defaults)
        def _lot_label(i, lc):
            k = i + 1
            em = defs.get(f"lot{k}_exit_modes", lc.get("exit_modes", "fixed"))
            tp = defs.get(f"lot{k}_tp_points",  lc.get("tp_points",  0.0))
            is_runner = float(tp) <= 0
            return f"Lot{k}={'runner:'+str(em) if is_runner else 'fixed:'+str(tp)+'xATR'}"
        _lot_sum = ", ".join(_lot_label(i, _dlc[i]) for i in range(_dn))
        print(f"  Default: {_dn} lots  ({_lot_sum})")
        if input("  Change multi-lot config? [y/N]: ").strip().lower() not in ("y", "yes"):
            print()
            out = {"multi_lot": [True]}
            for k in range(1, _dn + 1):
                cfg = _dlc[k - 1]  # class defaults as fallback
                out[f"lot{k}_size"]          = [1]
                out[f"lot{k}_tp_points"]     = [defs.get(f"lot{k}_tp_points",   cfg.get("tp_points", 0.0))]
                out[f"lot{k}_stop_mode"]     = [defs.get(f"lot{k}_stop_mode",   cfg.get("stop_mode", "breakeven"))]
                out[f"lot{k}_exit_modes"]    = [defs.get(f"lot{k}_exit_modes",  cfg.get("exit_modes", "fixed"))]
                out[f"lot{k}_atr_stop_mult"] = [defs.get(f"lot{k}_atr_stop_mult", cfg.get("atr_stop_mult", 0.0))]
            for k in range(_dn + 1, 4):
                out[f"lot{k}_size"]       = [0]
                out[f"lot{k}_tp_points"]  = [0.0]
                out[f"lot{k}_stop_mode"]  = [defs.get(f"lot{k}_stop_mode", "breakeven")]
                out[f"lot{k}_exit_modes"] = [defs.get(f"lot{k}_exit_modes", "fixed")]
            out["runner_ema_fast"]       = [defs.get("runner_ema_fast", 9)]
            out["runner_ema_slow"]       = [defs.get("runner_ema_slow", 21)]
            out["runner_time_bars"]      = [defs.get("runner_time_bars", 120)]
            out["runner_activation_atr"] = [defs.get("runner_activation_atr", 0.0)]
            return out
        print()
    use_ml = input("  Enable multi-lot? [y/N]: ").strip().lower() in ("y", "yes")

    if not use_ml:
        out = {"multi_lot": [False]}
        for mp in ["lot1_size", "lot1_tp_points",
                   "lot2_size", "lot2_tp_points", "lot2_stop_mode", "lot2_exit_modes",
                   "lot3_size", "lot3_tp_points", "lot3_stop_mode", "lot3_exit_modes",
                   "runner_ema_fast", "runner_ema_slow", "runner_time_bars",
                   "runner_activation_atr"]:
            out[mp] = [defs.get(mp)]
        return out

    _default_n = getattr(STRATEGY_CLASS, "default_n_lots", 2)
    raw_n = input(f"  How many lots? [{_default_n}]: ").strip()
    n_lots = int(raw_n) if raw_n.isdigit() and int(raw_n) >= 1 else _default_n
    print()

    out = {"multi_lot": [True]}
    has_runner = False

    for k in range(1, n_lots + 1):
        print(f"  Lot {k}:")
        is_runner = input(f"    Runner? [y/N]: ").strip().lower() in ("y", "yes")
        out[f"lot{k}_size"] = [1]

        if is_runner:
            has_runner = True
            em_def = defs.get(f"lot{k}_exit_modes", "ema_flip,time_exit")
            _valid_modes = {"ema_flip", "time_exit", "fixed", "chandelier"}
            while True:
                raw_em = input(f"    exit_modes [{em_def}]: ").strip()
                em_val = raw_em if raw_em else em_def
                # Each option may be comma-separated combination e.g. "ema_flip,time_exit"
                _all_valid = all(
                    m.strip() in _valid_modes
                    for opt in ([em_val] if "," not in em_val or raw_em == "" else [em_val])
                    for m in opt.split(",")
                )
                if _all_valid:
                    vals = [v.strip() for v in raw_em.split(",")] if raw_em else [em_def]
                    out[f"lot{k}_exit_modes"] = vals
                    break
                print(f"      Invalid mode(s). Valid: {sorted(_valid_modes)}")
            out[f"lot{k}_tp_points"]  = [0.0]
            # Activation threshold per runner lot
            act_def = defs.get("runner_activation_atr", 0.0)
            raw_min = input(f"    activation_atr min  [{act_def}]: ").strip()
            raw_max = input(f"    activation_atr max  [{act_def}]: ").strip()
            a_min   = float(raw_min) if raw_min else act_def
            a_max   = float(raw_max) if raw_max else act_def
            if a_min >= a_max - 1e-9:
                a_step = 0.5
            else:
                raw_stp = input(f"    activation_atr step [0.5]: ").strip()
                a_step  = float(raw_stp) if raw_stp else 0.5
            out["runner_activation_atr"] = _safe_float_range(a_min, a_max, a_step, 0.5)
        else:
            tp_def  = defs.get(f"lot{k}_tp_points", 2.0)
            raw_min = input(f"    tp_atr_mult min  [{tp_def}]: ").strip()
            raw_max = input(f"    tp_atr_mult max  [{tp_def}]: ").strip()
            p_min   = float(raw_min) if raw_min else tp_def
            p_max   = float(raw_max) if raw_max else tp_def
            if p_min >= p_max - 1e-9:
                p_step = tick_size
            else:
                raw_stp = input(f"    tp_atr_mult step [{tick_size}]: ").strip()
                p_step  = float(raw_stp) if raw_stp else tick_size
            out[f"lot{k}_tp_points"]  = _safe_float_range(p_min, p_max, p_step, tick_size)
            out[f"lot{k}_exit_modes"] = [defs.get(f"lot{k}_exit_modes", "fixed")]

        if k > 1:
            sm_def  = defs.get(f"lot{k}_stop_mode", "breakeven")
            raw_sm  = input(f"    stop_mode [{sm_def}]: ").strip()
            sm_vals = [v.strip() for v in raw_sm.split(",")] if raw_sm else [sm_def]
            out[f"lot{k}_stop_mode"] = sm_vals
        else:
            out[f"lot{k}_stop_mode"] = [defs.get(f"lot{k}_stop_mode", "none")]

        # ATR hard stop per lot
        atr_stop_def = defs.get(f"lot{k}_atr_stop_mult", 0.0)
        raw_atr_min = input(f"    atr_stop_mult min  (0=off) [{atr_stop_def}]: ").strip()
        raw_atr_max = input(f"    atr_stop_mult max  [{atr_stop_def}]: ").strip()
        atr_min = float(raw_atr_min) if raw_atr_min else float(atr_stop_def)
        atr_max = float(raw_atr_max) if raw_atr_max else float(atr_stop_def)
        if atr_min == atr_max:
            out[f"lot{k}_atr_stop_mult"] = [atr_min]
        else:
            raw_stp = input(f"    atr_stop_mult step [0.1]: ").strip()
            stp = float(raw_stp) if raw_stp else 0.1
            out[f"lot{k}_atr_stop_mult"] = _safe_float_range(atr_min, atr_max, stp, 0.1)
        print()

    # Disable unused lots
    for k in range(n_lots + 1, 4):
        out[f"lot{k}_size"]       = [0]
        out[f"lot{k}_tp_points"]  = [0.0]
        out[f"lot{k}_stop_mode"]  = [defs.get(f"lot{k}_stop_mode", "breakeven")]
        out[f"lot{k}_exit_modes"] = [defs.get(f"lot{k}_exit_modes", "ema_flip,time_exit")]

    if has_runner:
        print("  -- Runner EMA params --")
        for rp, rdef in [("runner_ema_fast",  defs.get("runner_ema_fast",  9)),
                         ("runner_ema_slow",  defs.get("runner_ema_slow",  21)),
                         ("runner_time_bars", defs.get("runner_time_bars", 120))]:
            raw_min = input(f"  {rp} min  [{rdef}]: ").strip()
            raw_max = input(f"  {rp} max  [{rdef}]: ").strip()
            p_min   = int(raw_min) if raw_min.isdigit() else rdef
            p_max   = int(raw_max) if raw_max.isdigit() else rdef
            if p_min >= p_max:
                p_step = 1
            else:
                raw_stp = input(f"  {rp} step [1]: ").strip()
                p_step  = int(raw_stp) if raw_stp.isdigit() else 1
            out[rp] = _safe_int_range(p_min, p_max, p_step)
        print()
    else:
        out["runner_ema_fast"]        = [defs.get("runner_ema_fast", 9)]
        out["runner_ema_slow"]        = [defs.get("runner_ema_slow", 21)]
        out["runner_time_bars"]       = [defs.get("runner_time_bars", 120)]
        if "runner_activation_atr" not in out:
            out["runner_activation_atr"] = [defs.get("runner_activation_atr", 0.0)]

    return out


def select_param_grid(timeframe: str, assets: list[str] | None = None) -> dict[str, list]:
    _section("PARAM GRID")
    print(f"  Strategy: {STRATEGY_CLASS.__name__}")
    print()
    print("  For each parameter enter min, max, step to generate a range.")
    print("  Press Enter to accept the default shown in brackets.")
    print()

    # Determine tick size from first asset — used as default step for price params
    _tick_size = 0.25  # fallback
    if assets:
        _tick_size = DATASETS.get(assets[0], {}).get("tick_size", 0.25)

    # Params that represent prices/points — default step = tick size
    PRICE_PARAMS = {"target_points", "stop_buf_atr"}

    dummy    = STRATEGY_CLASS()
    defaults = dummy.params

    # Warm-start: offer to load previous best params as defaults
    if assets:
        prior = load_best_params(STRATEGY_CLASS.__name__, assets, "train")
        if not prior:
            # No exact/fallback match — offer manual choose if any saves exist
            all_saved_check = list_best_params(STRATEGY_CLASS.__name__)
            if all_saved_check:
                print(f"  No previous best found for {', '.join(assets)}.")
                use_choose = input("  Load params from another asset? [y/N]: ").strip().lower()
                if use_choose in ("y", "yes"):
                    print()
                    print("  Available saved results:")
                    print(f"  {'#':<4} {'Asset(s)':<20} {'PF':>6} {'Trades':>7} {'PnL':>12}  {'Date'}")
                    print(f"  {'-'*4} {'-'*20} {'-'*6} {'-'*7} {'-'*12}  {'-'*16}")
                    for _i, _s in enumerate(all_saved_check, 1):
                        _a = ", ".join(_s.get("assets", []))
                        _m2 = _s.get("metrics", {})
                        try:
                            _dt2 = datetime.strptime(
                                _s.get("saved_at",""), "%Y%m%d_%H%M%S"
                            ).strftime("%Y-%m-%d %H:%M")
                        except ValueError:
                            _dt2 = _s.get("saved_at","")
                        print(f"  {_i:<4} {_a:<20} "
                              f"{_m2.get('profit_factor',0):>6.2f} "
                              f"{_m2.get('num_trades',0):>7,} "
                              f"${_m2.get('total_pnl',0):>11,.0f}  "
                              f"{_dt2}")
                    print()
                    _rc = input(f"  Select [1]: ").strip()
                    try:
                        _chosen = all_saved_check[int(_rc)-1 if _rc else 0]
                        defaults = {**defaults, **_chosen["params"]}
                        print(f"  → Using params from {', '.join(_chosen.get('assets',[]))}.")
                    except (ValueError, IndexError):
                        print("  Invalid — using strategy defaults.")
                print()
        if prior:
            m = prior["metrics"]
            saved = prior["saved_at"]
            # Format timestamp "20260306_085439" -> "2026-03-06 08:54"
            try:
                saved_dt = datetime.strptime(saved, "%Y%m%d_%H%M%S")
                saved_fmt = saved_dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                saved_fmt = saved

            # Find params that differ from strategy defaults
            strategy_defaults = dummy.params
            changed = {
                k: v for k, v in prior["params"].items()
                if k in strategy_defaults and str(v) != str(strategy_defaults[k])
            }

            fallback_assets = prior.get("_fallback_from")
            if fallback_assets:
                print(f"  Best result on record  ({saved_fmt})  "
                      f"[from {', '.join(fallback_assets)}]:")
            else:
                print(f"  Best result on record  ({saved_fmt}):")
            print(f"    PF={m['profit_factor']:.2f}  WR={m['win_rate']:.1%}  "
                  f"Trades={m['num_trades']:,}  PnL=${m['total_pnl']:,.0f}")
            if changed:
                changed_str = "  ".join(f"{k}={v}" for k, v in changed.items())
                print(f"    Changed vs defaults: {changed_str}")
            else:
                print(f"    No params differ from strategy defaults.")
            print()
            use_prior = input("  Use previous best as defaults? [y/N/choose]: ").strip().lower()
            if use_prior == "y":
                defaults = {**defaults, **prior["params"]}
                print("  → Using previous best params as defaults.")
            elif use_prior in ("c", "choose"):
                # Show all saved results for this strategy and let user pick
                all_saved = list_best_params(STRATEGY_CLASS.__name__)
                if not all_saved:
                    print("  No saved results found.")
                else:
                    print()
                    print("  Available saved results:")
                    print(f"  {'#':<4} {'Asset(s)':<20} {'PF':>6} {'Trades':>7} {'PnL':>12}  {'Date'}")
                    print(f"  {'-'*4} {'-'*20} {'-'*6} {'-'*7} {'-'*12}  {'-'*16}")
                    for i, s in enumerate(all_saved, 1):
                        a_str = ", ".join(s.get("assets", []))
                        m2    = s.get("metrics", {})
                        try:
                            dt_str = datetime.strptime(
                                s.get("saved_at",""), "%Y%m%d_%H%M%S"
                            ).strftime("%Y-%m-%d %H:%M")
                        except ValueError:
                            dt_str = s.get("saved_at","")
                        print(f"  {i:<4} {a_str:<20} "
                              f"{m2.get('profit_factor',0):>6.2f} "
                              f"{m2.get('num_trades',0):>7,} "
                              f"${m2.get('total_pnl',0):>11,.0f}  "
                              f"{dt_str}")
                    print()
                    raw_choice = input(f"  Select [1]: ").strip()
                    try:
                        idx = int(raw_choice) - 1 if raw_choice else 0
                        chosen = all_saved[idx]
                        defaults = {**defaults, **chosen["params"]}
                        chosen_assets = ", ".join(chosen.get("assets", []))
                        print(f"  → Using params from {chosen_assets}.")
                    except (ValueError, IndexError):
                        print("  Invalid selection — using strategy defaults.")
            print()
    param_grid = {}

    # --- Multi-lot section FIRST ---
    ml_grid = _select_multi_lot_grid(_tick_size, PRICE_PARAMS, prior_params=defaults)
    param_grid.update(ml_grid)

    print()
    print("  ── Strategy params ────────────────────────────────")
    print()
    _use_strat_defaults = input("  Use all strategy defaults? [Y/n]: ").strip().lower() not in ("n", "no")
    print()

    for param, default in defaults.items():
        if param == "timeframe":
            param_grid[param] = [timeframe]
            continue

        # Skip all multi-lot params — already handled above
        _multi_lot_keys = {"multi_lot",
                           "lot1_size", "lot1_tp_points", "lot1_exit_modes", "lot1_stop_mode", "lot1_atr_stop_mult",
                           "lot2_size", "lot2_tp_points", "lot2_stop_mode", "lot2_exit_modes", "lot2_atr_stop_mult",
                           "lot3_size", "lot3_tp_points", "lot3_stop_mode", "lot3_exit_modes", "lot3_atr_stop_mult",
                           "runner_ema_fast", "runner_ema_slow", "runner_time_bars",
                           "runner_activation_atr"}
        if param in _multi_lot_keys:
            continue

        # Use defaults without prompting if user chose to skip
        if _use_strat_defaults:
            param_grid[param] = [default]
            continue

        print(f"  — {param} (default: {default})")

        if isinstance(default, bool):
            raw    = input(f"      options [y/n] (default: {'y' if default else 'n'}): ").strip().lower()
            if raw in ("y", "yes", "true", "1"):
                values = [True]
            elif raw in ("n", "no", "false", "0"):
                values = [False]
            elif raw == "":
                values = [default]
            else:
                values = [v.strip().lower() in ("y", "yes", "true", "1")
                          for v in raw.split(",")]
            print(f"      → {values}")

        elif isinstance(default, int):
            raw_min  = input(f"      min  [{default}]: ").strip()
            raw_max  = input(f"      max  [{default}]: ").strip()
            p_min    = int(raw_min)  if raw_min  else default
            p_max    = int(raw_max)  if raw_max  else default
            if p_min >= p_max:
                p_step = 1
            else:
                raw_step = input(f"      step [1]: ").strip()
                p_step   = int(raw_step) if raw_step else 1
            values   = _safe_int_range(p_min, p_max, p_step)
            print(f"      → {values}")

        elif isinstance(default, float):
            _default_step = _tick_size if param in PRICE_PARAMS else 0.1
            raw_min  = input(f"      min  [{default}]: ").strip()
            raw_max  = input(f"      max  [{default}]: ").strip()
            p_min    = float(raw_min)  if raw_min  else default
            p_max    = float(raw_max)  if raw_max  else default
            if p_min >= p_max - 1e-9:
                p_step = _default_step
            else:
                raw_step = input(f"      step [{_default_step}]: ").strip()
                p_step   = float(raw_step) if raw_step else _default_step
            values   = _safe_float_range(p_min, p_max, p_step, _default_step)
            print(f"      → {values}")

        else:
            raw    = input(f"      options [{default}]: ").strip()
            values = [v.strip() for v in raw.split(",")] if raw else [default]
            print(f"      → {values}")

        print()
        param_grid[param] = values

    import itertools
    total = len(list(itertools.product(*param_grid.values())))
    print(f"  → {total} combination(s) to test.")

    # Guard against large grid searches
    WARN_THRESHOLD  = 100
    BLOCK_THRESHOLD = 500
    est_secs = total * 4  # rough estimate: ~4s per backtest
    est_mins = est_secs / 60
    est_str  = f"~{est_mins:.0f} min" if est_mins >= 1 else f"~{est_secs:.0f} sec"

    while total > WARN_THRESHOLD:
        if total > BLOCK_THRESHOLD:
            print(f"\n  WARNING: {total:,} combinations — estimated {est_str}.")
            print(f"  Consider narrowing your ranges.")
            confirm = input(f"  Proceed anyway? [y/N]: ").strip().lower()
        else:
            print(f"\n  Note: {total:,} combinations — estimated {est_str}.")
            confirm = input(f"  Proceed? [Y/n]: ").strip().lower()

        if confirm in ("y", "yes") or (total <= BLOCK_THRESHOLD and confirm == ""):
            break

        # Re-prompt only the swept params (those with more than one value)
        print("\n  Re-enter ranges for swept parameters:")
        swept = {k: v for k, v in param_grid.items() if isinstance(v, list) and len(v) > 1}
        for param, current_vals in swept.items():
            default = current_vals[0]
            print(f"  — {param}  (current: {current_vals})")
            if isinstance(default, bool):
                raw = input(f"      options [{default}]: ").strip()
                new_vals = [v.strip().lower() in ("y","yes","true","1") for v in raw.split(",")] if raw else [default]
            elif isinstance(default, int):
                raw_min  = input(f"      min  [{default}]: ").strip()
                raw_max  = input(f"      max  [{current_vals[-1]}]: ").strip()
                raw_step = input(f"      step [1]: ").strip()
                p_min    = int(raw_min)  if raw_min  else default
                p_max    = int(raw_max)  if raw_max  else current_vals[-1]
                p_step   = int(raw_step) if raw_step else 1
                new_vals = _safe_int_range(p_min, p_max, p_step)
            elif isinstance(default, float):
                raw_min  = input(f"      min  [{default}]: ").strip()
                raw_max  = input(f"      max  [{current_vals[-1]}]: ").strip()
                raw_step = input(f"      step [0.1]: ").strip()
                p_min    = float(raw_min)  if raw_min  else default
                p_max    = float(raw_max)  if raw_max  else current_vals[-1]
                p_step   = float(raw_step) if raw_step else 0.1
                new_vals = _safe_float_range(p_min, p_max, p_step, 0.1)
            else:
                raw      = input(f"      options [{','.join(str(v) for v in current_vals)}]: ").strip()
                new_vals = [v.strip() for v in raw.split(",")] if raw else current_vals
            print(f"      → {new_vals}")
            param_grid[param] = new_vals

        # Recompute
        total    = len(list(itertools.product(*param_grid.values())))
        est_secs = total * 4
        est_mins = est_secs / 60
        est_str  = f"~{est_mins:.0f} min" if est_mins >= 1 else f"~{est_secs:.0f} sec"
        print(f"  → {total} combination(s) to test.")

    return param_grid



# ---------------------------------------------------------------------------
# Run header
# ---------------------------------------------------------------------------

def print_run_header(assets, split, timeframe, mode, params_or_grid) -> None:
    print()
    print("=" * 65)
    print("  BACKTEST RUN CONFIGURATION")
    print("=" * 65)
    print(f"  Run time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Strategy       : {STRATEGY_CLASS.__name__}")
    print(f"  Resample mode  : {STRATEGY_CLASS.resample_mode}")
    print(f"  Timeframe      : {timeframe}")
    print(f"  Split          : {split}")
    print(f"  Mode           : {mode}")
    print(f"  Trade size     : {TRADE_SIZE} contract(s)")
    print(f"  Commission     : ${COMMISSION:.2f}/side  (${COMMISSION * 2:.2f} round-trip)")
    print()
    if mode == "single":
        print("  Params         :")
        _multi_lot = params_or_grid.get("multi_lot", False)
        _multi_lot_prefixes = ("lot1_", "lot2_", "lot3_", "runner_")
        for k, v in params_or_grid.items():
            # Hide multi-lot detail params when multi_lot=False
            if not _multi_lot and any(k.startswith(p) for p in _multi_lot_prefixes):
                continue
            print(f"    {k:<16} = {v}")
    else:
        print("  Param grid     :")
        _multi_lot_vals = params_or_grid.get("multi_lot", [False])
        _any_multi_lot  = any(_multi_lot_vals) if isinstance(_multi_lot_vals, list) else _multi_lot_vals
        _multi_lot_prefixes = ("lot1_", "lot2_", "lot3_", "runner_")
        for k, v in params_or_grid.items():
            if not _any_multi_lot and any(k.startswith(p) for p in _multi_lot_prefixes):
                continue
            print(f"    {k:<16} = {v}")
        import itertools
        total = len(list(itertools.product(*params_or_grid.values())))
        print(f"  Combinations   : {total}")
        print(f"  Rank by        : {RANK_BY}")
        print(f"  Min trades     : {MIN_TRADES}")
    print()
    print(f"  Assets ({len(assets)}):")
    for a in assets:
        ds = DATASETS.get(a, {})
        print(f"    {a:<14} tick={ds.get('tick_size')}  "
              f"tick_val=${ds.get('tick_value')}  ccy={ds.get('currency')}")
    print("=" * 65)
    print()


# ---------------------------------------------------------------------------
# Single run results
# ---------------------------------------------------------------------------

def print_single_results(results: dict, assets: list[str]) -> None:
    print(f"  {'Asset':<14} {'Bars':>7}  {'1m Bars':>8}  {'Trades':>6}  {'Net PnL':>12}  "
          f"{'WR':>6}  {'PF':>6}  {'Sharpe':>7}  {'MaxDD':>12}")
    print("-" * 112)

    for key in assets:
        result = results.get(key, {})
        if "error" in result:
            print(f"  {key:<14} ERROR: {result['error']}")
        else:
            print(
                f"  {key:<14} "
                f"{result['bars']:>7,}  "
                f"{result.get('bars_1m', result['bars']):>8,}  "
                f"{result['num_trades']:>6,}  "
                f"${result['total_pnl']:>11,.0f}  "
                f"{result['win_rate']:>5.1%}  "
                f"{result['profit_factor']:>6.2f}  "
                f"{result['sharpe']:>7.2f}  "
                f"${result['max_drawdown']:>11,.0f}"
            )
            if result.get("skipped_entries", 0) > 0:
                print(f"  {'':14} → {result['skipped_entries']} signal(s) skipped (gap > 4 ticks at entry)")

    valid = {k: v for k, v in results.items() if "error" not in v}
    if not valid:
        return

    all_pnl      = pd.concat([v["trades"]["net_pnl"] for v in valid.values()], ignore_index=True)
    total_pnl    = sum(v["total_pnl"] for v in valid.values())
    total_trades = sum(v["num_trades"] for v in valid.values())
    total_bars   = sum(v["bars"] for v in valid.values())
    total_comm   = sum(v["total_commission"] for v in valid.values())
    agg_sharpe   = (all_pnl.mean() / all_pnl.std() * np.sqrt(len(all_pnl))
                    if all_pnl.std() > 0 else 0)
    agg_dd       = sum(v["max_drawdown"] for v in valid.values())
    agg_wr       = (all_pnl > 0).sum() / len(all_pnl)
    winners      = all_pnl[all_pnl > 0].sum()
    losers       = all_pnl[all_pnl <= 0].sum()
    agg_pf       = (winners / abs(losers)) if losers != 0 else np.inf

    print("-" * 112)
    print(
        f"  {'AGGREGATE':<14} "
        f"{total_bars:>7,}  "
        f"{'':>8}  "
        f"{total_trades:>6,}  "
        f"${total_pnl:>11,.0f}  "
        f"{agg_wr:>5.1%}  "
        f"{agg_pf:>6.2f}  "
        f"{agg_sharpe:>7.2f}  "
        f"${agg_dd:>11,.0f}"
    )
    print()
    print(f"  Errors           : {[k for k, v in results.items() if 'error' in v] or 'none'}")
    print(f"  Total commission : ${total_comm:,.0f}")

    # --- Exit reason breakdown ---
    all_reasons = {}
    for v in valid.values():
        for reason, count in v.get("exit_reasons", {}).items():
            all_reasons[reason] = all_reasons.get(reason, 0) + count

    if all_reasons:
        total_closed = sum(all_reasons.values())
        parts = []
        for reason in ["target", "stop", "time", "signal", "end_of_data"]:
            if reason in all_reasons:
                n   = all_reasons[reason]
                pct = n / total_closed * 100
                parts.append(f"{reason.capitalize()}: {n:,} ({pct:.1f}%)")
        for reason, n in all_reasons.items():
            if reason not in ["target", "stop", "time", "signal", "end_of_data"]:
                pct = n / total_closed * 100
                parts.append(f"{reason.capitalize()}: {n:,} ({pct:.1f}%)")
        print(f"  Exit breakdown   : {' | '.join(parts)}")

    # --- Drawdown analytics ---
    # Aggregate across assets (weighted by trade count)
    dd_rows = [(k, v) for k, v in valid.items() if "dd_n_periods" in v]
    if dd_rows:
        print()
        print(f"  {'─' * 110}")
        print(f"  DRAWDOWN ANALYSIS")
        print(f"  {'─' * 110}")
        print(f"  {'Asset':<16} {'MaxDD':>10} {'AvgDD':>10} {'Periods':>8} {'MaxLoseStrk':>12} "
              f"{'MaxUWDays':>10} {'TotalUWDays':>12} {'DatasetDays':>12} {'CurDD':>10} {'Recovery':>10}")
        print(f"  {'─' * 110}")
        for sym, v in dd_rows:
            # Compute total calendar days in dataset from date_range tuple
            _dr = v.get("date_range")
            if _dr:
                try:
                    import pandas as _pd
                    _dataset_days = (_pd.Timestamp(_dr[1]) - _pd.Timestamp(_dr[0])).days + 1
                except Exception:
                    _dataset_days = 0
            else:
                _dataset_days = 0
            print(
                f"  {sym:<16} "
                f"${v['max_drawdown']:>9,.0f} "
                f"${v['dd_avg_depth']:>9,.0f} "
                f"{v['dd_n_periods']:>8,} "
                f"{v.get('dd_max_consec_loss', 0):>12,} "
                f"{v.get('dd_max_underwater_days', 0):>10,} "
                f"{v.get('dd_total_underwater_days', 0):>12,} "
                f"{_dataset_days:>12,} "
                f"${v['dd_current']:>9,.0f} "
                f"{v['recovery_factor']:>10.2f}"
            )
        print(f"  {'─' * 110}")
        print(f"  Periods=equity DD periods | MaxLoseStrk=max consec losing trades | MaxUWDays=longest days below peak | TotalUWDays=all days underwater")
    print()


# ---------------------------------------------------------------------------
# Train/test side-by-side results
# ---------------------------------------------------------------------------

def print_train_test_results(
    train_results: list[dict],   # list of {params, per_asset} for top N
    test_results:  list[dict],   # parallel list, same order
    assets: list[str],
    param_grid: dict,
) -> None:
    """Display train vs test side-by-side for top N param sets."""
    param_cols = [p for p in param_grid.keys() if p != "timeframe"]
    GATE_PF = 1.0

    _section("TRAIN / TEST VALIDATION — TOP 3 PARAM SETS")
    print(f"  Gate: test PF > {GATE_PF}")
    print()

    passed = 0
    for rank, (train_run, test_run) in enumerate(zip(train_results, test_results), 1):
        params = train_run["params"]

        # Aggregate train metrics
        t_trades = sum(train_run["per_asset"].get(s, {}).get("num_trades", 0) for s in assets)
        t_pnl    = sum(train_run["per_asset"].get(s, {}).get("total_pnl",   0) for s in assets)
        t_wr     = np.mean([train_run["per_asset"][s]["win_rate"]      for s in assets if s in train_run["per_asset"]])
        t_pf     = np.mean([train_run["per_asset"][s]["profit_factor"] for s in assets if s in train_run["per_asset"]])
        t_sh     = np.mean([train_run["per_asset"][s]["sharpe"]        for s in assets if s in train_run["per_asset"]])
        t_dd     = sum(train_run["per_asset"].get(s, {}).get("max_drawdown", 0) for s in assets)

        # Aggregate test metrics
        x_trades = sum(test_run["per_asset"].get(s, {}).get("num_trades", 0) for s in assets)
        x_pnl    = sum(test_run["per_asset"].get(s, {}).get("total_pnl",   0) for s in assets)
        x_wr     = np.mean([test_run["per_asset"][s]["win_rate"]      for s in assets if s in test_run["per_asset"]]) if test_run["per_asset"] else 0
        x_pf     = np.mean([test_run["per_asset"][s]["profit_factor"] for s in assets if s in test_run["per_asset"]]) if test_run["per_asset"] else 0
        x_sh     = np.mean([test_run["per_asset"][s]["sharpe"]        for s in assets if s in test_run["per_asset"]]) if test_run["per_asset"] else 0
        x_dd     = sum(test_run["per_asset"].get(s, {}).get("max_drawdown", 0) for s in assets)

        # Gate check — PF > 1.0 required
        gate_pass = x_pf > GATE_PF
        gate_tag  = "✓ PASS" if gate_pass else f"✗ FAIL (PF={x_pf:.2f})"
        if gate_pass:
            passed += 1

        # Param summary
        param_str = "  ".join(f"{p}={params.get(p)}" for p in param_cols)
        print(f"  Rank {rank}  [{gate_tag}]  {param_str}")
        print(f"  {'':>20} {'Trades':>8} {'Net PnL':>12} {'WR':>7} {'PF':>6} {'Sharpe':>8} {'MaxDD':>12}")
        print(f"  {'  Train':<20} {t_trades:>8,} ${t_pnl:>11,.0f} {t_wr:>6.1%} {t_pf:>6.2f} {t_sh:>8.2f} ${t_dd:>11,.0f}")
        print(f"  {'  Test':<20} {x_trades:>8,} ${x_pnl:>11,.0f} {x_wr:>6.1%} {x_pf:>6.2f} {x_sh:>8.2f} ${x_dd:>11,.0f}")
        print()

    print(f"  {passed}/{len(train_results)} param set(s) passed the test gate.")
    print()


# ---------------------------------------------------------------------------
# Grid results
# ---------------------------------------------------------------------------

def print_grid_results(results: dict, assets: list[str], param_grid: dict) -> None:
    shared    = results["shared_ranking"]
    per_asset = results["per_asset_best"]
    param_cols = [p for p in param_grid.keys() if p != "timeframe"]

    print(f"  SHARED PARAM RANKING (top {TOP_N}) — ranked by {RANK_BY} across all assets")
    print("-" * 100)
    print(f"  {'Rank':<5} "
          + "  ".join(f"{p:<8}" for p in param_cols)
          + f"  {'Trades':>6}  {'Net PnL':>12}  {'WR':>6}  "
            f"{'PF':>6}  {'Sharpe':>7}  {'MaxDD':>12}  {'Assets':>6}")
    print("-" * 100)

    for rank, row in shared.head(TOP_N).iterrows():
        param_str = "  ".join(f"{str(row[p]):<8}" for p in param_cols)
        print(
            f"  {rank:<5} {param_str}"
            f"  {int(row['num_trades']):>6,}"
            f"  ${row['total_pnl']:>11,.0f}"
            f"  {row['win_rate']:>5.1%}"
            f"  {row['profit_factor']:>6.2f}"
            f"  {row['sharpe']:>7.2f}"
            f"  ${row['max_drawdown']:>11,.0f}"
            f"  {int(row['assets_valid']):>3}/{len(assets)}"
        )

    print()
    print(f"  PER-ASSET BEST PARAMS — ranked by {RANK_BY} independently")
    print("-" * 100)
    print(f"  {'Asset':<14} "
          + "  ".join(f"{p:<8}" for p in param_cols)
          + f"  {'Trades':>6}  {'Net PnL':>12}  {'WR':>6}  "
            f"{'PF':>6}  {'Sharpe':>7}  {'MaxDD':>12}")
    print("-" * 100)

    for asset in assets:
        df = per_asset.get(asset)
        if df is None or df.empty:
            print(f"  {asset:<14} no valid results")
            continue
        row       = df.iloc[0]
        param_str = "  ".join(f"{str(row[p]):<8}" for p in param_cols)
        print(
            f"  {asset:<14} {param_str}"
            f"  {int(row['num_trades']):>6,}"
            f"  ${row['total_pnl']:>11,.0f}"
            f"  {row['win_rate']:>5.1%}"
            f"  {row['profit_factor']:>6.2f}"
            f"  {row['sharpe']:>7.2f}"
            f"  ${row['max_drawdown']:>11,.0f}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _offer_cross_split_run(current_split, assets, timeframe, params, base_config, timestamp):
    """After a train run, offer to immediately run the same params on test (and vice versa)."""
    other_split = "test" if current_split == "train" else "train"
    print()
    ans = input(f"  Run {other_split} split with same params? [y/N]: ").strip().lower()
    if ans not in ("y", "yes"):
        return

    from backtest import load_data as _load_data
    print(f"\n  Running {other_split} split...")
    cross_results = {}
    for key in assets:
        dataset = DATASETS.get(key, {})
        path = dataset.get(other_split)
        if not path:
            print(f"  WARNING: No {other_split} data for {key} — skipping.")
            continue
        try:
            df = _load_data(path, dataset["date_format"])
        except Exception as e:
            print(f"  WARNING: Could not load {key} {other_split}: {e}")
            continue

        result = run_backtest(
            dataset=dataset,
            strategy=STRATEGY_CLASS(params=params),
            split=other_split,
            trade_size=sum(int(params.get(f"lot{k}_size", 0)) for k in range(1, 4)
                           if int(params.get(f"lot{k}_size", 0)) > 0) or 1,
            commission_per_side=2.0,
        )
        cross_results[key] = result

    if not cross_results:
        print("  No results.")
        return

    # Print comparison
    print()
    print(f"  {'':─<110}")
    print(f"  TRAIN vs TEST COMPARISON")
    print(f"  {'':─<110}")
    print(f"  {'Split':<8} {'Asset':<10} {'Trades':>7} {'Net PnL':>12} {'WR':>7} {'PF':>6} {'Sharpe':>8} {'MaxDD':>12}")
    print(f"  {'':─<110}")


    for key in assets:
        # Train result (already computed, passed in base_config context)
        cr = cross_results.get(key, {})
        if "error" not in cr:
            print(f"  {'test':<8} {key:<10} {cr['num_trades']:>7,} ${cr['total_pnl']:>11,.0f} "
                  f"{cr['win_rate']:>6.1%} {cr['profit_factor']:>6.2f} {cr['sharpe']:>8.2f} "
                  f"${cr['max_drawdown']:>11,.0f}")
    print(f"  {'':─<110}")

    # Export cross-split results
    export_results(
        results=cross_results,
        config={**base_config, "params": params, "split": other_split},
        timestamp=timestamp,
    )
    run_validation(cross_results, other_split, timeframe,
                   strategy_name=STRATEGY_CLASS.__name__,
                   strategy_params=params)

def main():
    global STRATEGY_CLASS
    STRATEGY_CLASS = select_strategy()

    assets = select_assets()
    if not assets:
        print("No assets selected — exiting.")
        return {}

    split     = select_split()
    # Strategies that handle their own resampling always execute on 1m
    if getattr(STRATEGY_CLASS, "manages_own_lots", False):
        timeframe = "1m"
        print(f"  → Timeframe: 1m  (fixed for {STRATEGY_CLASS.__name__})")
    else:
        timeframe = select_timeframe()
    mode      = select_mode()

    # Shared config dict — passed to exporter for run_config.txt
    base_config = {
        "strategy":      STRATEGY_CLASS.__name__,
        "resample_mode": STRATEGY_CLASS.resample_mode,
        "timeframe":     timeframe,
        "split":         split,
        "mode":          mode,
        "trade_size":    TRADE_SIZE,
        "commission":    COMMISSION,
        "assets":        assets,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if mode == "single":
        # Load prior best so multi-lot prompt can use them as defaults
        _single_prior = load_best_params(STRATEGY_CLASS.__name__, assets, "train")
        _single_defaults = {**STRATEGY_CLASS().params,
                            **(_single_prior["params"] if _single_prior else {})}
        # Offer choose option for single mode too
        if _single_prior:
            _sp_m = _single_prior.get("metrics", {})
            _sp_fa = _single_prior.get("_fallback_from")
            try:
                _sp_dt = datetime.strptime(_single_prior.get("saved_at",""),
                                           "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")
            except ValueError:
                _sp_dt = _single_prior.get("saved_at","")
            _sp_label = (f"[from {', '.join(_sp_fa)}]" if _sp_fa else "")
            print(f"\n  Best result on record  ({_sp_dt}) {_sp_label}:")
            print(f"    PF={_sp_m.get('profit_factor',0):.2f}  "
                  f"WR={_sp_m.get('win_rate',0):.1%}  "
                  f"Trades={_sp_m.get('num_trades',0):,}  "
                  f"PnL=${_sp_m.get('total_pnl',0):,.0f}")
            print()
            _sp_choice = input("  Use previous best as defaults? [y/N/choose]: ").strip().lower()
            if _sp_choice == "y":
                pass  # already merged above
            elif _sp_choice in ("c", "choose"):
                _all_saved = list_best_params(STRATEGY_CLASS.__name__)
                if _all_saved:
                    print()
                    print("  Available saved results:")
                    print(f"  {'#':<4} {'Asset(s)':<20} {'PF':>6} {'Trades':>7} {'PnL':>12}  {'Date'}")
                    print(f"  {'-'*4} {'-'*20} {'-'*6} {'-'*7} {'-'*12}  {'-'*16}")
                    for _i, _s in enumerate(_all_saved, 1):
                        _a = ", ".join(_s.get("assets", []))
                        _m2 = _s.get("metrics", {})
                        try:
                            _dt2 = datetime.strptime(
                                _s.get("saved_at",""), "%Y%m%d_%H%M%S"
                            ).strftime("%Y-%m-%d %H:%M")
                        except ValueError:
                            _dt2 = _s.get("saved_at","")
                        print(f"  {_i:<4} {_a:<20} "
                              f"{_m2.get('profit_factor',0):>6.2f} "
                              f"{_m2.get('num_trades',0):>7,} "
                              f"${_m2.get('total_pnl',0):>11,.0f}  "
                              f"{_dt2}")
                    print()
                    _rc = input(f"  Select [1]: ").strip()
                    try:
                        _chosen = _all_saved[int(_rc)-1 if _rc else 0]
                        _single_defaults = {**STRATEGY_CLASS().params, **_chosen["params"]}
                        print(f"  → Using params from {', '.join(_chosen.get('assets',[]))}.\n")
                    except (ValueError, IndexError):
                        print("  Invalid — using strategy defaults.\n")
                else:
                    print("  No saved results found.")
            else:
                _single_defaults = STRATEGY_CLASS().params  # reset to defaults
            print()

        # Multi-lot selection first
        ml_params, ml_trade_size = _select_multi_lot_single(prior_params=_single_defaults)

        # Use strategy defaults, override with SINGLE_PARAMS and multi-lot, then set timeframe
        defaults = STRATEGY_CLASS().params
        params = {**defaults, **SINGLE_PARAMS, **ml_params, "timeframe": timeframe}
        effective_trade_size = ml_trade_size if ml_params.get("multi_lot") else TRADE_SIZE
        print_run_header(assets, split, timeframe, mode, params)

        results = {}
        for key in assets:
            results[key] = run_backtest(
                dataset=DATASETS[key],
                strategy=STRATEGY_CLASS(params=params),
                split=split,
                trade_size=effective_trade_size,
                commission_per_side=COMMISSION,
            )

        print_single_results(results, assets)

        # Export
        export_results(
            results=results,
            config={**base_config, "params": params},
            timestamp=timestamp,
        )

        # Validation
        run_validation(results, split, timeframe,
                       strategy_name=STRATEGY_CLASS.__name__,
                       strategy_params=params)

        # Offer to run the opposite split with same params
        if split in ("train", "test"):
            _offer_cross_split_run(split, assets, timeframe, params,
                                   base_config, timestamp)

        return results

    else:
        param_grid = select_param_grid(timeframe, assets=assets)

        # Derive trade_size from lot sizes if multi_lot enabled
        _any_ml = any(param_grid.get("multi_lot", [False]))
        if _any_ml:
            # Use max possible trade size (sum of max lot sizes) for grid
            _grid_trade_size = sum(
                max(param_grid.get(f"lot{k}_size", [0]))
                for k in range(1, 4)
            )
        else:
            _grid_trade_size = TRADE_SIZE

        # For "both": always run grid on train, then validate top 3 on test
        train_split = "train" if split == "both" else split
        print_run_header(assets, train_split, timeframe, mode, param_grid)

        results = run_grid_search(
            param_grid=param_grid,
            strategy_class=STRATEGY_CLASS,
            assets=assets,
            split=train_split,
            trade_size=_grid_trade_size,
            commission=COMMISSION,
            rank_by=RANK_BY,
            min_trades=MIN_TRADES,
        )

        print_grid_results(results, assets, param_grid)

        shared   = results["shared_ranking"]
        all_runs = results["all_results"]

        if not shared.empty:
            best_params = {
                p: shared.iloc[0][p]
                for p in param_grid.keys()
                if p in shared.columns
            }

            # Save best params for warm-start on future runs
            save_best_params(
                strategy_name=STRATEGY_CLASS.__name__,
                assets=assets,
                split=train_split,
                params=best_params,
                metrics={
                    "profit_factor": shared.iloc[0].get("profit_factor", 0),
                    "num_trades":    shared.iloc[0].get("num_trades", 0),
                    "total_pnl":     shared.iloc[0].get("total_pnl", 0),
                    "win_rate":      shared.iloc[0].get("win_rate", 0),
                    "sharpe":        shared.iloc[0].get("sharpe", 0),
                    "max_drawdown":  shared.iloc[0].get("max_drawdown", 0),
                },
                timestamp=timestamp,
            )
            # Find matching result set for best combo (for export/validation)
            best_results = {}
            for r in all_runs:
                if all(str(r["params"].get(k)) == str(v)
                       for k, v in best_params.items()):
                    for sym, res in r["per_asset"].items():
                        best_results[sym] = res
                    break

            export_results(
                results=best_results,
                config={**base_config, "param_grid": param_grid,
                        "best_params": best_params,
                        "rank_by": RANK_BY},
                timestamp=timestamp,
            )

            # --- Train/test validation when split == "both" ---
            if split == "both":
                TRAIN_MIN_PF = 1.0
                TOP_N = 3

                # Only test param sets where train PF > minimum
                top_runs = []
                for _, row in shared.iterrows():
                    if len(top_runs) >= TOP_N:
                        break
                    if row.get("profit_factor", 0) <= TRAIN_MIN_PF:
                        continue
                    row_params = {p: row[p] for p in param_grid.keys() if p in row.index}
                    for r in all_runs:
                        if all(str(r["params"].get(k)) == str(v)
                               for k, v in row_params.items()):
                            top_runs.append(r)
                            break

                if not top_runs:
                    print(f"\n  No train param sets met the minimum PF > {TRAIN_MIN_PF} threshold.")
                    print(f"  Best train PF: {shared.iloc[0]['profit_factor']:.2f} — strategy needs work before test validation.")
                else:
                    print(f"\n  Running top {len(top_runs)} param set(s) (train PF > {TRAIN_MIN_PF}) on test split...")
                    test_runs = []
                    for r in top_runs:
                        test_per_asset = {}
                        for key in assets:
                            test_res = run_backtest(
                                dataset=DATASETS[key],
                                strategy=STRATEGY_CLASS(params=r["params"]),
                                split="test",
                                trade_size=TRADE_SIZE,
                                commission_per_side=COMMISSION,
                            )
                            test_per_asset[key] = test_res
                        test_runs.append({"params": r["params"], "per_asset": test_per_asset})

                    print_train_test_results(top_runs, test_runs, assets, param_grid)

                    run_validation(test_runs[0]["per_asset"], "test", timeframe,
                                   strategy_name=STRATEGY_CLASS.__name__,
                                   strategy_params=top_runs[0]["params"])
            else:
                # Single split — validation on best params as before
                run_validation(best_results, split, timeframe,
                               strategy_name=STRATEGY_CLASS.__name__,
                               strategy_params=best_params)

                # Offer to run the opposite split with same params
                if split in ("train", "test"):
                    _offer_cross_split_run(split, assets, timeframe, best_params,
                                           base_config, timestamp)

        return results


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()   # required for Windows multiprocessing
    results = main()
