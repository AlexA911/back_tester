# Directory refactor
"""Shared paths anchored to the project, independent of the working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_DIR = Path(__file__).resolve().parent / "config"
