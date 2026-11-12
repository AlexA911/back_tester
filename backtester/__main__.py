# Directory refactor
"""Launch the interactive backtester."""
import multiprocessing as mp

from backtester.cli import main


if __name__ == "__main__":
    mp.freeze_support()
    main()
