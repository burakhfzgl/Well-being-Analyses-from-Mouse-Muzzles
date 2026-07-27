"""Thin wrapper. Prefer: python main.py --mode check"""

from src.check_setup import run_check

if __name__ == "__main__":
    run_check()
