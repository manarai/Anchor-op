"""Minimal smoke test for an isolated wheel installation."""

import anchorop

if __name__ == "__main__":
    assert anchorop.__version__ == "0.1.0"
    print(f"isolated wheel import succeeded: anchorop {anchorop.__version__}")
