"""Convenience entry point: python scripts/run_benchmarks.py [same options as pathfinding.benchmark]."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathfinding.benchmark import main

if __name__ == '__main__':
    sys.exit(main())
