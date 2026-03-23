#!/usr/bin/env python3
"""Evaluate all trained models and generate KPI comparison tables."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics           import compute_all_metrics
from src.evaluation.financial_metrics import sharpe_ratio, max_drawdown, win_rate

print("\n  📊  Evaluation pipeline — coming in Phase 7 of the roadmap.\n")
print("  Metrics available:")
import inspect, src.evaluation.metrics as m
for name, fn in inspect.getmembers(m, inspect.isfunction):
    print(f"    ✔  {name}")
