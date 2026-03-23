#!/usr/bin/env python3
"""End-to-end data collection and preprocessing pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n  📡  Data pipeline — implementation in Phase 2.\n")
print("  Will collect: OHLCV · On-chain · Sentiment · Macro · LTST features")
print("  Assets: BTC · ETH · SOL · SUI · XRP")
