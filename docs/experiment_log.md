# Experiment Log
**Deep Learning for Cryptocurrency Price Forecasting**
*Muluh Penn Junior Patrick — M.Tech. Thesis 2026*

---

## Phase 1 — Project Setup (2026-03)
- [x] KPI framework defined: RMSE, MAE, MAPE (<5%), R² (>0.85), DA (>60%), Sharpe, Max DD, Win Rate
- [x] Project scaffold created (114 files, 35 directories)
- [x] Thesis proposal written and submitted
- [x] GitHub repo initialised: `ImaJin14/crypto-forecast-thesis`

## Phase 2 — Data Collection (2026-03)
- [x] BTC, ETH, SOL, SUI, XRP daily OHLCV data collected via Binance API (free tier)
- [x] On-chain metrics collected via Blockchain.info (BTC), Etherscan (ETH)
- [x] Sentiment: Fear & Greed Index via Alternative.me; CoinGecko market data
- [x] Macro: Yahoo Finance (S&P 500, DXY, Gold, VIX)
- [x] Data validator built; timezone bug fixed (UTC alignment)

## Phase 3 — Feature Engineering (2026-03)
- [x] 104 technical indicators implemented (`src/preprocessing/technical_indicators.py`)
- [x] LTST decomposition: 35 features via MA-based trend extraction
- [x] Zero-leakage MinMax normalizer (`src/preprocessing/normalizer.py`)
- [x] Sliding window sequence builder (seq_len, horizon, stride configurable)
- [x] ADF + KPSS stationarity tests
- [x] Final feature count: **149 features** per timestep

## Phase 4 — Training Infrastructure (2026-03)
- [x] PyTorch Lightning `CryptoForecasterModule` and `CryptoDataModule`
- [x] `CombinedLoss`: MSE + Huber + Directional (thesis primary loss)
- [x] 7 callbacks: early stopping (patience=15), checkpoint, LR monitor, CSV logger,
      gradient monitor, timing, MetricsLogger
- [x] Temporal 70/15/15 walk-forward split
- [x] Log-return targets with USD price reconstruction via `test_close.npy`
- [x] Tensor Core precision enabled (`torch.set_float32_matmul_precision("medium")`)

## Phase 5 — Hyperparameter Tuning + Training (2026-03)

### BTC 1d h=1 Results (all 6 models)

| Model | Best val_loss | MAPE | RMSE | DA | Params | Training |
|-------|-------------|------|------|----|--------|----------|
| LSTM | 0.067 | 1.61% | $2,085 | 50.1% | 143K | 24 epochs, 15s |
| GRU | 0.060 | 1.60% | $2,077 | 50.1% | ~130K | 22 epochs, 12s |
| BiLSTM | 0.078 | 1.71% | $2,165 | 49.3% | ~285K | 30 epochs, 25s |
| CNN-LSTM | 0.087 | 1.72% | $2,205 | 50.6% | 574K | 18 epochs, 30s |
| Attention-LSTM | 0.068 | 1.67% | $2,161 | 48.5% | ~285K | 32 epochs, 73s |
| Transformer | 0.069 | 1.67% | $2,142 | 50.1% | 418K | 29 epochs, 41s |

### Tuning notes
- `combined` loss is critical — pure MSE/Huber collapses to near-zero on log returns
- `plateau` scheduler removed from all search spaces (collapses in short tuning runs)
- GRU: best seq_len=90, hidden_size=128, num_layers=1, lr=0.00448 (Adam)
- Transformer: best seq_len=90, d_model=128, nhead=2, num_encoder_layers=4 (warmup_cosine)
- CNN-LSTM: switched to Huber loss to prevent collapse

### BTC 1d h=30 (in progress)
- [x] All 6 models tuned (15 trials × 30 epochs)
- [ ] Full training with best params pending

## Phase 6 — Evaluation (2026-03)
- [x] `evaluator.py`: KPI comparison tables for all model × asset × horizon combinations
- [x] `diebold_mariano.py`: DM test with real inference errors (5/6 models)
- [x] `save_predictions.py`: saves per-sample prediction errors as .npy for DM testing
- [x] `ablation_study.py`: feature group ablation (LTST, on-chain, sentiment, macro)
- [x] Ablation result: ΔMAPE ≈ 0 for all groups at h=1 (features matter more at longer horizons)

### Key DM test results (BTC 1d h=1)
| Pair | DM stat | Significant? |
|------|---------|-------------|
| GRU vs LSTM | −0.285 | No (p>0.05) |
| GRU vs Transformer | −3.103 | **Yes (p<0.05)** |
| LSTM vs Transformer | −2.511 | **Yes (p<0.05)** |
| BiLSTM vs all others | +4.4 to +8.4 | **Yes (all pairs)** |

## Phase 7 — Visualization (2026-03)
- [x] `thesis_figures.py`: 6 publication-quality figures (PDF + PNG at 300 DPI)
  - Fig 1: Training convergence curves
  - Fig 2: Price prediction vs actual (GRU, test period)
  - Fig 3: MAPE/RMSE model comparison bar chart
  - Fig 4: Diebold-Mariano heatmap
  - Fig 5: Ablation study
  - Fig 6: Feature engineering overview

## Phase 8 — Thesis Writing (upcoming)
- [ ] Chapter 1: Introduction
- [ ] Chapter 2: Literature Review
- [ ] Chapter 3: Methodology
- [ ] Chapter 4: Results
- [ ] Chapter 5: Discussion
- [ ] Chapter 6: Conclusion
- [ ] Abstract
- [ ] References

## Known Issues / Technical Notes

1. **CNN-LSTM checkpoint mismatch**: Ablation run (use_ltst=False) overwrites the scaler
   with 114 features. Workaround: delete checkpoint and retrain before DM testing.
   Fix: ablation should use a separate scaler path.

2. **BiLSTM causal concern**: BiLSTM uses backward-pass future information, which violates
   causality for live trading. Results are valid for research comparison but BiLSTM should
   not be deployed in a live trading system.

3. **DA at 50% for h=1**: All models predict direction near-randomly at 1-day horizon,
   consistent with weak-form EMH. DA expected to improve at h=7, 14, 30.

4. **CNN-LSTM collapse with combined loss**: CNN-LSTM tends to collapse (val_loss→0)
   with `combined` loss. Switched to `huber` loss for stable training.
