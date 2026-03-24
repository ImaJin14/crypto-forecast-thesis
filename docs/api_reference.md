# API Reference
**Deep Learning for Cryptocurrency Price Forecasting**
*Muluh Penn Junior Patrick — M.Tech. Thesis 2026*

---

## `src.models`

### `get_model(name, **kwargs) → nn.Module`
Instantiate a model by name.

```python
from src.models import get_model
model = get_model('lstm', input_size=149, hidden_size=128, num_layers=1, dropout=0.3)
```

**Available models:** `lstm`, `gru`, `bilstm`, `cnn_lstm`, `attention_lstm`, `transformer`

### `LSTMForecaster(input_size, hidden_size=256, num_layers=2, dropout=0.2, output_size=1)`
Stacked LSTM. Input: `(batch, seq_len, input_size)`. Output: `(batch, output_size)`.

### `GRUForecaster(...)` — same signature as LSTM.

### `BiLSTMForecaster(input_size, hidden_size=128, num_layers=2, dropout=0.2, output_size=1)`
Bidirectional LSTM. Output dimension = `hidden_size * 2`.

### `CNNLSTMForecaster(input_size, num_filters=64, kernel_size=3, hidden_size=128, num_layers=2, dropout=0.2, output_size=1)`
CNN feature extractor → LSTM sequence model.

### `AttentionLSTMForecaster(input_size, hidden_size=256, num_layers=2, dropout=0.2, output_size=1)`
LSTM with Bahdanau-style soft attention over the sequence.

### `TransformerForecaster(input_size, d_model=128, nhead=8, num_encoder_layers=3, dim_feedforward=512, dropout=0.1, output_size=1)`
Transformer encoder with positional encoding.

---

## `src.training.trainer`

### `train_model(model_name, asset, interval, horizon, ...) → dict`
Full training pipeline: data → features → normalise → train → evaluate.

```python
from src.training.trainer import train_model
results = train_model(
    model_name='gru', asset='BTC', interval='1d', horizon=1,
    seq_len=90, batch_size=64, lr=0.00448, max_epochs=200,
    loss_name='combined',
    model_kwargs={'hidden_size': 128, 'num_layers': 1, 'dropout': 0.296},
)
# results['test_metrics'] → dict with rmse, mape, r2, directional_accuracy, sharpe_ratio, ...
```

### `CryptoDataModule`
PyTorch Lightning DataModule. Handles data loading, feature engineering, normalisation, and sequence building.

```python
from src.training.trainer import CryptoDataModule
dm = CryptoDataModule(asset='BTC', interval='1d', seq_len=90, horizon=1, batch_size=64)
dm.setup()
# dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()
# dm.n_features → int (number of input features)
```

---

## `src.evaluation`

### `evaluator.print_comparison_table(asset, interval, horizon)`
Print formatted KPI table for all 6 models.

### `evaluator.rank_models(asset, interval, horizon) → pd.DataFrame`
Return models ranked by average KPI rank.

### `diebold_mariano.dm_test(errors_a, errors_b, h=1, loss='mse') → (stat, p_value)`
Diebold-Mariano test for equal predictive accuracy.
- Negative stat → model A more accurate
- p < 0.05 → statistically significant difference

### `diebold_mariano.dm_matrix(errors_dict, h=1) → (dm_stats, p_values)`
Pairwise DM test for all model combinations.

### `ablation_study.run_full_ablation(model_name, asset, ...) → pd.DataFrame`
Run ablation removing each feature group (ltst, onchain, sentiment, macro) one at a time.

---

## `src.preprocessing`

### `TechnicalIndicatorComputer.compute(df) → pd.DataFrame`
Add 104 technical indicator features to OHLCV DataFrame.

### `LTSTDecomposer(methods=['ma']).decompose(df) → pd.DataFrame`
Add 35 LTST decomposition features.

### `CryptoNormalizer(method='minmax').fit(df) / .transform(df) / .inverse_transform(df)`
Zero-leakage normalizer. Always `fit` on training data only.

### `SequenceBuilder(seq_len, horizon, stride).build(df, target_col) → (X, y)`
Build sliding window sequences. Returns `(n_samples, seq_len, n_features)` and `(n_samples,)`.

---

## `src.visualization.thesis_figures`

```python
from src.visualization.thesis_figures import (
    fig1_training_curves,   # val_loss curves for all 6 models
    fig2_price_prediction,  # actual vs predicted BTC price
    fig3_model_comparison,  # MAPE + RMSE bar charts
    fig4_dm_heatmap,        # Diebold-Mariano significance matrix
    fig5_ablation,          # feature ablation ΔMAPE
    fig6_feature_overview,  # 149-feature pie chart
)
# All functions save PDF + PNG to experiments/figures/
```

---

## CLI Reference

```bash
# Training
python scripts/run_training.py --model gru --asset BTC --interval 1d --horizon 1
python scripts/run_training.py --model lstm --asset ETH --use-best-params

# Hyperparameter tuning
python scripts/run_tuning.py --model lstm --asset BTC --trials 30 --epochs 50
python scripts/run_tuning.py --model gru  --asset ETH --no-resume

# Evaluation
python scripts/run_evaluation.py --asset BTC --horizon 1 --dm
python scripts/run_evaluation.py --all

# Ablation study
python scripts/run_ablation.py --model lstm --asset BTC --epochs 100

# All 120 experiments (overnight)
python scripts/run_all_experiments.py --trials 15 --epochs 30
python scripts/run_all_experiments.py --asset BTC  # BTC only
python scripts/run_all_experiments.py --dry-run    # print plan only

# Save prediction arrays for DM testing
python save_predictions.py --asset BTC --interval 1d --horizon 1
```
