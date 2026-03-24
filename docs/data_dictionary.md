# Data Dictionary
**Deep Learning for Cryptocurrency Price Forecasting**
*Muluh Penn Junior Patrick — M.Tech. Thesis 2026*

Total features: **149** per timestep, across 5 categories.

---

## 1. OHLCV Base Features (10)

| Feature | Description | Unit |
|---------|-------------|------|
| `open` | Opening price | USD |
| `high` | Session high | USD |
| `low` | Session low | USD |
| `close` | Closing price | USD |
| `volume` | Trade volume (base asset) | Coins |
| `quote_volume` | Trade volume (USDT) | USD |
| `log_return` | log(close_t / close_{t-1}) | — |
| `log_return_h` | log(high / open) | — |
| `log_return_l` | log(low / open) | — |
| `hlcc4` | (high + low + close + close) / 4 | USD |

---

## 2. Technical Indicators (104)

### Trend (25)
| Feature | Formula | Period(s) |
|---------|---------|-----------|
| `ema_9/21/50/100/200` | Exponential Moving Average | 9, 21, 50, 100, 200 |
| `sma_10/20/50/200` | Simple Moving Average | 10, 20, 50, 200 |
| `macd` | EMA(12) − EMA(26) | 12, 26 |
| `macd_signal` | EMA(9) of MACD | 9 |
| `macd_hist` | MACD − Signal | — |
| `adx` | Average Directional Index | 14 |
| `adx_pos / adx_neg` | +DI / −DI components | 14 |
| `aroon_up / aroon_down` | Aroon oscillator | 25 |
| `cci` | Commodity Channel Index | 20 |
| `dpo` | Detrended Price Oscillator | 20 |
| `vortex_pos / vortex_neg` | Vortex Indicator | 14 |
| `trix` | 1-period % change of triple EMA | 15 |
| `mass_index` | Mass Index | 9, 25 |
| `price_vs_sma200` | close / SMA(200) − 1 | — |

### Momentum (22)
| Feature | Formula | Period(s) |
|---------|---------|-----------|
| `rsi` | Relative Strength Index | 14 |
| `rsi_6 / rsi_24` | RSI variants | 6, 24 |
| `stoch_k / stoch_d` | Stochastic Oscillator | 14, 3 |
| `stoch_rsi_k / stoch_rsi_d` | Stochastic RSI | 14 |
| `roc` | Rate of Change | 12 |
| `roc_5 / roc_20` | ROC variants | 5, 20 |
| `williams_r` | Williams %R | 14 |
| `ultimate_oscillator` | Ultimate Oscillator | 7, 14, 28 |
| `awesome_oscillator` | AO = SMA(5, median) − SMA(34, median) | 5, 34 |
| `ppo / ppo_signal / ppo_hist` | Percentage Price Oscillator | 12, 26, 9 |
| `kama` | Kaufman's Adaptive Moving Average | 10 |
| `tsi` | True Strength Index | 25, 13 |

### Volatility (18)
| Feature | Formula | Period(s) |
|---------|---------|-----------|
| `atr` | Average True Range | 14 |
| `atr_pct` | ATR / close | 14 |
| `bb_high / bb_low / bb_mid` | Bollinger Bands | 20, 2σ |
| `bb_width / bb_pct` | BB width and position | 20 |
| `kc_high / kc_low / kc_mid` | Keltner Channel | 20 |
| `kc_width / kc_pct` | KC width and position | 20 |
| `donchian_high / donchian_low` | Donchian Channel | 20 |
| `ulcer_index` | Ulcer Index | 14 |
| `realized_vol_5 / realized_vol_20` | Rolling std of log returns | 5, 20 |

### Volume (18)
| Feature | Formula | Period(s) |
|---------|---------|-----------|
| `obv` | On-Balance Volume | — |
| `obv_ema` | EMA of OBV | 21 |
| `cmf` | Chaikin Money Flow | 20 |
| `mfi` | Money Flow Index | 14 |
| `fi` | Force Index | 2, 13 |
| `ease` | Ease of Movement | 14 |
| `vpt` | Volume Price Trend | — |
| `nvi` | Negative Volume Index | — |
| `vwap` | Volume Weighted Average Price | — |
| `volume_ema` | Volume EMA | 21 |
| `volume_ratio` | volume / SMA(volume, 20) | 20 |
| `ad` | Accumulation/Distribution | — |
| `adosc` | Chaikin A/D Oscillator | 3, 10 |

### Derived / Cross (21)
| Feature | Description |
|---------|-------------|
| `price_efficiency` | Daily move / Total path |
| `body_size / shadow_upper / shadow_lower` | Candlestick anatomy |
| `gap_open` | Open − previous close |
| `high_low_range_pct` | (high − low) / close |
| `close_to_high / close_to_low` | Intrabar position |
| `log_return_lag_1..5` | Lagged log returns |
| `vol_adjusted_return` | Log return / ATR |

---

## 3. LTST Decomposition Features (35)

Long-term trend (LTT) and short-term trend (STT) extracted via moving averages.

| Feature | Description | Window |
|---------|-------------|--------|
| `ltt_50 / ltt_100 / ltt_200` | Long-term SMA trends | 50, 100, 200 |
| `stt_5 / stt_10 / stt_20` | Short-term EMA trends | 5, 10, 20 |
| `ltt_stt_spread_*` | LTT − STT differences | Various |
| `ltt_slope_* / stt_slope_*` | Period-over-period slope | Various |
| `ma_residual_*` | price − LTT | Various |
| `above_ltt_*` | Binary: price > LTT | Various |
| `ltt_strength_*` | (price − LTT) / LTT | Various |
| `trend_consensus` | % of MAs with price above | — |
| `mean_reversion_signal` | Distance from 200d mean in σ | — |

---

## 4. On-Chain Metrics (variable, BTC/ETH only)

Sourced from Blockchain.info (BTC) and Etherscan (ETH) free APIs.

| Feature | Description | Asset |
|---------|-------------|-------|
| `hash_rate` | Network hash rate | BTC |
| `difficulty` | Mining difficulty | BTC |
| `n_transactions` | Daily confirmed transactions | BTC |
| `mempool_size` | Mempool transaction count | BTC |
| `transaction_volume` | USD value transferred | BTC |
| `miner_revenue` | Miner fees + block reward | BTC |
| `eth_tx_count` | Daily Ethereum transactions | ETH |

---

## 5. Sentiment & Macro (variable)

| Feature | Source | Description |
|---------|--------|-------------|
| `fg_value` | Alternative.me | Fear & Greed Index (0–100) |
| `fg_classification` | Alternative.me | Categorical label |
| `fg_pct_change` | Derived | Day-over-day FG change |
| `fg_ma_7` | Derived | 7-day rolling FG mean |
| `cg_market_cap` | CoinGecko | Global crypto market cap |
| `cg_total_volume` | CoinGecko | Global 24h volume |
| `btc_dominance` | CoinGecko | BTC market cap share |
| `sp500_return` | Yahoo Finance | S&P 500 daily return |
| `dxy_return` | Yahoo Finance | Dollar Index return |
| `gold_return` | Yahoo Finance | Gold spot return |
| `vix` | Yahoo Finance | CBOE Volatility Index |

---

## Train / Val / Test Split

| Set | Period | Rows | Proportion |
|-----|--------|------|-----------|
| Train | 2018-01-01 → 2023-10-02 | 2,101 | 70% |
| Validation | 2023-10-03 → 2024-12-25 | 450 | 15% |
| Test | 2024-12-26 → 2026-03-21 | 451 | 15% |

*Normalizer (MinMax) fitted on train set only — no leakage.*
