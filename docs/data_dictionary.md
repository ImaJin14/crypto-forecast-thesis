# Data Dictionary

## Price / Volume Features (OHLCV)
| Feature | Formula | Source |
|---------|---------|--------|
| open    | Opening price | Binance |
| high    | Session high  | Binance |
| low     | Session low   | Binance |
| close   | Closing price | Binance |
| volume  | Trade volume (base asset) | Binance |
| quote_volume | Trade volume (quote asset, USDT) | Binance |

## Technical Indicators
| Feature | Formula | Period |
|---------|---------|--------|
| rsi     | Relative Strength Index | 14 |
| macd    | EMA(12) − EMA(26) | — |
| macd_signal | EMA(9) of MACD | 9 |
| bb_upper | SMA(20) + 2σ | 20 |
| bb_lower | SMA(20) − 2σ | 20 |
| bb_width | (upper − lower) / SMA | 20 |
| ema_9   | Exponential MA | 9 |
| ema_21  | Exponential MA | 21 |
| atr     | Average True Range | 14 |
| obv     | On-Balance Volume | — |

## On-Chain Metrics
| Feature | Description | Source |
|---------|-------------|--------|
| active_addresses | Unique daily active addresses | Glassnode |
| nvts    | NVT Signal (Market Cap / Tx Volume) | Glassnode |
| hash_rate | Network hash rate | Glassnode |
| mvrv    | Market Value to Realized Value | Glassnode |
| sopr    | Spent Output Profit Ratio | Glassnode |

## Sentiment Features
| Feature | Description | Source |
|---------|-------------|--------|
| fear_greed | Fear & Greed Index (0–100) | Alternative.me |
| tweet_sentiment | FinBERT-scored tweet aggregate | Twitter/X API |
| sentiment_lag_1 | 1-period lagged sentiment | Derived |
| sentiment_lag_24 | 24-period lagged sentiment | Derived |

## Macro / Correlating Assets
| Feature | Description | Source |
|---------|-------------|--------|
| sp500_return | Daily S&P 500 return | Yahoo Finance |
| dxy_return | DXY Dollar Index return | Yahoo Finance |
| gold_return | Gold spot return | Yahoo Finance |
| vix | CBOE Volatility Index | Yahoo Finance |
| btc_dom | BTC market dominance % | CoinGecko |

## LTST Decomposition
| Feature | Description | Method |
|---------|-------------|--------|
| ltt_50  | Long-term trend (50-period) | SMA |
| ltt_200 | Long-term trend (200-period) | SMA |
| stt_5   | Short-term trend (5-period)  | EMA |
| stt_20  | Short-term trend (20-period) | EMA |
| hp_trend | HP filter trend component | Hodrick-Prescott |
| residual | Price − HP trend | Derived |
