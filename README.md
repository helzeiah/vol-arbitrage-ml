# Machine Learning Volatility Arbitrage (MVP), currenty WIP

This project explores **volatility arbitrage** strategies by comparing **implied volatility (VIX)** with realized and predicted volatility estimates.  
Currently, the project implements a full **end-to-end MVP pipeline**: data collection, feature engineering, signal generation, backtesting, and performance evaluation.

---

## Features Implemented
- **Data Collection**
  - Downloads and cleans adjusted OHLCV equity prices (via `yfinance`).
  - Pulls VIX data and converts implied vol to decimal.
  - Merges price and VIX data into a single table indexed by date.

- **Feature Engineering**
  - Computes **log returns**.
  - Creates **forward realized volatility labels** (10-day horizon).
  - Adds baseline predictors:
    - **HV10**: backward 10-day realized volatility.
    - **EWMA**: RiskMetrics-style exponentially weighted vol (λ = 0.94).

- **Signal Generation**
  - Long/short vol signals from predicted vs implied vol with noise-band filtering.
  - T+1 signal execution to prevent look-ahead bias.

- **Backtesting**
  - Variance-spread proxy: compares realized vs implied daily variance.
  - Simulates daily PnL and cumulative equity curve.
  - Saves processed data to Parquet for fast reloads.

- **Performance Metrics**
  - Annualized Sharpe ratio.
  - Max drawdown.
  - Hit rate (fraction of positive PnL days).
  - Total PnL.

- **Visualization**
  - Equity curve plots to quickly check strategy sanity.

---

## Usage
Run the end-to-end MVP pipeline with:
```bash
python -m vol_arbitrage_ml.research.mvp_signals
