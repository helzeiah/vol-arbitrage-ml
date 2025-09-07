from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vol_arbitrage_ml.data.prices import download_prices, download_vix, save_parquet
from vol_arbitrage_ml.features.returns import compute_log_returns, hv_vol, ewma_vol
from vol_arbitrage_ml.labels.realized_vol import forward_realized_vol


def load_join_prices_vix(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    purpose:
        build a single table with prices, returns, labels, and vix iv proxy

    details:
        - downloads clean prices
        - computes log returns and rv10 label
        - joins vix implied vol (decimal)

    returns:
        pd.DataFrame: merged table indexed by date
    """
    # isolate i/o and transforms for reuse
    prices = download_prices(ticker, start, end)
    prices["log_ret"] = compute_log_returns(prices["adj_close"])
    prices["rv10"] = forward_realized_vol(prices["log_ret"], horizon=10, annualize=True)

    vix = download_vix(start, end)

    # inner join keeps aligned days for both series
    df = pd.merge(prices, vix, on="date", how="inner")
    df = df.set_index("date").sort_index()
    return df


def add_baseline_vols(df: pd.DataFrame) -> pd.DataFrame:
    """
    purpose:
        add simple baseline volatility estimates for comparison

    details:
        - hv10: backward 10-day realized vol
        - ewma: riskmetrics lambda=0.94
    """
    # baselines give a benchmark and initial predictor
    df = df.copy()
    df["hv10"] = hv_vol(df["log_ret"], window=10)
    df["ewma"] = ewma_vol(df["log_ret"], lam=0.94)
    return df


def make_signals(df: pd.DataFrame, thresh: float = 0.03, source: str = "ewma") -> pd.DataFrame:
    """
    purpose:
        create long/short volatility signals from predicted vs implied vol

    details:
        - if iv - pred_vol > threshold → short vol (sell) = -1
        - if pred_vol - iv > threshold → long vol (buy) = +1
        - otherwise flat = 0

    parameters:
        thresh (float): band to avoid noise, in vol points (e.g., 0.03 = 3%)
        source (str): column name to use for predicted vol ('ewma' or 'hv10')
    """
    # small band reduces churn and transaction noise
    df = df.copy()
    pred = df[source]
    diff = df["iv"] - pred
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[diff > thresh] = -1
    sig[diff < -thresh] = 1
    df["signal"] = sig
    return df


def backtest_var_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    purpose:
        simulate a simple variance-spread p&l from the vol signal

    details:
        - position acts with 1-day delay (no look-ahead)
        - pnl_daily ≈ position * (realized_var - implied_var) * dt * notional
        - realized_var_daily = log_ret^2
        - implied_var_daily  = iv^2 / 252

    returns:
        pd.DataFrame: adds pnl_daily and equity columns
    """
    df = df.copy()

    # enforce signal delay to avoid look-ahead bias
    pos = df["signal"].shift(1).fillna(0)

    # daily variance comparison is a clean, model-free proxy
    dt = 1.0 / 252.0
    realized_var_daily = df["log_ret"].pow(2)
    implied_var_daily = (df["iv"].pow(2)) / 252.0

    var_spread = realized_var_daily - implied_var_daily
    df["pnl_daily"] = pos * var_spread * dt

    # equity curve shows cumulative effect of daily pnl
    df["equity"] = df["pnl_daily"].cumsum()
    return df


def summarize_performance(df: pd.DataFrame) -> Dict[str, float]:
    """
    purpose:
        compute quick performance stats for the backtest

    returns:
        dict: sharpe (ann), max drawdown, hit rate, total pnl
    """
    # small set of intuitive, comparable metrics
    pnl = df["pnl_daily"].dropna()
    sharpe = 0.0 if pnl.std(ddof=0) == 0 else (pnl.mean() / pnl.std(ddof=0)) * np.sqrt(252.0)

    # drawdown quantifies worst peak-to-trough loss
    eq = df["equity"].fillna(0.0)
    roll_max = eq.cummax()
    dd = eq - roll_max
    max_dd = float(dd.min())

    hit = float((pnl > 0).mean())
    total = float(pnl.sum())

    return {"sharpe": sharpe, "max_dd": max_dd, "hit_rate": hit, "total_pnl": total}


def plot_equity(df: pd.DataFrame, title: str) -> None:
    """
    purpose:
        visualize the cumulative equity curve over time
    """
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(df.index, df["equity"])
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("equity (arbitrary units)")
    fig.tight_layout()
    plt.show()


def run_mvp(ticker: str = "SPY", start: str = "2015-01-01", end: str = "2025-01-01") -> None:
    """
    purpose:
        run the end-to-end mvp: data, labels, baseline, signals, backtest, report
    """
    # build a single, reproducible table for the pipeline
    df = load_join_prices_vix(ticker, start, end)
    df = add_baseline_vols(df)
    df = make_signals(df, thresh=0.03, source="ewma")
    df = backtest_var_proxy(df)

    # basic quality checks
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique

    # persist to processed for later modeling
    out_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_parquet(df.reset_index(), out_dir / f"mvp_{ticker}.parquet")

    # quick metrics to know if it’s sane
    stats = summarize_performance(df)
    print(f"sharpe: {stats['sharpe']:.2f} | max_dd: {stats['max_dd']:.6f} | hit: {stats['hit_rate']:.1%} | total: {stats['total_pnl']:.6f}")

    # a single chart tells you if signals roughly make sense
    plot_equity(df, title=f"{ticker} variance-spread proxy (signal from ewma vs vix)")


if __name__ == "__main__":
    run_mvp()