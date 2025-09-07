import numpy as np
import pandas as pd


def compute_log_returns(adj_close: pd.Series) -> pd.Series:
    """
    purpose:
        compute log returns r_t = ln(P_t) - ln(P_{t-1})

    parameters:
        adj_close (pd.Series): adjusted close prices

    returns:
        pd.Series: log returns with first value NaN
    """
    # prevent log(0) which would create -inf
    s = adj_close.astype(float).clip(lower=1e-12)
    logp = np.log(s)
    r = logp.diff()
    return r


def hv_vol(log_returns: pd.Series, window: int) -> pd.Series:
    """
    purpose:
        historical realized volatility over a backward window

    details:
        - sample std over 'window' days
        - annualized with sqrt(252)

    returns:
        pd.Series: annualized hv series
    """
    # ddof=1 gives sample std, common in finance
    std = log_returns.rolling(window, min_periods=window).std(ddof=1)
    return std * np.sqrt(252.0)


def ewma_vol(log_returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """
    purpose:
        riskmetrics-style ewma volatility

    details:
        - ewma of squared returns with alpha = 1 - lambda
        - annualized with sqrt(252)

    parameters:
        log_returns (pd.Series): daily log returns
        lam (float): decay lambda in [0,1), default 0.94

    returns:
        pd.Series: ewma volatility (annualized)
    """
    # ewma captures persistence without hard windows
    alpha = 1.0 - lam
    ewvar = log_returns.pow(2).ewm(alpha=alpha).mean()
    return np.sqrt(ewvar * 252.0)