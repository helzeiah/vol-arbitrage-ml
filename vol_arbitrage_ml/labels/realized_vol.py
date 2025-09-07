import numpy as np
import pandas as pd


def forward_realized_vol(log_returns: pd.Series, horizon: int = 10, annualize: bool = True) -> pd.Series:
    """
    purpose:
        create a forward-looking realized volatility label over the next horizon days

    details:
        - compute rolling std over 'horizon' (backward)
        - shift by -horizon so label at t uses r_{t+1..t+h}
        - optional annualization by sqrt(252)

    parameters:
        log_returns (pd.Series): daily log returns
        horizon (int): number of future days summarized
        annualize (bool): multiply by sqrt(252) if True

    returns:
        pd.Series: forward realized vol label; last `horizon` are NaN
    """
    # start with backward std to reuse fast pandas window
    back_std = log_returns.rolling(window=horizon, min_periods=horizon).std(ddof=1)
    # shift left to align with the start (forward label, no leakage)
    fwd = back_std.shift(-horizon)
    # wmake units comparable with implied vol
    if annualize:
        fwd = fwd * np.sqrt(252.0)
    return fwd.rename(f"rv{horizon}_ann" if annualize else f"rv{horizon}")