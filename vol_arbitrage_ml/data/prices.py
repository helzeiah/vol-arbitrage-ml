from pathlib import Path

import pandas as pd
import yfinance as yf


def download_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    purpose:
        download and clean adjusted daily ohlcv for a single equity

    details:
        - uses yfinance auto_adjust so 'Close' is adjusted
        - flattens any multiindex columns
        - returns columns: date, open, high, low, adj_close, volume

    parameters:
        ticker (str): symbol like "SPY"
        start (str): iso date "YYYY-MM-DD"
        end   (str): iso date "YYYY-MM-DD"

    returns:
        pd.DataFrame: tidy daily prices with adjusted close
    """
    # request adjusted data and stable flat columns
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        actions=False,
        group_by="column",
        progress=False,
    )

    # yfinance may return multiindex (field, ticker)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=1, drop_level=True)
        except Exception:
            df = df.droplevel(0, axis=1)

    # put date in a column and standardize names
    df = df.reset_index().rename(columns={"Date": "date", "Close": "adj_close"})
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Volume": "volume"})

    cols = ["date", "open", "high", "low", "adj_close", "volume"]
    df = df[cols]

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    for c in ["open", "high", "low", "adj_close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    return df

def download_vix(start: str, end: str) -> pd.DataFrame:
    """
    purpose:
        download vix close and return as decimal implied vol

    details:
        - loads ^VIX daily close
        - sets index name to 'date' (tz-naive)
        - converts percent to decimal (e.g., 18.5 -> 0.185)
        - returns columns: date, iv
    """
    # fetch series of closes only
    s = yf.download("^VIX", start=start, end=end, auto_adjust=False, progress=False)["Close"]

    # ensure it's a 1d series and name the value
    if isinstance(s, pd.DataFrame):
        s = s.squeeze(axis=1)
    s = s.astype("float64")
    s.name = "iv"

    # clean index → column with stable names
    idx = pd.to_datetime(s.index).tz_localize(None)
    df = pd.DataFrame({"date": idx, "iv": s.to_numpy()})

    # convert percent to decimal and sort
    df["iv"] = pd.to_numeric(df["iv"], errors="coerce") / 100.0
    df = df.dropna(subset=["iv"]).sort_values("date").reset_index(drop=True)
    return df


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    purpose:
        persist a dataframe to parquet for fast reload

    parameters:
        df (pd.DataFrame): table to save
        path (Path): destination file path
    """
    # create parent dirs to avoid io errors
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    """
    purpose:
        load a parquet if it exists, else raise

    parameters:
        path (Path): source file path

    returns:
        pd.DataFrame: loaded table
    """
    if not path.exists():
        raise FileNotFoundError(f"parquet not found: {path}")
    return pd.read_parquet(path)