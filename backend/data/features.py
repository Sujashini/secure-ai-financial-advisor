import pandas as pd
import numpy as np


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators used by the RL agent.
    """
    df = df.copy()

    # -------------------------
    # Ensure 'close' is a clean Series
    # -------------------------
    close_col = df["close"]

    if isinstance(close_col, pd.DataFrame):
        # If multi-column (e.g. from yfinance), take the first column
        close_col = close_col.iloc[:, 0]

    # Force it to be a Series and overwrite
    df["close"] = close_col.astype(float)

    # -------------------------
    # Returns
    # -------------------------
    df["return_1"] = df["close"].pct_change()

    # -------------------------
    # Moving averages
    # -------------------------
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["sma_20"] = df["close"].rolling(window=20).mean()

    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

    # -------------------------
    # Volatility
    # -------------------------
    df["volatility_10"] = df["return_1"].rolling(window=10).std()

    # -------------------------
    # RSI
    # -------------------------
    df["rsi_14"] = compute_rsi(df["close"], window=14)

    # -------------------------
    # Clean NaNs
    # -------------------------
    df = df.dropna().reset_index(drop=True)

    assert df["rsi_14"].between(0, 100).all(), "RSI out of bounds"
    assert df["volatility_10"].ge(0).all(), "Negative volatility detected"

    return df


if __name__ == "__main__":
    from market_data import fetch_stock_data

    data = fetch_stock_data("AAPL")
    data = add_technical_indicators(data)

    print(data.tail())
