import yfinance as yf
import pandas as pd


def fetch_stock_data(
    ticker: str,
    start: str = "2018-01-01",
    end: str = None
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given stock ticker.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column"
    )

    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")

    # Reset index so Date becomes a column
    df = df.reset_index()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0].lower() if isinstance(col, tuple) else col.lower()
            for col in df.columns
        ]
    else:
        df.columns = [c.lower() for c in df.columns]

    return df


if __name__ == "__main__":
    data = fetch_stock_data("AAPL")
    print(data.head())
