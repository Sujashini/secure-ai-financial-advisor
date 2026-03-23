# backend/Evaluation/run_all_backtests.py

import os
import pandas as pd

from backend.Evaluation.backtest import backtest_ticker
from backend.data.market_data import fetch_stock_data
from backend.data.features import add_technical_indicators
from frontend.app import add_drawdowns, simulate_rsi_strategy_equity  # or copy these helpers here

TICKERS = ["AAPL", "MSFT", "NVDA"]
# Common starting capital for all strategies
INITIAL_CASH = 100_000.0
MODELS_DIR = os.path.join("models")

def compute_metrics_for_equity(equity_series: pd.Series):
    """Return total_return, max_drawdown, sharpe for a single equity curve."""
    equity = equity_series.dropna()
    if equity.empty:
        return None, None, None

    # Total return
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0

    # Max drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()  # negative

    # Sharpe (daily)
    returns = equity.pct_change().dropna()
    if returns.empty or returns.std() == 0:
        sharpe = None
    else:
        sharpe = (returns.mean() / returns.std()) * (252 ** 0.5)

    return total_return, max_dd, sharpe

def run_for_ticker(ticker: str):
    """
    Run evaluation for a single stock ticker across:
    - reinforcement learning strategy,
    - buy-and-hold strategy,
    - RSI-based baseline strategy.

    Parameters:
        ticker (str): Stock ticker symbol.

    Returns:
        dict: Summary metrics for all strategies for the given ticker.
    """
    model_path = os.path.join(MODELS_DIR, f"dqn_{ticker}.pth")

    # 1) RL vs Buy & Hold from existing backtest
    equity_df, metrics = backtest_ticker(
        ticker=ticker,
        model_path=model_path,
        initial_cash=INITIAL_CASH,
    )
    equity_df = add_drawdowns(equity_df)
    equity_df["date"] = pd.to_datetime(equity_df["date"])

    # 2) RSI strategy on same data
    price_data = fetch_stock_data(ticker)
    price_data = add_technical_indicators(price_data)
    rsi_df = simulate_rsi_strategy_equity(price_data, initial_cash=INITIAL_CASH)

    merged = pd.merge(equity_df, rsi_df, on="date", how="inner")

    # Compute metrics for each strategy
    res = {"ticker": ticker}

    for col, label in [
        ("equity_ai", "rl"),
        ("equity_bh", "buy_hold"),
        ("equity_rsi", "rsi"),
    ]:
        total_ret, max_dd, sharpe = compute_metrics_for_equity(merged[col])
        res[f"{label}_final_value"] = merged[col].iloc[-1]
        res[f"{label}_total_return"] = total_ret
        res[f"{label}_max_drawdown"] = max_dd
        res[f"{label}_sharpe"] = sharpe

    # Save per-ticker equity curve for plotting later 
    out_dir = os.path.join("backend", "Evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    merged.to_csv(os.path.join(out_dir, f"equity_curves_{ticker}.csv"), index=False)

    return res

def main():
    rows = []
    for t in TICKERS:
        print(f"Running backtest for {t}...")
        rows.append(run_for_ticker(t))

    df = pd.DataFrame(rows)
    out_dir = os.path.join("backend", "Evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False)
    print("Saved summary_metrics.csv")

if __name__ == "__main__":
    main()