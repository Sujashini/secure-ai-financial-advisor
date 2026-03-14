import os
import pandas as pd

from backend.data.market_data import fetch_stock_data
from backend.data.features import add_technical_indicators
from backend.Evaluation.backtest import backtest_ticker
from frontend.utils.constants import TECHY_TICKERS


def get_latest_price_and_change(ticker: str):
    data = fetch_stock_data(ticker)
    data = add_technical_indicators(data)
    data["date"] = pd.to_datetime(data["date"])

    if len(data) < 2:
        current_price = float(data["close"].iloc[-1])
        return current_price, None

    last_close = float(data["close"].iloc[-1])
    prev_close = float(data["close"].iloc[-2])
    change_pct = (last_close - prev_close) / prev_close * 100.0
    return last_close, change_pct


def compute_portfolio_unrealised(portfolio):
    if not portfolio:
        return 0.0, 0.0, 0.0

    total_value = 0.0
    cost_basis = 0.0

    for pos in portfolio:
        try:
            price, _ = get_latest_price_and_change(pos.ticker)
        except Exception:
            continue

        if price is None:
            continue

        total_value += float(price) * float(pos.shares)
        cost_basis += float(pos.avg_price) * float(pos.shares)

    unrealised_pl = total_value - cost_basis
    return total_value, cost_basis, unrealised_pl


def compute_realised_pl(trade_history):
    realised = 0.0
    for trade in trade_history:
        if trade["action"] in ("SELL", "SELL_ALL"):
            cost_price = float(trade.get("cost_price", trade["price"]))
            realised += (float(trade["price"]) - cost_price) * float(trade["shares"])
    return realised


def generate_portfolio_alerts(portfolio):
    alerts = []
    if not portfolio:
        return alerts

    values = []
    total_value = 0.0
    for pos in portfolio:
        try:
            price, _ = get_latest_price_and_change(pos.ticker)
        except Exception:
            continue
        if price is None:
            continue
        v = float(price) * float(pos.shares)
        values.append((pos.ticker, v))
        total_value += v

    if total_value <= 0:
        return alerts

    for ticker, v in values:
        share = v / total_value
        if share >= 0.5:
            alerts.append(
                f"More than 50% of your portfolio value is in {ticker}. This is a very concentrated position."
            )

    tech_value = sum(v for t, v in values if t in TECHY_TICKERS)
    if tech_value / total_value >= 0.7:
        alerts.append(
            "Over 70% of your portfolio is in technology / growth stocks (e.g. AAPL, MSFT, NVDA, TSLA, GOOGL). "
            "This can make your portfolio more sensitive to that sector."
        )

    if len(values) == 1:
        alerts.append(
            "You currently hold only one stock. A single company can be very risky on its own."
        )

    return alerts


def add_drawdowns(equity_df: pd.DataFrame) -> pd.DataFrame:
    df = equity_df.copy().sort_values("date")
    df["peak_ai"] = df["equity_ai"].cummax()
    df["peak_bh"] = df["equity_bh"].cummax()
    df["dd_ai"] = (df["equity_ai"] - df["peak_ai"]) / df["peak_ai"]
    df["dd_bh"] = (df["equity_bh"] - df["peak_bh"]) / df["peak_bh"]
    return df


def compute_risk_metrics_for_ticker(ticker: str):
    try:
        equity_df, metrics = backtest_ticker(
            ticker=ticker,
            model_path=os.path.join("models", f"dqn_{ticker}.pth"),
            initial_cash=100_000.0,
        )
    except Exception as e:
        return None, str(e)

    equity_df = add_drawdowns(equity_df)
    returns = equity_df["equity_ai"].pct_change().dropna()

    if returns.empty:
        sharpe = None
    else:
        mean_daily = returns.mean()
        std_daily = returns.std()
        if std_daily == 0:
            sharpe = None
        else:
            sharpe = (mean_daily / std_daily) * (252 ** 0.5)

    result = {
        "final_value": metrics["final_ai"],
        "total_return": metrics["return_ai"],
        "max_drawdown": metrics["max_drawdown_ai"],
        "sharpe": sharpe,
    }
    return result, None