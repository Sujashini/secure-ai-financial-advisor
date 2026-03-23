import os
import pandas as pd

from backend.data.market_data import fetch_stock_data
from backend.data.features import add_technical_indicators
from backend.Evaluation.backtest import backtest_ticker
from frontend.utils.constants import TECHY_TICKERS


def get_latest_price_and_change(ticker: str):
    """
    Classify current stock risk level based on recent volatility.

    Parameters:
        data (pd.DataFrame): Market data containing volatility_10.

    Returns:
        tuple:
            risk_label (str): Low / Medium / High / Unknown
            risk_text (str): Plain-English explanation
    """
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
    """
    Fetch the latest available stock price and daily percentage change.

    Parameters:
        ticker (str): Stock ticker symbol.

    Returns:
        tuple:
            current_price (float): Latest closing price
            change_pct (float | None): Daily percentage change
    """
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
    """
    Compute total portfolio value, cost basis, and unrealised profit/loss.

    Parameters:
        portfolio: List of user portfolio positions.

    Returns:
        tuple:
            total_value (float)
            cost_basis (float)
            unrealised_pl (float)
    """
    realised = 0.0
    for trade in trade_history:
        if trade["action"] in ("SELL", "SELL_ALL"):
            cost_price = float(trade.get("cost_price", trade["price"]))
            realised += (float(trade["price"]) - cost_price) * float(trade["shares"])
    return realised


def build_portfolio_positions(portfolio):
    """
    Compute realised profit/loss from recorded sell transactions.

    Parameters:
        trade_history (list): Session trade history records.

    Returns:
        float: Total realised profit/loss
    """
    if not portfolio:
        return []

    rows = []
    total_value = 0.0

    for pos in portfolio:
        try:
            price, change_pct = get_latest_price_and_change(pos.ticker)
        except Exception:
            continue

        if price is None:
            continue

        shares = float(pos.shares)
        avg_price = float(pos.avg_price)
        position_value = price * shares
        cost_value = avg_price * shares
        unrealised_pl = position_value - cost_value
        unrealised_pl_pct = ((price - avg_price) / avg_price * 100.0) if avg_price else 0.0

        rows.append(
            {
                "ticker": pos.ticker,
                "shares": shares,
                "avg_price": avg_price,
                "current_price": float(price),
                "daily_change_pct": change_pct,
                "position_value": position_value,
                "unrealised_pl": unrealised_pl,
                "unrealised_pl_pct": unrealised_pl_pct,
            }
        )
        total_value += position_value

    if total_value > 0:
        for row in rows:
            row["weight"] = row["position_value"] / total_value
    else:
        for row in rows:
            row["weight"] = 0.0

    rows.sort(key=lambda x: x["position_value"], reverse=True)
    return rows


def build_holdings_dataframe(portfolio):
    """
    Build a formatted holdings DataFrame for table display in the UI.
    """
    rows = build_portfolio_positions(portfolio)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["Shares"] = df["shares"].map(lambda x: f"{x:.1f}")
    df["Avg Price ($)"] = df["avg_price"].map(lambda x: f"{x:,.2f}")
    df["Current Price ($)"] = df["current_price"].map(lambda x: f"{x:,.2f}")
    df["Daily Change"] = df["daily_change_pct"].map(
        lambda x: "N/A" if x is None else f"{x:+.2f}%"
    )
    df["Position Value ($)"] = df["position_value"].map(lambda x: f"{x:,.2f}")
    df["Unrealised P/L ($)"] = df["unrealised_pl"].map(lambda x: f"{x:+,.2f}")
    df["Unrealised P/L (%)"] = df["unrealised_pl_pct"].map(lambda x: f"{x:+.2f}%")
    df["Weight"] = df["weight"].map(lambda x: f"{x * 100:.1f}%")

    return df[
        [
            "ticker",
            "Shares",
            "Avg Price ($)",
            "Current Price ($)",
            "Daily Change",
            "Position Value ($)",
            "Unrealised P/L ($)",
            "Unrealised P/L (%)",
            "Weight",
        ]
    ].rename(columns={"ticker": "Ticker"})


def generate_portfolio_alerts(portfolio):
    """
    Returns structured alerts:
    {severity: 'high'|'medium'|'low', title: str, message: str}
    """
    alerts = []
    positions = build_portfolio_positions(portfolio)

    if not positions:
        return alerts

    total_value = sum(p["position_value"] for p in positions)
    if total_value <= 0:
        return alerts

    largest = positions[0]

    if largest["weight"] >= 0.50:
        alerts.append(
            {
                "severity": "high",
                "title": "High concentration risk",
                "message": (
                    f"{largest['ticker']} makes up {largest['weight'] * 100:.1f}% "
                    "of your portfolio. A large single-stock position can increase risk."
                ),
            }
        )
    elif largest["weight"] >= 0.35:
        alerts.append(
            {
                "severity": "medium",
                "title": "Moderate concentration",
                "message": (
                    f"{largest['ticker']} makes up {largest['weight'] * 100:.1f}% "
                    "of your portfolio. Consider whether this matches your intended diversification."
                ),
            }
        )

    tech_value = sum(p["position_value"] for p in positions if p["ticker"] in TECHY_TICKERS)
    tech_share = tech_value / total_value if total_value > 0 else 0.0

    if tech_share >= 0.70:
        alerts.append(
            {
                "severity": "medium",
                "title": "Sector concentration",
                "message": (
                    f"{tech_share * 100:.1f}% of your portfolio is in technology / growth stocks. "
                    "This may make performance more sensitive to one sector."
                ),
            }
        )

    if len(positions) == 1:
        alerts.append(
            {
                "severity": "high",
                "title": "Single-stock portfolio",
                "message": (
                    "You currently hold only one stock. This means your portfolio is not diversified."
                ),
            }
        )
    elif len(positions) == 2:
        alerts.append(
            {
                "severity": "low",
                "title": "Limited diversification",
                "message": (
                    "You hold only two stocks. Diversification is better than a single holding, "
                    "but portfolio-specific risk is still fairly high."
                ),
            }
        )

    losers = [p for p in positions if p["unrealised_pl_pct"] <= -8]
    if losers:
        worst = min(losers, key=lambda x: x["unrealised_pl_pct"])
        alerts.append(
            {
                "severity": "medium",
                "title": "Position under pressure",
                "message": (
                    f"{worst['ticker']} is down {worst['unrealised_pl_pct']:.1f}% versus "
                    "your average purchase price. Review whether the position still fits your strategy."
                ),
            }
        )

    gainers = [p for p in positions if p["unrealised_pl_pct"] >= 12]
    if gainers:
        best = max(gainers, key=lambda x: x["unrealised_pl_pct"])
        alerts.append(
            {
                "severity": "low",
                "title": "Strong winner",
                "message": (
                    f"{best['ticker']} is up {best['unrealised_pl_pct']:.1f}% versus "
                    "your average purchase price. You may want to review if its weight has grown too large."
                ),
            }
        )

    return alerts


def generate_portfolio_takeaway(portfolio):
    """
    Generate structured portfolio alerts for concentration,
    diversification, and strong winners/losers.

    Returns:
        list[dict]: Alerts with severity, title, and message
    """
    positions = build_portfolio_positions(portfolio)

    if not positions:
        return (
            "Your portfolio is currently empty.",
            "Add a position to begin tracking allocation, concentration, and risk signals."
        )

    total_value = sum(p["position_value"] for p in positions)
    total_unrealised = sum(p["unrealised_pl"] for p in positions)
    largest = positions[0]
    num_positions = len(positions)

    performance_text = (
        "overall unrealised gain"
        if total_unrealised > 0
        else "overall unrealised loss"
        if total_unrealised < 0
        else "roughly flat performance"
    )

    diversification_text = (
        "highly concentrated"
        if largest["weight"] >= 0.5
        else "moderately concentrated"
        if largest["weight"] >= 0.35
        else "reasonably spread"
    )

    headline = (
        f"Your portfolio is worth ${total_value:,.2f} across {num_positions} holding"
        f"{'' if num_positions == 1 else 's'}."
    )

    detail = (
        f"It is currently showing an {performance_text} of ${total_unrealised:,.2f} "
        f"and looks {diversification_text}, with {largest['ticker']} making up "
        f"{largest['weight'] * 100:.1f}% of total value."
    )

    return headline, detail


def generate_suggested_next_steps(portfolio):
    """
    Generate a short headline and detail summary describing
    the current portfolio state.
    """
    positions = build_portfolio_positions(portfolio)
    steps = []

    if not positions:
        return [
            "Start with one or two watchlist ideas before placing simulated trades.",
            "Use the explanation page to understand why SAFE-Bot recommends BUY, SELL, or HOLD.",
        ]

    largest = positions[0]

    if largest["weight"] >= 0.5:
        steps.append(
            f"Review whether your exposure to {largest['ticker']} is too high for your risk tolerance."
        )

    if len(positions) <= 2:
        steps.append(
            "Consider whether adding more uncorrelated holdings would improve diversification."
        )

    losing_positions = [p for p in positions if p["unrealised_pl_pct"] <= -8]
    if losing_positions:
        worst = min(losing_positions, key=lambda x: x["unrealised_pl_pct"])
        steps.append(
            f"Re-check the explanation and recent trend for {worst['ticker']} before deciding whether to hold or reduce it."
        )

    winning_positions = [p for p in positions if p["weight"] >= 0.4 and p["unrealised_pl_pct"] >= 10]
    if winning_positions:
        top = max(winning_positions, key=lambda x: x["weight"])
        steps.append(
            f"Ask whether {top['ticker']} has become too dominant after gains and should be rebalanced."
        )

    if not steps:
        steps.append(
            "Your portfolio currently looks fairly balanced. Continue monitoring allocation, sector exposure, and new recommendation changes."
        )

    return steps[:3]


def add_drawdowns(equity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a short list of suggested next actions based on
    the current portfolio condition.
    """
    df = equity_df.copy().sort_values("date")
    df["peak_ai"] = df["equity_ai"].cummax()
    df["peak_bh"] = df["equity_bh"].cummax()
    df["dd_ai"] = (df["equity_ai"] - df["peak_ai"]) / df["peak_ai"]
    df["dd_bh"] = (df["equity_bh"] - df["peak_bh"]) / df["peak_bh"]
    return df


def compute_risk_metrics_for_ticker(ticker: str):
    """
    Compute historical risk metrics for a specific ticker using
    the RL backtest equity curve.

    Returns:
        tuple:
            result (dict | None): Historical metrics
            err (str | None): Error message if calculation fails
    """
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


def explain_risk_metrics(metrics: dict):
    """
    Generate plain-English explanations for return, drawdown,
    and Sharpe ratio values.
    """
    if not metrics:
        return {}

    total_return = metrics.get("total_return")
    max_drawdown = metrics.get("max_drawdown")
    sharpe = metrics.get("sharpe")

    if total_return is None:
        return_text = "Return is not available."
    elif total_return > 0:
        return_text = (
            f"The strategy produced a positive historical return of {total_return * 100:.1f}%. "
            "This suggests growth over the backtest period, though past results do not guarantee future performance."
        )
    elif total_return < 0:
        return_text = (
            f"The strategy produced a negative historical return of {total_return * 100:.1f}%. "
            "This suggests the approach struggled over the backtest period."
        )
    else:
        return_text = "The strategy finished roughly flat over the backtest period."

    if max_drawdown is None:
        drawdown_text = "Worst drop is not available."
    else:
        drawdown_text = (
            f"The worst peak-to-trough decline was {abs(max_drawdown) * 100:.1f}%. "
            "A larger drawdown means the portfolio experienced a deeper temporary loss at some point."
        )

    if sharpe is None:
        sharpe_text = "Sharpe ratio is not available."
    elif sharpe >= 1.0:
        sharpe_text = (
            f"The Sharpe ratio is {sharpe:.2f}, which suggests relatively strong return for the level of volatility taken."
        )
    elif sharpe >= 0.5:
        sharpe_text = (
            f"The Sharpe ratio is {sharpe:.2f}, which suggests moderate risk-adjusted performance."
        )
    elif sharpe >= 0:
        sharpe_text = (
            f"The Sharpe ratio is {sharpe:.2f}, which suggests returns were modest relative to volatility."
        )
    else:
        sharpe_text = (
            f"The Sharpe ratio is {sharpe:.2f}, which suggests the strategy was not well rewarded for the risk taken."
        )

    overall = (
        "These are historical backtest metrics for transparency only. "
        "They help the user understand past behaviour, not predict guaranteed future outcomes."
    )

    return {
        "return_text": return_text,
        "drawdown_text": drawdown_text,
        "sharpe_text": sharpe_text,
        "overall": overall,
    }