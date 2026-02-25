import os
import sys
import altair as alt
from functools import reduce
from datetime import datetime

# --- Make sure we can import from project root (backend package) --- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import streamlit as st
import pandas as pd
import re
import json

from backend.data.market_data import fetch_stock_data
from backend.data.features import add_technical_indicators
from backend.RL.trading_env import TradingEnv
from backend.RL.dqn_agent import DQNAgent
from backend.XAI.explainer import SurrogateExplainer
from backend.users.service import (
    authenticate_user,
    create_user,
    get_portfolio,
    change_password,
    reset_password,
    AccountLockedError,
    get_user_by_id,
    buy_shares,
    sell_shares,
)
from backend.Evaluation.backtest import backtest_ticker
from backend.LLM.ollama_chat import chat_with_advisor, summarize_conversation
from backend.LLM.chat_store import (
    init_chat_db,
    load_chat_history,
    save_message,
    clear_chat_history,
)

# Initialise chat DB (creates table if needed)
init_chat_db()

# --- Simple watchlist configuration --- #
WATCHLIST_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]
COMPANY_NAMES = {
    "AAPL": "Apple, Inc",
    "MSFT": "Microsoft Corp",
    "NVDA": "NVIDIA Corp",
    "TSLA": "Tesla, Inc",
    "GOOGL": "Alphabet, Inc",
}

# Simple "sector-ish" bucket for alerts
TECHY_TICKERS = {"AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"}

REMEMBER_ME_PATH = os.path.join(os.path.dirname(__file__), "remember_me.json")


def save_remember_me(user_id: int, remember: bool) -> None:
    """
    Persist or clear the 'remember me' state on disk.
    This is a simple local-file mechanism appropriate for a single-user FYP prototype.
    """
    if remember:
        data = {"remember": True, "user_id": user_id}
        try:
            with open(REMEMBER_ME_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            # Fail quietly – 'remember me' is convenience only
            pass
    else:
        # Clear file if it exists
        try:
            if os.path.exists(REMEMBER_ME_PATH):
                os.remove(REMEMBER_ME_PATH)
        except Exception:
            pass


def evaluate_password_strength(password: str):
    """
    Very simple password strength check.
    Returns (label, score_0_to_1, help_text).
    This is for UX only – actual security is from hashing (bcrypt) in the backend.
    """
    if not password:
        return "Too short", 0.0, "Enter a password to see the strength."

    length = len(password)
    score = 0

    # Length
    if length >= 8:
        score += 1
    if length >= 12:
        score += 1

    # Character classes
    if re.search(r"[a-z]", password):
        score += 1
    if re.search(r"[A-Z]", password):
        score += 1
    if re.search(r"\d", password):
        score += 1
    if re.search(r"[^\w\s]", password):  # special characters
        score += 1

    max_score = 6
    norm = score / max_score

    if length < 8:
        label = "Too short"
        help_text = "Use at least 8 characters."
    elif norm < 0.4:
        label = "Weak"
        help_text = "Add upper/lowercase letters, numbers and a symbol."
    elif norm < 0.75:
        label = "Medium"
        help_text = "Pretty good – you can make it even stronger with more variety."
    else:
        label = "Strong"
        help_text = "This looks like a strong password."

    return label, norm, help_text


def is_valid_email(email: str) -> bool:
    """Very simple email pattern check for nicer error messages."""
    if not email:
        return False
    # This is intentionally simple – just avoids obvious typos.
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))


def generate_plain_english_explanation(
    ticker: str,
    action: int,
    explanation: dict,
) -> str:
    """
    Turn the action + SHAP explanation into a short, user-friendly paragraph.
    """
    action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
    action_text = action_map.get(action, "HOLD")

    friendly_names = {
        "return_1": "very recent price movement",
        "sma_10": "short-term price trend",
        "sma_20": "medium-term price trend",
        "ema_10": "short-term trend (EMA)",
        "ema_20": "medium-term trend (EMA)",
        "volatility_10": "recent price volatility",
        "rsi_14": "momentum (RSI)",
        "open": "recent opening prices",
        "high": "recent highs",
        "low": "recent lows",
        "close": "recent closing prices",
    }

    def format_feature_list(items):
        names = []
        for item in items:
            raw_name = item["feature"]
            name = friendly_names.get(raw_name, raw_name)
            names.append(name)
        if not names:
            return "no single dominant factor"
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return f"{', '.join(names[:-1])}, and {names[-1]}"

    positives = explanation.get("top_positive", [])
    negatives = explanation.get("top_negative", [])

    pos_text = format_feature_list(positives)
    neg_text = format_feature_list(negatives)

    if action_text == "BUY":
        summary = (
            f"For {ticker}, the system currently leans towards **BUY**. "
            f"This is mainly because indicators related to {pos_text} "
            f"look similar to past situations where the price often went up. "
        )
        if negatives:
            summary += (
                f"However, it also sees some caution signals from {neg_text}, "
                "so this is not a guaranteed outcome and is for learning purposes only."
            )
        else:
            summary += (
                "There are no strong opposing signals, but this still does not guarantee any future performance."
            )

    elif action_text == "SELL":
        summary = (
            f"For {ticker}, the system currently leans towards **SELL** or reducing exposure. "
            f"It has detected risk signals from {neg_text}, which look similar to past situations "
            "where the price often fell or became unstable. "
        )
        if positives:
            summary += (
                f"Some positive signs from {pos_text} are still present, "
                "so the picture is mixed and this is not a certainty."
            )
        else:
            summary += "Overall, the balance of signals is tilted towards caution."

    else:  # HOLD
        summary = (
            f"For {ticker}, the system suggests **HOLD**. "
            f"Signals from {pos_text} and {neg_text} are relatively balanced, "
            "so it does not see a strong reason to buy more or to sell right now. "
            "This is meant to indicate uncertainty rather than a clear prediction."
        )

    summary += (
        "\n\nThis explanation is based on patterns in historical data and is provided "
        "for educational and transparency purposes only. It is **not** financial advice."
    )

    return summary


def compute_signal_strength_and_confidence(explanation: dict):
    """
    Heuristic 'confidence' score based on how strong the explanation values are.
    Returns (label, percentage, subtitle).
    """
    abs_vals = []

    for item in explanation.get("top_positive", []):
        abs_vals.append(abs(float(item["value"])))
    for item in explanation.get("top_negative", []):
        abs_vals.append(abs(float(item["value"])))

    if not abs_vals:
        return "Unclear", 0, "Signals are weak or mixed."

    raw_strength = sum(abs_vals)

    # Normalise to 0–100; the constant is just a heuristic scaling factor.
    max_reasonable_strength = 0.1
    confidence_pct = min(raw_strength / max_reasonable_strength * 100.0, 100.0)

    if confidence_pct < 33:
        label = "Low"
        subtitle = "Signals are present but weak or conflicting."
    elif confidence_pct < 66:
        label = "Medium"
        subtitle = "Signals are moderate; there is some uncertainty."
    else:
        label = "High"
        subtitle = "Signals are strong and consistent with past patterns."

    return label, round(confidence_pct), subtitle


def classify_risk_level(data: pd.DataFrame):
    """
    Classify risk level based on recent price volatility.
    Returns (label, explanation).
    """
    if "volatility_10" not in data.columns or data["volatility_10"].dropna().empty:
        return "Unknown", "Not enough data to estimate risk."

    recent_vol = float(data["volatility_10"].dropna().iloc[-1])

    # Thresholds are heuristic, tuned for daily stock data.
    if recent_vol < 0.015:
        return "Low", "Price has been relatively stable in recent history."
    elif recent_vol < 0.035:
        return "Medium", "Price moves up and down moderately."
    else:
        return "High", "Price has been quite jumpy; expect larger swings."


def build_allocation_chart(portfolio):
    """
    Build a pie chart showing percentage allocation per ticker
    using latest prices.
    """
    if not portfolio:
        return None

    rows = []
    for pos in portfolio:
        try:
            price, _ = get_latest_price_and_change(pos.ticker)
            if price is None:
                continue
            value = float(price) * float(pos.shares)
            rows.append({"Ticker": pos.ticker, "Value": value})
        except Exception:
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows)
    total = df["Value"].sum()
    df["Share"] = df["Value"] / total

    chart = (
        alt.Chart(df)
        .mark_arc()
        .encode(
            theta=alt.Theta("Value:Q", stack=True),
            color=alt.Color("Ticker:N", legend=alt.Legend(title="Ticker")),
            tooltip=[
                alt.Tooltip("Ticker:N"),
                alt.Tooltip("Value:Q", format=",.2f", title="Value ($)"),
                alt.Tooltip("Share:Q", format=".1%", title="Portfolio share"),
            ],
        )
        .properties(height=260)
    )
    return chart


def compute_risk_metrics_for_ticker(ticker: str):
    """
    Run the historical backtest for this ticker and compute a small set of
    risk/return metrics for the AI strategy.
    Returns (metrics_dict, error_message_or_None).
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

    # Daily returns of the AI strategy
    returns = equity_df["equity_ai"].pct_change().dropna()
    if returns.empty:
        sharpe = None
    else:
        mean_daily = returns.mean()
        std_daily = returns.std()
        if std_daily == 0:
            sharpe = None
        else:
            # Approx annualised Sharpe with 252 trading days
            sharpe = (mean_daily / std_daily) * (252**0.5)

    result = {
        "final_value": metrics["final_ai"],
        "total_return": metrics["return_ai"],  # fraction
        "max_drawdown": metrics["max_drawdown_ai"],  # fraction
        "sharpe": sharpe,
    }
    return result, None


def compute_portfolio_unrealised(portfolio):
    """
    Returns (total_value, cost_basis, unrealised_pl).
    Uses avg_price * shares as cost basis for each open position.
    """
    if not portfolio:
        return 0.0, 0.0, 0.0

    total_value = 0.0
    cost_basis = 0.0

    for pos in portfolio:
        try:
            price, _ = get_latest_price_and_change(pos.ticker)
        except Exception:
            # If we can't get price, skip that position
            continue

        if price is None:
            continue

        position_value = float(price) * float(pos.shares)
        total_value += position_value
        cost_basis += float(pos.avg_price) * float(pos.shares)

    unrealised_pl = total_value - cost_basis
    return total_value, cost_basis, unrealised_pl


def compute_realised_pl(trade_history):
    """
    Very simple realised P/L estimator based on trade_history.
    We only count SELL / SELL_ALL trades, using the logged cost_price.
    """
    realised = 0.0
    for trade in trade_history:
        if trade["action"] in ("SELL", "SELL_ALL"):
            cost_price = float(trade.get("cost_price", trade["price"]))
            realised += (float(trade["price"]) - cost_price) * float(trade["shares"])
    return realised


def generate_portfolio_alerts(portfolio):
    """
    Return a list of plain-English alerts about concentration / basic risk.
    """
    alerts = []
    if not portfolio:
        return alerts

    # Compute value per position
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

    # Single-stock concentration
    for ticker, v in values:
        share = v / total_value
        if share >= 0.5:
            alerts.append(
                f"More than 50% of your portfolio value is in {ticker}. "
                "This is a very concentrated position."
            )

    # Very simple tech-like concentration
    tech_value = sum(v for t, v in values if t in TECHY_TICKERS)
    if tech_value / total_value >= 0.7:
        alerts.append(
            "Over 70% of your portfolio is in technology / growth stocks "
            "(e.g. AAPL, MSFT, NVDA, TSLA, GOOGL). "
            "This can make your portfolio more sensitive to that sector."
        )

    # Very few holdings
    if len(values) == 1:
        alerts.append(
            "You currently hold only one stock. A single company can be very risky on its own."
        )

    return alerts


def get_latest_price_and_change(ticker: str):
    """
    Return (current_price, daily_change_pct) for the given ticker
    based on the last two closing prices.
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


def build_price_action_chart(data: pd.DataFrame, agent: DQNAgent):
    """
    Simulate the trained agent over the price history and build
    a chart showing close price + BUY/SELL/HOLD markers.
    """
    env = TradingEnv(data)
    state, _ = env.reset()

    rows = []
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _, info = env.step(action)

        idx = env.current_step
        if idx >= len(data):
            break

        rows.append(
            {
                "date": data.iloc[idx]["date"],
                "close": float(data.iloc[idx]["close"]),
                "action": int(action),
            }
        )

        state = next_state

    if not rows:
        return None

    df_traj = pd.DataFrame(rows)
    df_traj["action_label"] = df_traj["action"].map({0: "HOLD", 1: "BUY", 2: "SELL"})

    base = alt.Chart(df_traj).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("close:Q", title="Close price"),
    )

    price_line = base.mark_line()

    action_points = base.mark_point(size=60).encode(
        shape=alt.Shape("action_label:N", title="Action"),
        color=alt.Color(
            "action_label:N",
            title="Action",
            scale=alt.Scale(
                domain=["BUY", "SELL", "HOLD"],
                range=["#2ecc71", "#e74c3c", "#f1c40f"],
            ),
        ),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("close:Q", title="Price"),
            alt.Tooltip("action_label:N", title="Action"),
        ],
    )

    return (price_line + action_points).interactive()


def build_indicator_chart(data: pd.DataFrame, selected_series=None):
    """
    Build a chart showing close price and selected trend indicators over time.
    """
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])

    default_series = ["close", "sma_10", "sma_20", "ema_10", "ema_20"]
    if selected_series is None:
        selected_series = default_series

    existing_cols = [c for c in selected_series if c in df.columns]
    if len(existing_cols) == 0:
        return None

    plot_df = df[["date"] + existing_cols]

    melted = plot_df.melt(
        id_vars="date",
        value_vars=existing_cols,
        var_name="series",
        value_name="value",
    )

    color_domain = ["close", "sma_10", "sma_20", "ema_10", "ema_20"]
    color_range = [
        "#2563eb",
        "#f97316",
        "#22c55e",
        "#a855f7",
        "#6b7280",
    ]

    chart = (
        alt.Chart(melted)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Price / Indicator"),
            color=alt.Color(
                "series:N",
                title="Series",
                scale=alt.Scale(domain=color_domain, range=color_range),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("value:Q", title="Value"),
            ],
        )
        .properties(height=260)
        .interactive()
    )

    return chart


def build_shap_bar_chart(explanation: dict):
    """
    Build a horizontal bar chart showing the most important
    positive and negative SHAP contributors.
    """
    rows = []

    for item in explanation.get("top_positive", []):
        rows.append(
            {
                "feature": item["feature"],
                "value": float(item["value"]),
                "direction": "Positive",
            }
        )

    for item in explanation.get("top_negative", []):
        rows.append(
            {
                "feature": item["feature"],
                "value": float(item["value"]),
                "direction": "Negative",
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", title="Contribution to this decision"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            color=alt.Color(
                "direction:N",
                title="Direction",
                scale=alt.Scale(
                    domain=["Positive", "Negative"],
                    range=["#2ecc71", "#e74c3c"],
                ),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("value:Q", title="Contribution"),
                alt.Tooltip("direction:N", title="Direction"),
            ],
        )
        .properties(height=250)
        .interactive()
    )

    return chart


def add_drawdowns(equity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add drawdown columns for AI and Buy&Hold strategies.
    """
    df = equity_df.copy()
    df = df.sort_values("date")

    df["peak_ai"] = df["equity_ai"].cummax()
    df["peak_bh"] = df["equity_bh"].cummax()

    df["dd_ai"] = (df["equity_ai"] - df["peak_ai"]) / df["peak_ai"]
    df["dd_bh"] = (df["equity_bh"] - df["peak_bh"]) / df["peak_bh"]

    return df


def build_equity_chart(equity_df: pd.DataFrame):
    """
    Altair chart of equity over time for AI strategy vs Buy & Hold.
    """
    df = equity_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    plot_df = df[["date", "equity_ai", "equity_bh"]]

    melted = plot_df.melt(
        id_vars="date",
        value_vars=["equity_ai", "equity_bh"],
        var_name="strategy",
        value_name="equity",
    )

    strategy_name = {
        "equity_ai": "AI strategy",
        "equity_bh": "Buy & hold",
    }
    melted["strategy_label"] = melted["strategy"].map(strategy_name)

    chart = (
        alt.Chart(melted)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("equity:Q", title="Portfolio value ($)"),
            color=alt.Color("strategy_label:N", title="Strategy"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("strategy_label:N", title="Strategy"),
                alt.Tooltip("equity:Q", title="Portfolio value", format=",.0f"),
            ],
        )
        .interactive()
    )

    return chart


def build_drawdown_chart(equity_df: pd.DataFrame):
    """
    Altair chart of drawdown (%) over time for AI vs Buy & Hold.
    """
    df = equity_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    plot_df = df[["date", "dd_ai", "dd_bh"]]

    melted = plot_df.melt(
        id_vars="date",
        value_vars=["dd_ai", "dd_bh"],
        var_name="strategy",
        value_name="drawdown",
    )

    strategy_name = {
        "dd_ai": "AI strategy",
        "dd_bh": "Buy & hold",
    }
    melted["strategy_label"] = melted["strategy"].map(strategy_name)

    chart = (
        alt.Chart(melted)
        .mark_area(opacity=0.6)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(
                "drawdown:Q",
                title="Drawdown (fraction of peak)",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[-1, 0]),
            ),
            color=alt.Color("strategy_label:N", title="Strategy"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("strategy_label:N", title="Strategy"),
                alt.Tooltip("drawdown:Q", title="Drawdown", format=".1%"),
            ],
        )
        .properties(height=250)
        .interactive()
    )

    return chart


def build_portfolio_performance_chart(portfolio, freq_code: str = "M"):
    """
    Build a line chart of total portfolio value over time,
    aggregated to the selected frequency.
    """
    if not portfolio:
        return None

    frames = []
    for pos in portfolio:
        try:
            df = fetch_stock_data(pos.ticker)
            df["date"] = pd.to_datetime(df["date"])
            df = df[["date", "close"]].rename(columns={"close": pos.ticker})
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return None

    merged = reduce(
        lambda left, right: pd.merge(left, right, on="date", how="outer"),
        frames,
    )
    merged = merged.sort_values("date")
    merged = merged.ffill()

    price_cols = [p.ticker for p in portfolio]
    merged = merged.dropna(subset=price_cols, how="all")
    if merged.empty:
        return None

    merged["portfolio_value"] = 0.0
    for pos in portfolio:
        if pos.ticker in merged.columns:
            merged["portfolio_value"] += merged[pos.ticker] * pos.shares

    df = merged[["date", "portfolio_value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    if freq_code in ("M", "Q", "Y"):
        df = df.resample(freq_code).last()
        df = df.dropna()

    df = df.reset_index()

    if freq_code == "M":
        df["label"] = df["date"].dt.strftime("%b %y")
        x_enc = alt.X("label:N", title="Month", sort=list(df["label"]))
    elif freq_code == "Q":
        df["label"] = df["date"].dt.to_period("Q").astype(str)
        x_enc = alt.X("label:N", title="Quarter", sort=list(df["label"]))
    elif freq_code == "Y":
        df["label"] = df["date"].dt.year.astype(str)
        x_enc = alt.X("label:N", title="Year", sort=list(df["label"]))
    else:
        x_enc = alt.X("date:T", title="Date")

    base = alt.Chart(df).encode(
        x=x_enc,
        y=alt.Y("portfolio_value:Q", title="Portfolio value ($)"),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("portfolio_value:Q", title="Portfolio value", format=",.0f"),
        ],
    )

    area = base.mark_area(opacity=0.12)
    line = base.mark_line()

    chart = (area + line).properties(height=260).interactive()
    return chart


def simulate_rsi_strategy_equity(
    data: pd.DataFrame, initial_cash: float = 100_000.0
) -> pd.DataFrame:
    """
    Very simple RSI strategy:
      - If RSI < 30: fully invest (buy with all cash)
      - If RSI > 70: sell all and move to cash
    Returns a DataFrame with columns ['date', 'equity_rsi'].
    """
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])

    if "rsi_14" not in df.columns:
        df = add_technical_indicators(df)

    cash = initial_cash
    shares = 0.0
    equity_values = []

    for _, row in df.iterrows():
        price = float(row["close"])
        rsi = float(row["rsi_14"])

        # Buy signal
        if rsi < 30 and cash > 0:
            shares = cash / price
            cash = 0.0

        # Sell signal
        elif rsi > 70 and shares > 0:
            cash = shares * price
            shares = 0.0

        equity = cash + shares * price
        equity_values.append(equity)

    return pd.DataFrame({"date": df["date"], "equity_rsi": equity_values})


def build_strategy_comparison_chart(ticker: str):
    """
    Build an Altair chart comparing equity over time for:
      - RL AI strategy
      - Buy & hold
      - Simple RSI strategy
    """
    try:
        # Use the existing backtest to get AI vs Buy&Hold
        equity_df, _ = backtest_ticker(
            ticker=ticker,
            model_path=os.path.join("models", f"dqn_{ticker}.pth"),
            initial_cash=100_000.0,
        )
    except Exception:
        return None

    equity_df = equity_df.copy()
    equity_df["date"] = pd.to_datetime(equity_df["date"])

    # Simulate RSI on the same price data
    data = fetch_stock_data(ticker)
    data = add_technical_indicators(data)
    rsi_df = simulate_rsi_strategy_equity(data, initial_cash=100_000.0)

    merged = pd.merge(equity_df, rsi_df, on="date", how="inner")

    plot_df = merged[["date", "equity_ai", "equity_bh", "equity_rsi"]]
    melted = plot_df.melt(
        id_vars="date",
        value_vars=["equity_ai", "equity_bh", "equity_rsi"],
        var_name="strategy",
        value_name="equity",
    )

    name_map = {
        "equity_ai": "RL AI strategy",
        "equity_bh": "Buy & hold",
        "equity_rsi": "RSI strategy",
    }
    melted["strategy_label"] = melted["strategy"].map(name_map)

    chart = (
        alt.Chart(melted)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("equity:Q", title="Portfolio value ($)"),
            color=alt.Color("strategy_label:N", title="Strategy"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("strategy_label:N", title="Strategy"),
                alt.Tooltip("equity:Q", title="Value ($)", format=",.0f"),
            ],
        )
        .properties(height=260)
        .interactive()
    )

    return chart


# ----------------------------
# Auth page helper
# ----------------------------
def show_auth_page():
    """Render a standalone login / sign-up page and handle auth logic."""

    # Give the page a bit of vertical breathing room
    st.empty()
    st.markdown("## 👋 Welcome to the Secure Explainable AI Financial Advisor Bot")
    st.caption(
        "Create an account to save your portfolio and see a personalised dashboard. "
        "This app is for educational purposes only and is **not** financial advice."
    )

    # Center the auth card
    left_spacer, center_col, right_spacer = st.columns([1, 2, 1])
    with center_col:
        login_tab, signup_tab = st.tabs(["Log in", "Sign Up"])

        # ---------- LOGIN TAB ----------
        with login_tab:
            st.subheader("Log in")
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                remember_me = st.checkbox("Remember Me", value=False)
                submitted = st.form_submit_button("Log in")

            if submitted:
                clean_email = email.strip()
                if not clean_email or not password:
                    st.error("Please enter both email and password.")
                elif not is_valid_email(clean_email):
                    st.error("Please enter a valid email address (e.g. name@example.com).")
                else:
                    try:
                        user = authenticate_user(email=email, password=password)
                        if user:
                            st.session_state["user"] = user
                            save_remember_me(user.id, remember_me)
                            st.success(f"Welcome back, {user.username}!")
                            st.rerun()
                        else:
                            st.error(
                                "We could not sign you in. Check your email and password and try again."
                            )
                    except AccountLockedError:
                        st.error(
                            "Your account has been locked due to too many failed login attempts. "
                            "Please use **'Forgot your password?'** below to reset your password and unlock the account."
                        )

            # ---------------- Forgot password ----------------
            with st.expander("Forgot your password?"):
                with st.form("forgot_pw_form"):
                    fp_email = st.text_input("Registered email", key="fp_email")
                    fp_new1 = st.text_input("New password", type="password", key="fp_new1")
                    fp_new2 = st.text_input("Confirm new password", type="password", key="fp_new2")

                    # Show strength for new password
                    fp_label, fp_score, fp_help = evaluate_password_strength(fp_new1)
                    st.markdown(
                        f"<small>Strength: <b>{fp_label}</b> – {fp_help}</small>",
                        unsafe_allow_html=True,
                    )

                    fp_submit = st.form_submit_button("Reset password")

                if fp_submit:
                    clean_fp_email = fp_email.strip()

                    if not clean_fp_email or not fp_new1 or not fp_new2:
                        st.error("Please fill in all the fields.")
                    elif fp_new1 != fp_new2:
                        st.error("New passwords do not match.")
                    elif len(fp_new1) < 8:
                        st.error("New password must be at least 8 characters long.")
                    elif fp_label in ("Too short", "Weak"):
                        st.error("New password is too weak. Please choose a stronger one.")
                    else:
                        try:
                            reset_password(clean_fp_email.strip(), fp_new1)
                            st.success(
                                "Your password has been reset successfully. You can now log in with your new password."
                            )
                        except ValueError as e:
                            # e.g. email not found
                            st.error(str(e))
                        except Exception:
                            st.error(
                                "Something went wrong while resetting your password. Please try again."
                            )

        # ---------- SIGN-UP TAB ----------
        with signup_tab:
            st.subheader("Create a new account")
            with st.form("signup_form"):
                email = st.text_input("Email", key="signup_email")
                username = st.text_input("Username", key="signup_username")
                password = st.text_input(
                    "Password", type="password", key="signup_password"
                )
                # --- Password strength indicator ---
                strength_label, strength_score, strength_help = evaluate_password_strength(
                    password
                )
                st.markdown(
                    f"<small>Strength: <b>{strength_label}</b> – {strength_help}</small>",
                    unsafe_allow_html=True,
                )
                confirm = st.text_input(
                    "Confirm password", type="password", key="signup_confirm"
                )
                submitted = st.form_submit_button("Signup")

            if submitted:
                clean_email = email.strip()
                clean_username = username.strip()

                if not clean_email or not clean_username or not password or not confirm:
                    st.error("Please fill in all fields.")
                elif not is_valid_email(clean_email):
                    st.error("Please enter a valid email address (e.g. name@example.com).")
                elif password != confirm:
                    st.error("Passwords do not match.")
                elif len(password) < 8:
                    st.error("Password must be at least 8 characters long.")
                elif strength_label in ("Too short", "Weak"):
                    st.error("Password is too weak. Please use a stronger password.")
                else:
                    try:
                        user = create_user(
                            email=clean_email.strip(),
                            username=clean_username.strip(),
                            password=password,
                        )
                        st.session_state["user"] = user
                        st.success(f"Account created. Welcome, {user.username}!")
                        st.rerun()
                    except ValueError as e:
                        # e.g. "Email already registered"
                        st.error(str(e))
                    except Exception:
                        st.error(
                            "Something went wrong while creating your account. Please try again."
                        )

    st.markdown("</div>", unsafe_allow_html=True)


# --- Streamlit page config --- #
st.set_page_config(page_title="AI Financial Advisor", layout="wide")

# --- Session state --- #
if "user" not in st.session_state:
    st.session_state["user"] = None

if "trade_history" not in st.session_state:
    st.session_state["trade_history"] = []

    # Try auto-login via remember-me file (only when session starts empty)
    if os.path.exists(REMEMBER_ME_PATH):
        try:
            with open(REMEMBER_ME_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("remember") and data.get("user_id") is not None:
                remembered_user = get_user_by_id(int(data["user_id"]))
                if remembered_user:
                    st.session_state["user"] = remembered_user
        except Exception:
            # If anything goes wrong, just ignore and fall back to normal login
            pass

user = st.session_state["user"]

# =========================
# Top bar: title + user icon
# =========================
top_col1, top_col2, top_col3 = st.columns([0.7, 0.15, 0.15])

with top_col1:
    st.title("📈 Secure Explainable AI Financial Advisor Bot")
    st.caption("Educational prototype – not financial advice.")

with top_col2:
    st.write("")  # spacer

with top_col3:
    if user:
        st.markdown(f"**👤 {user.username}**")
        if st.button("Logout", key="logout_button"):
            save_remember_me(user.id, remember=False)
            st.session_state.clear()
            st.rerun()
    else:
        st.write("")


@st.cache_resource
def load_explainer(ticker: str):
    model_path = os.path.join("models", f"dqn_{ticker}.pth")
    return SurrogateExplainer.build_from_trained_agent(
        model_path=model_path,
        ticker=ticker,
        episodes=5,
    )


# If no user is logged in, show the auth page and stop.
if st.session_state["user"] is None:
    show_auth_page()
    st.stop()

# Refresh local variable after a possible login in show_auth_page
user = st.session_state["user"]

# =========================
# Main Dashboard
# =========================
if user:
    # Choose stock ticker (for detailed charts)
    ticker = st.selectbox(
        "Choose stock to inspect",
        ["AAPL", "MSFT", "NVDA"],
        index=0,
    )

    data = fetch_stock_data(ticker)
    data = add_technical_indicators(data)

    env = TradingEnv(data)
    state, _ = env.reset()

    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    )
    agent.load(os.path.join("models", f"dqn_{ticker}.pth"))
    agent.epsilon = 0.0

    action = agent.select_action(state)
    action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
    action_text = action_map[action]

    # Compute explanation once
    explainer = load_explainer(ticker)
    _, explanation = explainer.explain_state(state)

    tab_rec, tab_expl, tab_chat, tab_profile, tab_help = st.tabs(
        [
            "Dashboard",
            "Explanation",
            "Chat with Advisor",
            "Profile / Settings",
            "Help / Glossary",
        ]
    )

    # ----------------------
    # DASHBOARD TAB
    # ----------------------
    with tab_rec:
        st.metric("Recommended Action", f"{action_text} ({ticker})")

        # ---- AI signal & risk summary row ----
        sig_col, risk_col = st.columns(2)

        with sig_col:
            conf_label, conf_pct, conf_subtitle = compute_signal_strength_and_confidence(
                explanation
            )
            st.markdown("#### 🤖 AI signal strength")
            st.metric("Confidence", f"{conf_pct}%", conf_label)
            st.caption(conf_subtitle)

        with risk_col:
            risk_label, risk_text = classify_risk_level(data)
            st.markdown("#### ⚠️ Risk level (this stock)")
            st.metric("Risk", risk_label)
            st.caption(risk_text)

        with st.expander("Why this suggestion? (key factors)", expanded=False):
            pos = explanation.get("top_positive", [])
            neg = explanation.get("top_negative", [])

            st.markdown("**Main positive signals**")
            if pos:
                for item in pos:
                    st.write(f"- {item['feature']} (pushed towards this action)")
            else:
                st.write("_No strong positive signals identified._")

            st.markdown("**Main caution signals**")
            if neg:
                for item in neg:
                    st.write(f"- {item['feature']} (pushed against this action)")
            else:
                st.write("_No strong caution signals identified._")

            st.caption(
                "These are the main data points the AI looked at. "
                "Technical details (like SHAP values) are documented in the project report."
            )

        st.markdown("### 🛠 Act on this recommendation")

        portfolio = get_portfolio(user.id)

        current_price, _ = get_latest_price_and_change(ticker)

        if action_text == "BUY":
            shares_to_buy = st.number_input(
                "Number of shares to buy",
                min_value=1.0,
                step=1.0,
                key="buy_shares_input",
            )

            if st.button("✅ Buy shares", key="buy_btn"):
                try:
                    buy_shares(
                        user_id=user.id,
                        ticker=ticker,
                        shares=shares_to_buy,
                        price=current_price,
                    )
                    st.session_state["trade_history"].append(
                        {
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "ticker": ticker,
                            "action": "BUY",
                            "shares": float(shares_to_buy),
                            "price": float(current_price),
                        }
                    )
                    st.success(f"Bought {shares_to_buy} shares of {ticker}.")
                    st.rerun()
                except Exception as e:
                    st.error("Could not complete purchase.")
                    st.caption(str(e))

        elif action_text == "SELL":
            owned_position = next((p for p in portfolio if p.ticker == ticker), None)

            if owned_position:
                max_shares = float(owned_position.shares)

                shares_to_sell = st.number_input(
                    "Number of shares to sell",
                    min_value=1.0,
                    max_value=max_shares,
                    step=1.0,
                    key="sell_shares_input",
                )

                if st.button("🔻 Sell shares", key="sell_btn"):
                    try:
                        cost_price = float(owned_position.avg_price)
                        sell_shares(
                            user_id=user.id,
                            ticker=ticker,
                            shares=shares_to_sell,
                            price=current_price,
                        )
                        st.session_state["trade_history"].append(
                            {
                                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "ticker": ticker,
                                "action": "SELL",
                                "shares": float(shares_to_sell),
                                "price": float(current_price),
                                "cost_price": cost_price,
                            }
                        )
                        st.success(f"Sold {shares_to_sell} shares of {ticker}.")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error("Could not complete sale.")
                        st.caption(str(e))
            else:
                st.info("You do not currently own this stock.")
        else:
            st.info("The AI suggests holding. No action required.")

        data["date"] = pd.to_datetime(data["date"])
        latest_date = data["date"].iloc[-1]
        st.caption(
            f"Market data for {ticker} is shown up to {latest_date.date()} "
            "(latest available daily closing prices)."
        )

        portfolio = get_portfolio(user.id)

        # ==========================
        # ROW 1 – OVERVIEW STRIP
        # ==========================
        overview_left, overview_right = st.columns([2, 1])

        # --- Portfolio snapshot cards ---
        with overview_left:
            st.markdown("### 📌 Portfolio snapshot")

            if portfolio:
                snap_cols = st.columns(2)
                for i, pos in enumerate(portfolio):
                    with snap_cols[i % 2]:
                        t = pos.ticker
                        price, change_pct = None, None
                        try:
                            price, change_pct = get_latest_price_and_change(t)
                        except Exception:
                            pass

                        price_str = f"${price:.2f}" if price is not None else "N/A"

                        if change_pct is not None:
                            arrow = "▲" if change_pct >= 0 else "▼"
                            color = "#22c55e" if change_pct >= 0 else "#ef4444"
                            change_html = (
                                f'<span style="color:{color};">{arrow} {change_pct:.2f}%</span>'
                            )
                        else:
                            change_html = '<span style="color:#6b7280;">N/A</span>'

                        ai_label = "BUY"

                        card_html = f"""
<div style="background:#ffffff;border-radius:16px;padding:16px 18px;
            box-shadow:0 2px 6px rgba(15,23,42,0.08);
            margin-bottom:12px;">
  <div style="font-weight:600;font-size:16px;">{t}</div>
  <div style="color:#6b7280;font-size:12px;">Shares: {pos.shares}</div>

  <div style="font-size:20px;font-weight:600;margin-top:6px;">{price_str}</div>

  <div style="font-size:12px;margin-top:4px;">{change_html}</div>

  <span style="display:inline-block;margin-top:8px;
               background:#dcfce7;color:#16a34a;
               padding:4px 10px;border-radius:999px;
               font-size:11px;font-weight:600;">
    AI: {ai_label}
  </span>
</div>
"""
                        st.markdown(card_html, unsafe_allow_html=True)

                        quick_buy_col, quick_sell_col, quick_close_col = st.columns(3)

                        with quick_buy_col:
                            if st.button("Buy 1", key=f"quick_buy_{t}"):
                                try:
                                    buy_shares(
                                        user_id=user.id,
                                        ticker=t,
                                        shares=1.0,
                                        price=price if price is not None else pos.avg_price,
                                    )
                                    st.session_state["trade_history"].append(
                                        {
                                            "time": datetime.now().strftime(
                                                "%Y-%m-%d %H:%M:%S"
                                            ),
                                            "ticker": t,
                                            "action": "BUY",
                                            "shares": 1.0,
                                            "price": float(price)
                                            if price is not None
                                            else float(pos.avg_price),
                                        }
                                    )
                                    st.success(f"Bought 1 share of {t}.")
                                    st.rerun()
                                except Exception as e:
                                    st.error("Could not complete quick buy.")
                                    st.caption(str(e))

                        with quick_sell_col:
                            if pos.shares >= 1 and st.button(
                                "Sell 1", key=f"quick_sell_{t}"
                            ):
                                try:
                                    cost_price = float(pos.avg_price)
                                    sell_shares(
                                        user_id=user.id,
                                        ticker=t,
                                        shares=1.0,
                                        price=price if price is not None else pos.avg_price,
                                    )
                                    st.session_state["trade_history"].append(
                                        {
                                            "time": datetime.now().strftime(
                                                "%Y-%m-%d %H:%M:%S"
                                            ),
                                            "ticker": t,
                                            "action": "SELL",
                                            "shares": 1.0,
                                            "price": float(price)
                                            if price is not None
                                            else float(pos.avg_price),
                                            "cost_price": cost_price,
                                        }
                                    )
                                    st.success(f"Sold 1 share of {t}.")
                                    st.rerun()
                                except ValueError as e:
                                    st.error(str(e))
                                except Exception as e:
                                    st.error("Could not complete quick sell.")
                                    st.caption(str(e))

                        with quick_close_col:
                            if st.button(
                                "Close position",
                                key=f"close_{t}",
                                help="Sell all shares of this stock at the latest available price (simulated).",
                            ):
                                try:
                                    cost_price = float(pos.avg_price)
                                    sell_shares(
                                        user_id=user.id,
                                        ticker=t,
                                        shares=float(pos.shares),
                                        price=price
                                        if price is not None
                                        else pos.avg_price,
                                    )
                                    st.session_state["trade_history"].append(
                                        {
                                            "time": datetime.now().strftime(
                                                "%Y-%m-%d %H:%M:%S"
                                            ),
                                            "ticker": t,
                                            "action": "SELL_ALL",
                                            "shares": float(pos.shares),
                                            "price": float(price)
                                            if price is not None
                                            else float(pos.avg_price),
                                            "cost_price": cost_price,
                                        }
                                    )
                                    st.success(f"Closed position in {t}.")
                                    st.rerun()
                                except ValueError as e:
                                    st.error(str(e))
                                except Exception as e:
                                    st.error("Could not close this position.")
                                    st.caption(str(e))
            else:
                st.info("You don't have any holdings yet.")

        # --- Account summary card ---
        with overview_right:
            st.markdown("### 🧾 Account summary")

            if portfolio:
                total_value, cost_basis, unrealised_pl = compute_portfolio_unrealised(
                    portfolio
                )
                unrealised_pct = 0.0
                if cost_basis > 0:
                    unrealised_pct = unrealised_pl / cost_basis * 100.0

                realised_pl = compute_realised_pl(
                    st.session_state.get("trade_history", [])
                )

                def format_pl(amount):
                    sign = "+" if amount >= 0 else "-"
                    return f"{sign}${abs(amount):,.2f}"

                unrealised_color = "#22c55e" if unrealised_pl >= 0 else "#ef4444"
                realised_color = "#22c55e" if realised_pl >= 0 else "#ef4444"

                card_html = f"""
<div style="background:#ffffff;border-radius:16px;padding:16px 18px;
            box-shadow:0 2px 6px rgba(15,23,42,0.10);
            margin-bottom:12px;">
  <div style="font-size:12px;color:#6b7280;margin-bottom:4px;">Total portfolio value</div>
  <div style="font-size:24px;font-weight:700;">${total_value:,.2f}</div>

  <div style="font-size:12px;margin-top:8px;">
    Unrealised gain/loss:
    <span style="color:{unrealised_color};font-weight:600;">
      {format_pl(unrealised_pl)} ({unrealised_pct:.1f}%)
    </span>
  </div>

  <div style="font-size:12px;margin-top:4px;">
    Realised gain/loss (this session):
    <span style="color:{realised_color};font-weight:600;">
      {format_pl(realised_pl)}
    </span>
  </div>

  <div style="font-size:11px;margin-top:6px;color:#6b7280;">
    Based on simulated trades in this prototype.
  </div>
</div>
"""
                st.markdown(card_html, unsafe_allow_html=True)

                st.caption(
                    "This prototype is for **educational purposes only** and does not constitute financial advice."
                )

                st.markdown("### ⚠️ Portfolio alerts")
                alerts = generate_portfolio_alerts(portfolio)
                if alerts:
                    for msg in alerts:
                        st.warning(msg)
                else:
                    st.success("No obvious concentration alerts based on your current holdings.")
            else:
                st.info("No holdings yet, so the account summary is empty.")

        # Allocation & risk metrics row
        if portfolio:
            alloc_col, riskmetrics_col = st.columns([1, 1])

            with alloc_col:
                st.markdown("### 🧩 Portfolio allocation")
                alloc_chart = build_allocation_chart(portfolio)
                if alloc_chart is not None:
                    st.altair_chart(alloc_chart, use_container_width=True)
                else:
                    st.info("Could not compute allocation chart.")

            with riskmetrics_col:
                st.markdown("### 📊 Historical risk & return (AI strategy)")
                metrics, err = compute_risk_metrics_for_ticker(ticker)
                if err or metrics is None:
                    st.info("Historical risk metrics are not available right now.")
                else:
                    total_return_pct = metrics["total_return"] * 100.0
                    max_dd_pct = metrics["max_drawdown"] * 100.0
                    sharpe = metrics["sharpe"]

                    st.metric("Total return (backtest)", f"{total_return_pct:.1f}%")
                    st.metric("Worst historical drop", f"{max_dd_pct:.1f}%")

                    if sharpe is not None:
                        st.metric("Risk/return score", f"{sharpe:.2f}")
                        st.caption(
                            "Higher is generally better: this score balances historical returns "
                            "against how bumpy the journey was. "
                            "It is based on the Sharpe ratio, explained in the report."
                        )
                    else:
                        st.caption(
                            "Risk/return score is not available (insufficient variation in backtest returns)."
                        )

        # ==========================
        # ROW 2 – MAIN CHARTS
        # ==========================
        st.markdown("### 📊 Portfolio performance & trends")
        st.caption(
            "Total value of your holdings over time, plus technical trends for the selected stock."
        )

        row2_left, row2_right = st.columns([2, 1])

        with row2_left:
            st.markdown("#### Portfolio Performance")
            ctrl_col, chart_col = st.columns([1, 4])

            with ctrl_col:
                st.caption("View by")
                freq_label = st.radio(
                    "",
                    options=["Monthly", "Quarterly", "Annually"],
                    index=0,
                    horizontal=False,
                    key="perf_freq",
                )
                freq_map = {"Monthly": "M", "Quarterly": "Q", "Annually": "Y"}
                freq_code = freq_map[freq_label]

            with chart_col:
                perf_chart = build_portfolio_performance_chart(portfolio, freq_code)
                if perf_chart is not None:
                    st.altair_chart(perf_chart, use_container_width=True)
                else:
                    st.info("Not enough data to show portfolio performance yet.")

        with row2_right:
            st.markdown(f"#### {ticker} trend indicators")
            ctrl_col, chart_col = st.columns([1, 4])

            with ctrl_col:
                st.caption("Show these series:")
                indicator_options = {
                    "Close price": "close",
                    "SMA 10": "sma_10",
                    "SMA 20": "sma_20",
                    "EMA 10": "ema_10",
                    "EMA 20": "ema_20",
                }
                default_checked = {"Close price", "SMA 20"}

                selected_labels = []
                for label, col_name in indicator_options.items():
                    checked = st.checkbox(
                        label,
                        value=(label in default_checked),
                        key=f"ind_{label}",
                    )
                    if checked:
                        selected_labels.append(label)

            with chart_col:
                selected_cols = [indicator_options[label] for label in selected_labels]
                if selected_cols:
                    ind_chart = build_indicator_chart(data, selected_cols)
                    if ind_chart is not None:
                        st.altair_chart(ind_chart, use_container_width=True)
                    else:
                        st.info("Trend indicators are not available for this stock.")
                else:
                    st.info("Select at least one indicator to display.")

        # ==========================
        # Strategy performance comparison (historical)
        # ==========================
        st.markdown("### 🧪 Strategy performance comparison (historical data)")
        st.caption(
            "This compares, on past data only, three simple strategies starting from the same amount: "
            "the RL AI strategy, a basic buy-&-hold, and a simple RSI-based trading rule. "
            "It is for transparency and education only – it does **not** predict the future."
        )

        comp_chart = build_strategy_comparison_chart(ticker)
        if comp_chart is not None:
            st.altair_chart(comp_chart, use_container_width=True)
        else:
            st.info("Could not load historical comparison for this stock.")

        # ==========================
        # ROW 3 – TABLE + MARKET OVERVIEW
        # ==========================
        row3_left, row3_right = st.columns([2, 1])

        # --- LEFT: advisor suggestions table ---
        with row3_left:
            if portfolio:
                st.markdown("### 🔄 Advisor suggestions for your holdings")
                st.caption(
                    "For each stock you hold, the advisor looks at recent market data and "
                    "suggests whether to BUY more, SELL, or HOLD. The table also shows "
                    "your position (shares and average price)."
                )

                rows = []

                for pos in portfolio:
                    t = pos.ticker
                    model_path = os.path.join("models", f"dqn_{t}.pth")

                    if not os.path.exists(model_path):
                        rows.append(
                            {
                                "Ticker": t,
                                "Shares": pos.shares,
                                "Avg price ($)": f"{pos.avg_price:.2f}",
                                "Current price ($)": "N/A",
                                "AI action": "N/A",
                                "Short explanation": "No trained model is available for this stock yet.",
                            }
                        )
                        continue

                    try:
                        data_t = fetch_stock_data(t)
                        data_t = add_technical_indicators(data_t)
                        data_t["date"] = pd.to_datetime(data_t["date"])

                        current_price = float(data_t["close"].iloc[-1])

                        env_t = TradingEnv(data_t)
                        state_t, _ = env_t.reset()

                        agent_t = DQNAgent(
                            state_dim=env_t.observation_space.shape[0],
                            action_dim=env_t.action_space.n,
                        )
                        agent_t.load(model_path)
                        agent_t.epsilon = 0.0

                        action_t = agent_t.select_action(state_t)
                        action_text_t = action_map.get(action_t, "HOLD")

                        explainer_t = load_explainer(t)
                        _, explanation_t = explainer_t.explain_state(state_t)

                        full_expl = generate_plain_english_explanation(
                            ticker=t,
                            action=action_t,
                            explanation=explanation_t,
                        )

                        short_expl = full_expl.split(".")[0].strip()
                        if short_expl:
                            short_expl = short_expl + "."
                        if len(short_expl) > 200:
                            short_expl = short_expl[:197] + "..."

                        rows.append(
                            {
                                "Ticker": t,
                                "Shares": pos.shares,
                                "Avg price ($)": f"{pos.avg_price:.2f}",
                                "Current price ($)": f"{current_price:.2f}",
                                "AI action": action_text_t,
                                "Short explanation": short_expl,
                            }
                        )

                    except Exception:
                        rows.append(
                            {
                                "Ticker": t,
                                "Shares": pos.shares,
                                "Avg price ($)": f"{pos.avg_price:.2f}",
                                "Current price ($)": "N/A",
                                "AI action": "N/A",
                                "Short explanation": "Could not generate a recommendation for this stock.",
                            }
                        )

                rec_df = pd.DataFrame(rows)
                st.dataframe(rec_df, use_container_width=True)
            else:
                st.info("No advisor suggestions yet – your portfolio is empty.")

        # --- RIGHT: trending + watchlist stack ---
        with row3_right:
            watchlist_rows = []
            for wt in WATCHLIST_TICKERS:
                try:
                    price, change_pct = get_latest_price_and_change(wt)
                    watchlist_rows.append(
                        {
                            "Ticker": wt,
                            "Name": COMPANY_NAMES.get(wt, ""),
                            "Last price": price,
                            "Daily change (%)": change_pct,
                        }
                    )
                except Exception:
                    continue

            st.markdown("#### 🔥 Trending stocks")
            if watchlist_rows:
                trending_list = sorted(
                    watchlist_rows,
                    key=lambda r: abs(r["Daily change (%)"] or 0),
                    reverse=True,
                )[:3]

                card_cols = st.columns(len(trending_list))
                for i, r in enumerate(trending_list):
                    with card_cols[i]:
                        t = r["Ticker"]
                        name = r["Name"]
                        price = r["Last price"]
                        change_pct = r["Daily change (%)"]

                        price_str = f"${price:.2f}" if price is not None else "N/A"
                        if change_pct is not None:
                            arrow = "▲" if change_pct >= 0 else "▼"
                            change_color = "#22c55e" if change_pct >= 0 else "#ef4444"
                            change_html = (
                                f'<span style="color:{change_color};">'
                                f"{arrow} {change_pct:.2f}%</span>"
                            )
                        else:
                            change_html = '<span style="color:#6b7280;">N/A</span>'

                        card_html = f"""
<div style="background-color:#ffffff;border-radius:18px;
            padding:16px 18px;box-shadow:0 2px 6px rgba(15,23,42,0.10);
            margin-bottom:12px;">
  <div style="font-weight:600;font-size:16px;">{t}</div>
  <div style="color:#6b7280;font-size:12px;margin-bottom:10px;">{name}</div>
  <div style="display:flex;justify-content:space-between;align-items:center;
              margin-bottom:12px;">
    <span style="font-size:18px;font-weight:600;">{price_str}</span>
    <span style="font-size:13px;">{change_html}</span>
  </div>
  <div style="display:flex;gap:8px;margin-top:4px;">
    <div style="flex:1;border-radius:999px;border:1px solid #e5e7eb;
                padding:6px 0;text-align:center;font-size:12px;color:#374151;">
      Simulate short
    </div>
    <div style="flex:1;border-radius:999px;background-color:#4f46e5;
                color:#ffffff;padding:6px 0;text-align:center;font-size:12px;">
      Simulate buy
    </div>
  </div>
</div>
"""
                        st.markdown(card_html, unsafe_allow_html=True)
            else:
                st.info("No watchlist data available yet.")

            # My watchlist
            st.markdown("#### 👀 My watchlist")
            if watchlist_rows:
                # Outer card containing *all* rows
                rows_html = (
                    '<div style="background-color:#ffffff;border-radius:18px;'
                    'padding:12px 16px;box-shadow:0 2px 6px rgba(15,23,42,0.10);'
                    'margin-bottom:12px;">'
                )

                for idx, r in enumerate(watchlist_rows):
                    t = r["Ticker"]
                    name = r["Name"]
                    price = r["Last price"]
                    change_pct = r["Daily change (%)"]

                    price_str = f"${price:,.2f}" if price is not None else "N/A"
                    if change_pct is not None:
                        arrow = "▲" if change_pct >= 0 else "▼"
                        change_color = "#22c55e" if change_pct >= 0 else "#ef4444"
                        change_str = (
                            f'<span style="color:{change_color};">'
                            f"{arrow} {change_pct:.2f}%</span>"
                        )
                    else:
                        change_str = '<span style="color:#6b7280;">N/A</span>'

                    # Bottom border for all but last row
                    border_style = (
                        "border-bottom:1px solid #e5e7eb;"
                        if idx < len(watchlist_rows) - 1
                        else ""
                    )

                    rows_html += (
                        '<div style="display:flex;justify-content:space-between;'
                        f'align-items:center;padding:10px 0;{border_style}">'
                        "<div>"
                        f'<div style="font-weight:600;font-size:14px;">{t}</div>'
                        f'<div style="color:#6b7280;font-size:11px;">{name}</div>'
                        "</div>"
                        '<div style="text-align:right;">'
                        f'<div style="font-size:13px;font-weight:500;">{price_str}</div>'
                        f'<div style="font-size:12px;margin-top:2px;">{change_str}</div>'
                        "</div>"
                        "</div>"
                    )

                rows_html += "</div>"
                st.markdown(rows_html, unsafe_allow_html=True)
            else:
                st.info("Your watchlist is empty or could not be loaded.")

    # ----------------------
    # EXPLANATION TAB
    # ----------------------
    with tab_expl:
        st.subheader("🧠 Why this recommendation?")

        explainer = load_explainer(ticker)
        _, explanation = explainer.explain_state(state)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Positive contributors**")
            if explanation["top_positive"]:
                for item in explanation["top_positive"]:
                    st.write(f"- {item['feature']}: {item['value']:.4f}")
            else:
                st.write("_No strong positive contributors identified._")

        with col2:
            st.markdown("**Negative contributors**")
            if explanation["top_negative"]:
                for item in explanation["top_negative"]:
                    st.write(f"- {item['feature']}: {item['value']:.4f}")
            else:
                st.write("_No strong negative contributors identified._")

        shap_chart = build_shap_bar_chart(explanation)
        if shap_chart is not None:
            st.subheader("📊 Feature importance for this decision")
            st.caption(
                "Each bar shows how strongly a feature influenced this decision. "
                "Green bars pushed the AI more towards the chosen action, "
                "red bars pushed against it."
            )
            st.altair_chart(shap_chart, use_container_width=True)
        else:
            st.info("Not enough explanation data to show a feature importance chart.")

        st.markdown(f"### 📈 Price & AI decisions for {ticker}")

        price_chart = build_price_action_chart(data, agent)
        if price_chart is not None:
            st.altair_chart(price_chart, use_container_width=True)
        else:
            st.info("Not enough data to display the price chart.")

        st.subheader("📝 What this means in plain English")

        plain_text = generate_plain_english_explanation(
            ticker=ticker,
            action=action,
            explanation=explanation,
        )
        st.write(plain_text)

        # ----------------------
        # CHAT TAB
        # ----------------------
        with tab_chat:
            st.subheader("💬 Chat with the Advisor")

            # --- Strong educational disclaimer --- #
            st.info(
                "This is an **educational prototype**, not financial advice.\n\n"
                "- It explains how the demo model is thinking about a single stock.\n"
                "- It **cannot** consider your full financial situation.\n"
                "- Do not make real trading decisions based on this tool."
            )

            # --- Scope / capabilities --- #
            with st.expander("What this chat can and cannot do"):
                st.markdown(
                    """
                    **✅ This chat *can* help you:**
                    - Understand *why* the model suggests BUY / SELL / HOLD.
                    - Interpret technical indicators and risk levels.
                    - Learn basic investing and risk management concepts.
                    - Understand limitations of the RL model and backtests.

                    **🚫 This chat *cannot* do:**
                    - Tell you exactly what to buy or sell in real life.
                    - Give personalised financial advice or consider your full situation.
                    - Guarantee profits or predict the future with certainty.
                    """
                )

            # --- Context banner: ticker, AI action, confidence, risk --- #
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current ticker", ticker)
            with col2:
                st.metric("Model action", action_text)
            with col3:
                st.metric("Signal confidence", f"{conf_pct}%")

            st.caption(
                f"Risk level for {ticker}: **{risk_label}**. "
                "Responses are generated by a local language model (Ollama) and are for **educational purposes only**."
            )

            st.divider()

            # --- Load conversation history from SQLite for this user + ticker --- #
            chat_history = load_chat_history(
                user_id=user.id,
                ticker=ticker,
                limit=50,
            )

            # --- Optional summarise button if chat is long --- #
            if len(chat_history) >= 8:
                if st.button("🧾 Summarise conversation so far"):
                    convo_text_for_summary = ""
                    for m in chat_history:
                        speaker = "User" if m["role"] == "user" else "Advisor"
                        convo_text_for_summary += f"{speaker}: {m['content']}\n"

                    with st.spinner("Summarising conversation..."):
                        try:
                            summary = summarize_conversation(
                                ticker=ticker,
                                conversation_history=convo_text_for_summary,
                            )
                            st.markdown("**Conversation summary so far:**")
                            st.write(summary)
                        except Exception:
                            st.warning(
                                "Sorry, I couldn't summarise the conversation right now."
                            )

            # --- Show conversation (chat bubbles: user left, advisor right) --- #
            st.markdown("### Conversation")

            if not chat_history:
                st.write(
                    "No questions yet. Try asking: *“Why is it suggesting HOLD for this stock?”*"
                )
            else:
                for msg in chat_history:
                    if msg["role"] == "user":
                        # User message on the LEFT
                        col_left, col_right = st.columns([0.7, 0.3])
                        with col_left:
                            st.markdown(
                                f"""
                                <div style="
                                    background-color:#e8f4ff;
                                    padding:8px 12px;
                                    border-radius:16px;
                                    margin-bottom:6px;
                                    max-width:100%;
                                ">
                                    <strong>You:</strong><br>{msg['content']}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        with col_right:
                            st.write("")
                    else:
                        # Advisor message on the RIGHT
                        col_left, col_right = st.columns([0.3, 0.7])
                        with col_left:
                            st.write("")
                        with col_right:
                            st.markdown(
                                f"""
                                <div style="
                                    background-color:#f4f4f4;
                                    padding:8px 12px;
                                    border-radius:16px;
                                    margin-bottom:6px;
                                    max-width:100%;
                                ">
                                    <strong>Advisor:</strong><br>{msg['content']}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            st.divider()

            # --- Build explanation context for LLM (grounding) --- #
            pos = explanation.get("top_positive", [])
            neg = explanation.get("top_negative", [])

            pos_features = [item["feature"] for item in pos]
            neg_features = [item["feature"] for item in neg]

            pos_text = (
                ", ".join(pos_features) if pos_features else "no strong positive signals"
            )
            neg_text = (
                ", ".join(neg_features) if neg_features else "no strong negative signals"
            )

            backtest_summary = (
                "Strategy comparison summary for this stock:\n"
                "- RL strategy performance: see the 'Strategy performance comparison' chart.\n"
                "- Buy & Hold baseline: holding the stock over the same period.\n"
                "- RSI strategy: simple technical indicator-based strategy.\n"
                "These are **historical tests** and do not guarantee future performance."
            )

            rl_confidence = conf_pct / 100.0 if conf_pct is not None else None

            # -------------------------
            # Quick questions (auto-send)
            # -------------------------
            st.markdown("### Quick questions")

            qp_col1, qp_col2 = st.columns(2)
            qp_col3, qp_col4 = st.columns(2)

            quick_question = None

            with qp_col1:
                if st.button("📈 Why is it recommending this?"):
                    quick_question = (
                        "Why is the model recommending this action for this stock?"
                    )

            with qp_col2:
                if st.button("⚠️ What are the risks?"):
                    quick_question = (
                        "What are the main risks for this stock according to the model?"
                    )

            with qp_col3:
                if st.button("📊 How does this compare to Buy & Hold?"):
                    quick_question = (
                        "How does the RL strategy compare to a simple Buy and Hold strategy on past data?"
                    )

            with qp_col4:
                if st.button("🔎 What indicators influenced this?"):
                    quick_question = (
                        "Which indicators or features most influenced this recommendation?"
                    )

            # -------------------------
            # Free-text question
            # -------------------------
            st.markdown("### Ask your own question")

            user_q = st.text_area(
                "Your question",
                placeholder="Ask something about the recommendation, risk, or historical testing...",
                key="advisor_text",
            )

            btn_col1, btn_col2 = st.columns([0.6, 0.4])
            with btn_col1:
                ask_clicked = st.button("Ask the advisor", key="ask_advisor_btn")
            with btn_col2:
                clear_clicked = st.button("🗑️ Clear conversation", key="clear_convo_btn")

            # Clear chat for this user + ticker
            if clear_clicked:
                clear_chat_history(user.id, ticker)
                st.success("Conversation cleared for this stock.")
                st.rerun()

            # Decide if we are sending a question this run
            question_to_send = None

            # Priority 1: quick question buttons (auto-send)
            if quick_question is not None:
                question_to_send = quick_question
            # Priority 2: manual "Ask the advisor" button
            elif ask_clicked and user_q.strip():
                question_to_send = user_q.strip()

            # If we have something to send, call the model once
            if question_to_send is not None:
                # Build short conversation history string for LLM (for grounding)
                recent_msgs = load_chat_history(
                    user_id=user.id,
                    ticker=ticker,
                    limit=20,
                )
                convo_text = ""
                for m in recent_msgs:
                    speaker = "User" if m["role"] == "user" else "Advisor"
                    convo_text += f"{speaker}: {m['content']}\n"

                try:
                    answer = chat_with_advisor(
                        user_question=question_to_send,
                        ticker=ticker,
                        action_text=action_text,
                        pos_text=pos_text,
                        neg_text=neg_text,
                        backtest_summary=backtest_summary,
                        conversation_history=convo_text,
                        rl_confidence=rl_confidence,
                        risk_label=risk_label,
                    )

                    # Save both user and advisor messages to SQLite
                    save_message(user.id, ticker, "user", question_to_send)
                    save_message(user.id, ticker, "assistant", answer)

                    st.rerun()

                except Exception as e:
                    st.error("Could not reach the local language model.")
                    st.caption(str(e))

            st.divider()

            # -------------------------
            # Suggested follow-up questions (also auto-send)
            # -------------------------
            st.markdown("### 🔁 Suggested follow-up questions")

            fu_col1, fu_col2 = st.columns(2)
            follow_up_question = None

            with fu_col1:
                if st.button("What could cause this to change?"):
                    follow_up_question = (
                        "What could cause the model's recommendation for this stock to change?"
                    )

            with fu_col2:
                if st.button("How does the risk level affect this?"):
                    follow_up_question = (
                        "How does the current risk level affect the interpretation of this recommendation?"
                    )

            if follow_up_question is not None:
                # Build short conversation history again
                recent_msgs = load_chat_history(
                    user_id=user.id,
                    ticker=ticker,
                    limit=20,
                )
                convo_text = ""
                for m in recent_msgs:
                    speaker = "User" if m["role"] == "user" else "Advisor"
                    convo_text += f"{speaker}: {m['content']}\n"

                try:
                    answer = chat_with_advisor(
                        user_question=follow_up_question,
                        ticker=ticker,
                        action_text=action_text,
                        pos_text=pos_text,
                        neg_text=neg_text,
                        backtest_summary=backtest_summary,
                        conversation_history=convo_text,
                        rl_confidence=rl_confidence,
                        risk_label=risk_label,
                    )

                    save_message(user.id, ticker, "user", follow_up_question)
                    save_message(user.id, ticker, "assistant", answer)
                    st.rerun()

                except Exception as e:
                    st.error("Could not reach the local language model.")
                    st.caption(str(e))

    # ----------------------
    # PROFILE / SETTINGS TAB
    # ----------------------
    with tab_profile:
        st.subheader("👤 Profile & Settings")

        st.markdown(f"**Username:** {user.username}")
        st.markdown(f"**Email:** {user.email}")

        st.markdown("---")
        st.markdown("### Change password")

        old_pw = st.text_input("Current password", type="password", key="prof_old_pw")
        new_pw1 = st.text_input("New password", type="password", key="prof_new_pw1")
        new_pw2 = st.text_input("Confirm new password", type="password", key="prof_new_pw2")

        if st.button("Update password", key="btn_change_pw"):
            if not old_pw or not new_pw1 or not new_pw2:
                st.warning("Please fill in all the fields.")
            elif new_pw1 != new_pw2:
                st.error("New passwords do not match.")
            elif len(new_pw1) < 8:
                st.error("New password must be at least 8 characters long.")
            else:
                try:
                    change_password(user.id, old_pw, new_pw1)
                    st.success("Password updated successfully.")
                except ValueError as e:
                    st.error(str(e))
                except Exception:
                    st.error(
                        "Something went wrong while updating your password. Please try again."
                    )

    # ----------------------
    # HELP / GLOSSARY TAB
    # ----------------------
    with tab_help:
        st.subheader("❓ Help / Glossary")

        st.markdown(
            """
**What do BUY / SELL / HOLD mean here?**

- **BUY** – The system has found patterns similar to past situations where the price often went up.  
- **SELL** – The system has detected patterns similar to past situations where the price often fell or became unstable.  
- **HOLD** – Signals are mixed or weak.

These are *educational signals only* and **not** financial advice.

---

**Indicators used**

- **SMA / EMA** – Short- and medium-term price trends  
- **RSI** – Momentum indicator  
- **Volatility** – How much prices fluctuate

---

**Testing**

- **AI strategy** – Simulated past performance of the AI  
- **Buy & hold** – Buying once and holding

Everything shown in this app is based on **historical data** and is intended **only for learning**.
"""
        )

else:
    st.info("Please log in using the 👤 User menu at the top-right to use the advisor.")