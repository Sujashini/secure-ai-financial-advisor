import os
import altair as alt
import pandas as pd
from functools import reduce
import streamlit as st

from backend.data.market_data import fetch_stock_data
from backend.data.features import add_technical_indicators
from backend.RL.trading_env import TradingEnv
from backend.RL.dqn_agent import DQNAgent
from backend.XAI.explainer import SurrogateExplainer
from backend.Evaluation.backtest import backtest_ticker

from frontend.utils.portfolio_helpers import get_latest_price_and_change


def build_allocation_chart(portfolio):
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


def build_indicator_chart(data: pd.DataFrame, selected_series=None):
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])

    default_series = ["close", "sma_10", "sma_20", "ema_10", "ema_20"]
    if selected_series is None:
        selected_series = default_series

    existing_cols = [c for c in selected_series if c in df.columns]
    if not existing_cols:
        return None

    plot_df = df[["date"] + existing_cols]
    melted = plot_df.melt(
        id_vars="date",
        value_vars=existing_cols,
        var_name="series",
        value_name="value",
    )

    color_domain = ["close", "sma_10", "sma_20", "ema_10", "ema_20"]
    color_range = ["#2563eb", "#f97316", "#22c55e", "#a855f7", "#6b7280"]

    return (
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
        .properties(height=300)
        .interactive()
    )


def build_shap_bar_chart(explanation: dict):
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

    return (
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
                    range=["#22c55e", "#ef4444"],
                ),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("value:Q", title="Contribution"),
                alt.Tooltip("direction:N", title="Direction"),
            ],
        )
        .properties(height=280)
        .interactive()
    )


def build_price_action_chart(data: pd.DataFrame, agent: DQNAgent):
    env = TradingEnv(data)
    state, _ = env.reset()

    rows = []
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
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
                range=["#22c55e", "#ef4444", "#eab308"],
            ),
        ),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("close:Q", title="Price"),
            alt.Tooltip("action_label:N", title="Action"),
        ],
    )

    return (price_line + action_points).interactive()


def build_portfolio_performance_chart(portfolio, freq_code="M"):
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
    merged = merged.sort_values("date").ffill()

    price_cols = [p.ticker for p in portfolio]
    merged = merged.dropna(subset=price_cols, how="all")
    if merged.empty:
        return None

    merged["portfolio_value"] = 0.0
    for pos in portfolio:
        if pos.ticker in merged.columns:
            merged["portfolio_value"] += merged[pos.ticker] * pos.shares

    df = merged[["date", "portfolio_value"]].copy()
    df = df.set_index("date")

    if freq_code in ("M", "Q", "Y"):
        df = df.resample(freq_code).last().dropna()

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

    return (base.mark_area(opacity=0.12) + base.mark_line()).properties(height=310).interactive()


def simulate_rsi_strategy_equity(data: pd.DataFrame, initial_cash=100_000.0):
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

        if rsi < 30 and cash > 0:
            shares = cash / price
            cash = 0.0
        elif rsi > 70 and shares > 0:
            cash = shares * price
            shares = 0.0

        equity = cash + shares * price
        equity_values.append(equity)

    return pd.DataFrame({"date": df["date"], "equity_rsi": equity_values})


def build_strategy_comparison_chart(ticker: str):
    try:
        equity_df, _ = backtest_ticker(
            ticker=ticker,
            model_path=os.path.join("models", f"dqn_{ticker}.pth"),
            initial_cash=100_000.0,
        )
    except Exception:
        return None

    equity_df = equity_df.copy()
    equity_df["date"] = pd.to_datetime(equity_df["date"])

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

    return (
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
        .properties(height=300)
        .interactive()
    )


@st.cache_resource
def load_explainer(ticker: str):
    model_path = os.path.join("models", f"dqn_{ticker}.pth")
    return SurrogateExplainer.build_from_trained_agent(
        model_path=model_path,
        ticker=ticker,
        episodes=5,
    )