import os
import sys
import altair as alt
from functools import reduce

# --- Make sure we can import from project root (backend package) --- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
import streamlit as st
import pandas as pd
import re

from backend.data.market_data import fetch_stock_data
from backend.data.features import add_technical_indicators
from backend.RL.trading_env import TradingEnv
from backend.RL.dqn_agent import DQNAgent
from backend.XAI.explainer import SurrogateExplainer
from backend.users.service import authenticate_user, create_user, get_portfolio, change_password, reset_password, AccountLockedError
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

        login_tab, signup_tab = st.tabs(["Log in", "Create account"])

        # ---------- LOGIN TAB ----------
        with login_tab:
            st.subheader("Log in")
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log in")

            if submitted:
                if not email or not password:
                    st.error("Please enter both email and password.")
                else:
                    try:
                        user = authenticate_user(email=email, password=password)
                        if user:
                            st.session_state["user"] = user
                            st.success(f"Welcome back, {user.username}!")
                            st.rerun()
                        else:
                            st.error("Invalid email or password.")
                    except AccountLockedError as e:
                        st.error("Your account has been locked due to too many failed login attempts. "
                "Please use 'Forgot your password?' below to reset your password and unlock the account.")

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
                    if not fp_email or not fp_new1 or not fp_new2:
                        st.error("Please fill in all the fields.")
                    elif fp_new1 != fp_new2:
                        st.error("New passwords do not match.")
                    elif len(fp_new1) < 8:
                        st.error("New password must be at least 8 characters long.")
                    elif fp_label in ("Too short", "Weak"):
                        st.error("New password is too weak. Please choose a stronger one.")
                    else:
                        try:
                            reset_password(fp_email.strip(), fp_new1)
                            st.success(
                                "Password reset successfully. You can now log in with your new password."
                            )
                        except ValueError as e:
                            # e.g. email not found
                            st.error(str(e))
                        except Exception:
                            st.error("Something went wrong while resetting your password.")

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
                strength_label, strength_score, strength_help = evaluate_password_strength(password)
                # simple text indicator (you could add a progress bar too)
                st.markdown(
                    f"<small>Strength: <b>{strength_label}</b> – {strength_help}</small>",
                    unsafe_allow_html=True,
                )
                confirm = st.text_input(
                    "Confirm password", type="password", key="signup_confirm"
                )
                submitted = st.form_submit_button("Create account")

            if submitted:
                if not email or not username or not password or not confirm:
                    st.error("Please fill in all fields.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                elif len(password) < 8:
                    st.error("Password must be at least 8 characters long.")
                elif strength_label in ("Too short", "Weak"):
                    st.error("Password is too weak. Please use a stronger password.")
                else:
                    try:
                        user = create_user(
                            email=email.strip(),
                            username=username.strip(),
                            password=password,
                        )
                        st.session_state["user"] = user
                        st.success(f"Account created. Welcome, {user.username}!")
                        st.rerun()
                    except ValueError as e:
                        # e.g. "Email already registered"
                        st.error(str(e))
                    except Exception:
                        st.error("Something went wrong while creating your account.")

        st.markdown("</div>", unsafe_allow_html=True)


# --- Streamlit page config --- #
st.set_page_config(page_title="AI Financial Advisor", layout="wide")

## --- Session state for user --- #
if "user" not in st.session_state:
    st.session_state["user"] = None

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
    # Label on the little user pill
    if user:
        st.markdown(f"**👤 {user.username}**")
        if st.button("Logout", key="logout_button"):
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

# --- Initialise chat history from DB when user logs in --- #
if user and "chat_history" not in st.session_state:
    st.session_state["chat_history"] = load_chat_history(user.id, ticker=None)


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

    tab_rec, tab_expl, tab_eval, tab_chat, tab_profile, tab_help = st.tabs(
        ["Dashboard", "Explanation", "Evaluation", "Chat with Advisor", "Profile / Settings", "Help / Glossary"]
    )

    # ----------------------
    # DASHBOARD TAB
    # ----------------------
    with tab_rec:
        st.metric("Recommended Action", f"{action_text} ({ticker})")

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

                        # For now, simple AI label; you can wire in per-ticker logic.
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
            else:
                st.info("You don't have any holdings yet.")

        # --- Account summary card ---
        with overview_right:
            st.markdown("### 🧾 Account summary")

            if portfolio:
                total_value = 0.0
                daily_pl = 0.0

                for pos in portfolio:
                    try:
                        price, change_pct = get_latest_price_and_change(pos.ticker)
                    except Exception:
                        continue

                    if price is None:
                        continue

                    position_value = price * pos.shares
                    total_value += position_value

                    if change_pct is not None:
                        daily_pl += position_value * (change_pct / 100.0)

                holdings_count = len(portfolio)

                pl_color = "#22c55e" if daily_pl >= 0 else "#ef4444"
                sign = "+" if daily_pl >= 0 else "-"
                pl_str = f"{sign}${abs(daily_pl):.2f}"

                card_html = f"""
<div style="background:#ffffff;border-radius:16px;padding:16px 18px;
            box-shadow:0 2px 6px rgba(15,23,42,0.10);
            margin-bottom:12px;">
  <div style="font-size:12px;color:#6b7280;margin-bottom:4px;">Total portfolio value</div>
  <div style="font-size:24px;font-weight:700;">${total_value:,.2f}</div>

  <div style="font-size:12px;margin-top:8px;">
    Daily P&amp;L:
    <span style="color:{pl_color};font-weight:600;">{pl_str}</span>
  </div>

  <div style="font-size:12px;margin-top:4px;color:#6b7280;">
    Holdings: {holdings_count}
  </div>
</div>
"""
                st.markdown(card_html, unsafe_allow_html=True)

                st.caption(
                    "This prototype is for **educational purposes only** and does not constitute financial advice."
                )
            else:
                st.info("No holdings yet, so the account summary is empty.")
        # ==========================
        # ROW 2 – MAIN CHARTS
        # ==========================
        st.markdown("### 📊 Portfolio performance & trends")
        st.caption(
            "Total value of your holdings over time, plus technical trends for the selected stock."
        )

        row2_left, row2_right = st.columns([2, 1])

        # --- LEFT: portfolio performance with side controls ---
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

        # --- RIGHT: trend indicators with side controls ---
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

            # Trending stocks
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
    # EVALUATION TAB
    # ----------------------
    with tab_eval:
        st.subheader("📜 How this was tested")

        st.markdown(
            """
This system was tested using **historical price data** for the selected stock.

The chart below shows what would have happened **in the past** if:

- The AI system’s decisions were followed (**AI strategy**), and  
- Someone simply bought the stock once and held it (**Buy & hold**).

This is for **transparency only**.  
It does **not** predict the future and is **not financial advice**.
"""
        )

        with st.spinner("Running historical simulation..."):
            equity_df, metrics = backtest_ticker(
                ticker=ticker,
                model_path=os.path.join("models", f"dqn_{ticker}.pth"),
                initial_cash=100_000.0,
            )

        equity_df = add_drawdowns(equity_df)

        st.markdown("### 💵 Equity over time")
        st.caption(
            "This chart shows how the portfolio value would have changed over time "
            "for the AI strategy vs simply buying and holding the stock."
        )

        equity_chart = build_equity_chart(equity_df)
        st.altair_chart(equity_chart, use_container_width=True)

        st.markdown("### 📊 Summary metrics")

        metrics_table = pd.DataFrame(
            [
                {
                    "Strategy": "AI strategy",
                    "Final value ($)": f"{metrics['final_ai']:.0f}",
                    "Total return (%)": f"{metrics['return_ai'] * 100:.2f}",
                    "Max drawdown (%)": f"{metrics['max_drawdown_ai'] * 100:.2f}",
                },
                {
                    "Strategy": "Buy & hold",
                    "Final value ($)": f"{metrics['final_bh']:.0f}",
                    "Total return (%)": f"{metrics['return_bh'] * 100:.2f}",
                    "Max drawdown (%)": f"{metrics['max_drawdown_bh'] * 100:.2f}",
                },
            ]
        )

        st.table(metrics_table)

        st.markdown("### 📉 Drawdowns over time")
        st.caption(
            "This chart shows how far each strategy fell from its previous peak value. "
            "Larger negative values mean deeper historical losses."
        )

        dd_chart = build_drawdown_chart(equity_df)
        st.altair_chart(dd_chart, use_container_width=True)

    # ----------------------
    # CHAT TAB
    # ----------------------
    with tab_chat:
        st.subheader("💬 Chat with the Advisor")

        st.info(
            "This is an interactive assistant. You can ask follow-up questions "
            "about the recommendation, indicators, or historical testing. "
            "Responses are generated by a local language model (Ollama) and are "
            "for educational purposes only."
        )

        chat_history = st.session_state.get("chat_history", [])

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

        for msg in chat_history:
            if msg["role"] == "user":
                col1, col2 = st.columns([0.3, 0.7])
                with col1:
                    st.write("")
                with col2:
                    st.markdown(
                        f"""
                        <div style="
                            background-color:#e8f4ff;
                            padding:8px 12px;
                            border-radius:12px;
                            margin-bottom:4px;
                            max-width:100%;
                        ">
                            <strong>You:</strong><br>{msg['content']}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                col1, col2 = st.columns([0.7, 0.3])
                with col1:
                    st.markdown(
                        f"""
                        <div style="
                            background-color:#f4f4f4;
                            padding:8px 12px;
                            border-radius:12px;
                            margin-bottom:4px;
                            max-width:100%;
                        ">
                            <strong>Advisor:</strong><br>{msg['content']}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.write("")

        explainer = load_explainer(ticker)
        _, explanation = explainer.explain_state(state)

        pos_features = [item["feature"] for item in explanation.get("top_positive", [])]
        neg_features = [item["feature"] for item in explanation.get("top_negative", [])]

        pos_text = ", ".join(pos_features) if pos_features else "no strong positive signals"
        neg_text = ", ".join(neg_features) if neg_features else "no strong negative signals"

        backtest_summary = (
            "The system was historically tested by comparing the AI strategy "
            "to a simple buy-and-hold approach on past data. Results are shown "
            "in the 'How this was tested' tab."
        )

        user_q = st.text_area(
            "Your question",
            placeholder="e.g. Why is it suggesting HOLD for this stock?",
        )

        btn_col1, btn_col2 = st.columns([0.6, 0.4])

        with btn_col1:
            ask_clicked = st.button("Ask the advisor", key="ask_advisor_btn")

        with btn_col2:
            clear_clicked = st.button("🗑️ Clear conversation", key="clear_convo_btn")

        if clear_clicked:
            clear_chat_history(user.id)
            st.session_state["chat_history"] = []
            st.success("Conversation cleared.")
            st.rerun()

        if ask_clicked:
            clean_q = user_q.strip()
            if not clean_q:
                st.warning("Please enter a question.")
            else:
                chat_history = st.session_state.get("chat_history", [])
                recent_msgs = chat_history[-6:]
                convo_text = ""
                for m in recent_msgs:
                    speaker = "User" if m["role"] == "user" else "Advisor"
                    convo_text += f"{speaker}: {m['content']}\n"

                try:
                    answer = chat_with_advisor(
                        user_question=clean_q,
                        ticker=ticker,
                        action_text=action_text,
                        pos_text=pos_text,
                        neg_text=neg_text,
                        backtest_summary=backtest_summary,
                        conversation_history=convo_text,
                    )

                    save_message(user.id, ticker, "user", clean_q)
                    save_message(user.id, ticker, "assistant", answer)

                    st.session_state["chat_history"] = load_chat_history(user.id)
                    st.rerun()

                except Exception as e:
                    st.error("Could not reach the local language model.")
                    st.caption(str(e))

    with tab_profile:
        st.subheader("👤 Profile & Settings")

        # Basic profile info
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
                    st.error("Something went wrong while updating your password.")
               

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
