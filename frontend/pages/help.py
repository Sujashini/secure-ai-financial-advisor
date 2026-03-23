import streamlit as st


def _section_card(title: str, body: str, icon: str = "ℹ️"):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(30,41,59,0.92), rgba(15,23,42,0.96));
            border: 1px solid rgba(148,163,184,0.14);
            border-radius: 16px;
            padding: 1rem 1rem 0.9rem 1rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 10px 22px rgba(0,0,0,0.16);
        ">
            <div style="
                font-size: 1rem;
                font-weight: 700;
                color: #f8fafc;
                margin-bottom: 0.45rem;
            ">{icon} {title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(body)


def _mini_card(title: str, body: str):
    st.markdown(
        f"""
        <div style="
            background: rgba(15,23,42,0.72);
            border: 1px solid rgba(148,163,184,0.14);
            border-radius: 14px;
            padding: 0.9rem 0.95rem;
            min-height: 185px;
            box-shadow: 0 8px 18px rgba(0,0,0,0.14);
            margin-bottom: 0.8rem;
        ">
            <div style="
                font-size: 0.98rem;
                font-weight: 700;
                color: #f8fafc;
                margin-bottom: 0.5rem;
            ">{title}</div>
            <div style="
                color: #cbd5e1;
                line-height: 1.6;
                font-size: 0.95rem;
            ">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_help_page():
    st.subheader("❓ Help / Glossary")

    # -------------------------
    # Introductory purpose section
    # -------------------------

    _section_card(
        "What this page is for",
        """
This page explains how to read SAFE-Bot’s outputs in plain English. It helps you understand the app’s recommendations, indicators, and historical risk metrics.

SAFE-Bot is an educational prototype focused on explainability and decision support. It is **not** automated financial advice.
        """,
        icon="🧭",
    )
    # -------------------------
    # Recommendation signals
    # -------------------------

    st.markdown("### Understanding recommendation signals")

    c1, c2, c3 = st.columns(3)

    with c1:
        _mini_card(
            "📈 BUY",
            """
The model has found patterns similar to past situations where price movement was more favorable.

This does **not** guarantee the stock will rise. It only means the current signal leans positive based on historical data.
            """,
        )

    with c2:
        _mini_card(
            "📉 SELL",
            """
The model has found patterns similar to past situations where prices often weakened or became less stable.

This does **not** mean the stock will definitely fall, but it is a caution signal.
            """,
        )

    with c3:
        _mini_card(
            "⏸️ HOLD",
            """
The model does not currently see a strong edge in either direction.

Signals may be mixed, weak, or uncertain, which can happen when the market is noisy or trend direction is unclear.
            """,
        )

    st.warning(
        "These signals are for learning and transparency only. They should not be treated as personal financial advice."
    )
    # -------------------------
    # Usage guidance
    # -------------------------

    st.markdown("### How to use SAFE-Bot")

    h1, h2, h3 = st.columns(3)

    with h1:
        _mini_card(
            "🏠 1. Start from the Dashboard",
            """
Use the dashboard to review the latest recommendation, watchlist, and quick stock information.

This gives you a fast overview before going deeper.
            """,
        )

    with h2:
        _mini_card(
            "🔍 2. Check the Explanation page",
            """
The Explanation page shows why the system leaned toward BUY, SELL, or HOLD.

Look at the plain-English explanation, positive signals, caution signals, and charts together.
            """,
        )

    with h3:
        _mini_card(
            "💬 3. Review Portfolio and Chat",
            """
The Portfolio page helps you understand allocation, concentration risk, and historical metrics.

The Chat page helps you ask follow-up questions in simpler language.
            """,
        )
    # -------------------------
    # Indicator glossary
    # -------------------------

    st.markdown("### Glossary of key indicators")

    g1, g2 = st.columns(2)

    with g1:
        _mini_card(
            "📊 SMA / EMA",
            """
SMA and EMA are moving averages. They help show the short- or medium-term trend of a stock price.

EMA reacts faster to recent price changes than SMA.
            """,
        )

        _mini_card(
            "⚡ RSI",
            """
RSI stands for Relative Strength Index. It is a momentum indicator that helps show whether a stock has moved up or down too quickly recently.

It is often used to highlight potentially overbought or oversold conditions.
            """,
        )

    with g2:
        _mini_card(
            "🌊 Volatility",
            """
Volatility describes how much price moves around over time.

A more volatile stock tends to swing more sharply up and down, which usually means higher uncertainty and risk.
            """,
        )

        _mini_card(
            "🧠 Trend vs Momentum",
            """
Trend indicators describe general direction over time.

Momentum indicators describe how strong or weak recent movement is. SAFE-Bot uses both so it does not rely on only one type of signal.
            """,
        )
    # -------------------------
    # Backtest metric glossary
    # -------------------------

    st.markdown("### Backtest and risk metrics")

    r1, r2, r3 = st.columns(3)

    with r1:
        _mini_card(
            "💹 Return",
            """
Return shows how much the strategy gained or lost over the backtest period.

A positive return means the strategy ended higher than it started.
            """,
        )

    with r2:
        _mini_card(
            "📉 Worst drop",
            """
Worst drop means the largest peak-to-trough decline during the backtest.

It helps show how painful the biggest temporary loss was.
            """,
        )

    with r3:
        _mini_card(
            "⚖️ Sharpe ratio",
            """
The Sharpe ratio compares return against volatility.

A higher Sharpe ratio usually means the strategy delivered better return for the amount of risk taken.
            """,
        )
    # -------------------------
    # Prototype limits and value
    # -------------------------

    st.markdown("### Limits of this prototype")

    st.info(
        """
SAFE-Bot does not predict the future with certainty, does not know your personal financial situation, and does not replace human financial advice.

It learns from historical patterns, which may stop working when market conditions change.
        """
    )

    st.success(
        """
Even with those limits, SAFE-Bot is useful as an explainable AI learning tool.

It helps users explore how model-driven recommendations can be presented clearly, responsibly, and with understandable supporting signals.
        """
    )