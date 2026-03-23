import pandas as pd
import streamlit as st

from backend.users.service import get_portfolio
from frontend.utils.constants import ACTION_MAP
from frontend.utils.explanation_helpers import (
    compute_signal_strength_and_confidence,
    classify_risk_level,
)
from frontend.utils.portfolio_helpers import get_latest_price_and_change
from frontend.utils.chart_builders import (
    build_indicator_chart,
    build_portfolio_performance_chart,
)
from frontend.components.dashboard_sections import (
    render_hero_section,
    render_trade_panel,
    render_watchlist,
)

FRIENDLY_FEATURE_NAMES = {
    "return_1": "very recent price movement",
    "sma_10": "short-term price trend",
    "sma_20": "medium-term price trend",
    "ema_10": "short-term trend (EMA)",
    "ema_20": "smoothed medium-term trend",
    "volatility_10": "recent price volatility",
    "rsi_14": "momentum",
    "open": "opening price behaviour",
    "high": "recent high price",
    "low": "recent low price",
    "close": "closing price behaviour",
    "volume": "trading volume",
    "position_flag": "current position status",
}


def friendly_feature_name(feature: str) -> str:
    return FRIENDLY_FEATURE_NAMES.get(feature, feature.replace("_", " ").lower())


def build_factor_summary(explanation: dict) -> str:
    """
    Build a short human-readable summary of the strongest
    positive explanation factors.
    """
    pos = explanation.get("top_positive", [])[:3]
    if not pos:
        return "mixed signals across recent price, trend, and momentum indicators"

    parts = [friendly_feature_name(item["feature"]) for item in pos]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{parts[0]}, {parts[1]}, and {parts[2]}"


def render_dashboard_page(user, ticker, data, action, explanation):
    """
    Render the main dashboard page, including:
    - current recommendation and market context,
    - plain-language explanation summary,
    - portfolio performance chart,
    - trend indicator chart,
    - action panel,
    - watchlist.
    """
    action_text = ACTION_MAP.get(action, "HOLD")
    portfolio = get_portfolio(user.id)
    current_price, price_change_pct = get_latest_price_and_change(ticker)

    conf_label, conf_pct, conf_subtitle = compute_signal_strength_and_confidence(explanation)
    risk_label, risk_text = classify_risk_level(data)
    st.subheader("📊 Dashboard")
    st.caption(
        "Review the latest recommendation, confidence, risk signals, portfolio performance, and quick actions for the selected stock."
    )

    render_hero_section(
        ticker=ticker,
        action_text=action_text,
        conf_label=conf_label,
        conf_pct=conf_pct,
        conf_subtitle=conf_subtitle,
        risk_label=risk_label,
        risk_text=risk_text,
        explanation=explanation,
        factor_summary=build_factor_summary(explanation),
        current_price=current_price,
        price_change_pct=price_change_pct,
    )

    # Expandable plain-language explanation section
    with st.expander("Why this suggestion? (plain explanation)", expanded=False):
        pos = explanation.get("top_positive", [])
        neg = explanation.get("top_negative", [])

        st.markdown(
            f"""
            The model currently leans toward **{action_text}** because it sees more supportive than cautionary
            signals for **{ticker}** right now.
            """
        )

        st.markdown("**Main supporting signals**")
        if pos:
            for item in pos:
                st.write(f"- {friendly_feature_name(item['feature']).capitalize()}")
        else:
            st.write("_No strong supporting signals were identified._")

        st.markdown("**Main reasons for caution**")
        if neg:
            for item in neg:
                st.write(f"- {friendly_feature_name(item['feature']).capitalize()}")
        else:
            st.write("_No strong caution signals were identified._")

        st.caption(
            "These signals are derived from recent price behaviour and technical indicators. "
            "They help explain the recommendation, but they do not guarantee future performance."
        )

    # Show the latest available market date in the dataset
    latest_date = pd.to_datetime(data["date"]).iloc[-1]
    st.caption(
        f"Market data for {ticker} is shown up to {latest_date.date()} "
        "(latest available daily closing prices)."
    )

    st.markdown(
        '<div class="section-title">Portfolio performance & trends</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">View your simulated portfolio performance and compare the selected stock against simple trend indicators.</div>',
        unsafe_allow_html=True,
    )

    c_left, c_right = st.columns([2.15, 1])

    with c_left:
        st.markdown("#### Portfolio performance")
        freq_label = st.radio(
            "View by",
            options=["Monthly", "Quarterly", "Annually"],
            index=0,
            horizontal=True,
            key="perf_freq",
        )
        freq_map = {"Monthly": "M", "Quarterly": "Q", "Annually": "Y"}
        perf_chart = build_portfolio_performance_chart(portfolio, freq_map[freq_label])

        if perf_chart is not None:
            st.altair_chart(perf_chart, use_container_width=True)
            st.info(
                "💡 Tip: Rising values suggest stronger historical portfolio growth over the selected time view."
            )
        else:
            st.info(
                "Your portfolio chart will appear after your first simulated trade. "
                "Try buying a stock from the action panel below."
            )

    with c_right:
        st.markdown(f"#### {ticker} trend indicators")
        st.caption("Choose which indicators to compare against the stock price.")

        selected_cols = []
        if st.checkbox("Close price", value=True, key="close_price"):
            selected_cols.append("close")
        if st.checkbox("Short-term average (SMA 20)", value=True, key="sma_20"):
            selected_cols.append("sma_20")
        if st.checkbox("Smoothed trend (EMA 20)", value=False, key="ema_20"):
            selected_cols.append("ema_20")

        if selected_cols:
            ind_chart = build_indicator_chart(data, selected_cols)
            if ind_chart is not None:
                st.altair_chart(ind_chart, use_container_width=True)
                st.info(
                    "💡 Tip: When price stays above the moving average, it can suggest stronger short-term momentum."
                )
            else:
                st.info("Not enough indicator data is available right now.")

    bottom_left, bottom_right = st.columns([1.55, 1])

    with bottom_left:
        render_trade_panel(
            user=user,
            ticker=ticker,
            action_text=action_text,
            current_price=current_price,
            portfolio=portfolio,
        )

    with bottom_right:
        render_watchlist(selected_ticker=ticker)