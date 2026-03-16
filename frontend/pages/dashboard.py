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
    render_trade_panel,
    render_watchlist,
)


def render_dashboard_page(user, ticker, data, action, explanation):
    action_text = ACTION_MAP[action]
    portfolio = get_portfolio(user.id)
    current_price, _ = get_latest_price_and_change(ticker)

    conf_label, conf_pct, conf_subtitle = compute_signal_strength_and_confidence(explanation)
    risk_label, risk_text = classify_risk_level(data)

    risk_pill_class = (
        "pill-green"
        if risk_label == "Low"
        else "pill-amber"
        if risk_label == "Medium"
        else "pill-red"
    )

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="mini-label">AI recommendation for selected stock</div>
            <div class="big-value">{action_text} ({ticker})</div>
            <div style="margin-top:0.45rem;">
                <span class="pill pill-green">Confidence: {conf_pct}% ({conf_label})</span>
                <span class="pill {risk_pill_class}">Risk: {risk_label}</span>
            </div>
            <div style="margin-top:0.55rem;color:#94a3b8;font-size:0.95rem;">
                Main factors behind this suggestion include:
                {", ".join([x["feature"] for x in explanation.get("top_positive", [])[:3]]) or "mixed signals"}.
            </div>
            <div style="margin-top:0.2rem;color:#94a3b8;font-size:0.95rem;">
                {conf_subtitle} {risk_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Why this suggestion? (key factors)", expanded=False):
        pos = explanation.get("top_positive", [])
        neg = explanation.get("top_negative", [])

        st.markdown("**Main positive signals**")
        if pos:
            for item in pos:
                st.write(f"- {item['feature']}")
        else:
            st.write("_No strong positive signals identified._")

        st.markdown("**Main caution signals**")
        if neg:
            for item in neg:
                st.write(f"- {item['feature']}")
        else:
            st.write("_No strong caution signals identified._")

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
        '<div class="section-caption">Core portfolio behaviour and selected technical indicators for the chosen stock.</div>',
        unsafe_allow_html=True,
    )

    c_left, c_right = st.columns([2.2, 1])

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
        else:
            st.info("Portfolio performance will appear once you have holdings.")

    with c_right:
        st.markdown(f"#### {ticker} trend indicators")
        selected_cols = []
        if st.checkbox("Close price", value=True, key="close_price"):
            selected_cols.append("close")
        if st.checkbox("SMA 20", value=True, key="sma_20"):
            selected_cols.append("sma_20")
        if st.checkbox("EMA 20", value=False, key="ema_20"):
            selected_cols.append("ema_20")

        if selected_cols:
            ind_chart = build_indicator_chart(data, selected_cols)
            if ind_chart is not None:
                st.altair_chart(ind_chart, use_container_width=True)

    bottom_left, bottom_right = st.columns([1.6, 1])

    with bottom_left:
        render_trade_panel(
            user=user,
            ticker=ticker,
            action_text=action_text,
            current_price=current_price,
            portfolio=portfolio,
        )

    with bottom_right:
        render_watchlist()