import pandas as pd
import streamlit as st

from backend.users.service import get_portfolio
from frontend.utils.constants import ACTION_MAP
from frontend.utils.explanation_helpers import (
    compute_signal_strength_and_confidence,
    classify_risk_level,
)
from frontend.utils.portfolio_helpers import (
    get_latest_price_and_change,
    compute_risk_metrics_for_ticker,
)
from frontend.utils.chart_builders import (
    build_allocation_chart,
    build_indicator_chart,
    build_portfolio_performance_chart,
)
from frontend.components.dashboard_sections import (
    render_portfolio_snapshot,
    render_account_summary,
    render_watchlist,
    render_holdings_table,
)


def render_dashboard_page(user, ticker, data, action, explanation):
    action_text = ACTION_MAP[action]
    portfolio = get_portfolio(user.id)
    current_price, _ = get_latest_price_and_change(ticker)

    conf_label, conf_pct, conf_subtitle = compute_signal_strength_and_confidence(explanation)
    risk_label, risk_text = classify_risk_level(data)

    risk_pill_class = "pill-green" if risk_label == "Low" else "pill-amber" if risk_label == "Medium" else "pill-red"

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="mini-label">AI recommendation for selected stock</div>
            <div class="big-value">{action_text} ({ticker})</div>
            <div style="margin-top:0.45rem;">
                <span class="pill pill-green">Confidence: {conf_pct}% ({conf_label})</span>
                <span class="pill {risk_pill_class}">Risk: {risk_label}</span>
            </div>
            <div style="margin-top:0.55rem;color:#6b7280;font-size:0.95rem;">
                Main factors behind this suggestion include:
                {", ".join([x["feature"] for x in explanation.get("top_positive", [])[:3]]) or "mixed signals"}.
            </div>
            <div style="margin-top:0.2rem;color:#6b7280;font-size:0.95rem;">
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

    st.markdown('<div class="section-title">Act on this recommendation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">You can simulate a simple trade here for the selected stock.</div>', unsafe_allow_html=True)

    if action_text == "BUY":
        c1, c2 = st.columns([4, 1.4])
        with c1:
            shares_to_buy = st.number_input(
                "Number of shares to buy",
                min_value=1.0,
                step=1.0,
                key="buy_shares_input",
            )
        with c2:
            st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)
            if st.button("✅ Buy shares", key="buy_btn", use_container_width=True):
                from backend.users.service import buy_shares
                try:
                    buy_shares(user.id, ticker, shares_to_buy, current_price)
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    elif action_text == "SELL":
        owned_position = next((p for p in portfolio if p.ticker == ticker), None)
        if owned_position:
            c1, c2 = st.columns([4, 1.4])
            with c1:
                shares_to_sell = st.number_input(
                    "Number of shares to sell",
                    min_value=1.0,
                    max_value=float(owned_position.shares),
                    step=1.0,
                    key="sell_shares_input",
                )
            with c2:
                st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)
                if st.button("🔻 Sell shares", key="sell_btn", use_container_width=True):
                    from backend.users.service import sell_shares
                    try:
                        sell_shares(user.id, ticker, shares_to_sell, current_price)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
        else:
            st.info("You do not currently own this stock.")
    else:
        st.info("The AI suggests holding. No action is required right now.")

    latest_date = pd.to_datetime(data["date"]).iloc[-1]
    st.caption(f"Market data for {ticker} is shown up to {latest_date.date()} (latest available daily closing prices).")

    row1_left, row1_right = st.columns([2, 1])

    with row1_left:
        render_portfolio_snapshot(user, portfolio)

    with row1_right:
        render_account_summary(portfolio)

    if portfolio:
        row2_left, row2_right = st.columns(2)

        with row2_left:
            st.markdown('<div class="section-title">Portfolio allocation</div>', unsafe_allow_html=True)
            alloc_chart = build_allocation_chart(portfolio)
            if alloc_chart is not None:
                st.altair_chart(alloc_chart, use_container_width=True)

        with row2_right:
            st.markdown('<div class="section-title">Historical risk & return</div>', unsafe_allow_html=True)
            metrics, err = compute_risk_metrics_for_ticker(ticker)
            if err or metrics is None:
                st.info("Historical risk metrics are not available right now.")
            else:
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Return", f"{metrics['total_return'] * 100:.1f}%")
                with m2:
                    st.metric("Worst drop", f"{metrics['max_drawdown'] * 100:.1f}%")
                with m3:
                    st.metric("Sharpe", f"{metrics['sharpe']:.2f}" if metrics["sharpe"] is not None else "N/A")
                st.caption("Historical backtest metrics shown for transparency only.")

    st.markdown('<div class="section-title">Portfolio performance & trends</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Core portfolio behaviour and selected technical indicators for the chosen stock.</div>', unsafe_allow_html=True)

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

    row3_left, row3_right = st.columns([2, 1])

    with row3_left:
        render_holdings_table(portfolio)

    with row3_right:
        render_watchlist()