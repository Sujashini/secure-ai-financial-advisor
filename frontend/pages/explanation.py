import os
import streamlit as st

from frontend.utils.constants import ACTION_MAP
from frontend.utils.explanation_helpers import (
    generate_plain_english_explanation,
    compute_signal_strength_and_confidence,
    classify_risk_level,
)
from frontend.utils.chart_builders import (
    load_explainer,
    build_shap_bar_chart,
    build_price_action_chart,
    build_strategy_comparison_chart,
)
from frontend.utils.portfolio_helpers import compute_risk_metrics_for_ticker


# Friendly fallback names for non-technical users
FRIENDLY_FEATURE_NAMES = {
    "return_1": "Very recent price movement",
    "sma_10": "Short-term price trend",
    "sma_20": "Medium-term price trend",
    "ema_10": "Short-term trend (EMA)",
    "ema_20": "Medium-term trend (EMA)",
    "volatility_10": "Recent price volatility",
    "rsi_14": "Momentum (RSI)",
    "open": "Opening price",
    "high": "Recent high price",
    "low": "Recent low price",
    "close": "Closing price",
    "volume": "Trading volume",
    "position_flag": "Current position status",
}


def friendly_feature_name(feature: str) -> str:
    return FRIENDLY_FEATURE_NAMES.get(feature, feature.replace("_", " ").title())


def render_signal_card(title, body):
    st.markdown(
        f"""
        <div class="card" style="min-height:120px;">
            <div class="mini-label">{title}</div>
            <div style="color:#f8fafc;font-weight:700;font-size:1rem;line-height:1.55;margin-top:0.35rem;">
                {body}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_explanation_page(ticker, data, agent, state, action):
    action_text = ACTION_MAP.get(action, "HOLD")

    explainer = load_explainer(ticker)
    _, explanation = explainer.explain_state(state)

    conf_label, conf_pct, conf_subtitle = compute_signal_strength_and_confidence(explanation)
    risk_label, risk_text = classify_risk_level(data)

    st.markdown("## Explanation")
    st.caption(
        "This page explains why the model suggested its current action, which signals mattered most, "
        "and how the strategy behaved on historical data."
    )

    st.success(
        f"📊 The AI currently suggests **{action_text} {ticker}** with **{conf_pct}% confidence** "
        f"and **{risk_label.lower()} risk**."
    )

    # ----------------------------
    # Decision overview
    # ----------------------------
    st.markdown('<div class="section-title">Decision overview</div>', unsafe_allow_html=True)

    o1, o2, o3 = st.columns(3)
    with o1:
        render_signal_card("Current action", f"{action_text} ({ticker})")
    with o2:
        render_signal_card("How confident the AI is", f"{conf_pct}% — {conf_label}")
    with o3:
        render_signal_card("Risk view", f"{risk_label} — {risk_text}")

    # ----------------------------
    # Plain English explanation
    # ----------------------------
    st.markdown(
        '<div class="section-title">What this means in plain English</div>',
        unsafe_allow_html=True,
    )

    plain_text = generate_plain_english_explanation(
        ticker=ticker,
        action=action,
        explanation=explanation,
    )

    bullet_points = [
        point.strip()
        for point in plain_text.replace("\n", " ").split(". ")
        if point.strip()
    ]

    bullets_html = "".join(
        [f"<li style='margin-bottom:0.45rem;'>{point.rstrip('.')}</li>" for point in bullet_points]
    )

    st.markdown(
        f"""
        <div class="card">
            <div style="color:#cbd5e1;line-height:1.75;font-size:0.98rem;">
                <ul style="padding-left:1.2rem; margin:0;">
                    {bullets_html}
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------------------
    # Decision drivers
    # ----------------------------
    st.markdown('<div class="section-title">Key reasons behind this decision</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">These are the strongest signals that supported the recommendation and the main reasons for caution.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Reasons supporting this action")
        if explanation.get("top_positive"):
            for item in explanation["top_positive"]:
                feature_name = friendly_feature_name(item["feature"])
                st.markdown(
                    f"""
                    <div class="card" style="padding:12px 14px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
                            <span style="color:#f8fafc;font-weight:600;">{feature_name}</span>
                            <span class="pill pill-green">{float(item['value']):.4f}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No strong supporting signals were identified.")

    with col2:
        st.markdown("#### Reasons for caution")
        if explanation.get("top_negative"):
            for item in explanation["top_negative"]:
                feature_name = friendly_feature_name(item["feature"])
                st.markdown(
                    f"""
                    <div class="card" style="padding:12px 14px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
                            <span style="color:#f8fafc;font-weight:600;">{feature_name}</span>
                            <span class="pill pill-red">{float(item['value']):.4f}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No strong caution signals were identified.")

    # ----------------------------
    # Feature importance chart
    # ----------------------------
    st.markdown('<div class="section-title">Which signals mattered most?</div>', unsafe_allow_html=True)
    st.caption(
        "Green bars pushed the AI more toward the chosen action, while red bars pushed against it."
    )

    shap_chart = build_shap_bar_chart(explanation)
    if shap_chart is not None:
        st.altair_chart(shap_chart, use_container_width=True)
        st.info(
            "💡 Tip: If most bars are green, the AI sees more support for its decision. "
            "If several red bars appear, the model is seeing conflicting signals."
        )
    else:
        st.info("Not enough explanation data to show which signals mattered most.")

    # ----------------------------
    # Price and AI actions
    # ----------------------------
    st.markdown('<div class="section-title">Price behaviour and AI actions</div>', unsafe_allow_html=True)
    st.caption(
        "This chart shows the stock price alongside the actions selected by the AI over time."
    )

    price_chart = build_price_action_chart(data, agent)
    if price_chart is not None:
        st.altair_chart(price_chart, use_container_width=True)
        st.info(
            "💡 Tip: This helps you see whether the AI tended to buy, hold, or sell during rising and falling periods."
        )
    else:
        st.info("Not enough data to display the price and action chart.")

    # ----------------------------
    # Strategy comparison
    # ----------------------------
    st.markdown('<div class="section-title">Historical strategy comparison</div>', unsafe_allow_html=True)
    st.caption(
        "This compares the reinforcement learning strategy with a buy-and-hold baseline and a simple RSI strategy "
        "on historical data. It is shown for transparency only."
    )

    comp_chart = build_strategy_comparison_chart(ticker)
    if comp_chart is not None:
        st.altair_chart(comp_chart, use_container_width=True)
        st.info(
            "💡 Tip: If the RL strategy line stays above the others, it performed better historically. "
            "However, past performance does not guarantee future results."
        )
    else:
        st.info("Could not load historical comparison for this stock.")

    # ----------------------------
    # Risk and performance metrics
    # ----------------------------
    st.markdown('<div class="section-title">Risk and performance summary</div>', unsafe_allow_html=True)

    metrics, err = compute_risk_metrics_for_ticker(ticker)
    if err or metrics is None:
        st.info("Historical risk metrics are not available right now.")
    else:
        m1, m2, m3 = st.columns(3)
        with m1:
            render_signal_card("Total return", f"{metrics['total_return'] * 100:.1f}%")
        with m2:
            render_signal_card("Maximum drawdown", f"{metrics['max_drawdown'] * 100:.1f}%")
        with m3:
            sharpe_text = f"{metrics['sharpe']:.2f}" if metrics["sharpe"] is not None else "N/A"
            render_signal_card("Sharpe ratio", sharpe_text)

        st.caption(
            "Total return shows overall growth, maximum drawdown shows the worst drop from a previous peak, "
            "and Sharpe ratio indicates return relative to risk."
        )

    # ----------------------------
    # Model transparency note
    # ----------------------------
    st.markdown('<div class="section-title">Model transparency note</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="card">
            <div style="color:#94a3b8;line-height:1.7;">
                The current recommendation for <b>{ticker}</b> is generated using a reinforcement learning model trained on
                historical market data. The explanation on this page is based on the features that most influenced the
                model's decision at the current state. A higher confidence score suggests that the current pattern looks
                more similar to past situations where the model found a stronger action signal. This does not guarantee
                future performance and is shown for educational transparency only.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )