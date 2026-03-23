import streamlit as st

from frontend.utils.constants import ACTION_MAP
from frontend.utils.explanation_helpers import (
    generate_plain_english_explanation,
    compute_signal_strength_and_confidence,
    classify_risk_level,
    get_risk_pill_class,
    get_confidence_pill_class,
    generate_takeaway_text,
)
from frontend.utils.chart_builders import (
    load_explainer,
    build_shap_bar_chart,
    build_price_action_chart,
    build_strategy_comparison_chart,
)
from frontend.utils.portfolio_helpers import compute_risk_metrics_for_ticker


FRIENDLY_FEATURE_NAMES = {
    "return_1": "Very recent price movement",
    "sma_10": "Short-term price trend",
    "sma_20": "Medium-term price trend",
    "ema_10": "Short-term trend (EMA)",
    "ema_20": "Smoothed medium-term trend",
    "volatility_10": "Recent price volatility",
    "rsi_14": "Momentum (RSI)",
    "open": "Opening price behaviour",
    "high": "Recent high price",
    "low": "Recent low price",
    "close": "Closing price behaviour",
    "volume": "Trading volume",
    "position_flag": "Current position status",
}


def friendly_feature_name(feature: str) -> str:
    return FRIENDLY_FEATURE_NAMES.get(feature, feature.replace("_", " ").title())


def render_signal_card(title, body, helper=None):
    """
    Render a small summary card showing one key explanation signal.
    """
    helper_html = (
        f'<div style="color:#94a3b8;font-size:0.88rem;line-height:1.6;margin-top:0.4rem;">{helper}</div>'
        if helper
        else ""
    )

    st.markdown(
        f"""
        <div class="card" style="min-height:138px;">
            <div class="mini-label">{title}</div>
            <div style="color:#f8fafc;font-weight:700;font-size:1rem;line-height:1.5;margin-top:0.35rem;">
                {body}
            </div>
            {helper_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_takeaway_banner(ticker, action_text, conf_pct, risk_label, conf_subtitle, risk_text):
    """
    Render the top takeaway banner summarising:
    - the recommendation,
    - confidence,
    - risk,
    - short natural-language takeaway.
    """
    conf_class = get_confidence_pill_class(
        "High" if conf_pct >= 66 else "Medium" if conf_pct >= 33 else "Low"
    )
    risk_class = get_risk_pill_class(risk_label)
    takeaway = generate_takeaway_text(
        ticker=ticker,
        action_text=action_text,
        conf_pct=conf_pct,
        risk_label=risk_label,
        conf_subtitle=conf_subtitle,
        risk_text=risk_text,
    )

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(30,41,59,0.9));
            border: 1px solid rgba(148,163,184,0.14);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 24px rgba(0,0,0,0.16);
        ">
            <div style="
                display:flex;
                align-items:center;
                justify-content:space-between;
                gap:0.8rem;
                flex-wrap:wrap;
                margin-bottom:0.65rem;
            ">
                <div style="font-size:1.02rem;font-weight:800;color:#f8fafc;">
                    Recommendation summary
                </div>
                <div style="display:flex;gap:0.45rem;flex-wrap:wrap;">
                    <span class="{conf_class}">Confidence: {conf_pct}%</span>
                    <span class="{risk_class}">Risk: {risk_label}</span>
                </div>
            </div>
            <div style="color:#cbd5e1;line-height:1.72;">
                {takeaway}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_reason_chip(feature_name, value, positive=True):
    """
    Render a compact card showing one explanation feature
    and whether it supported or opposed the recommendation.
    """
    pill_class = "pill pill-green" if positive else "pill pill-red"
    st.markdown(
        f"""
        <div class="card" style="padding:12px 14px;">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
                <span style="color:#f8fafc;font-weight:600;">{feature_name}</span>
                <span class="{pill_class}">{float(value):.4f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_explanation_paragraphs(paragraphs):
    """
    Render multiple explanation paragraphs inside a styled content card.
    """
    html_blocks = ""
    for p in paragraphs:
        html_blocks += f"""
        <div style="margin-bottom:0.8rem;line-height:1.75;color:#cbd5e1;font-size:0.98rem;">
            {p}
        </div>
        """

    st.markdown(
        f"""
        <div class="card">
            {html_blocks}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_explanation_page(ticker, data, agent, state, action):
    action_text = ACTION_MAP.get(action, "HOLD")
    explanation_style = st.session_state.get("explanation_style", "Balanced")

    explainer = load_explainer(ticker)
    _, explanation = explainer.explain_state(state)

    conf_label, conf_pct, conf_subtitle = compute_signal_strength_and_confidence(explanation)
    risk_label, risk_text = classify_risk_level(data)

    st.markdown("## Explanation")
    st.caption(
        "This page explains why the model suggested its current action, which signals mattered most, "
        "and how the strategy behaved on historical data."
    )

    render_takeaway_banner(
        ticker=ticker,
        action_text=action_text,
        conf_pct=conf_pct,
        risk_label=risk_label,
        conf_subtitle=conf_subtitle,
        risk_text=risk_text,
    )

    # ----------------------------
    # Decision overview
    # ----------------------------
    st.markdown('<div class="section-title">Decision overview</div>', unsafe_allow_html=True)

    o1, o2, o3 = st.columns(3)
    with o1:
        render_signal_card(
            "Current action",
            f"{action_text} ({ticker})",
            helper="This is the action currently preferred by the model.",
        )
    with o2:
        render_signal_card(
            "How confident the AI is",
            f"{conf_pct}% — {conf_label}",
            helper=conf_subtitle,
        )
    with o3:
        render_signal_card(
            "Risk view",
            f"{risk_label} risk",
            helper=risk_text,
        )

    # ----------------------------
    # Plain English explanation
    # ----------------------------
    st.markdown(
        '<div class="section-title">What this means in plain English</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Explanation style: {explanation_style}. This can later be personalised from the Trader Profile page."
    )

    paragraphs = generate_plain_english_explanation(
        ticker=ticker,
        action=action,
        explanation=explanation,
        style=explanation_style,
    )
    render_explanation_paragraphs(paragraphs)

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
                render_reason_chip(feature_name, item["value"], positive=True)
        else:
            st.info("No strong supporting signals were identified.")

    with col2:
        st.markdown("#### Reasons for caution")
        if explanation.get("top_negative"):
            for item in explanation["top_negative"]:
                feature_name = friendly_feature_name(item["feature"])
                render_reason_chip(feature_name, item["value"], positive=False)
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
            render_signal_card(
                "Total return",
                f"{metrics['total_return'] * 100:.1f}%",
                helper="Shows the overall historical growth or loss of the strategy.",
            )
        with m2:
            render_signal_card(
                "Maximum drawdown",
                f"{metrics['max_drawdown'] * 100:.1f}%",
                helper="Shows the worst peak-to-trough drop during the backtest.",
            )
        with m3:
            sharpe_text = f"{metrics['sharpe']:.2f}" if metrics["sharpe"] is not None else "N/A"
            render_signal_card(
                "Sharpe ratio",
                sharpe_text,
                helper="Shows return relative to volatility; higher is usually better.",
            )

    # ----------------------------
    # Model transparency note
    # ----------------------------
    st.markdown('<div class="section-title">Model transparency note</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="card">
            <div style="color:#94a3b8;line-height:1.75;">
                The current recommendation for <b>{ticker}</b> is generated using a reinforcement learning model trained on
                historical market data. The explanation on this page is based on the features that most influenced the
                model's decision in the current market state. A higher confidence score suggests that the present pattern
                looks more similar to past situations where the model identified a stronger action signal. This improves
                transparency, but it does not guarantee future performance.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )