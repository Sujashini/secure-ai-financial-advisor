import streamlit as st

from frontend.utils.explanation_helpers import generate_plain_english_explanation
from frontend.utils.chart_builders import (
    load_explainer,
    build_shap_bar_chart,
    build_price_action_chart,
    build_strategy_comparison_chart,
)


def render_explanation_page(ticker, data, agent, state, action):
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
            "Green bars pushed the AI more towards the chosen action; red bars pushed against it."
        )
        st.altair_chart(shap_chart, use_container_width=True)
    else:
        st.info("Not enough explanation data to show a feature importance chart.")

    st.subheader(f"📈 Price & AI decisions for {ticker}")
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

    st.subheader("🧪 Strategy performance comparison (historical data)")
    st.caption(
        "This compares, on past data only, three simple strategies starting from the same amount: "
        "the RL AI strategy, a basic buy-and-hold, and a simple RSI-based trading rule."
    )
    comp_chart = build_strategy_comparison_chart(ticker)
    if comp_chart is not None:
        st.altair_chart(comp_chart, use_container_width=True)
    else:
        st.info("Could not load historical comparison for this stock.")