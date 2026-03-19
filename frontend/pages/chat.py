import html
import re
import streamlit as st

from backend.LLM.ollama_chat import chat_with_advisor, summarize_conversation
from backend.LLM.chat_store import (
    load_chat_history,
    save_message,
    clear_chat_history,
)

FRIENDLY_FEATURE_NAMES = {
    "return_1": "very recent price movement",
    "sma_10": "short-term price trend",
    "sma_20": "medium-term price trend",
    "ema_10": "short-term trend",
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


def _format_message(text: str) -> str:
    if not text:
        return ""

    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    safe_paragraphs = [html.escape(p).replace("\n", "<br>") for p in paragraphs]

    return "".join(
        f'<div style="margin-bottom:0.72rem; line-height:1.72;">{p}</div>'
        for p in safe_paragraphs
    )


def _build_quick_questions(action_text, risk_label, pos_features):
    quick_questions = [
        "Why is the model recommending this action for this stock?",
        "What are the main risks for this stock according to the model?",
        "How does the RL strategy compare to a simple Buy and Hold strategy on past data?",
        "Which signal mattered most for this recommendation?",
    ]

    if action_text == "BUY":
        quick_questions[0] = "Why does the model currently think this stock looks attractive?"
    elif action_text == "SELL":
        quick_questions[0] = "Why does the model currently think reducing exposure makes sense?"
    else:
        quick_questions[0] = "Why does the model prefer to wait instead of buying or selling?"

    if str(risk_label).lower() == "high":
        quick_questions[1] = "Why is the risk level high for this stock right now?"
    elif str(risk_label).lower() == "low":
        quick_questions[1] = "Why is the risk level low for this stock right now?"

    if pos_features:
        quick_questions[3] = f"Why does {pos_features[0]} matter so much for this recommendation?"

    return quick_questions


def _risk_badge_color(risk_label: str) -> str:
    label = str(risk_label).lower()
    if label == "high":
        return "#ef4444"
    if label == "medium":
        return "#f59e0b"
    return "#22c55e"


def _render_context_card(ticker, action_text, conf_pct, risk_label, pos_text, neg_text):
    risk_color = _risk_badge_color(risk_label)

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(30,41,59,0.94), rgba(15,23,42,0.98));
            border: 1px solid rgba(148,163,184,0.14);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 24px rgba(0,0,0,0.18);
        ">
            <div style="
                display:flex;
                justify-content:space-between;
                align-items:flex-start;
                gap:1rem;
                flex-wrap:wrap;
                margin-bottom:0.7rem;
            ">
                <div>
                    <div style="font-size:0.78rem; font-weight:700; color:#a5b4fc; letter-spacing:0.04em;">
                        CHAT CONTEXT
                    </div>
                    <div style="font-size:1.1rem; font-weight:800; color:#f8fafc; margin-top:0.22rem;">
                        {ticker} • {action_text}
                    </div>
                </div>
                <div style="display:flex; gap:0.45rem; flex-wrap:wrap;">
                    <span style="
                        background:rgba(99,102,241,0.18);
                        color:#e0e7ff;
                        padding:0.28rem 0.62rem;
                        border-radius:999px;
                        font-size:0.8rem;
                        font-weight:700;
                    ">
                        Confidence: {conf_pct}%
                    </span>
                    <span style="
                        background:{risk_color};
                        color:white;
                        padding:0.28rem 0.62rem;
                        border-radius:999px;
                        font-size:0.8rem;
                        font-weight:700;
                    ">
                        Risk: {risk_label}
                    </span>
                </div>
            </div>
            <div style="color:#cbd5e1; line-height:1.68;">
                The advisor is answering questions about the current recommendation using the strongest
                supportive signals (<b>{html.escape(pos_text)}</b>) and the main caution signals
                (<b>{html.escape(neg_text)}</b>). Strategy comparisons are historical and shown for transparency only.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_empty_state():
    st.markdown(
        """
        <div class="chat-bot" style="max-width:82%;">
            <div class="chat-label">Advisor</div>
            <div style="line-height:1.7;">
                Ask a question about the current recommendation, the risk level, or the historical strategy comparison.
                <br><br>
                A good place to start is: <em>Why is the model recommending this action right now?</em>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_page(user, ticker, action_text, conf_pct, risk_label, explanation):
    st.subheader("💬 Chat with the Advisor")

    st.info(
        "This is an educational prototype, not financial advice.\n\n"
        "- It explains how the demo model is thinking about a stock.\n"
        "- It cannot consider your full financial situation.\n"
        "- Do not make real trading decisions based on this tool."
    )

    with st.expander("What this chat can and cannot do"):
        st.markdown(
            """
            **✅ This chat can help you**
            - understand why the model suggests BUY / SELL / HOLD
            - interpret indicators, confidence, and risk
            - compare the RL strategy with simple historical baselines
            - understand the recommendation in plain English

            **🚫 This chat cannot do**
            - give personalised financial advice
            - guarantee profits or predict the future with certainty
            - replace professional financial advice
            """
        )

    pos = explanation.get("top_positive", [])
    neg = explanation.get("top_negative", [])

    pos_features = [friendly_feature_name(item["feature"]) for item in pos]
    neg_features = [friendly_feature_name(item["feature"]) for item in neg]

    pos_text = ", ".join(pos_features[:3]) if pos_features else "no strong positive signals"
    neg_text = ", ".join(neg_features[:3]) if neg_features else "no strong caution signals"

    _render_context_card(
        ticker=ticker,
        action_text=action_text,
        conf_pct=conf_pct,
        risk_label=risk_label,
        pos_text=pos_text,
        neg_text=neg_text,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current ticker", ticker)
    with col2:
        st.metric("Model action", action_text)
    with col3:
        st.metric("Signal confidence", f"{conf_pct}%")

    st.caption(
        f"Market context for {ticker}: risk level is {risk_label.lower()}. Responses come from a local language model and are intended for learning and transparency."
    )

    st.divider()

    chat_history = load_chat_history(user_id=user.id, ticker=ticker, limit=20)

    convo_title_col, convo_btn_col = st.columns([0.72, 0.28])

    with convo_title_col:
        st.markdown("### Conversation")

    with convo_btn_col:
        summarize_disabled = len(chat_history) < 6
        summarize_clicked = st.button(
            "🧾 Summarise conversation",
            key="summarise_conversation_btn",
            disabled=summarize_disabled,
            use_container_width=True,
        )

    if summarize_disabled:
        st.caption(
            f"Summaries become available after a longer conversation "
            f"(currently {len(chat_history)} messages; need at least 6)."
        )

    if summarize_clicked and not summarize_disabled:
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
                st.markdown("#### Conversation summary")
                st.markdown(
                    f"""
                    <div class="chat-bot" style="max-width:100%;">
                        <div class="chat-label">Summary</div>
                        {_format_message(summary)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception:
                st.warning("Sorry, I couldn't summarise the conversation right now.")

    if not chat_history:
        _render_empty_state()
    else:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        for msg in chat_history:
            safe_text = _format_message(msg["content"])

            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div class="chat-user">
                        <div class="chat-label">You</div>
                        {safe_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="chat-bot">
                        <div class="chat-label">Advisor</div>
                        {safe_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    backtest_summary = (
        "The RL strategy is compared with two simple historical baselines: "
        "Buy and Hold, which keeps the stock throughout the period, and an RSI strategy, "
        "which uses a basic momentum-style technical signal. These comparisons are historical only."
    )

    rl_confidence = conf_pct / 100.0 if conf_pct is not None else None
    quick_questions = _build_quick_questions(action_text, risk_label, pos_features)

    st.markdown("### Quick questions")
    qp_col1, qp_col2 = st.columns(2)
    qp_col3, qp_col4 = st.columns(2)

    quick_question = None
    with qp_col1:
        if st.button(f"📈 {quick_questions[0]}", use_container_width=True):
            quick_question = quick_questions[0]
    with qp_col2:
        if st.button(f"⚠️ {quick_questions[1]}", use_container_width=True):
            quick_question = quick_questions[1]
    with qp_col3:
        if st.button(f"📊 {quick_questions[2]}", use_container_width=True):
            quick_question = quick_questions[2]
    with qp_col4:
        if st.button(f"🔎 {quick_questions[3]}", use_container_width=True):
            quick_question = quick_questions[3]

    st.markdown("### Ask your own question")
    user_q = st.text_area(
        "Your question",
        placeholder="Ask about the recommendation, confidence, risk, indicators, or historical strategy testing...",
        key="advisor_text",
        height=110,
    )

    btn_col1, btn_col2 = st.columns([0.32, 0.68])
    with btn_col1:
        ask_clicked = st.button("Ask the advisor", key="ask_advisor_btn", use_container_width=True)
    with btn_col2:
        clear_clicked = st.button("🗑️ Clear conversation", key="clear_convo_btn", use_container_width=False)

    if clear_clicked:
        clear_chat_history(user.id, ticker)
        st.success("Conversation cleared for this stock.")
        st.rerun()

    question_to_send = None
    if quick_question is not None:
        question_to_send = quick_question
    elif ask_clicked and user_q.strip():
        question_to_send = user_q.strip()
    elif ask_clicked and not user_q.strip():
        st.warning("Please enter a question first.")

    if question_to_send is not None:
        recent_msgs = load_chat_history(user_id=user.id, ticker=ticker, limit=12)
        convo_text = ""
        for m in recent_msgs:
            speaker = "User" if m["role"] == "user" else "Advisor"
            convo_text += f"{speaker}: {m['content']}\n"

        try:
            with st.spinner("Thinking..."):
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

            save_message(user.id, ticker, "user", question_to_send)
            save_message(user.id, ticker, "assistant", answer)
            st.rerun()

        except Exception as e:
            st.error("⚠️ The AI model is currently unavailable or taking too long. Please try again.")
            st.caption(str(e))