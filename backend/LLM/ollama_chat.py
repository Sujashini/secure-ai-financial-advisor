import requests
import textwrap
from typing import Optional

# ======================================
# Ollama configuration
# ======================================

OLLAMA_URL = "http://localhost:11434/api/generate"

# Make sure this model is pulled, for example:
#   ollama pull mistral:7b
MODEL_NAME = "mistral:7b"


# ======================================
# Security helpers: sanitisation & filtering
# ======================================

def sanitize_user_question(user_question: str) -> str:
    """
    Basic input sanitisation to reduce prompt injection and unsafe requests.
    Returns either the (possibly truncated) question or a block message.
    """
    lowered = user_question.lower()

    banned_phrases = [
        "ignore previous instructions",
        "act as a financial advisor",
        "guarantee profits",
        "tell me exactly what to buy",
        "tell me exactly what to sell",
        "bypass the safety rules",
        "pretend the safety rules do not apply",
    ]

    for phrase in banned_phrases:
        if phrase in lowered:
            return (
                "User query blocked due to unsafe or non-compliant request. "
                "The system can only provide high-level educational explanations, "
                "not direct financial advice."
            )

    # Optionally enforce a maximum length to avoid prompt stuffing
    max_len = 1000
    if len(user_question) > max_len:
        return user_question[:max_len]

    return user_question


def filter_llm_response(raw_response: str) -> str:
    """
    Post-process the model output to remove explicit financial advice
    or overly strong action-oriented language.
    """
    lowered = raw_response.lower()
    banned_terms = [
        "you should buy",
        "you should sell",
        "you must buy",
        "you must sell",
        "definitely buy",
        "definitely sell",
        "this is guaranteed",
        "guaranteed profit",
    ]

    for term in banned_terms:
        if term in lowered:
            return (
                "This explanation was blocked because it appeared to contain "
                "direct financial advice or guarantees. The system is intended "
                "for educational purposes only and cannot tell you what trades "
                "to make."
            )

    return raw_response


def is_personal_advice_request(text: str) -> bool:
    """
    Quick guardrail for 'what should I buy/sell' type questions.
    """
    text_l = text.lower()
    trigger_phrases = [
        "should i buy",
        "should i sell",
        "should i hold",
        "what stock should i buy",
        "tell me what to buy",
        "tell me what to sell",
        "exactly what should i do",
    ]
    return any(p in text_l for p in trigger_phrases)


# ======================================
# Main advisor chat entry point
# ======================================

def chat_with_advisor(
    user_question: str,
    ticker: str,
    action_text: str,
    pos_text: str,
    neg_text: str,
    backtest_summary: str | None = None,
    conversation_history: str | None = None,
    rl_confidence: float | None = None,
    risk_label: str | None = None,
) -> str:
    """
    Generate a conversational, educational response using a local Ollama model.

    Parameters
    ----------
    user_question : str
        The user's current question.
    ticker : str
        Stock ticker currently being discussed.
    action_text : str
        One of BUY / SELL / HOLD (model recommendation).
    pos_text : str
        Human-readable summary of positive contributing factors.
    neg_text : str
        Human-readable summary of negative or cautionary factors.
    backtest_summary : Optional[str]
        Optional short description of historical testing results.
    conversation_history : Optional[str]
        Recent conversation turns, formatted as:
            User: ...
            Advisor: ...
    rl_confidence : Optional[float]
        Model confidence for this ticker (0–1). If None, treated as unknown.
    risk_label : Optional[str]
        Risk classification for this ticker, e.g. "Low", "Medium", "High".

    Returns
    -------
    str
        Advisor's natural-language response.
    """

    # --- 1) Hard guardrail for explicit personal advice requests ---
    if is_personal_advice_request(user_question):
        return (
            "I can’t tell you exactly what to buy or sell – this tool is only for **education**.\n\n"
            "- I can explain how the model is thinking about **"
            f"{ticker}** and what the key risks are.\n"
            "- I can help you understand concepts like indicators, risk levels and backtesting.\n\n"
            "For real investment decisions, please speak with a licensed financial professional "
            "who can consider your full financial situation."
        )

    # --- 2) Input sanitisation (prompt injection etc.) ---
    cleaned_question = sanitize_user_question(user_question)

    # If the question was blocked, return the block message directly
    if cleaned_question.startswith("User query blocked"):
        return cleaned_question

    # =========================
    # 3) Core context for the LLM (structured prompting)
    # =========================
    if rl_confidence is not None:
        confidence_text = f"{rl_confidence:.0%} (approximate)"
    else:
        confidence_text = "Not available"

    risk_text = risk_label or "Not available"

    base_context = textwrap.dedent(
        f"""
        You are an **educational AI financial advisor** inside a university Final Year Project app
        called "Secure Explainable AI Financial Advisor Bot".

        Your role:
        - Help a non-technical user understand an AI-based trading advisor.
        - Be clear, neutral, and transparent.
        - Never provide real financial advice or guarantees.

        Context for this session:
        - Current stock under discussion: {ticker}
        - Advisor's recommended action: {action_text}
        - Model confidence for this recommendation: {confidence_text}
        - Risk level classification: {risk_text}

        Key positive signals detected by the system:
        - {pos_text}

        Key negative or caution signals detected:
        - {neg_text}
        """
    )

    # Optional historical testing context
    if backtest_summary:
        base_context += textwrap.dedent(
            f"""

            Historical testing / strategy comparison (high level):
            {backtest_summary}
            """
        )

    # Conversation memory
    history_block = ""
    if conversation_history:
        history_block = textwrap.dedent(
            f"""
            Previous conversation with this user (most recent first):
            {conversation_history}
            """
        )

    # Final prompt with explicit behavioural constraints & response structure
    prompt = textwrap.dedent(
        f"""
        {base_context}
        {history_block}

        The user now asks:

        "{cleaned_question}"

        Instructions for your answer:
        1. Start with a short 1–2 sentence **summary**.
        2. Then give **3–6 bullet points** explaining the reasoning:
           - Refer to the model's BUY/SELL/HOLD recommendation.
           - Mention important positive and negative factors.
           - Mention risk level and uncertainty where relevant.
        3. Add a short section titled **"🔎 What this means in plain English"**
           that re-explains the idea simply.
        4. Do NOT invent specific numbers, prices, or predictions.
        5. Do NOT give personal financial advice or tell the user what trades to make.
        6. End with a line starting with **"Educational note:"** reminding the user
           that this is for learning only and not real financial advice.
        7. Do not encourage real trading actions or guarantee profits.

        Now respond directly to the user.
        """
    )

    # =========================
    # 4) Call Ollama API
    # =========================
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "num_predict": 256,
        },
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()

    raw_text = data.get("response", "").strip()

    # 5) Output filtering
    return filter_llm_response(raw_text)


# ======================================
# Conversation summarisation (unchanged)
# ======================================

def summarize_conversation(
    ticker: str,
    conversation_history: str,
) -> str:
    """
    Use the local LLM to produce a short summary of a long conversation.

    Parameters
    ----------
    ticker : str
        The stock the user is mainly asking about.
    conversation_history : str
        Full conversation text in the form:
            User: ...
            Advisor: ...
    """

    prompt = textwrap.dedent(
        f"""
        You are an educational AI assistant.

        The following is a conversation between a user and an AI advisor
        about stock {ticker}. Your task is to summarise the discussion in a
        way that is easy for a non-technical user to review later.

        Conversation:
        {conversation_history}

        Instructions:
        1. Summarise the main topics and questions the user asked.
        2. Summarise how the advisor responded (at a high level).
        3. Focus on understanding and learning, not on trading actions.
        4. Use 3–6 short sentences or bullet points.
        5. Do NOT give financial advice or tell the user what to do.
        6. Do NOT invent specific numbers or performance claims.

        Now write the summary for the user.
        """
    )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()