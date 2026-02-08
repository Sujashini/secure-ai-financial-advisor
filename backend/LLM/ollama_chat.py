import requests
import textwrap
from typing import Optional

# ======================================
# Ollama configuration
# ======================================

OLLAMA_URL = "http://localhost:11434/api/generate"

# Make sure this model is pulled:
#   ollama pull llama3:8b
MODEL_NAME = "mistral:7b"


def chat_with_advisor(
    user_question: str,
    ticker: str,
    action_text: str,
    pos_text: str,
    neg_text: str,
    backtest_summary: Optional[str] = None,
    conversation_history: Optional[str] = None,
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
        One of BUY / SELL / HOLD.
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

    Returns
    -------
    str
        Advisor's natural-language response.
    """

    # =========================
    # Core context for the LLM
    # =========================
    base_context = textwrap.dedent(
        f"""
        You are an educational AI financial advisor.

        Your role:
        - Help a non-technical user understand an AI-based trading advisor.
        - Be clear, neutral, and transparent.
        - Never provide real financial advice or guarantees.

        Current stock under discussion: {ticker}
        Advisor's recommended action: {action_text}

        Key positive signals detected by the system:
        - {pos_text}

        Key negative or caution signals detected:
        - {neg_text}
        """
    )

    # =========================
    # Optional historical testing context
    # =========================
    if backtest_summary:
        base_context += textwrap.dedent(
            f"""

            Historical testing information:
            {backtest_summary}
            """
        )

    # =========================
    # Conversation memory
    # =========================
    history_block = ""
    if conversation_history:
        history_block = textwrap.dedent(
            f"""
            Previous conversation with this user:
            {conversation_history}
            """
        )

    # =========================
    # Final prompt
    # =========================
    prompt = textwrap.dedent(
        f"""
        {base_context}
        {history_block}

        The user now asks:

        "{user_question}"

        Instructions:
        1. Answer in 3–6 short sentences.
        2. Use simple, non-technical language.
        3. Refer to previous messages only if relevant.
        4. Do NOT invent specific numbers or predictions.
        5. Always remind the user this is educational and NOT financial advice.
        6. Do not encourage real trading actions.

        Respond directly to the user.
        """
    )

    # =========================
    # Call Ollama API
    # =========================
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "num_predict": 256,
        },
        timeout=120,
    )

    response.raise_for_status()
    data = response.json()

    # Ollama returns text in the "response" field
    return data.get("response", "").strip()

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

