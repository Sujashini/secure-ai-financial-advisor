import re
import ollama

MODEL_NAME = "mistral:7b"


def _build_system_prompt():
    return """
You are a clear, calm, educational AI financial advisor assistant inside a student project demo.

Your purpose:
- Explain a stock recommendation in plain English
- Help non-technical users understand signals, risk, and historical strategy comparisons
- Be informative, concise, and transparent

Important rules:
- Do NOT give personalised financial advice
- Do NOT tell the user what they personally should invest
- Do NOT promise profits or certainty
- Do NOT pretend future performance is guaranteed
- Keep answers practical and easy to understand
- Avoid unnecessary jargon
- Prefer short paragraphs over long bullet-heavy responses
- Mention only the most relevant signals for the user's question
- Use a brief educational note only when relevant, not in an overly repetitive way

Response style:
- Start with a direct answer in 1-2 sentences
- Then explain the 2-4 most relevant reasons
- Use short paragraphs, not one large block
- Keep most answers around 100-180 words unless the user asks for more detail
- If comparing strategies, clearly say results are historical only
- End naturally without repeating the same disclaimer too many times
"""


def _build_user_prompt(
    user_question,
    ticker,
    action_text,
    pos_text,
    neg_text,
    backtest_summary,
    conversation_history="",
    rl_confidence=None,
    risk_label=None,
):
    confidence_text = "N/A" if rl_confidence is None else f"{rl_confidence:.2f}"
    risk_text = risk_label if risk_label else "Unknown"

    return f"""
Context for this question:

Ticker: {ticker}
Current model recommendation: {action_text}
Model confidence score: {confidence_text}
Risk level: {risk_text}

Main supportive signals:
{pos_text}

Main caution signals:
{neg_text}

Historical strategy comparison:
{backtest_summary}

Recent conversation history:
{conversation_history}

User question:
{user_question}

Please answer in plain English for a non-technical user.
Only mention the most relevant signals for this question.
"""


def _clean_answer(answer: str) -> str:
    answer = answer.strip()

    answer = re.sub(r"\n{3,}", "\n\n", answer)
    answer = re.sub(r"[ \t]+", " ", answer)

    if len(answer) > 1600:
        answer = answer[:1600].rstrip() + "..."

    return answer


def chat_with_advisor(
    user_question,
    ticker,
    action_text,
    pos_text,
    neg_text,
    backtest_summary,
    conversation_history="",
    rl_confidence=None,
    risk_label=None,
):
    prompt = _build_user_prompt(
        user_question=user_question,
        ticker=ticker,
        action_text=action_text,
        pos_text=pos_text,
        neg_text=neg_text,
        backtest_summary=backtest_summary,
        conversation_history=conversation_history,
        rl_confidence=rl_confidence,
        risk_label=risk_label,
    )

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": 0.45,
        },
    )

    answer = response["message"]["content"]
    return _clean_answer(answer)


def summarize_conversation(ticker, conversation_history):
    summary_prompt = f"""
Summarise this educational chat about {ticker} for a non-technical user.

Requirements:
- Keep it short and clear
- Focus on the main recommendation, risk, and most important signals discussed
- Mention that results are educational and historical where relevant
- Use plain English
- Keep it under 120 words

Conversation:
{conversation_history}
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You create short, clear educational summaries for non-technical users. "
                    "Use 2 short paragraphs maximum."
                ),
            },
            {"role": "user", "content": summary_prompt},
        ],
        options={
            "temperature": 0.35,
        },
    )

    return _clean_answer(response["message"]["content"])