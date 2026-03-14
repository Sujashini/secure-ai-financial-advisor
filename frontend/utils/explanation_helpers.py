import pandas as pd
from frontend.utils.constants import ACTION_MAP


def generate_plain_english_explanation(ticker: str, action: int, explanation: dict) -> str:
    action_text = ACTION_MAP.get(action, "HOLD")

    friendly_names = {
        "return_1": "very recent price movement",
        "sma_10": "short-term price trend",
        "sma_20": "medium-term price trend",
        "ema_10": "short-term trend (EMA)",
        "ema_20": "medium-term trend (EMA)",
        "volatility_10": "recent price volatility",
        "rsi_14": "momentum (RSI)",
        "open": "recent opening prices",
        "high": "recent highs",
        "low": "recent lows",
        "close": "recent closing prices",
    }

    def format_feature_list(items):
        names = []
        for item in items:
            raw_name = item["feature"]
            names.append(friendly_names.get(raw_name, raw_name))
        if not names:
            return "no single dominant factor"
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return f"{', '.join(names[:-1])}, and {names[-1]}"

    positives = explanation.get("top_positive", [])
    negatives = explanation.get("top_negative", [])

    pos_text = format_feature_list(positives)
    neg_text = format_feature_list(negatives)

    if action_text == "BUY":
        summary = (
            f"For {ticker}, the system currently leans towards **BUY**. "
            f"This is mainly because indicators related to {pos_text} "
            f"look similar to past situations where the price often went up. "
        )
        if negatives:
            summary += (
                f"However, it also sees some caution signals from {neg_text}, "
                "so this is not a guaranteed outcome and is for learning purposes only."
            )
        else:
            summary += (
                "There are no strong opposing signals, but this still does not guarantee future performance."
            )

    elif action_text == "SELL":
        summary = (
            f"For {ticker}, the system currently leans towards **SELL** or reducing exposure. "
            f"It has detected risk signals from {neg_text}, which look similar to past situations "
            "where the price often fell or became unstable. "
        )
        if positives:
            summary += (
                f"Some positive signs from {pos_text} are still present, "
                "so the picture is mixed and this is not a certainty."
            )
        else:
            summary += "Overall, the balance of signals is tilted towards caution."

    else:
        summary = (
            f"For {ticker}, the system suggests **HOLD**. "
            f"Signals from {pos_text} and {neg_text} are relatively balanced, "
            "so it does not see a strong reason to buy more or to sell right now. "
            "This indicates uncertainty rather than a clear prediction."
        )

    summary += (
        "\n\nThis explanation is based on patterns in historical data and is provided "
        "for educational and transparency purposes only. It is **not** financial advice."
    )
    return summary


def compute_signal_strength_and_confidence(explanation: dict):
    abs_vals = []
    for item in explanation.get("top_positive", []):
        abs_vals.append(abs(float(item["value"])))
    for item in explanation.get("top_negative", []):
        abs_vals.append(abs(float(item["value"])))

    if not abs_vals:
        return "Unclear", 0, "Signals are weak or mixed."

    raw_strength = sum(abs_vals)
    max_reasonable_strength = 0.1
    confidence_pct = min(raw_strength / max_reasonable_strength * 100.0, 100.0)

    if confidence_pct < 33:
        label = "Low"
        subtitle = "Signals are present but weak or conflicting."
    elif confidence_pct < 66:
        label = "Medium"
        subtitle = "Signals are moderate; there is some uncertainty."
    else:
        label = "High"
        subtitle = "Signals are strong and consistent with past patterns."

    return label, round(confidence_pct), subtitle


def classify_risk_level(data: pd.DataFrame):
    if "volatility_10" not in data.columns or data["volatility_10"].dropna().empty:
        return "Unknown", "Not enough data to estimate risk."

    recent_vol = float(data["volatility_10"].dropna().iloc[-1])

    if recent_vol < 0.015:
        return "Low", "Price has been relatively stable in recent history."
    elif recent_vol < 0.035:
        return "Medium", "Price moves up and down moderately."
    return "High", "Price has been quite jumpy; expect larger swings."


def get_risk_pill_class(risk_label: str) -> str:
    if risk_label == "Low":
        return "pill pill-green"
    if risk_label == "Medium":
        return "pill pill-amber"
    if risk_label == "High":
        return "pill pill-red"
    return "pill pill-blue"


def get_confidence_pill_class(conf_label: str) -> str:
    if conf_label == "High":
        return "pill pill-green"
    if conf_label == "Medium":
        return "pill pill-amber"
    if conf_label == "Low":
        return "pill pill-red"
    return "pill pill-blue"