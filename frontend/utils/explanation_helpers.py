import pandas as pd
from frontend.utils.constants import ACTION_MAP


FRIENDLY_NAMES = {
    "return_1": "very recent price movement",
    "sma_10": "short-term price trend",
    "sma_20": "medium-term price trend",
    "ema_10": "short-term trend (EMA)",
    "ema_20": "medium-term trend (EMA)",
    "volatility_10": "recent price volatility",
    "rsi_14": "momentum (RSI)",
    "open": "recent opening prices",
    "high": "recent high prices",
    "low": "recent low prices",
    "close": "recent closing prices",
    "volume": "trading volume",
    "macd": "trend momentum (MACD)",
    "macd_signal": "MACD signal trend",
    "bollinger_upper": "upper Bollinger band level",
    "bollinger_lower": "lower Bollinger band level",
    "position_flag": "current position status",
}


def friendly_feature_name(name: str) -> str:
    return FRIENDLY_NAMES.get(name, name.replace("_", " ").lower())


def format_feature_list(items):
    names = [friendly_feature_name(item["feature"]) for item in items if "feature" in item]

    if not names:
        return "no single dominant factor"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def generate_plain_english_explanation(
    ticker: str,
    action: int,
    explanation: dict,
    style: str = "Balanced",
) -> list[str]:
    """
    Returns a list of explanation paragraphs.
    style: Simple | Balanced | Technical
    """
    action_text = ACTION_MAP.get(action, "HOLD")
    style = (style or "Balanced").strip().title()

    positives = explanation.get("top_positive", [])
    negatives = explanation.get("top_negative", [])

    pos_text = format_feature_list(positives)
    neg_text = format_feature_list(negatives)

    paragraphs = []

    if style == "Simple":
        if action_text == "BUY":
            paragraphs.append(
                f"The model currently leans toward BUY for {ticker}. "
                f"The strongest supportive signals come from {pos_text}, which look more favorable right now."
            )
            if negatives:
                paragraphs.append(
                    f"There are still some caution signals from {neg_text}, so this is not a guaranteed outcome."
                )
        elif action_text == "SELL":
            paragraphs.append(
                f"The model currently leans toward SELL for {ticker}. "
                f"The strongest caution signals come from {neg_text}, which make the setup look less favorable."
            )
            if positives:
                paragraphs.append(
                    f"There are still some supportive signs from {pos_text}, so the picture is not completely one-sided."
                )
        else:
            paragraphs.append(
                f"The model currently suggests HOLD for {ticker}. "
                f"The signals are mixed, so it does not see a strong reason to buy more or sell right now."
            )
            paragraphs.append(
                f"Some signals support the stock, such as {pos_text}, while others create caution, such as {neg_text}."
            )

    elif style == "Technical":
        if action_text == "BUY":
            paragraphs.append(
                f"For {ticker}, the model currently leans toward BUY because the highest positive feature contributions "
                f"come from {pos_text}."
            )
            if negatives:
                paragraphs.append(
                    f"The main negative contributors are {neg_text}, which means the feature profile is supportive overall "
                    f"but still contains some conflicting signals."
                )
        elif action_text == "SELL":
            paragraphs.append(
                f"For {ticker}, the model currently leans toward SELL because the negative feature contributions from "
                f"{neg_text} outweigh the supportive effects."
            )
            if positives:
                paragraphs.append(
                    f"At the same time, {pos_text} still provides some positive signal, so the setup is cautionary rather than absolute."
                )
        else:
            paragraphs.append(
                f"For {ticker}, the model currently suggests HOLD because positive and negative feature contributions are relatively balanced."
            )
            paragraphs.append(
                f"The supporting factors include {pos_text}, while the cautionary factors include {neg_text}."
            )

    else:  # Balanced
        if action_text == "BUY":
            paragraphs.append(
                f"For {ticker}, the system currently leans toward BUY. "
                f"This is mainly because indicators related to {pos_text} look similar to past situations where price behaviour was more favorable."
            )
            if negatives:
                paragraphs.append(
                    f"However, it also sees some caution signals from {neg_text}, so this is not a guaranteed outcome."
                )
        elif action_text == "SELL":
            paragraphs.append(
                f"For {ticker}, the system currently leans toward SELL or reducing exposure. "
                f"It has detected cautionary patterns from {neg_text}, which look less favorable based on historical behaviour."
            )
            if positives:
                paragraphs.append(
                    f"Some positive signs from {pos_text} are still present, so the picture is mixed rather than completely one-directional."
                )
        else:
            paragraphs.append(
                f"For {ticker}, the system suggests HOLD. "
                f"The current signals are relatively balanced, so it does not see a strong reason to buy more or sell right now."
            )
            paragraphs.append(
                f"Some factors support the stock, such as {pos_text}, while others create caution, such as {neg_text}."
            )

    paragraphs.append(
        "This explanation is based on patterns in historical market data and is shown for educational transparency. "
        "It should not be treated as financial advice."
    )

    return paragraphs


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
        subtitle = "Signals are moderate, so there is still some uncertainty."
    else:
        label = "High"
        subtitle = "Signals are strong and more consistent with past patterns."

    return label, round(confidence_pct), subtitle


def classify_risk_level(data: pd.DataFrame):
    if "volatility_10" not in data.columns or data["volatility_10"].dropna().empty:
        return "Unknown", "Not enough recent data to estimate risk."

    recent_vol = float(data["volatility_10"].dropna().iloc[-1])

    if recent_vol < 0.015:
        return "Low", "Price has been relatively stable in recent history."
    elif recent_vol < 0.035:
        return "Medium", "Price has shown moderate ups and downs."
    return "High", "Price has been more jumpy, so larger swings are more likely."


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


def generate_takeaway_text(ticker, action_text, conf_pct, risk_label, conf_subtitle, risk_text):
    return (
        f"The model currently suggests {action_text} for {ticker} with {conf_pct}% confidence and {risk_label.lower()} risk. "
        f"{conf_subtitle} {risk_text}"
    )