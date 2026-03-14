import json
import os
import re
from frontend.utils.constants import REMEMBER_ME_PATH


def save_remember_me(user_id: int, remember: bool) -> None:
    if remember:
        data = {"remember": True, "user_id": user_id}
        try:
            with open(REMEMBER_ME_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass
    else:
        try:
            if os.path.exists(REMEMBER_ME_PATH):
                os.remove(REMEMBER_ME_PATH)
        except Exception:
            pass


def evaluate_password_strength(password: str):
    if not password:
        return "Too short", 0.0, "Enter a password to see the strength."

    length = len(password)
    score = 0

    if length >= 8:
        score += 1
    if length >= 12:
        score += 1
    if re.search(r"[a-z]", password):
        score += 1
    if re.search(r"[A-Z]", password):
        score += 1
    if re.search(r"\d", password):
        score += 1
    if re.search(r"[^\w\s]", password):
        score += 1

    max_score = 6
    norm = score / max_score

    if length < 8:
        label = "Too short"
        help_text = "Use at least 8 characters."
    elif norm < 0.4:
        label = "Weak"
        help_text = "Add upper/lowercase letters, numbers and a symbol."
    elif norm < 0.75:
        label = "Medium"
        help_text = "Pretty good — you can make it even stronger with more variety."
    else:
        label = "Strong"
        help_text = "This looks like a strong password."

    return label, norm, help_text


def is_valid_email(email: str) -> bool:
    if not email:
        return False
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))