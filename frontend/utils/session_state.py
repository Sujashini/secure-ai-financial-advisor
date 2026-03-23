import os
import json
import streamlit as st

from backend.users.service import get_user_by_id
from frontend.utils.constants import REMEMBER_ME_PATH


def initialize_session_state():
    """
    Build a historical performance comparison chart between:
    - the RL strategy,
    - buy-and-hold,
    - RSI baseline strategy.

    Parameters:
        ticker (str): Stock ticker symbol.

    Returns:
        Altair chart or None if comparison data cannot be loaded.
    """
    if "user" not in st.session_state:
        st.session_state["user"] = None

    if "trade_history" not in st.session_state:
        st.session_state["trade_history"] = []

    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Dashboard"

    if "auth_view" not in st.session_state:
        st.session_state["auth_view"] = "landing"


def try_auto_login():
    """
    Initialise the main Streamlit session state variables
    used across the application.

    This ensures that key values always exist before the UI
    starts reading or updating them.
    """
    if st.session_state["user"] is None and os.path.exists(REMEMBER_ME_PATH):
        try:
            with open(REMEMBER_ME_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("remember") and data.get("user_id") is not None:
                remembered_user = get_user_by_id(int(data["user_id"]))
                if remembered_user:
                    st.session_state["user"] = remembered_user
        except Exception:
            pass