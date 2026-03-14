import os
import json
import streamlit as st

from backend.users.service import get_user_by_id
from frontend.utils.constants import REMEMBER_ME_PATH


def initialize_session_state():
    if "user" not in st.session_state:
        st.session_state["user"] = None

    if "trade_history" not in st.session_state:
        st.session_state["trade_history"] = []

    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Dashboard"

    if "auth_view" not in st.session_state:
        st.session_state["auth_view"] = "landing"


def try_auto_login():
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