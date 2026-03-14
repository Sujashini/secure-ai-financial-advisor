import os
import sys
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from backend.data.market_data import fetch_stock_data
from backend.data.features import add_technical_indicators
from backend.RL.trading_env import TradingEnv
from backend.RL.dqn_agent import DQNAgent
from backend.LLM.chat_store import init_chat_db

from frontend.utils.ui_styles import apply_global_styles
from frontend.utils.session_state import initialize_session_state, try_auto_login
from frontend.utils.constants import ACTION_MAP, AVAILABLE_TICKERS
from frontend.utils.chart_builders import load_explainer
from frontend.utils.explanation_helpers import (
    compute_signal_strength_and_confidence,
    classify_risk_level,
)

from frontend.components.auth_views import show_landing_page, show_auth_page
from frontend.components.navbar import render_app_shell_topbar

from frontend.pages.dashboard import render_dashboard_page
from frontend.pages.explanation import render_explanation_page
from frontend.pages.chat import render_chat_page
from frontend.pages.profile import render_profile_page
from frontend.pages.help import render_help_page

init_chat_db()

st.set_page_config(
    page_title="Secure Explainable AI Financial Advisor Bot",
    layout="wide",
    page_icon="📈",
)

apply_global_styles()
initialize_session_state()
try_auto_login()

if st.session_state["user"] is None:
    if st.session_state.get("auth_view") == "landing":
        show_landing_page()
    else:
        show_auth_page()
    st.stop()

user = st.session_state["user"]
selected_page = render_app_shell_topbar(user)

ticker = st.selectbox(
    "Choose stock to inspect",
    AVAILABLE_TICKERS,
    index=0,
)

data = fetch_stock_data(ticker)
data = add_technical_indicators(data)

env = TradingEnv(data)
state, _ = env.reset()

agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
)
agent.load(os.path.join("models", f"dqn_{ticker}.pth"))
agent.epsilon = 0.0

action = agent.select_action(state)
action_text = ACTION_MAP[action]

explainer = load_explainer(ticker)
_, explanation = explainer.explain_state(state)

conf_label, conf_pct, conf_subtitle = compute_signal_strength_and_confidence(explanation)
risk_label, risk_text = classify_risk_level(data)

if selected_page == "Dashboard":
    render_dashboard_page(user, ticker, data, action, explanation)

elif selected_page == "Explanation":
    render_explanation_page(ticker, data, agent, state, action)

elif selected_page == "Chat with Advisor":
    render_chat_page(
        user=user,
        ticker=ticker,
        action_text=action_text,
        conf_pct=conf_pct,
        risk_label=risk_label,
        explanation=explanation,
    )

elif selected_page == "Profile / Settings":
    render_profile_page(user)

elif selected_page == "Help / Glossary":
    render_help_page()