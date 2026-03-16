import streamlit as st

from backend.users.service import change_password, get_portfolio
from frontend.utils.portfolio_helpers import compute_risk_metrics_for_ticker
from frontend.utils.chart_builders import build_allocation_chart
from frontend.components.dashboard_sections import (
    render_portfolio_snapshot,
    render_account_summary,
    render_holdings_table,
)


def render_portfolio_page(user, ticker):
    st.subheader("📁My Portfolio")

    portfolio = get_portfolio(user.id)

    top_left, top_right = st.columns([2, 1])

    with top_left:
        render_holdings_table(portfolio)

    with top_right:
        render_account_summary(portfolio)

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    lower_left, lower_right = st.columns([2, 1])

    with lower_left:
        render_portfolio_snapshot(user, portfolio)

    with lower_right:
        st.markdown('<div class="section-title">Portfolio allocation</div>', unsafe_allow_html=True)
        alloc_chart = build_allocation_chart(portfolio)
        if alloc_chart is not None:
            st.altair_chart(alloc_chart, use_container_width=True)
        else:
            st.info("No allocation chart available yet.")

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Historical risk & return</div>', unsafe_allow_html=True)
    metrics, err = compute_risk_metrics_for_ticker(ticker)

    if err or metrics is None:
        st.info("Historical risk metrics are not available right now.")
    else:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Return", f"{metrics['total_return'] * 100:.1f}%")
        with m2:
            st.metric("Worst drop", f"{metrics['max_drawdown'] * 100:.1f}%")
        with m3:
            st.metric("Sharpe", f"{metrics['sharpe']:.2f}" if metrics["sharpe"] is not None else "N/A")
        st.caption("Historical backtest metrics shown for transparency only.")

    st.markdown("---")
    st.markdown("### Account settings")
    st.markdown(f"**Username:** {user.username}")
    st.markdown(f"**Email:** {user.email}")

    st.markdown("---")
    st.markdown("### Change password")

    old_pw = st.text_input("Current password", type="password", key="prof_old_pw")
    new_pw1 = st.text_input("New password", type="password", key="prof_new_pw1")
    new_pw2 = st.text_input("Confirm new password", type="password", key="prof_new_pw2")

    if st.button("Update password", key="btn_change_pw"):
        if not old_pw or not new_pw1 or not new_pw2:
            st.warning("Please fill in all the fields.")
        elif new_pw1 != new_pw2:
            st.error("New passwords do not match.")
        elif len(new_pw1) < 8:
            st.error("New password must be at least 8 characters long.")
        else:
            try:
                change_password(user.id, old_pw, new_pw1)
                st.success("Password updated successfully.")
            except ValueError as e:
                st.error(str(e))
            except Exception:
                st.error("Something went wrong while updating your password. Please try again.")