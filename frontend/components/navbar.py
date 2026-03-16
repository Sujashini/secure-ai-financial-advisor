import streamlit as st
from frontend.utils.auth_helpers import save_remember_me


def render_app_shell_topbar(user):
    pages = [
        "Dashboard",
        "Explanation",
        "Chat with Advisor",
        "Portfolio",
        "Help / Glossary",
    ]

    page_labels = {
        "Dashboard": "Dashboard",
        "Explanation": "Explanation",
        "Chat with Advisor": "Chat",
        "Portfolio": "Portfolio",
        "Help / Glossary": "Help",
    }

    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Dashboard"

    brand_col, nav_col, user_col, logout_col = st.columns([1.6, 4.8, 1.1, 0.9])

    with brand_col:
        st.markdown(
            """
            <div style="
                display:flex;
                align-items:center;
                gap:10px;
                font-size:1.1rem;
                font-weight:700;
                color:#f8fafc;
                white-space:nowrap;
                padding-top:0.2rem;
            ">
                <span style="font-size:1.35rem;">📈</span>
                <span>SAFE-Bot</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with nav_col:
        nav_inner_cols = st.columns(len(pages))
        for col, page in zip(nav_inner_cols, pages):
            with col:
                is_active = st.session_state["active_page"] == page
                label = page_labels[page]
                button_type = "secondary" if is_active else "tertiary"

                if st.button(
                    label,
                    key=f"topnav_{page}",
                    use_container_width=True,
                    type=button_type,
                ):
                    st.session_state["active_page"] = page
                    st.rerun()

    with user_col:
        if st.button(
            f"👤 {user.username}",
            key="open_trader_profile_btn",
            use_container_width=True,
            type="tertiary",
        ):
            st.session_state["active_page"] = "Trader Profile"
            st.rerun()

    with logout_col:
        st.markdown(
            """
            <div style="
                display:flex;
                align-items:center;
                height:100%;
                margin-top:0.25rem;
            ">
            """,
            unsafe_allow_html=True,
        )
        if st.button("Logout", key="logout_button", use_container_width=True, type="tertiary"):
            save_remember_me(user.id, remember=False)
            st.session_state.clear()
            st.rerun()

    st.markdown(
        """
        <div style="
            margin-top:0.55rem;
            margin-bottom:0.9rem;
            border-bottom:1px solid rgba(255,255,255,0.10);
        "></div>
        """,
        unsafe_allow_html=True,
    )

    return st.session_state["active_page"]