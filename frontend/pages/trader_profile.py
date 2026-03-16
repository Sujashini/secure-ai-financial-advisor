import streamlit as st


def render_trader_profile_page(user):
    # Default saved preferences in session state
    defaults = {
        "risk_tolerance": "Moderate",
        "trading_style": "Swing Trading",
        "investment_horizon": "Medium-term",
        "starting_capital": 5000.0,
        "preferred_assets": ["Stocks"],
        "preferred_sectors": ["Technology", "Healthcare"],
        "preferred_markets": ["US"],
        "explanation_style": "Balanced",
        "show_confidence": True,
        "show_risk_warnings": True,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    st.markdown("## 👤 Your Trader Profile")
    st.caption(
        "Configure your trading preferences so the advisor can present recommendations in a way that matches your style."
    )

    left_col, right_col = st.columns([1, 1.4])

    with left_col:

        initials = user.username[:2].upper() if user.username else "TP"

        with st.container(border=True):

            c1, c2 = st.columns([1,3])

            with c1:
                st.markdown(
                    f"""
                    <div style="
                    width:64px;
                    height:64px;
                    border-radius:50%;
                    background:linear-gradient(135deg,#7c3aed,#4f46e5);
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    font-size:1.4rem;
                    font-weight:700;
                    color:white;">
                    {initials}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c2:
                st.markdown(f"**{user.username}**")
                st.caption(user.email)

            st.divider()

            st.caption("Account type")
            st.write("Simulated trader")

            st.caption("Profile purpose")
            st.write(
                "This profile stores your trading preferences so the AI advisor can "
                "present recommendations, explanations, and warnings that match "
                "your trading behaviour."
            )

            st.divider()

            st.caption(
                "Educational prototype only. This system supports AI-driven "
                "analysis but does not provide financial advice."
            )
    with right_col:
        st.markdown('<div class="section-title">Trading Preferences</div>', unsafe_allow_html=True)

        pref_left, pref_right = st.columns(2)

        with pref_left:
            risk_tolerance = st.radio(
                "Risk tolerance",
                ["Conservative", "Moderate", "Aggressive"],
                index=["Conservative", "Moderate", "Aggressive"].index(
                    st.session_state["risk_tolerance"]
                ),
                key="risk_tolerance_input",
            )

            trading_style = st.selectbox(
                "Trading style",
                ["Day Trading", "Swing Trading", "Position Trading"],
                index=["Day Trading", "Swing Trading", "Position Trading"].index(
                    st.session_state["trading_style"]
                ),
                key="trading_style_input",
            )

            starting_capital = st.number_input(
                "Initial trading capital ($)",
                min_value=100.0,
                step=100.0,
                value=float(st.session_state["starting_capital"]),
                key="starting_capital_input",
            )

        with pref_right:
            investment_horizon = st.radio(
                "Trading horizon",
                ["Short-term", "Medium-term", "Long-term"],
                index=["Short-term", "Medium-term", "Long-term"].index(
                    st.session_state["investment_horizon"]
                ),
                key="investment_horizon_input",
            )

            explanation_style = st.selectbox(
                "Explanation style",
                ["Simple", "Balanced", "Technical"],
                index=["Simple", "Balanced", "Technical"].index(
                    st.session_state["explanation_style"]
                ),
                key="explanation_style_input",
            )

            show_confidence = st.checkbox(
                "Show confidence scores",
                value=st.session_state["show_confidence"],
                key="show_confidence_input",
            )

            show_risk_warnings = st.checkbox(
                "Show risk warnings",
                value=st.session_state["show_risk_warnings"],
                key="show_risk_warnings_input",
            )

        st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

        sec_left, sec_right = st.columns(2)

        with sec_left:
            st.markdown("#### Preferred asset classes")
            preferred_assets = []
            for asset in ["Stocks", "Crypto", "ETFs", "Forex"]:
                checked = asset in st.session_state["preferred_assets"]
                if st.checkbox(asset, value=checked, key=f"asset_{asset}"):
                    preferred_assets.append(asset)

            st.markdown("#### Preferred sectors")
            preferred_sectors = []
            for sector in [
                "Technology",
                "Healthcare",
                "Finance",
                "Energy",
                "Consumer",
                "Real Estate",
            ]:
                checked = sector in st.session_state["preferred_sectors"]
                if st.checkbox(sector, value=checked, key=f"sector_{sector}"):
                    preferred_sectors.append(sector)

        with sec_right:
            st.markdown("#### Preferred markets")
            preferred_markets = []
            for market in ["US", "Europe", "Asia", "Global"]:
                checked = market in st.session_state["preferred_markets"]
                if st.checkbox(market, value=checked, key=f"market_{market}"):
                    preferred_markets.append(market)

            st.markdown(
                """
                <div class="card" style="margin-top:0.7rem;">
                    <div class="mini-label">How this affects the advisor</div>
                    <div style="color:#94a3b8;line-height:1.65;">
                        These settings can be used to personalise how recommendations are framed,
                        which sectors are emphasised, and how much risk context is shown.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        btn_left, btn_right = st.columns([1, 1])

        with btn_left:
            if st.button("↺ Reset profile", use_container_width=True):
                for key, value in defaults.items():
                    st.session_state[key] = value
                st.rerun()

        with btn_right:
            if st.button("💾 Save preferences", use_container_width=True, type="primary"):
                st.session_state["risk_tolerance"] = risk_tolerance
                st.session_state["trading_style"] = trading_style
                st.session_state["investment_horizon"] = investment_horizon
                st.session_state["starting_capital"] = starting_capital
                st.session_state["preferred_assets"] = preferred_assets
                st.session_state["preferred_sectors"] = preferred_sectors
                st.session_state["preferred_markets"] = preferred_markets
                st.session_state["explanation_style"] = explanation_style
                st.session_state["show_confidence"] = show_confidence
                st.session_state["show_risk_warnings"] = show_risk_warnings
                st.success("Trader profile saved.")