import streamlit as st

from backend.users.service import change_password


def _profile_defaults():
    """
    Return the default trader profile settings used
    when the user has not saved any custom preferences yet.
    """
    return {
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


def _apply_profile_defaults():
    """
    Apply default trader profile settings into Streamlit session state
    for any keys that do not yet exist.
    """
    defaults = _profile_defaults()
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _advisor_mode_text(risk, style, horizon):
    """
    Build a plain-English description of how SAFE-Bot
    should frame recommendations based on the selected profile.
    """
    if risk == "Conservative":
        tone = "more cautious"
    elif risk == "Aggressive":
        tone = "more opportunity-seeking"
    else:
        tone = "balanced"

    return (
        f"SAFE-Bot is currently configured for a **{risk.lower()}** profile with a "
        f"**{style.lower()}** approach and a **{horizon.lower()}** focus. "
        f"This means explanations and warnings should feel more **{tone}** and aligned with that style."
    )


def _profile_summary_card():
    """
    Render a summary card showing the user's current
    trader profile settings at a glance.
    """
    risk = st.session_state.get("risk_tolerance", "Moderate")
    style = st.session_state.get("trading_style", "Swing Trading")
    horizon = st.session_state.get("investment_horizon", "Medium-term")
    capital = st.session_state.get("starting_capital", 5000.0)
    explanation = st.session_state.get("explanation_style", "Balanced")
    assets = st.session_state.get("preferred_assets", ["Stocks"])
    markets = st.session_state.get("preferred_markets", ["US"])

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(30,41,59,0.94), rgba(15,23,42,0.98));
            border: 1px solid rgba(99,102,241,0.22);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            box-shadow: 0 12px 28px rgba(0,0,0,0.18);
        ">
            <div style="
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                color: #a5b4fc;
                margin-bottom: 0.45rem;
            ">
                PROFILE SNAPSHOT
            </div>
            <div style="
                font-size: 1.05rem;
                font-weight: 700;
                color: #f8fafc;
                margin-bottom: 0.4rem;
            ">
                {risk} • {style} • {horizon}
            </div>
            <div style="color:#cbd5e1; line-height:1.68;">
                SAFE-Bot will frame recommendations for a <b>{risk.lower()}</b> investor with a
                <b>{style.lower()}</b> approach and a <b>{horizon.lower()}</b> focus.
                The current profile assumes starting capital of <b>${capital:,.0f}</b>,
                prioritises <b>{", ".join(assets) if assets else "no asset classes selected"}</b>,
                and focuses on <b>{", ".join(markets) if markets else "no markets selected"}</b>.
                Explanation style is set to <b>{explanation.lower()}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _preference_help_card():
    """
    Render a short explanatory card describing how
    profile preferences affect the behaviour of SAFE-Bot.
    """
    st.markdown(
        """
        <div class="card" style="margin-top:0.5rem;">
            <div class="mini-label">How this affects the advisor</div>
            <div style="color:#94a3b8;line-height:1.68;">
                These settings let SAFE-Bot personalise how recommendations are framed,
                how much risk context is shown, and which markets or sectors are emphasised.
                This makes the system feel more relevant while keeping the educational focus.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _preset_definitions():
    """
    Render a short explanatory card describing how
    profile preferences affect the behaviour of SAFE-Bot.
    """
    return {
        "Balanced learner": {
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
        },
        "Cautious beginner": {
            "risk_tolerance": "Conservative",
            "trading_style": "Position Trading",
            "investment_horizon": "Long-term",
            "starting_capital": 3000.0,
            "preferred_assets": ["Stocks", "ETFs"],
            "preferred_sectors": ["Healthcare", "Consumer"],
            "preferred_markets": ["US", "Global"],
            "explanation_style": "Simple",
            "show_confidence": True,
            "show_risk_warnings": True,
        },
        "Growth-focused explorer": {
            "risk_tolerance": "Aggressive",
            "trading_style": "Swing Trading",
            "investment_horizon": "Short-term",
            "starting_capital": 7000.0,
            "preferred_assets": ["Stocks", "Crypto", "ETFs"],
            "preferred_sectors": ["Technology", "Finance"],
            "preferred_markets": ["US", "Asia"],
            "explanation_style": "Technical",
            "show_confidence": True,
            "show_risk_warnings": True,
        },
    }


def _apply_preset(name: str):
    """
    Apply one of the predefined profile presets
    into session state and refresh the page.
    """
    preset = _preset_definitions()[name]
    for key, value in preset.items():
        st.session_state[key] = value
    st.success(f"Applied preset: {name}")
    st.rerun()


def render_trader_profile_page(user):
    """
    Render the trader profile page.

    This page allows the user to:
    - review their current profile summary,
    - apply a preset trading profile,
    - update account password,
    - customise trading preferences,
    - choose display and explanation options,
    - save or reset profile settings.
    """
    _apply_profile_defaults()
    defaults = _profile_defaults()

    st.markdown("## My Trader Profile")
    st.caption(
        "Configure your trading preferences so the advisor can present recommendations in a way that matches your style."
    )

    _profile_summary_card()

    st.info(_advisor_mode_text(
        st.session_state["risk_tolerance"],
        st.session_state["trading_style"],
        st.session_state["investment_horizon"],
    ))

    with st.expander("Use a recommended profile preset"):
        st.caption("This quickly fills the form with a ready-made learning style.")
        preset_name = st.selectbox(
            "Choose a preset",
            ["Balanced learner", "Cautious beginner", "Growth-focused explorer"],
            key="profile_preset_select",
        )
        if st.button("Apply preset", key="apply_profile_preset"):
            _apply_preset(preset_name)

    left_col, right_col = st.columns([1, 1.45])

    with left_col:
        initials = user.username[:2].upper() if user.username else "TP"

        with st.container(border=True):
            top1, top2 = st.columns([1, 3])

            with top1:
                st.markdown(
                    f"""
                    <div style="
                        width:68px;
                        height:68px;
                        border-radius:50%;
                        background:linear-gradient(135deg,#7c3aed,#4f46e5);
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-size:1.5rem;
                        font-weight:700;
                        color:white;
                        box-shadow:0 10px 22px rgba(79,70,229,0.28);
                    ">
                        {initials}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with top2:
                st.markdown(f"**{user.username}**")
                st.caption(user.email)

            st.divider()

            metric1, metric2 = st.columns(2)
            with metric1:
                st.caption("Account type")
                st.write("Simulated trader")
            with metric2:
                st.caption("Profile status")
                st.write("Active")

            st.caption("Profile purpose")
            st.write(
                "This profile stores your trading preferences so SAFE-Bot can present "
                "recommendations, explanations, and warnings in a way that better matches "
                "your intended investing style."
            )

            st.divider()

            st.markdown("### Account settings")
            st.caption("Update your password for this prototype account.")

            old_pw = st.text_input("Current password", type="password", key="prof_old_pw")
            new_pw1 = st.text_input("New password", type="password", key="prof_new_pw1")
            new_pw2 = st.text_input("Confirm new password", type="password", key="prof_new_pw2")

            if st.button("Update password", key="btn_change_pw", use_container_width=True):
                if not old_pw or not new_pw1 or not new_pw2:
                    st.warning("Please fill in all password fields.")
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

            st.divider()

            st.caption(
                "Educational prototype only. SAFE-Bot supports AI-driven analysis but does not provide personal financial advice."
            )

    with right_col:
        st.markdown('<div class="section-title">Trading Preferences</div>', unsafe_allow_html=True)

        pref_left, pref_right = st.columns(2)

        with pref_left:
            st.markdown("#### Risk & strategy")
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
            st.caption("This helps SAFE-Bot present recommendations in a style that matches your approach.")

            starting_capital = st.number_input(
                "Initial trading capital ($)",
                min_value=100.0,
                step=100.0,
                value=float(st.session_state["starting_capital"]),
                key="starting_capital_input",
            )

        with pref_right:
            st.markdown("#### Horizon & display")
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
            st.caption("Choose whether explanations should be easier to read or more detailed.")

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

        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

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

            _preference_help_card()

        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

        preview_assets = ", ".join(preferred_assets) if preferred_assets else "None selected"
        preview_markets = ", ".join(preferred_markets) if preferred_markets else "None selected"
        preview_sectors = ", ".join(preferred_sectors[:3]) if preferred_sectors else "None selected"
        if len(preferred_sectors) > 3:
            preview_sectors += f" +{len(preferred_sectors) - 3} more"

        st.markdown(
            f"""
            <div style="
                background: rgba(15,23,42,0.68);
                border: 1px solid rgba(148,163,184,0.14);
                border-radius: 16px;
                padding: 0.95rem 1rem;
                margin-bottom: 0.9rem;
            ">
                <div class="mini-label">Live preference summary</div>
                <div style="color:#cbd5e1; line-height:1.75;">
                    <b>Assets:</b> {preview_assets}<br>
                    <b>Markets:</b> {preview_markets}<br>
                    <b>Sectors:</b> {preview_sectors}<br>
                    <b>Explanation style:</b> {explanation_style}<br>
                    <b>Risk warnings:</b> {"Shown" if show_risk_warnings else "Hidden"}<br>
                    <b>Confidence scores:</b> {"Shown" if show_confidence else "Hidden"}
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
            if st.button("Save preferences", use_container_width=True, type="primary"):
                if not preferred_assets:
                    st.error("Please select at least one preferred asset class.")
                elif not preferred_markets:
                    st.error("Please select at least one preferred market.")
                else:
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

                    st.success(
                        f"Profile saved. SAFE-Bot will now use a {risk_tolerance.lower()} / "
                        f"{trading_style.lower()} / {investment_horizon.lower()} profile, "
                        f"with {explanation_style.lower()} explanations."
                    )