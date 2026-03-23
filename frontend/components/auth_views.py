import altair as alt
import pandas as pd
import streamlit as st

from backend.users.service import (
    authenticate_user,
    create_user,
    reset_password,
    AccountLockedError,
)

from frontend.utils.auth_helpers import (
    is_valid_email,
    evaluate_password_strength,
    save_remember_me,
)


def show_landing_page():
    """
    Render the public landing page shown before the user logs in.

    This page introduces the SAFE-Bot system, highlights key features,
    shows an illustrative backtest chart, and provides navigation
    to the login and registration views.
    """
    nav_left, nav_mid, nav_right1, nav_right2 = st.columns([6, 2.5, 1.2, 1.2])

    with nav_left:
        st.markdown(
            """
            <div style="font-size:1.2rem;font-weight:700;color:white;padding-top:0.25rem;">
                📈 SAFE-Bot
            </div>
            """,
            unsafe_allow_html=True,
        )

    with nav_mid:
        st.write("")

    with nav_right1:
        if st.button("↪ Login", key="top_login_btn", use_container_width=True, type="tertiary"):
            st.session_state["auth_view"] = "login"
            st.rerun()

    with nav_right2:
        if st.button("👤 Register", key="top_register_btn", use_container_width=True, type="tertiary"):
            st.session_state["auth_view"] = "signup"
            st.rerun()

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    hero_left, hero_right = st.columns([1.05, 1])

    with hero_left:
        st.markdown(
            """
            <div class="hero-title">
                Secure Explainable AI<br>Financial Advisor
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="hero-text">
                Explainable AI-powered financial guidance using reinforcement learning,
                technical indicators, and historical strategy evaluation. Understand not
                only <b>what</b> the model recommends, but also <b>why</b> it suggests
                BUY, SELL, or HOLD, with risk insights and transparent comparisons against
                baseline strategies.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="card" style="margin-top:0.8rem;">
                <div class="mini-label">Example recommendation</div>
                <div style="color:#f8fafc;font-size:1.15rem;font-weight:700;margin-bottom:0.45rem;">
                    BUY (AAPL)
                </div>
                <div style="margin-bottom:0.45rem;">
                    <span class="pill pill-green">Confidence: 90% (High)</span>
                    <span class="pill pill-green">Risk: Low</span>
                </div>
                <div style="color:#cbd5e1;line-height:1.65;font-size:0.96rem;">
                    “The model currently favours AAPL because short-term signals are positive,
                    recent price behaviour is strong, and historical patterns look similar to
                    earlier periods where the strategy performed well.”
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Create Account", key="landing_get_started", use_container_width=True, type="primary"):
                st.session_state["auth_view"] = "signup"
                st.rerun()
        with b2:
            if st.button("Sign In", key="landing_login", use_container_width=True, type="primary"):
                st.session_state["auth_view"] = "login"
                st.rerun()

    with hero_right:
        chart_df = pd.DataFrame(
            {
                "Month": list(range(1, 13)),
                "RL strategy": [100, 118, 114, 132, 149, 165, 178, 201, 197, 215, 231, 249],
                "Buy & Hold": [100, 106, 103, 111, 107, 116, 120, 124, 128, 137, 133, 141],
            }
        ).melt(id_vars="Month", var_name="Strategy", value_name="Portfolio Value")

        chart = (
            alt.Chart(chart_df)
            .mark_line(
                point=alt.OverlayMarkDef(filled=True, size=70),
                strokeWidth=3,
            )
            .encode(
                x=alt.X(
                    "Month:Q",
                    title="Month",
                    axis=alt.Axis(
                        grid=True,
                        gridColor="rgba(255,255,255,0.08)",
                        domain=False,
                        tickColor="rgba(255,255,255,0.15)",
                        labelColor="#cbd5e1",
                        titleColor="#94a3b8",
                    ),
                ),
                y=alt.Y(
                    "Portfolio Value:Q",
                    title="Portfolio value",
                    axis=alt.Axis(
                        grid=True,
                        gridColor="rgba(255,255,255,0.08)",
                        domain=False,
                        tickColor="rgba(255,255,255,0.15)",
                        labelColor="#cbd5e1",
                        titleColor="#94a3b8",
                    ),
                ),
                color=alt.Color(
                    "Strategy:N",
                    scale=alt.Scale(
                        domain=["RL strategy", "Buy & Hold"],
                        range=["#f43f5e", "#38bdf8"],
                    ),
                    legend=alt.Legend(
                        title="Strategy",
                        orient="top-right",
                        labelColor="#cbd5e1",
                        titleColor="#f8fafc",
                        fillColor="transparent",
                        padding=8,
                    ),
                ),
                tooltip=[
                    alt.Tooltip("Strategy:N"),
                    alt.Tooltip("Month:Q"),
                    alt.Tooltip("Portfolio Value:Q"),
                ],
            )
            .properties(height=320)
            .configure_view(
                stroke=None,
                fill="transparent",
            )
            .configure(background="transparent")
            .configure_axis(
                labelFontSize=12,
                titleFontSize=13,
            )
            .configure_legend(
                labelFontSize=12,
                titleFontSize=13,
            )
        )
        st.markdown(
            """
            <div class="card" style="padding:16px 18px;">
                <div class="mini-label">Illustrative backtest snapshot</div>
                <div style="color:#f8fafc;font-weight:700;font-size:1.05rem;margin-bottom:0.35rem;">
                    RL strategy vs Buy & Hold
                </div>
                <div style="color:#94a3b8;font-size:0.92rem;line-height:1.6;margin-bottom:0.6rem;">
                    Example historical comparison showing how the reinforcement learning
                    strategy can be evaluated against a simpler baseline.
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.altair_chart(chart, use_container_width=True, theme=None)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("## Why this system is different")
    diff1, diff2, diff3, diff4 = st.columns(4)

    differences = [
        (
            "🔍",
            "Transparent decisions",
            "See the main signals that pushed the model toward or against each recommendation.",
        ),
        (
            "📊",
            "Backtested strategies",
            "Compare the RL strategy against Buy & Hold and RSI-based baselines on historical data.",
        ),
        (
            "🧠",
            "Explainable AI",
            "Read plain-English summaries so non-technical users can understand the AI’s reasoning.",
        ),
        (
            "⚠️",
            "Risk-aware insights",
            "View confidence, risk interpretation, and portfolio alerts alongside each recommendation.",
        ),
    ]

    for col, (icon, title, text) in zip([diff1, diff2, diff3, diff4], differences):
        with col:
            st.markdown(
                f"""
                <div class="feature-card-dark">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("## Key Features")
    f1, f2, f3 = st.columns(3)

    features = [
        (
            "🤖",
            "AI-powered recommendations",
            "Get stock suggestions driven by reinforcement learning, technical indicators, and historical price patterns.",
        ),
        (
            "📈",
            "Portfolio and market insights",
            "Inspect price movement, indicators, historical performance, and portfolio behaviour in one place.",
        ),
        (
            "🧾",
            "Explanations for non-technical users",
            "Understand why the model made its decision using friendly summaries, contributor lists, and comparison charts.",
        ),
    ]

    for col, (icon, title, text) in zip([f1, f2, f3], features):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div style="font-size:2rem;margin-bottom:10px;">{icon}</div>
                    <div style="font-size:1.2rem;font-weight:700;margin-bottom:8px;color:#6366f1;">
                        {title}
                    </div>
                    <div style="color:#cbd5e1;font-size:0.95rem;line-height:1.6;">
                        {text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## How It Works")

    hw_left, hw_right = st.columns([1, 1])

    with hw_left:
        steps = [
            (
                "1",
                "Data processing",
                "Market prices and technical indicators are gathered and transformed into signals the system can analyse.",
            ),
            (
                "2",
                "Reinforcement learning decision engine",
                "The RL model evaluates the current market state and selects a BUY, SELL, or HOLD action.",
            ),
            (
                "3",
                "Explainability and user insight",
                "The recommendation is presented with confidence, risk, historical comparisons, and plain-English explanations.",
            ),
        ]

        for n, title, text in steps:
            st.markdown(
                f"""
                <div class="how-step-card">
                    <div class="step-row">
                        <div class="step-badge">{n}</div>
                        <div>
                            <div class="step-title">{title}</div>
                            <div class="step-text">{text}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with hw_right:
        flow_nodes = pd.DataFrame(
            {
                "label": ["Market data", "Indicators", "RL model", "Explanation", "Portfolio"],
                "x": [1.25, 2.25, 3.25, 4.15, 4.15],
                "y": [3.0, 3.0, 3.0, 3.9, 2.1],
            }
        )

        flow_lines = pd.DataFrame(
            {
                "x": [1.25, 2.25, 2.25, 3.25, 3.25, 4.15, 3.25, 4.15],
                "y": [3.0, 3.0, 3.0, 3.0, 3.0, 3.9, 3.0, 2.1],
                "group": ["a", "a", "b", "b", "c", "c", "d", "d"],
            }
        )

        base_x = alt.X(
            "x:Q",
            axis=None,
            scale=alt.Scale(domain=[0.6, 5.0], nice=False, zero=False),
        )
        base_y = alt.Y(
            "y:Q",
            axis=None,
            scale=alt.Scale(domain=[1.5, 4.3], nice=False, zero=False),
        )

        lines = (
            alt.Chart(flow_lines)
            .mark_line(strokeWidth=4, color="#34d399", strokeCap="round")
            .encode(
                x=base_x,
                y=base_y,
                detail="group:N",
            )
        )

        nodes = (
            alt.Chart(flow_nodes)
            .mark_circle(size=1800, color="#60a5fa", opacity=0.95)
            .encode(
                x=base_x,
                y=base_y,
            )
        )

        left_labels = (
            alt.Chart(flow_nodes[flow_nodes["label"].isin(["Market data", "Indicators", "RL model"])])
            .mark_text(
                color="#e2e8f0",
                dy=34,
                fontSize=14,
                fontWeight="bold",
                align="center",
            )
            .encode(
                x=base_x,
                y=base_y,
                text="label:N",
            )
        )

        right_labels = (
            alt.Chart(flow_nodes[flow_nodes["label"].isin(["Explanation", "Portfolio"])])
            .mark_text(
                color="#e2e8f0",
                dx=34,
                dy=4,
                fontSize=14,
                fontWeight="bold",
                align="left",
            )
            .encode(
                x=base_x,
                y=base_y,
                text="label:N",
            )
        )

        flow_chart = (
            (lines + nodes + left_labels + right_labels)
            .properties(height=320)
            .configure(background="transparent")
            .configure_view(
                strokeOpacity=0,
                fill="transparent",
            )
        )

        st.markdown(
            """
            <div class="card" style="padding:16px 18px;">
                <div class="mini-label">System pipeline</div>
                <div style="color:#f8fafc;font-weight:700;font-size:1.05rem;margin-bottom:0.35rem;">
                    From data to recommendation
                </div>
                <div style="color:#94a3b8;font-size:0.92rem;line-height:1.6;margin-bottom:0.6rem;">
                    A simplified view of how the advisor processes data, makes a decision, and presents results to the user.
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.altair_chart(flow_chart, use_container_width=True, theme=None)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Ready to explore explainable investing?")
    st.caption(
        "Use the dashboard to inspect recommendations, compare strategy performance, and understand how the AI reached its decision."
    )

    c1, c2, c3 = st.columns([1.2, 1.2, 2.6])
    with c1:
        if st.button("Create Account", key="landing_cta_signup", use_container_width=True, type="primary"):
            st.session_state["auth_view"] = "signup"
            st.rerun()
    with c2:
        if st.button("Sign In", key="landing_cta_login", use_container_width=True, type="primary"):
            st.session_state["auth_view"] = "login"
            st.rerun()

    st.markdown(
        """
        <div class="soft-note" style="margin-top:1rem;">
            ⚠️ Evaluation note: this educational prototype uses historical backtesting and baseline comparisons
            (such as Buy & Hold and RSI strategies) to assess recommendation behaviour. Past performance does
            not guarantee future results.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    foot1, foot2, foot3 = st.columns(3)

    with foot1:
        st.markdown("### Secure Explainable AI Financial Advisor")
        st.caption("Educational AI-powered investment assistant for transparent recommendation analysis.")

    with foot2:
        st.markdown("### Highlights")
        st.caption("• Reinforcement learning recommendations")
        st.caption("• Explainable AI insights")
        st.caption("• Historical strategy comparison")
        st.caption("• Simulated portfolio tracking")

    with foot3:
        st.markdown("### Project Scope")
        st.caption("Built as an educational final year project prototype.")
        st.caption("Designed for non-technical user interaction.")
        st.caption("Not financial advice.")


def show_auth_page():
    top1, top2 = st.columns([0.8, 0.2])
    with top1:
        st.markdown("## Welcome to Secure Explainable AI Financial Advisor")
        st.caption(
            "Create an account to save your portfolio and explore a personalised educational dashboard."
        )
    with top2:
        st.write("")
        if st.button("← Back", key="auth_back_btn", use_container_width=True):
            st.session_state["auth_view"] = "landing"
            st.rerun()

    left_spacer, center_col, right_spacer = st.columns([1, 1.6, 1])

    with center_col:
        default_index = 0 if st.session_state.get("auth_view") == "login" else 1
        login_tab, signup_tab = st.tabs(["Log in", "Sign Up"])

        if default_index == 0:
            st.caption("Currently showing: Log in")
        else:
            st.caption("Currently showing: Sign Up")

        with login_tab:
            st.subheader("Log in")
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                remember_me = st.checkbox("Remember Me", value=False)
                submitted = st.form_submit_button("Log in")

            if submitted:
                clean_email = email.strip()
                if not clean_email or not password:
                    st.error("Please enter both email and password.")
                elif not is_valid_email(clean_email):
                    st.error("Please enter a valid email address (e.g. name@example.com).")
                else:
                    try:
                        user = authenticate_user(email=email, password=password)
                        if user:
                            st.session_state["user"] = user
                            st.session_state["auth_view"] = "landing"
                            save_remember_me(user.id, remember_me)
                            st.success(f"Welcome back, {user.username}!")
                            st.rerun()
                        else:
                            st.error("We could not sign you in. Check your email and password and try again.")
                    except AccountLockedError:
                        st.error(
                            "Your account has been locked due to too many failed login attempts. "
                            "Use 'Forgot your password?' below to reset it."
                        )

            with st.expander("Forgot your password?"):
                with st.form("forgot_pw_form"):
                    fp_email = st.text_input("Registered email", key="fp_email")
                    fp_new1 = st.text_input("New password", type="password", key="fp_new1")
                    fp_new2 = st.text_input("Confirm new password", type="password", key="fp_new2")

                    fp_label, _, fp_help = evaluate_password_strength(fp_new1)
                    st.markdown(
                        f"<small>Strength: <b>{fp_label}</b> — {fp_help}</small>",
                        unsafe_allow_html=True,
                    )
                    fp_submit = st.form_submit_button("Reset password")

                if fp_submit:
                    clean_fp_email = fp_email.strip()
                    if not clean_fp_email or not fp_new1 or not fp_new2:
                        st.error("Please fill in all the fields.")
                    elif fp_new1 != fp_new2:
                        st.error("New passwords do not match.")
                    elif len(fp_new1) < 8:
                        st.error("New password must be at least 8 characters long.")
                    elif fp_label in ("Too short", "Weak"):
                        st.error("New password is too weak. Please choose a stronger one.")
                    else:
                        try:
                            reset_password(clean_fp_email.strip(), fp_new1)
                            st.success("Your password has been reset successfully. You can now log in.")
                        except ValueError as e:
                            st.error(str(e))
                        except Exception:
                            st.error("Something went wrong while resetting your password. Please try again.")

        with signup_tab:
            st.subheader("Create a new account")
            with st.form("signup_form"):
                email = st.text_input("Email", key="signup_email")
                username = st.text_input("Username", key="signup_username")
                password = st.text_input("Password", type="password", key="signup_password")

                strength_label, _, strength_help = evaluate_password_strength(password)
                st.markdown(
                    f"<small>Strength: <b>{strength_label}</b> — {strength_help}</small>",
                    unsafe_allow_html=True,
                )

                confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
                submitted = st.form_submit_button("Sign up")

            if submitted:
                clean_email = email.strip()
                clean_username = username.strip()

                if not clean_email or not clean_username or not password or not confirm:
                    st.error("Please fill in all fields.")
                elif not is_valid_email(clean_email):
                    st.error("Please enter a valid email address (e.g. name@example.com).")
                elif password != confirm:
                    st.error("Passwords do not match.")
                elif len(password) < 8:
                    st.error("Password must be at least 8 characters long.")
                elif strength_label in ("Too short", "Weak"):
                    st.error("Password is too weak. Please use a stronger password.")
                else:
                    try:
                        user = create_user(
                            email=clean_email.strip(),
                            username=clean_username.strip(),
                            password=password,
                        )
                        st.session_state["user"] = user
                        st.session_state["auth_view"] = "landing"
                        st.success(f"Account created. Welcome, {user.username}!")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
                    except Exception:
                        st.error("Something went wrong while creating your account. Please try again.")