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
    nav_left, nav_mid, nav_right1, nav_right2 = st.columns([6, 2.5, 1.2, 1.2])

    with nav_left:
        st.markdown(
            """
            <div style="font-size:1.2rem;font-weight:700;color:white;padding-top:0.25rem;">
                📈AI Financial Advisor Bot
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
        st.markdown("""
<div class="hero-title">
Secure Explainable AI<br>Financial Advisor
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="hero-text">
Personalized investment guidance powered by reinforcement learning,
explainable AI, and real-time market data. Explore recommendations,
manage a simulated portfolio, and understand why the model suggests
BUY, SELL, or HOLD.
</div>
""", unsafe_allow_html=True)

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Get Started", key="landing_get_started", use_container_width=True, type="primary"):
                st.session_state["auth_view"] = "signup"
                st.rerun()
        with b2:
            if st.button("Login", key="landing_login", use_container_width=True, type="primary"):
                st.session_state["auth_view"] = "login"
                st.rerun()

    with hero_right:
        chart_df = pd.DataFrame(
            {
                "Month": list(range(1, 13)),
                "AI Strategy": [22, 28, 27, 31, 35, 39, 42, 47, 46, 50, 53, 58],
                "Baseline": [16, 18, 17, 19, 18, 20, 21, 22, 23, 25, 24, 26],
            }
        ).melt(id_vars="Month", var_name="Series", value_name="Value")

        chart = (
            alt.Chart(chart_df)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X("Month:Q", axis=None),
                y=alt.Y("Value:Q", axis=None),
                color=alt.Color(
                    "Series:N",
                    scale=alt.Scale(
                        domain=["AI Strategy", "Baseline"],
                        range=["#f43f5e", "#38bdf8"],
                    ),
                    legend=None,
                ),
                tooltip=["Series", "Value"],
            )
            .properties(height=320)
            .configure_view(stroke=None)
        )

        st.markdown(
            """
            <div style="
                background:linear-gradient(135deg, #0f172a 0%, #172554 100%);
                border-radius:24px;
                padding:10px 14px 2px 14px;
                box-shadow:0 10px 24px rgba(15,23,42,0.18);
            ">
            """,
            unsafe_allow_html=True,
        )
        st.altair_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("## Key Features")
    f1, f2, f3 = st.columns(3)

    features = [
        ("🤖", "AI-Powered Recommendations",
         "Get stock suggestions driven by reinforcement learning, technical indicators, and historical price patterns."),
        ("📊", "Real-Time Market Data",
         "Inspect current price movement, trend indicators, historical performance, and strategy comparisons in one place."),
        ("🧠", "Explainable AI",
         "See which features influenced each decision so the recommendation is more transparent and easier to interpret."),
    ]

    for col, (icon, title, text) in zip([f1, f2, f3], features):
        with col:
            st.markdown(f"""
<div class="feature-card">
    <div style="font-size:2rem;margin-bottom:10px;">{icon}</div>
    <div style="font-size:1.2rem;font-weight:700;margin-bottom:8px;color:#6366f1;">
        {title}
    </div>
    <div style="color:#cbd5e1;font-size:0.95rem;line-height:1.6;">
        {text}
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## How It Works")

    hw_left, hw_right = st.columns([1, 1])

    with hw_left:
        steps = [
            ("1", "Create Your Profile",
             "Sign up securely to access your simulated portfolio, personalised dashboard, and advisor tools."),
            ("2", "Get Explainable Recommendations",
             "The system analyses market data and model signals to suggest BUY, SELL, or HOLD decisions."),
            ("3", "Track and Learn",
             "Monitor portfolio performance, compare strategies, and ask the advisor questions to understand the outputs."),
        ]

        for n, title, text in steps:
            st.markdown(
                f"""
                <div style="
                    background:#1e293b;
                    border:1px solid #e5e7eb;
                    border-radius:18px;
                    padding:16px 18px;
                    box-shadow:0 4px 12px rgba(15,23,42,0.05);
                    margin-bottom:0.8rem;
                ">
                    <div style="display:flex;gap:14px;align-items:flex-start;">
                        <div style="
                            min-width:44px;height:44px;border-radius:999px;
                            background:#7c3aed;color:white;font-weight:700;
                            display:flex;align-items:center;justify-content:center;
                            font-size:1.2rem;
                        ">{n}</div>
                        <div>
                            <div style="font-size:1.2rem;font-weight:700;color:#6366f1;margin-bottom:0.25rem;">{title}</div>
                            <div style="font-size:0.98rem;line-height:1.55;color:white;">{text}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with hw_right:
        flow_nodes = pd.DataFrame(
            {
                "label": ["User", "Profile", "Model", "Recommendation", "Portfolio"],
                "x": [1, 2, 3, 4, 4],
                "y": [3, 3, 3, 4, 2],
            }
        )

        flow_lines = pd.DataFrame(
            {
                "x": [1, 2, 2, 3, 3, 4, 3, 4],
                "y": [3, 3, 3, 3, 3, 4, 3, 2],
                "group": ["a", "a", "b", "b", "c", "c", "d", "d"],
            }
        )

        lines = (
            alt.Chart(flow_lines)
            .mark_line(strokeWidth=3, color="#34d399")
            .encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), detail="group:N")
        )
        nodes = (
            alt.Chart(flow_nodes)
            .mark_circle(size=1400, color="#60a5fa")
            .encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None))
        )
        labels = (
            alt.Chart(flow_nodes)
            .mark_text(color="#0f172a", dy=30, fontSize=12)
            .encode(x="x:Q", y="y:Q", text="label:N")
        )

        st.markdown(
            """
            <div style="
                background:#ffffff;
                border:1px solid #e5e7eb;
                border-radius:18px;
                padding:16px 16px 10px 16px;
                box-shadow:0 4px 12px rgba(15,23,42,0.05);
            ">
            """,
            unsafe_allow_html=True,
        )
        st.altair_chart((lines + nodes + labels).properties(height=320).configure_view(stroke=None), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Ready to Start Investing Smarter?")
    st.caption("Explore AI-powered recommendations, explanations, and portfolio insights in one educational platform.")

    c1, c2, c3 = st.columns([1.2, 1.2, 2.6])
    with c1:
        if st.button("Create Account", key="landing_cta_signup", use_container_width=True, type="primary"):
            st.session_state["auth_view"] = "signup"
            st.rerun()
    with c2:
        if st.button("Sign In", key="landing_cta_login", use_container_width=True, type="primary"):
            st.session_state["auth_view"] = "login"
            st.rerun()

    st.markdown("---")
    foot1, foot2, foot3 = st.columns(3)

    with foot1:
        st.markdown("### Secure Explainable AI Financial Advisor")
        st.caption("Your educational AI-powered investment assistant.")

    with foot2:
        st.markdown("### Highlights")
        st.caption("• Reinforcement learning recommendations")
        st.caption("• Explainable AI insights")
        st.caption("• Simulated portfolio tracking")

    with foot3:
        st.markdown("### Project Scope")
        st.caption("Built as an educational final year project prototype.")
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