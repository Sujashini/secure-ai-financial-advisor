import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from backend.users.service import get_portfolio
from frontend.utils.chart_builders import build_allocation_chart
from frontend.utils.portfolio_helpers import (
    compute_portfolio_unrealised,
    compute_risk_metrics_for_ticker,
    build_holdings_dataframe,
    generate_portfolio_alerts,
    generate_portfolio_takeaway,
    generate_suggested_next_steps,
    explain_risk_metrics,
)


def _render_alert_box(alert):
    severity = alert.get("severity", "low")
    title = alert.get("title", "Portfolio alert")
    message = alert.get("message", "")

    styles = {
        "high": {
            "bg": "rgba(127, 29, 29, 0.28)",
            "border": "#ef4444",
            "pill": "#ef4444",
            "label": "High",
        },
        "medium": {
            "bg": "rgba(120, 113, 18, 0.25)",
            "border": "#f59e0b",
            "pill": "#f59e0b",
            "label": "Medium",
        },
        "low": {
            "bg": "rgba(20, 83, 45, 0.28)",
            "border": "#22c55e",
            "pill": "#22c55e",
            "label": "Low",
        },
    }

    s = styles.get(severity, styles["low"])

    st.markdown(
        f"""
        <div style="
            background:{s['bg']};
            border-left:4px solid {s['border']};
            border-radius:14px;
            padding:0.95rem 1rem;
            margin-bottom:0.75rem;
        ">
            <div style="display:flex;align-items:center;gap:0.55rem;margin-bottom:0.35rem;">
                <span style="
                    background:{s['pill']};
                    color:white;
                    font-size:0.72rem;
                    font-weight:700;
                    padding:0.18rem 0.5rem;
                    border-radius:999px;
                ">{s['label']}</span>
                <span style="font-weight:700;color:#e5e7eb;">{title}</span>
            </div>
            <div style="color:#cbd5e1;line-height:1.55;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_portfolio_takeaway(portfolio):
    headline, detail = generate_portfolio_takeaway(portfolio)

    st.markdown(
        f"""
        <div style="
            background:linear-gradient(135deg, rgba(30,41,59,0.92), rgba(15,23,42,0.95));
            border:1px solid rgba(99,102,241,0.25);
            border-radius:18px;
            padding:1rem 1.1rem;
            margin-bottom:1rem;
            box-shadow:0 10px 24px rgba(0,0,0,0.20);
        ">
            <div style="font-size:0.82rem;color:#a5b4fc;font-weight:700;letter-spacing:0.04em;margin-bottom:0.45rem;">
                PORTFOLIO TAKEAWAY
            </div>
            <div style="font-size:1.05rem;font-weight:700;color:#f8fafc;margin-bottom:0.35rem;">
                {headline}
            </div>
            <div style="color:#cbd5e1;line-height:1.6;">
                {detail}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_holdings_table(portfolio):
    st.markdown('<div class="section-title">My holdings</div>', unsafe_allow_html=True)

    df = build_holdings_dataframe(portfolio)

    if df.empty:
        st.info("You do not have any holdings yet.")
        return

    def value_color(val):
        text = str(val).strip()
        if text.startswith("+"):
            return "#22c55e"
        if text.startswith("-"):
            return "#ef4444"
        return "#e2e8f0"

    rows_html = ""
    for _, row in df.iterrows():
        daily_color = value_color(row["Daily Change"])
        pl_color = value_color(row["Unrealised P/L ($)"])
        pl_pct_color = value_color(row["Unrealised P/L (%)"])

        rows_html += f"""
        <tr>
            <td>{row['Ticker']}</td>
            <td>{row['Shares']}</td>
            <td>{row['Avg Price ($)']}</td>
            <td>{row['Current Price ($)']}</td>
            <td style="color:{daily_color}; font-weight:600;">{row['Daily Change']}</td>
            <td>{row['Position Value ($)']}</td>
            <td style="color:{pl_color}; font-weight:600;">{row['Unrealised P/L ($)']}</td>
            <td style="color:{pl_pct_color}; font-weight:600;">{row['Unrealised P/L (%)']}</td>
            <td>{row['Weight']}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: Arial, sans-serif;
            }}

            .portfolio-table-wrap {{
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 16px;
                overflow: hidden;
                background: rgba(15, 23, 42, 0.96);
                box-shadow: 0 10px 24px rgba(0,0,0,0.18);
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
            }}

            thead th {{
                background: #172554;
                color: #f8fafc;
                text-align: left;
                padding: 14px 12px;
                font-weight: 700;
                border-bottom: 1px solid rgba(148, 163, 184, 0.16);
            }}

            tbody td {{
                background: #0f172a;
                color: #e2e8f0;
                padding: 14px 12px;
                border-bottom: 1px solid rgba(148, 163, 184, 0.10);
            }}

            tbody tr:hover td {{
                background: #1e293b;
            }}

            tbody tr:last-child td {{
                border-bottom: none;
            }}
        </style>
    </head>
    <body>
        <div class="portfolio-table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Ticker</th>
                        <th>Shares</th>
                        <th>Avg Price ($)</th>
                        <th>Current Price ($)</th>
                        <th>Daily Change</th>
                        <th>Position Value ($)</th>
                        <th>Unrealised P/L ($)</th>
                        <th>Unrealised P/L (%)</th>
                        <th>Weight</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """

    row_count = len(df)
    table_height = 78 + (row_count * 46)

    components.html(html, height=table_height, scrolling=False)
    
def _render_account_summary(portfolio):
    st.markdown('<div class="section-title">Account summary</div>', unsafe_allow_html=True)

    total_value, cost_basis, unrealised_pl = compute_portfolio_unrealised(portfolio)

    pct = (unrealised_pl / cost_basis * 100.0) if cost_basis > 0 else 0.0

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total value", f"${total_value:,.2f}")
    with c2:
        st.metric("Unrealised P/L", f"${unrealised_pl:,.2f}", f"{pct:+.1f}%")

    st.metric("Cost basis", f"${cost_basis:,.2f}")
    st.caption("Based on simulated trades in this prototype.")
    st.caption("Educational use only — not financial advice.")


def _render_next_steps(portfolio):
    steps = generate_suggested_next_steps(portfolio)

    st.markdown('<div class="section-title">Suggested next steps</div>', unsafe_allow_html=True)
    for i, step in enumerate(steps, start=1):
        st.markdown(
            f"""
            <div style="
                background:rgba(15,23,42,0.55);
                border:1px solid rgba(148,163,184,0.16);
                border-radius:14px;
                padding:0.85rem 0.95rem;
                margin-bottom:0.6rem;
                color:#cbd5e1;
                line-height:1.55;
            ">
                <span style="color:#a5b4fc;font-weight:700;">{i}.</span> {step}
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_risk_metrics(ticker):
    st.markdown('<div class="section-title">Historical risk & return</div>', unsafe_allow_html=True)

    metrics, err = compute_risk_metrics_for_ticker(ticker)

    if err or metrics is None:
        st.info("Historical risk metrics are not available right now.")
        return

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Return", f"{metrics['total_return'] * 100:.1f}%")
    with m2:
        st.metric("Worst drop", f"{metrics['max_drawdown'] * 100:.1f}%")
    with m3:
        st.metric(
            "Sharpe",
            f"{metrics['sharpe']:.2f}" if metrics["sharpe"] is not None else "N/A"
        )

    explanations = explain_risk_metrics(metrics)

    e1, e2, e3 = st.columns(3)
    with e1:
        st.caption(explanations["return_text"])
    with e2:
        st.caption(explanations["drawdown_text"])
    with e3:
        st.caption(explanations["sharpe_text"])

    st.caption(explanations["overall"])


def render_portfolio_page(user, ticker):
    st.subheader("📁 My Portfolio")

    portfolio = get_portfolio(user.id)

    _render_portfolio_takeaway(portfolio)

    top_left, top_right = st.columns([2, 1])

    with top_left:
        _render_holdings_table(portfolio)

    with top_right:
        _render_account_summary(portfolio)

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

    middle_left, middle_right = st.columns([2, 1])

    with middle_left:
        st.markdown('<div class="section-title">Portfolio alerts</div>', unsafe_allow_html=True)
        alerts = generate_portfolio_alerts(portfolio)

        if alerts:
            for alert in alerts:
                _render_alert_box(alert)
        else:
            st.success("No major concentration or diversification alerts detected right now.")

        _render_next_steps(portfolio)

    with middle_right:
        st.markdown('<div class="section-title">Portfolio allocation</div>', unsafe_allow_html=True)
        alloc_chart = build_allocation_chart(portfolio)
        if alloc_chart is not None:
            st.altair_chart(alloc_chart, use_container_width=True)
            st.caption(
                "This chart shows how your portfolio value is split across holdings. "
                "A very large slice may indicate concentration risk."
            )
        else:
            st.info("No allocation chart available yet.")

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

    _render_risk_metrics(ticker)