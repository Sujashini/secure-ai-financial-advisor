from datetime import datetime
import pandas as pd
import streamlit as st

from backend.users.service import buy_shares, sell_shares
from frontend.utils.explanation_helpers import (
    get_confidence_pill_class,
    get_risk_pill_class,
)
from frontend.utils.portfolio_helpers import (
    get_latest_price_and_change,
    compute_portfolio_unrealised,
    compute_realised_pl,
    generate_portfolio_alerts,
)
from frontend.utils.constants import WATCHLIST_TICKERS, COMPANY_NAMES


def render_hero_section(ticker, action_text, conf_label, conf_pct, conf_subtitle, risk_label, risk_text, explanation):
    pos = explanation.get("top_positive", [])
    top_features = [item["feature"] for item in pos[:3]]
    factors_text = ", ".join(top_features) if top_features else "mixed signals"

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="mini-label">AI recommendation for selected stock</div>
            <div class="big-value">{action_text} ({ticker})</div>
            <div style="margin-top: 0.55rem;">
                <span class="{get_confidence_pill_class(conf_label)}">Confidence: {conf_pct}% ({conf_label})</span>
                <span class="{get_risk_pill_class(risk_label)}" style="margin-left: 0.45rem;">Risk: {risk_label}</span>
            </div>
            <div class="soft-note">
                Main factors behind this suggestion include: {factors_text}.<br>
                {conf_subtitle} {risk_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trade_panel(user, ticker, action_text, current_price, portfolio):
    st.markdown('<div class="section-title">Act on this recommendation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">You can simulate a simple trade here for the selected stock.</div>',
        unsafe_allow_html=True,
    )

    if action_text == "BUY":
        trade_col1, trade_col2 = st.columns([3, 1])
        with trade_col1:
            shares_to_buy = st.number_input(
                "Number of shares to buy",
                min_value=1.0,
                step=1.0,
                key="buy_shares_input",
            )
        with trade_col2:
            st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)
            if st.button("✅ Buy shares", key="buy_btn", use_container_width=True):
                try:
                    buy_shares(
                        user_id=user.id,
                        ticker=ticker,
                        shares=shares_to_buy,
                        price=current_price,
                    )
                    st.session_state["trade_history"].append(
                        {
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "ticker": ticker,
                            "action": "BUY",
                            "shares": float(shares_to_buy),
                            "price": float(current_price),
                        }
                    )
                    st.success(f"Bought {shares_to_buy} shares of {ticker}.")
                    st.rerun()
                except Exception as e:
                    st.error("Could not complete purchase.")
                    st.caption(str(e))

    elif action_text == "SELL":
        owned_position = next((p for p in portfolio if p.ticker == ticker), None)
        if owned_position:
            max_shares = float(owned_position.shares)
            trade_col1, trade_col2 = st.columns([3, 1])

            with trade_col1:
                shares_to_sell = st.number_input(
                    "Number of shares to sell",
                    min_value=1.0,
                    max_value=max_shares,
                    step=1.0,
                    key="sell_shares_input",
                )

            with trade_col2:
                st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)
                if st.button("🔻 Sell shares", key="sell_btn", use_container_width=True):
                    try:
                        cost_price = float(owned_position.avg_price)
                        sell_shares(
                            user_id=user.id,
                            ticker=ticker,
                            shares=shares_to_sell,
                            price=current_price,
                        )
                        st.session_state["trade_history"].append(
                            {
                                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "ticker": ticker,
                                "action": "SELL",
                                "shares": float(shares_to_sell),
                                "price": float(current_price),
                                "cost_price": cost_price,
                            }
                        )
                        st.success(f"Sold {shares_to_sell} shares of {ticker}.")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error("Could not complete sale.")
                        st.caption(str(e))
        else:
            st.info("You do not currently own this stock.")
    else:
        st.info("The AI suggests holding. No action is required right now.")


def render_portfolio_snapshot(user, portfolio):
    st.markdown('<div class="section-title">Portfolio snapshot</div>', unsafe_allow_html=True)

    if not portfolio:
        st.info("You do not have any holdings yet.")
        return

    cols = st.columns(2)

    for i, pos in enumerate(portfolio):
        with cols[i % 2]:
            t = pos.ticker
            price, change_pct = None, None
            try:
                price, change_pct = get_latest_price_and_change(t)
            except Exception:
                pass

            price_str = f"${price:.2f}" if price is not None else "N/A"
            change_str = "N/A"
            change_color = "#6b7280"

            if change_pct is not None:
                arrow = "▲" if change_pct >= 0 else "▼"
                change_color = "#16a34a" if change_pct >= 0 else "#dc2626"
                change_str = f"{arrow} {change_pct:.2f}%"

            st.markdown(
                f"""
                <div class="card">
                    <div style="font-weight:700;font-size:1rem;">{t}</div>
                    <div class="mini-label">Shares: {pos.shares} · Avg: ${pos.avg_price:.2f}</div>
                    <div style="font-size:1.8rem;font-weight:750;margin-top:0.35rem;">{price_str}</div>
                    <div style="margin-top:0.2rem;color:{change_color};font-weight:600;">{change_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            b1, b2, b3 = st.columns(3)

            with b1:
                if st.button("Buy 1", key=f"quick_buy_{t}", use_container_width=True):
                    try:
                        buy_shares(
                            user_id=user.id,
                            ticker=t,
                            shares=1.0,
                            price=price if price is not None else pos.avg_price,
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

            with b2:
                if pos.shares >= 1 and st.button("Sell 1", key=f"quick_sell_{t}", use_container_width=True):
                    try:
                        sell_shares(
                            user_id=user.id,
                            ticker=t,
                            shares=1.0,
                            price=price if price is not None else pos.avg_price,
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

            with b3:
                if st.button("Close", key=f"close_{t}", use_container_width=True):
                    try:
                        sell_shares(
                            user_id=user.id,
                            ticker=t,
                            shares=float(pos.shares),
                            price=price if price is not None else pos.avg_price,
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))


def render_account_summary(portfolio):
    st.markdown('<div class="section-title">Account summary</div>', unsafe_allow_html=True)

    if not portfolio:
        st.info("No holdings yet, so the account summary is empty.")
        return

    total_value, cost_basis, unrealised_pl = compute_portfolio_unrealised(portfolio)
    unrealised_pct = (unrealised_pl / cost_basis * 100.0) if cost_basis > 0 else 0.0
    realised_pl = compute_realised_pl(st.session_state.get("trade_history", []))

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total value", f"${total_value:,.2f}")
    with c2:
        st.metric("Unrealised P/L", f"${unrealised_pl:,.2f}", f"{unrealised_pct:.1f}%")

    st.metric("Realised P/L (session)", f"${realised_pl:,.2f}")

    st.caption("Based on simulated trades in this prototype.")
    st.caption("Educational use only — not financial advice.")

    st.markdown('<div class="section-title">Portfolio alerts</div>', unsafe_allow_html=True)
    alerts = generate_portfolio_alerts(portfolio)
    if alerts:
        for msg in alerts:
            st.warning(msg)
    else:
        st.success("No obvious concentration alerts based on your current holdings.")


def render_watchlist():
    st.markdown('<div class="section-title">My watchlist</div>', unsafe_allow_html=True)

    watchlist_rows = []
    for wt in WATCHLIST_TICKERS:
        try:
            price, change_pct = get_latest_price_and_change(wt)
            watchlist_rows.append(
                {
                    "Ticker": wt,
                    "Name": COMPANY_NAMES.get(wt, ""),
                    "Last price": price,
                    "Daily change (%)": change_pct,
                }
            )
        except Exception:
            continue

    if watchlist_rows:
        for r in watchlist_rows:
            change_pct = r["Daily change (%)"]
            change_color = "#16a34a" if (change_pct or 0) >= 0 else "#dc2626"
            arrow = "▲" if (change_pct or 0) >= 0 else "▼"

            st.markdown(
                f'''
                <div class="card" style="padding:12px 14px;">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                        <div>
                            <div style="font-weight:700;">{r["Ticker"]}</div>
                            <div class="mini-label">{r["Name"]}</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-weight:700;">${r["Last price"]:.2f}</div>
                            <div style="color:{change_color};font-size:0.85rem;">{arrow} {change_pct:.2f}%</div>
                        </div>
                    </div>
                </div>
                ''',
                unsafe_allow_html=True,
            )


def render_holdings_table(portfolio):
    st.markdown('<div class="section-title">Advisor suggestions for your holdings</div>', unsafe_allow_html=True)
    if portfolio:
        rows = []
        for pos in portfolio:
            rows.append(
                {
                    "Ticker": pos.ticker,
                    "Shares": pos.shares,
                    "Avg price ($)": f"{pos.avg_price:.2f}",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No advisor suggestions yet — your portfolio is empty.")