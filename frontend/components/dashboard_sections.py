from datetime import datetime

import streamlit as st

from backend.users.service import buy_shares, sell_shares
from frontend.utils.portfolio_helpers import (
    get_latest_price_and_change,
    compute_portfolio_unrealised,
    compute_realised_pl,
    generate_portfolio_alerts,
)
from frontend.utils.constants import WATCHLIST_TICKERS, COMPANY_NAMES

# Friendly display names used to make technical features easier for
# non-technical users to understand in the interface
FRIENDLY_FEATURE_NAMES = {
    "return_1": "very recent price movement",
    "sma_10": "short-term price trend",
    "sma_20": "medium-term price trend",
    "ema_10": "short-term trend (EMA)",
    "ema_20": "smoothed medium-term trend",
    "volatility_10": "recent price volatility",
    "rsi_14": "momentum",
    "open": "opening price behaviour",
    "high": "recent high price",
    "low": "recent low price",
    "close": "closing price behaviour",
    "volume": "trading volume",
    "position_flag": "current position status",
}


def friendly_feature_name(feature: str) -> str:
    return FRIENDLY_FEATURE_NAMES.get(feature, feature.replace("_", " ").lower())


def get_confidence_pill_class(conf_label: str) -> str:
    if str(conf_label).lower() == "high":
        return "pill pill-green"
    if str(conf_label).lower() == "medium":
        return "pill pill-amber"
    return "pill pill-red"


def get_risk_pill_class(risk_label: str) -> str:
    if str(risk_label).lower() == "low":
        return "pill pill-green"
    if str(risk_label).lower() == "medium":
        return "pill pill-amber"
    return "pill pill-red"


def render_hero_section(
    ticker,
    action_text,
    conf_label,
    conf_pct,
    conf_subtitle,
    risk_label,
    risk_text,
    explanation,
    factor_summary=None,
    current_price=None,
    price_change_pct=None,
):
    """
    Render the main dashboard hero card showing:
    - selected ticker,
    - AI recommendation,
    - confidence and risk badges,
    - current market price,
    - short explanation summary.
    """
    pos = explanation.get("top_positive", [])
    top_features = [friendly_feature_name(item["feature"]) for item in pos[:3]]

    if factor_summary is None:
        factor_summary = ", ".join(top_features) if top_features else "mixed signals"

    # -------------------------
    # Build price display block
    # -------------------------
    price_html = ""
    if current_price is not None:
        change_html = ""
        if price_change_pct is not None:
            change_color = "#16a34a" if price_change_pct >= 0 else "#dc2626"
            arrow = "▲" if price_change_pct >= 0 else "▼"
            change_html = (
                f'<span style="color:{change_color};font-weight:600;margin-left:0.5rem;">'
                f'{arrow} {price_change_pct:.2f}%</span>'
            )

        price_html = (
            f'<div style="margin-top:0.5rem;color:#e2e8f0;font-size:0.98rem;">'
            f'Current price: <b>${current_price:.2f}</b> {change_html}'
            f"</div>"
        )

    hero_html = f"""<div class="hero-card">
<div class="mini-label">AI recommendation for selected stock</div>
<div class="big-value">{action_text} ({ticker})</div>

<div style="margin-top:0.55rem;">
    <span class="{get_confidence_pill_class(conf_label)}">Confidence: {conf_pct}% ({conf_label})</span>
    <span class="{get_risk_pill_class(risk_label)}" style="margin-left:0.45rem;">Risk: {risk_label}</span>
</div>

{price_html}

<div class="soft-note" style="margin-top:0.65rem;">
    Main reasons behind this suggestion include <b>{factor_summary}</b>.<br>
    {conf_subtitle} {risk_text}
</div>

<div style="margin-top:0.55rem;color:#94a3b8;font-size:0.94rem;">
    Best next step: review the explanation tab for a fuller breakdown before acting.
</div>
</div>"""

    st.markdown(hero_html, unsafe_allow_html=True)


def render_trade_panel(user, ticker, action_text, current_price, portfolio):
    """
    Render the simulated trading panel for the selected stock.

    Depending on the current recommendation, this section allows the user to:
    - buy shares,
    - sell shares,
    - or simply review the hold recommendation.

    Executed trades are stored in session state as part of a simulated
    portfolio experience.
    """
    st.markdown('<div class="section-title">Act on this recommendation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">You can simulate a simple trade here for the selected stock.</div>',
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("trade_history", [])


    # -------------------------
    # BUY workflow
    # -------------------------
    if action_text == "BUY":
        trade_col1, trade_col2 = st.columns([3, 1])

        with trade_col1:
            shares_to_buy = st.number_input(
                "Number of shares to buy",
                min_value=1.0,
                step=1.0,
                key="buy_shares_input",
            )
            estimated_cost = shares_to_buy * float(current_price)
            st.caption(f"Estimated cost at latest price: ${estimated_cost:,.2f}")

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
                    st.success(f"Bought {shares_to_buy:.0f} shares of {ticker}.")
                    st.rerun()
                except Exception as e:
                    st.error("Could not complete purchase.")
                    st.caption(str(e))

        st.info(
            "💡 The AI currently prefers buying. This simulated trade uses the latest available market price."
        )

    # -------------------------
    # SELL workflow
    # -------------------------
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
                estimated_value = shares_to_sell * float(current_price)
                st.caption(f"Estimated sale value at latest price: ${estimated_value:,.2f}")
                st.caption(f"You currently own {max_shares:.0f} shares of {ticker}.")

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
                        st.success(f"Sold {shares_to_sell:.0f} shares of {ticker}.")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error("Could not complete sale.")
                        st.caption(str(e))
            # -------------------------
            # HOLD workflow
            # -------------------------
            st.info(
                "💡 The AI currently prefers selling. This can help reduce exposure when caution signals are stronger."
            )
        else:
            st.info("You do not currently own this stock, so there is nothing available to sell.")

    else:
        st.info(
            "The AI currently prefers to hold. You do not need to take action right now, "
            "but you can still review the explanation tab to understand why."
        )


def render_portfolio_snapshot(user, portfolio):
    """
    Render a visual snapshot of the user's current portfolio holdings,
    including live price and daily percentage movement where available.
    """
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
                f"""<div class="card">
<div style="font-weight:700;font-size:1rem;">{t}</div>
<div class="mini-label">Shares: {pos.shares} · Avg: ${pos.avg_price:.2f}</div>
<div style="font-size:1.8rem;font-weight:750;margin-top:0.35rem;">{price_str}</div>
<div style="margin-top:0.2rem;color:{change_color};font-weight:600;">{change_str}</div>
</div>""",
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
    """
    Render account-level summary metrics for the user's portfolio,
    including:
    - total portfolio value,
    - unrealised profit/loss,
    - realised profit/loss,
    - basic portfolio alerts.
    """
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


def render_watchlist(selected_ticker=None):
    """
    Render the user's watchlist using a set of predefined stock tickers,
    together with latest price and daily movement information.
    """
    st.markdown('<div class="section-title">My watchlist</div>', unsafe_allow_html=True)
    st.caption("Daily move for selected stocks.")

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

    if not watchlist_rows:
        st.info("Watchlist prices are not available right now.")
        return

    for r in watchlist_rows:
        change_pct = r["Daily change (%)"] if r["Daily change (%)"] is not None else 0.0
        change_color = "#16a34a" if change_pct >= 0 else "#dc2626"
        arrow = "▲" if change_pct >= 0 else "▼"

        border_style = (
            "2px solid rgba(99,102,241,0.55)"
            if selected_ticker and r["Ticker"] == selected_ticker
            else "1px solid rgba(255,255,255,0.06)"
        )

        selected_badge = ""
        if selected_ticker and r["Ticker"] == selected_ticker:
            selected_badge = '<div style="font-size:0.76rem;color:#a5b4fc;font-weight:600;margin-top:0.15rem;">Currently selected</div>'

        watch_html = (
            f'<div class="card" style="padding:12px 14px;border:{border_style};">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
            f'<div>'
            f'<div style="font-weight:700;">{r["Ticker"]}</div>'
            f'<div class="mini-label">{r["Name"]}</div>'
            f'{selected_badge}'
            f'</div>'
            f'<div style="text-align:right;">'
            f'<div style="font-weight:700;">${r["Last price"]:.2f}</div>'
            f'<div style="color:{change_color};font-size:0.85rem;">{arrow} {change_pct:.2f}%</div>'
            f'</div>'
            f'</div>'
            f'</div>'
        )

        st.markdown(watch_html, unsafe_allow_html=True)
