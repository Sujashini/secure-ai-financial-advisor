import streamlit as st


def render_help_page():
    st.subheader("❓ Help / Glossary")

    st.markdown(
        """
**What do BUY / SELL / HOLD mean here?**

- **BUY** – The system has found patterns similar to past situations where the price often went up.  
- **SELL** – The system has detected patterns similar to past situations where the price often fell or became unstable.  
- **HOLD** – Signals are mixed or weak.

These are educational signals only and **not** financial advice.

---

**Indicators used**

- **SMA / EMA** – Short- and medium-term price trends  
- **RSI** – Momentum indicator  
- **Volatility** – How much prices fluctuate

---

**Testing**

- **AI strategy** – Simulated past performance of the AI  
- **Buy & hold** – Buying once and holding

Everything shown in this app is based on historical data and is intended only for learning.
"""
    )