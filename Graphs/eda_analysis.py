import matplotlib.pyplot as plt

from backend.data.market_data import fetch_stock_data
from backend.data.features import add_technical_indicators

# -----------------------------
# 1. Load data
# -----------------------------
ticker = "AAPL"
df = fetch_stock_data(ticker)
df = add_technical_indicators(df)

# -----------------------------
# 2. Price trend
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(df["date"], df["close"])
plt.title("Stock Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig("figure_price_trend.png")
plt.show()

# -----------------------------
# 3. Return distribution
# -----------------------------
plt.figure(figsize=(8,5))
plt.hist(df["return_1"], bins=40)
plt.title("Distribution of Daily Returns")
plt.xlabel("Return")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("figure_returns.png")
plt.show()

# -----------------------------
# 4. Moving averages
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(df["date"], df["close"], label="Close")
plt.plot(df["date"], df["sma_10"], label="SMA 10")
plt.plot(df["date"], df["ema_10"], label="EMA 10")
plt.legend()
plt.title("Price with Moving Averages")
plt.tight_layout()
plt.savefig("figure_indicators.png")
plt.show()

# -----------------------------
# 5. RSI (optional)
# -----------------------------
plt.figure(figsize=(10,4))
plt.plot(df["date"], df["rsi_14"])
plt.axhline(70, linestyle="--")
plt.axhline(30, linestyle="--")
plt.title("RSI Indicator (14)")
plt.tight_layout()
plt.savefig("figure_rsi.png")
plt.show()