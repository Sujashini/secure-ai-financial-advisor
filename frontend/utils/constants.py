import os

WATCHLIST_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]

COMPANY_NAMES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "NVDA": "NVIDIA Corp.",
    "TSLA": "Tesla Inc.",
    "GOOGL": "Alphabet Inc.",
}

TECHY_TICKERS = {"AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"}

REMEMBER_ME_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "remember_me.json")

ACTION_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}

AVAILABLE_TICKERS = ["AAPL", "MSFT", "NVDA"]