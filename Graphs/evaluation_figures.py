import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backend.data.market_data import fetch_stock_data
from backend.data.features import add_technical_indicators
from backend.RL.trading_env import TradingEnv
from backend.RL.dqn_agent import DQNAgent


# =========================================================
# CONFIG
# =========================================================
TICKER = "AAPL"
START_DATE = "2018-01-01"
TRAIN_END_DATE = "2023-12-31"
TEST_START_DATE = "2024-01-01"
END_DATE = None

MODEL_PATH = os.path.join("models", "dqn_aapl.pth")
OUTPUT_DIR = "chapter5_figures"

INITIAL_CASH = 10000.0


# =========================================================
# HELPERS
# =========================================================
def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_figure(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def compute_max_drawdown(series):
    series = np.array(series, dtype=float)
    peaks = np.maximum.accumulate(series)
    drawdowns = (series - peaks) / peaks
    return drawdowns.min()


def compute_sharpe_ratio(portfolio_values):
    values = pd.Series(portfolio_values).astype(float)
    returns = values.pct_change().dropna()
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    # Simple annualised Sharpe for daily data
    return (returns.mean() / returns.std()) * np.sqrt(252)


def evaluate_buy_and_hold(test_df, initial_cash=INITIAL_CASH):
    prices = test_df["close"].values
    if len(prices) == 0:
        return [], []

    shares = initial_cash / prices[0]
    portfolio_values = shares * prices
    actions = ["BUY"] + ["HOLD"] * (len(prices) - 1)
    return portfolio_values.tolist(), actions


def evaluate_rsi_strategy(test_df, initial_cash=INITIAL_CASH, buy_thresh=30, sell_thresh=70):
    cash = initial_cash
    shares = 0.0
    portfolio_values = []
    actions = []

    for _, row in test_df.iterrows():
        price = float(row["close"])
        rsi = float(row["rsi_14"])

        action = "HOLD"

        if rsi < buy_thresh and shares == 0:
            shares = cash / price
            cash = 0.0
            action = "BUY"
        elif rsi > sell_thresh and shares > 0:
            cash = shares * price
            shares = 0.0
            action = "SELL"

        portfolio_value = cash + shares * price
        portfolio_values.append(portfolio_value)
        actions.append(action)

    return portfolio_values, actions


def load_trained_agent(env, model_path):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    # Use the agent's own load method
    agent.load(model_path)

    # Disable exploration for evaluation
    agent.epsilon = 0.0

    # Put networks into eval mode
    agent.policy_net.eval()
    agent.target_net.eval()

    return agent

def action_to_label(action_int):
    mapping = {0: "HOLD", 1: "BUY", 2: "SELL"}
    return mapping.get(int(action_int), str(action_int))


def evaluate_rl_strategy(test_df, model_path):
    env = TradingEnv(test_df)
    agent = load_trained_agent(env, model_path)

    state, _ = env.reset()
    done = False

    portfolio_values = []
    actions = []

    while not done:
        # Inference only
        action = agent.select_action(state)

        next_state, reward, done, _, info = env.step(action)

        action_label = action_to_label(action)
        actions.append(action_label)

        # Try to read portfolio value safely
        if hasattr(env, "portfolio_value"):
            portfolio_values.append(float(env.portfolio_value))
        elif isinstance(info, dict) and "portfolio_value" in info:
            portfolio_values.append(float(info["portfolio_value"]))
        else:
            raise KeyError("Could not find portfolio value from env or info.")

        state = next_state

    return portfolio_values, actions


def summarise_strategy(name, portfolio_values, actions):
    start_val = float(portfolio_values[0]) if portfolio_values else INITIAL_CASH
    end_val = float(portfolio_values[-1]) if portfolio_values else INITIAL_CASH
    total_return_pct = ((end_val - start_val) / start_val) * 100 if start_val != 0 else 0.0
    max_dd_pct = compute_max_drawdown(portfolio_values) * 100 if portfolio_values else 0.0
    sharpe = compute_sharpe_ratio(portfolio_values) if portfolio_values else 0.0

    action_counts = pd.Series(actions).value_counts()
    buy_count = int(action_counts.get("BUY", 0))
    sell_count = int(action_counts.get("SELL", 0))
    hold_count = int(action_counts.get("HOLD", 0))

    return {
        "Strategy": name,
        "Start Value": round(start_val, 2),
        "End Value": round(end_val, 2),
        "Total Return (%)": round(total_return_pct, 2),
        "Max Drawdown (%)": round(max_dd_pct, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "BUY Count": buy_count,
        "SELL Count": sell_count,
        "HOLD Count": hold_count,
    }


# =========================================================
# MAIN GENERATION
# =========================================================
def main():
    ensure_output_dir()

    # 1. Load and process data
    df = fetch_stock_data(ticker=TICKER, start=START_DATE, end=END_DATE)
    df = add_technical_indicators(df)

    # 2. Test split
    test_df = df[df["date"] >= TEST_START_DATE].copy()
    if test_df.empty:
        raise ValueError("Test set is empty. Check TEST_START_DATE and data availability.")

    # 3. Evaluate strategies
    rl_values, rl_actions = evaluate_rl_strategy(test_df, MODEL_PATH)
    bh_values, bh_actions = evaluate_buy_and_hold(test_df)
    rsi_values, rsi_actions = evaluate_rsi_strategy(test_df)

    min_len = min(len(test_df), len(rl_values), len(bh_values), len(rsi_values))
    plot_df = test_df.iloc[:min_len].copy()

    rl_values = rl_values[:min_len]
    bh_values = bh_values[:min_len]
    rsi_values = rsi_values[:min_len]
    rl_actions = rl_actions[:min_len]

    # =====================================================
    # Figure 5.1: Portfolio value comparison
    # =====================================================
    plt.figure(figsize=(10, 5))
    plt.plot(plot_df["date"], rl_values, label="RL strategy")
    plt.plot(plot_df["date"], bh_values, label="Buy-and-Hold")
    plt.plot(plot_df["date"], rsi_values, label="RSI strategy")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value ($)")
    plt.title(f"Portfolio value comparison for {TICKER}")
    plt.legend()
    plt.xticks(rotation=45)
    save_figure("Figure_5_1_Portfolio_Value_Comparison.png")

    # =====================================================
    # Figure 5.2: RL action distribution
    # =====================================================
    action_counts = pd.Series(rl_actions).value_counts().reindex(["BUY", "SELL", "HOLD"], fill_value=0)

    plt.figure(figsize=(7, 5))
    plt.bar(action_counts.index, action_counts.values)
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.title(f"RL action distribution for {TICKER}")
    save_figure("Figure_5_2_RL_Action_Distribution.png")

    # =====================================================
    # Table 5.1: Summary metrics
    # =====================================================
    summary_rows = [
        summarise_strategy("RL strategy", rl_values, rl_actions),
        summarise_strategy("Buy-and-Hold", bh_values, bh_actions),
        summarise_strategy("RSI strategy", rsi_values, rsi_actions),
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "Table_5_1_Technical_Summary_Metrics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    print("\nDone. Technical evaluation figures and table saved to:")
    print(os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()