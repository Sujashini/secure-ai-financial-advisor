import numpy as np
import pandas as pd

from backend.data.market_data import fetch_stock_data
from backend.data.features import add_technical_indicators
from backend.RL.trading_env import TradingEnv
from backend.RL.dqn_agent import DQNAgent


def backtest_ticker(
    ticker: str,
    model_path: str,
    initial_cash: float = 100_000.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Run a simple backtest for a single ticker:
    - AI policy (DQN agent in TradingEnv)
    - Buy-and-hold baseline (fully invested at start)

    Returns:
        equity_df: DataFrame with columns:
            ['date', 'price', 'equity_ai', 'equity_bh']
        metrics: dict with simple summary stats.
    """
    # 1. Load data + indicators
    data = fetch_stock_data(ticker)
    data = add_technical_indicators(data)

    # 2. Set up environment and agent
    env = TradingEnv(data, initial_cash=initial_cash)
    state, _ = env.reset()

    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    )
    agent.load(model_path)
    agent.epsilon = 0.0  # greedy policy for backtest

    # 3. Buy-and-hold baseline: fully invested at first price
    first_price = data.loc[0, "close"]
    shares_bh = initial_cash / first_price

    dates = []
    prices = []
    equity_ai = []
    equity_bh = []

    done = False
    while not done:
        # Current price & date
        idx = env.current_step
        row = data.loc[idx]
        price = row["close"]
        date = row["date"]

        # AI portfolio value (based on env state)
        # We approximate by using current price + env cash/shares
        portfolio_value_ai = env._get_portfolio_value(price)

        # Buy-and-hold portfolio value
        portfolio_value_bh = shares_bh * price

        dates.append(date)
        prices.append(price)
        equity_ai.append(portfolio_value_ai)
        equity_bh.append(portfolio_value_bh)

        # AI action
        action = agent.select_action(state)
        next_state, reward, done, _, info = env.step(action)
        state = next_state

    # Build equity curve DataFrame
    equity_df = pd.DataFrame(
        {
            "date": dates,
            "price": prices,
            "equity_ai": equity_ai,
            "equity_bh": equity_bh,
        }
    )

    # 4. Compute simple metrics
    final_ai = equity_df["equity_ai"].iloc[-1]
    final_bh = equity_df["equity_bh"].iloc[-1]

    ret_ai = (final_ai / initial_cash) - 1.0
    ret_bh = (final_bh / initial_cash) - 1.0

    # Max drawdown (approximate)
    def max_drawdown(series: pd.Series) -> float:
        cum_max = series.cummax()
        drawdown = (series - cum_max) / cum_max
        return drawdown.min()

    mdd_ai = max_drawdown(equity_df["equity_ai"])
    mdd_bh = max_drawdown(equity_df["equity_bh"])

    metrics = {
        "initial_cash": initial_cash,
        "final_ai": final_ai,
        "final_bh": final_bh,
        "return_ai": ret_ai,
        "return_bh": ret_bh,
        "max_drawdown_ai": mdd_ai,
        "max_drawdown_bh": mdd_bh,
    }

    return equity_df, metrics


if __name__ == "__main__":
    # Quick manual test
    df, m = backtest_ticker("AAPL", "models/dqn_AAPL.pth")
    print(df.tail())
    print(m)
