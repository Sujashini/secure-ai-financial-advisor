import os

from backend.data.market_data import fetch_stock_data
from backend.data.features import add_technical_indicators
from backend.RL.trading_env import TradingEnv
from backend.RL.dqn_agent import DQNAgent


TICKERS = ["AAPL", "MSFT", "NVDA"]  # you can add more later


def train_for_ticker(ticker: str, episodes: int = 10):
    print(f"\n=== Training DQN for {ticker} ===")

    data = fetch_stock_data(ticker)
    data = add_technical_indicators(data)

    env = TradingEnv(data)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)

            agent.store_transition(
                state, action, reward, next_state, done
            )

            loss = agent.train_step()
            state = next_state
            total_reward += reward

        print(
            f"[{ticker}] Episode {episode + 1}/{episodes} | "
            f"Total Reward: {total_reward:.2f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

    # Save trained model for this ticker
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"dqn_{ticker}.pth")
    agent.save(model_path)
    print(f"[{ticker}] Saved trained DQN model to {model_path}")


def main():
    for ticker in TICKERS:
        train_for_ticker(ticker, episodes=10)


if __name__ == "__main__":
    main()
