import os

from backend.data.market_data import fetch_stock_data
from backend.data.features import add_technical_indicators
from backend.RL.trading_env import TradingEnv
from backend.RL.dqn_agent import DQNAgent


def train():
    data = fetch_stock_data("AAPL")
    data = add_technical_indicators(data)

    env = TradingEnv(data)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    episodes = 10  # keep small for now

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

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
            f"Episode {episode + 1}/{episodes} | "
            f"Total Reward: {total_reward:.2f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )
    # Save trained model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "dqn_aapl.pth")
    agent.save(model_path)
    print(f"Saved trained DQN model to {model_path}")



if __name__ == "__main__":
    train()
