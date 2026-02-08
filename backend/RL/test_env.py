from backend.data.market_data import fetch_stock_data
from backend.data.features import add_technical_indicators
from backend.RL.trading_env import TradingEnv



data = fetch_stock_data("AAPL")
data = add_technical_indicators(data)

env = TradingEnv(data)

obs, _ = env.reset()

print("Initial observation shape:", obs.shape)

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)

print("Final portfolio value:", info["portfolio_value"])
