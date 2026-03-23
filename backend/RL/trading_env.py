import gymnasium as gym
import numpy as np


class TradingEnv(gym.Env):
    """
    A simplified trading environment for a single asset.
    Actions:
        0 = HOLD
        1 = BUY
        2 = SELL
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data,
        initial_cash: float = 100_000,
    ):
        """
        Initialise the trading environment.

        Parameters:
            data: Preprocessed stock market data including technical indicators.
            initial_cash (float): Starting cash balance for the trading episode.
        """
        super().__init__()

        self.data = data.reset_index(drop=True)
        self.initial_cash = initial_cash

        self.current_step = 0
        self.cash = None
        self.shares_held = None
        self.entry_price = None

        # Actions: HOLD, BUY, SELL
        self.action_space = gym.spaces.Discrete(3)

        # Observations: feature vector + position flag
        self.num_features = data.shape[1] - 1  # exclude date column
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_features + 1,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state for a new episode.

        Returns:
            tuple:
                observation (np.ndarray): Initial observation vector.
                info (dict): Additional environment information.
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.cash = self.initial_cash
        self.shares_held = 0
        self.entry_price = 0.0

        return self._get_observation(), {}

    def _get_observation(self):
        """
        Construct the current observation vector.

        The observation consists of:
        - all feature values for the current timestep except the date,
        - a position flag showing whether the agent currently holds a stock.

        Returns:
            np.ndarray: Observation vector in float32 format.
        """
        features = self.data.iloc[self.current_step].drop("date").values
        position_flag = 1 if self.shares_held > 0 else 0

        obs = np.append(features, position_flag).astype(np.float32)
        return obs

    def _get_portfolio_value(self, price):
        """
        Compute total portfolio value at a given price.

        Parameters:
            price: Current stock price.

        Returns:
            float: Cash balance plus market value of held shares.
        """
        return self.cash + self.shares_held * price

    def step(self, action):
        """
        Execute one environment step based on the selected action.

        Parameters:
            action (int): Action chosen by the agent.

        Returns:
            tuple:
                obs (np.ndarray): Next observation.
                reward (float): Reward after taking the action.
                done (bool): Whether the episode has ended.
                truncated (bool): Whether the episode was truncated early.
                info (dict): Additional portfolio information.
        """
        done = False
        reward = 0.0

        price = self.data.loc[self.current_step, "close"]

        # Execute action
        if action == 1:  # BUY
            if self.cash >= price and self.shares_held == 0:
                self.shares_held = 1
                self.cash -= price
                self.entry_price = price

        elif action == 2:  # SELL
            if self.shares_held > 0:
                self.cash += price
                self.shares_held = 0
                self.entry_price = 0.0

        # Advance timestep
        self.current_step += 1

        if self.current_step >= len(self.data) - 1:
            done = True

        # Reward: change in portfolio value
        next_price = self.data.loc[self.current_step, "close"]
        portfolio_value = self._get_portfolio_value(next_price)
        reward = portfolio_value - self.initial_cash

        obs = self._get_observation()
        info = {
            "cash": self.cash,
            "shares_held": self.shares_held,
            "portfolio_value": portfolio_value,
        }

        return obs, reward, done, False, info

    def render(self):
        price = self.data.loc[self.current_step, "close"]
        print(
            f"Step: {self.current_step}, "
            f"Price: {price:.2f}, "
            f"Cash: {self.cash:.2f}, "
            f"Shares: {self.shares_held}"
        )
