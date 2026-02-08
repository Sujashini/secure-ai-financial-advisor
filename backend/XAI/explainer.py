import os
from typing import Tuple, Dict

import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier

from backend.data.market_data import fetch_stock_data
from backend.data.features import add_technical_indicators
from backend.RL.trading_env import TradingEnv
from backend.RL.dqn_agent import DQNAgent


class SurrogateExplainer:
    def __init__(
        self,
        surrogate_model: RandomForestClassifier,
        explainer: shap.TreeExplainer,
        feature_names,
    ):
        self.surrogate_model = surrogate_model
        self.explainer = explainer
        self.feature_names = feature_names

    @classmethod
    def build_from_trained_agent(
        cls,
        model_path: str,
        ticker: str = "AAPL",
        episodes: int = 5,
    ) -> "SurrogateExplainer":
        """
        Build a surrogate explainer by:
        - loading data
        - creating env
        - loading trained DQN
        - collecting (state, action) samples
        - training a RandomForest surrogate
        - creating a SHAP TreeExplainer
        """
        # 1. Load data and indicators
        data = fetch_stock_data(ticker)
        data = add_technical_indicators(data)

        env = TradingEnv(data)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # 2. Load trained DQN agent
        agent = DQNAgent(state_dim, action_dim)
        agent.load(model_path)
        # Use greedy policy for data collection
        agent.epsilon = 0.0

        # 3. Collect state-action samples
        states, actions = collect_policy_data(env, agent, episodes=episodes)

        # 4. Train surrogate RandomForest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
        )
        rf.fit(states, actions)

        # 5. Create SHAP TreeExplainer
        # Use a small background sample for efficiency
        background = shap.sample(states, 100) if states.shape[0] > 100 else states
        explainer = shap.TreeExplainer(rf, data=background)

        # Feature names: all columns except "date" plus "position_flag"
        feature_names = list(data.drop(columns=["date"]).columns) + ["position_flag"]

        return cls(rf, explainer, feature_names)

    def explain_state(self, state: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Given a single state vector, return:
        - SHAP values for each feature
        - A structured summary (e.g., top positive/negative features)
        """
        state_2d = state.reshape(1, -1)

        shap_values = self.explainer.shap_values(state_2d)

        # Predicted action from surrogate
        predicted_action = self.surrogate_model.predict(state_2d)[0]

        # Handle different SHAP return formats:
        # - List (one array per class)
        # - Single numpy array (no per-class split)
        if isinstance(shap_values, list):
            # Multi-class case: pick the SHAP values for the predicted class
            class_shap = shap_values[predicted_action][0]
        else:
            # Single-output case: shap_values is (1, n_features)
            class_shap = shap_values[0]

        # Build a simple summary
        # Ensure SHAP values are scalar floats (robust to different SHAP outputs)
        clean_shap = []
        for val in class_shap:
            if isinstance(val, (list, tuple, np.ndarray)):
                clean_shap.append(float(np.array(val).flatten()[0]))
            else:
                clean_shap.append(float(val))

        feature_contrib = list(zip(self.feature_names, clean_shap))
        feature_contrib_sorted = sorted(
            feature_contrib, key=lambda x: x[1], reverse=True
        )

        # Primary behaviour: filter by sign
        top_positive = [
            {"feature": name, "value": float(val)}
            for name, val in feature_contrib_sorted[:3]
            if val > 0
        ]
        top_negative = [
            {"feature": name, "value": float(val)}
            for name, val in feature_contrib_sorted[-3:]
            if val < 0
        ]

        # Fallback: if everything is ~0 or filtered out, just take top abs contributors
        if not top_positive and not top_negative:
            # Sort by absolute value
            feature_contrib_abs_sorted = sorted(
                feature_contrib, key=lambda x: abs(x[1]), reverse=True
            )
            top_any = feature_contrib_abs_sorted[:3]
            top_positive = [
                {"feature": name, "value": float(val)}
                for name, val in top_any
            ]
            top_negative = []  # none strictly negative in this case
         # 👉 Re-add the summary dict here
        summary = {
            "predicted_action": int(predicted_action),
            "top_positive": top_positive,
            "top_negative": top_negative,
        }    

        return np.array(class_shap), summary



def collect_policy_data(env: TradingEnv, agent: DQNAgent, episodes: int = 5):
    """
    Run the trained agent in the environment and collect (state, action) pairs.
    """
    all_states = []
    all_actions = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)

            all_states.append(state)
            all_actions.append(action)

            state = next_state

    return np.array(all_states), np.array(all_actions)


if __name__ == "__main__":
    # Quick manual test of the explainer
    model_path = os.path.join("models", "dqn_aapl.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained DQN model not found at {model_path}. "
            "Run backend.rl.train_dqn first."
        )

    explainer = SurrogateExplainer.build_from_trained_agent(
        model_path=model_path,
        ticker="AAPL",
        episodes=5,
    )

    # Get a sample state from fresh data
    data = fetch_stock_data("AAPL")
    data = add_technical_indicators(data)
    env = TradingEnv(data)
    state, _ = env.reset()

    shap_vals, summary = explainer.explain_state(state)

    print("Predicted action (0=HOLD,1=BUY,2=SELL):", summary["predicted_action"])
    print("Top positive contributors:")
    for item in summary["top_positive"]:
        print(f"  {item['feature']}: {item['value']:.4f}")

    print("Top negative contributors:")
    for item in summary["top_negative"]:
        print(f"  {item['feature']}: {item['value']:.4f}")
