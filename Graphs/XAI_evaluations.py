import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backend.data.market_data import fetch_stock_data
from backend.data.features import add_technical_indicators
from backend.RL.trading_env import TradingEnv
from backend.RL.dqn_agent import DQNAgent
from backend.XAI.explainer import SurrogateExplainer


# =========================================================
# CONFIG
# =========================================================
TICKER_MODEL_MAP = {
    "AAPL": os.path.join("models", "dqn_aapl.pth"),
    "MSFT": os.path.join("models", "dqn_MSFT.pth"),
    "NVDA": os.path.join("models", "dqn_NVDA.pth"),
}

OUTPUT_DIR = "chapter5_figures"
TEST_START_DATE = "2024-01-01"
MAX_HELDOUT_STATES = 200


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


def load_processed_data(ticker: str):
    df = fetch_stock_data(ticker=ticker)
    df = add_technical_indicators(df)
    return df


def split_test_data(df: pd.DataFrame, test_start_date: str = TEST_START_DATE):
    test_df = df[df["date"] >= test_start_date].copy()
    if test_df.empty:
        raise ValueError(f"Test set is empty for test_start_date={test_start_date}")
    return test_df


def load_rl_agent_for_inference(env, model_path: str):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    agent.load(model_path)
    agent.epsilon = 0.0
    agent.policy_net.eval()
    agent.target_net.eval()
    return agent


def collect_heldout_states_and_rl_actions(ticker: str, model_path: str, max_states: int = MAX_HELDOUT_STATES):
    df = load_processed_data(ticker)
    test_df = split_test_data(df)

    env = TradingEnv(test_df)
    agent = load_rl_agent_for_inference(env, model_path)

    states = []
    rl_actions = []

    state, _ = env.reset()
    done = False

    while not done and len(states) < max_states:
        action = agent.select_action(state)

        states.append(np.array(state, dtype=float))
        rl_actions.append(int(action))

        next_state, reward, done, _, info = env.step(action)
        state = next_state

    return np.array(states), np.array(rl_actions), test_df


def get_surrogate_predictions(explainer_obj: SurrogateExplainer, states: np.ndarray):
    # Random Forest surrogate stored inside explainer object
    surrogate_model = explainer_obj.surrogate_model
    preds = surrogate_model.predict(states)
    return np.array(preds, dtype=int)


def compute_agreement_rate(y_true: np.ndarray, y_pred: np.ndarray):
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).mean() * 100.0)


def compute_global_shap_importance(explainer_obj: SurrogateExplainer, states: np.ndarray):
    shap_values = explainer_obj.explainer.shap_values(states)
    shap_values = np.array(shap_values, dtype=float)

    # Possible shapes:
    # 1) (n_classes, n_samples, n_features)
    # 2) (n_samples, n_features, n_classes)
    # 3) (n_samples, n_features)

    if shap_values.ndim == 3:
        # Find the axis that matches the number of features
        n_features = len(explainer_obj.feature_names)

        if shap_values.shape[2] == n_features:
            # shape: (n_classes, n_samples, n_features)
            abs_shap = np.abs(shap_values).mean(axis=(0, 1))
        elif shap_values.shape[1] == n_features:
            # shape: (n_samples, n_features, n_classes)
            abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        else:
            raise ValueError(
                f"Could not identify feature axis in SHAP array with shape {shap_values.shape}"
            )

    elif shap_values.ndim == 2:
        # shape: (n_samples, n_features)
        abs_shap = np.abs(shap_values).mean(axis=0)

    else:
        raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")

    feature_names = list(explainer_obj.feature_names)

    min_len = min(len(feature_names), len(abs_shap))
    feature_names = feature_names[:min_len]
    abs_shap = abs_shap[:min_len]

    return feature_names, abs_shap


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_output_dir()

    agreement_rows = []
    shap_feature_store = []

    for ticker, model_path in TICKER_MODEL_MAP.items():
        if not os.path.exists(model_path):
            print(f"Skipping {ticker}: model not found at {model_path}")
            continue

        print(f"\nProcessing {ticker}...")

        # 1. Collect held-out states and RL actions
        states, rl_actions, test_df = collect_heldout_states_and_rl_actions(
            ticker=ticker,
            model_path=model_path,
            max_states=MAX_HELDOUT_STATES,
        )

        # 2. Build surrogate explainer from trained RL agent
        explainer_obj = SurrogateExplainer.build_from_trained_agent(
            model_path=model_path,
            ticker=ticker,
            episodes=5,
        )

        # 3. Surrogate predictions on held-out states
        surrogate_preds = get_surrogate_predictions(explainer_obj, states)
        agreement = compute_agreement_rate(rl_actions, surrogate_preds)

        agreement_rows.append({
            "Ticker": ticker,
            "Evaluated States": int(len(states)),
            "Agreement Rate (%)": round(agreement, 2),
        })

        # 4. Global SHAP importance
        feature_names, mean_abs_shap = compute_global_shap_importance(explainer_obj, states)

        for fname, fval in zip(feature_names, mean_abs_shap):
            shap_feature_store.append({
                "Ticker": ticker,
                "Feature": fname,
                "Mean Absolute SHAP": float(fval),
            })

    # =====================================================
    # Table 5.2
    # =====================================================
    if agreement_rows:
        agreement_df = pd.DataFrame(agreement_rows)
        table_path = os.path.join(OUTPUT_DIR, "Table_5_2_Surrogate_RL_Agreement.csv")
        agreement_df.to_csv(table_path, index=False)
        print(f"\nSaved: {table_path}")
    else:
        print("\nNo agreement results were generated.")
        return

    # =====================================================
    # Figure 5.3
    # Aggregate SHAP across all tickers
    # =====================================================
    shap_df = pd.DataFrame(shap_feature_store)

    if shap_df.empty:
        print("No SHAP values generated; skipping Figure 5.3.")
        return

    global_shap_df = (
        shap_df.groupby("Feature", as_index=False)["Mean Absolute SHAP"]
        .mean()
        .sort_values("Mean Absolute SHAP", ascending=False)
    )

    top_n = 10
    top_df = global_shap_df.head(top_n).copy()

    plt.figure(figsize=(9, 5))
    plt.barh(top_df["Feature"].iloc[::-1], top_df["Mean Absolute SHAP"].iloc[::-1])
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title("Global SHAP feature importance summary")

    save_figure("Figure_5_3_Global_SHAP_Importance.png")

    print("\nDone. Generated:")
    print("- Table_5_2_Surrogate_RL_Agreement.csv")
    print("- Figure_5_3_Global_SHAP_Importance.png")
    print(f"\nOutput folder: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()