import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from backend.data.market_data import fetch_stock_data
from backend.data.features import add_technical_indicators
from backend.RL.trading_env import TradingEnv
from backend.RL.dqn_agent import DQNAgent
from backend.XAI.explainer import SurrogateExplainer


# =========================================================
# CONFIG
# =========================================================
TICKER = "AAPL"
START_DATE = "2018-01-01"
END_DATE = None

# Folder where all generated figure images will be saved
OUTPUT_DIR = "chapter4_figures"

# Path to trained RL model
MODEL_PATH = os.path.join("models", "dqn_aapl.pth")

# Optional: paths to your existing UI screenshots
SCREENSHOTS = {
    "Figure_4_5_Chat_Interface.png": "Chat.png",
    "Figure_4_6_Dashboard.png": "Dashboard.png",
    "Figure_4_7_Explanation_Page.png": "Explanation.png",
    "Figure_4_8_Portfolio.png": "Portfolio.png",
    "Figure_4_9_Profile.png": "Profile.png",
}


# =========================================================
# HELPERS
# =========================================================
def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_current_figure(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def load_processed_data(ticker=TICKER, start=START_DATE, end=END_DATE):
    df = fetch_stock_data(ticker=ticker, start=start, end=end)
    df = add_technical_indicators(df)
    return df


# =========================================================
# FIGURE 4.2
# Example engineered feature output
# =========================================================
def generate_figure_4_2():
    df = load_processed_data()
    plot_df = df.tail(120).copy()

    plt.figure(figsize=(10, 5))
    plt.plot(plot_df["date"], plot_df["close"], label="Close price")
    plt.plot(plot_df["date"], plot_df["sma_10"], label="SMA 10")
    plt.plot(plot_df["date"], plot_df["ema_10"], label="EMA 10")
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"Figure 4.2: {TICKER} price with engineered trend indicators")
    plt.legend()

    save_current_figure("Figure_4_2_Engineered_Features.png")


# =========================================================
# FIGURE 4.3
# RL training / reward behaviour
# =========================================================
def generate_figure_4_3(num_episodes=50):
    df = load_processed_data()
    env = TradingEnv(df)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_episodes + 1), episode_rewards, label="Episode reward", alpha=0.5)

    if num_episodes >= 5:
        window = 5
        smoothed = np.convolve(
            episode_rewards,
            np.ones(window) / window,
            mode="valid"
        )
        plt.plot(
            range(window, num_episodes + 1),
            smoothed,
            label="5-episode moving average",
            linewidth=2
        )

    plt.xlabel("Training episode")
    plt.ylabel("Total episode reward")
    plt.title("Figure 4.3: RL training performance over episodes")
    plt.legend()

    save_current_figure("Figure_4_3_RL_Training_Performance.png")


# =========================================================
# FIGURE 4.4
# SHAP-style contribution visualisation from surrogate explainer
# =========================================================
def generate_figure_4_4():
    if not os.path.exists(MODEL_PATH):
        print(f"Skipping Figure 4.4 because model was not found at: {MODEL_PATH}")
        return

    explainer = SurrogateExplainer.build_from_trained_agent(
        model_path=MODEL_PATH,
        ticker=TICKER,
        episodes=5,
    )

    df = load_processed_data()
    env = TradingEnv(df)
    state, _ = env.reset()

    shap_values, summary = explainer.explain_state(state)

    # Flatten SHAP values robustly
    shap_values = np.array(shap_values).astype(float).flatten()
    feature_names = list(explainer.feature_names)

    # Make sure feature names and SHAP values have matching length
    min_len = min(len(feature_names), len(shap_values))

    if len(feature_names) != len(shap_values):
        print(
            f"Warning: feature_names has length {len(feature_names)} "
            f"but shap_values has length {len(shap_values)}. "
            f"Using first {min_len} entries."
        )

    feature_names = feature_names[:min_len]
    shap_values = shap_values[:min_len]

    # Sort by absolute contribution and keep top 8
    sort_idx = np.argsort(np.abs(shap_values))[::-1][:8]
    selected_features = [feature_names[i] for i in sort_idx]
    selected_values = shap_values[sort_idx]

    plt.figure(figsize=(10, 5))
    bars = plt.barh(selected_features[::-1], selected_values[::-1])

    # Just use alpha difference instead of manual colours
    for bar, val in zip(bars, selected_values[::-1]):
        if val >= 0:
            bar.set_alpha(0.85)
        else:
            bar.set_alpha(0.45)

    plt.xlabel("Contribution to predicted action")
    plt.ylabel("Feature")
    plt.title("Top feature contributions for the current recommendation")

    save_current_figure("Figure_4_4_SHAP_Contribution.png")


# =========================================================
# COPY UI SCREENSHOTS AS FIGURES 4.5 - 4.9
# =========================================================
def copy_ui_screenshots():
    for new_name, source_path in SCREENSHOTS.items():
        if os.path.exists(source_path):
            destination = os.path.join(OUTPUT_DIR, new_name)
            shutil.copy(source_path, destination)
            print(f"Copied: {destination}")
        else:
            print(f"Screenshot not found, skipped: {source_path}")


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_output_dir()

    print("Generating Chapter 4 figures...")
    generate_figure_4_2()
    generate_figure_4_3(num_episodes=50)
    generate_figure_4_4()
    copy_ui_screenshots()

    print("\nDone. All available figures are in:")
    print(os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()