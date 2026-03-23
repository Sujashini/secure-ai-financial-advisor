# Secure Explainable AI Financial Advisor Bot

## Overview
The **Secure Explainable AI Financial Advisor Bot** is an financial decision-support system that combines **reinforcement learning**, **explainable AI**, and a **secure user-facing interface** to generate and explain stock trading recommendations.

The system analyses historical market data, applies technical indicators, and uses a **Deep Q-Network (DQN)** trading agent to produce **BUY**, **SELL**, or **HOLD** actions. To improve transparency, the model’s behaviour is interpreted using a **surrogate explainability pipeline** based on **Random Forest** and **SHAP**. A constrained chat-based explanation module is also included to present recommendations in plain English for non-technical users.

This project was developed as an **educational final year project prototype** and is not intended for live trading.

---

## Key Features
- **Reinforcement Learning Trading Agent**  
  Uses a DQN-based agent to learn trading actions from historical stock market data.

- **Technical Indicator Engineering**  
  Generates features such as returns, moving averages, RSI, and volatility to represent market conditions.

- **Explainable AI Module**  
  Uses a surrogate model and SHAP values to identify the main factors influencing each recommendation.

- **Secure Explanation Interface**  
  Provides explanation outputs through a constrained and educational language interface.

- **Portfolio Simulation**  
  Supports simulated buy/sell actions, portfolio tracking, account summaries, and simple risk alerts.

- **Interactive Web Interface**  
  Includes dashboard, explanation, chat, portfolio, help, and trader profile pages implemented using Streamlit.

- **Historical Strategy Comparison**  
  Compares the RL strategy against baseline approaches such as Buy-and-Hold and RSI-based trading.

---

## System Architecture
The system is organised into the following main components:

- **Data Layer**  
  Market data retrieval and feature engineering

- **RL Decision Engine**  
  Trading environment and DQN agent training

- **Explainability Module**  
  Surrogate model construction and SHAP-based explanations

- **User Management Layer**  
  Authentication, account handling, and portfolio storage

- **Frontend Interface**  
  Streamlit-based dashboard and educational explanation views

---

## Technologies Used
- **Python**
- **Streamlit**
- **PyTorch**
- **SHAP**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **SQLAlchemy**
- **SQLite**
- **yFinance**
- **Altair**

---

## Project Structure
```text
backend/
├── data/              # Market data collection and feature engineering
├── RL/                # Trading environment and DQN agent
├── XAI/               # Surrogate explainer and SHAP-based explanation logic
├── Evaluation/        # Backtesting and strategy comparison
├── LLM/               # Chat storage and constrained language explanation logic
└── users/             # User models, authentication, and portfolio services

frontend/
├── components/        # Dashboard and UI section renderers
├── pages/             # Page-level views such as dashboard, explanation, portfolio
└── utils/             # Chart builders, helpers, constants, and session state