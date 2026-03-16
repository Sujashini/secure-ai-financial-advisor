import streamlit as st


def apply_global_styles():
    st.markdown("""
    <style>
    header[data-testid="stHeader"] {
        display: none;
    }

    div[data-testid="stToolbar"] {
        display: none;
    }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }

    .block-container {
        padding-top: 0.9rem;
        padding-bottom: 1.2rem;
        max-width: 1280px;
    }

    h1, h2, h3, h4 {
        color: white !important;
    }

    p, span, div {
        color: #cbd5f5;
    }

    .feature-card {
        background: #1e293b;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        color: white;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        color: white;
    }

    .hero-text {
        font-size: 1.05rem;
        color: #cbd5f5;
        line-height: 1.8;
    }

    .stButton > button {
        border-radius: 12px;
        height: 42px;
        font-weight: 600;
    }

    hr {
        border-color: rgba(255,255,255,0.1);
    }

    button[kind="tertiary"] {
        background: transparent !important;
        border: 1px solid transparent !important;
        color: #cbd5e1 !important;
    }

    button[kind="secondary"] {
        background: rgba(99,102,241,0.16) !important;
        border: 1px solid rgba(99,102,241,0.35) !important;
        color: #ffffff !important;
    }

    .stButton > button:hover {
        border-color: rgba(255,255,255,0.15) !important;
    }

    .app-subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-top: -0.25rem;
        margin-bottom: 0.75rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 0.3rem;
        margin-bottom: 0.25rem;
    }

    .section-caption {
        color: #6b7280;
        font-size: 0.92rem;
        margin-bottom: 0.75rem;
    }

    .hero-card {
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px 20px;
        backdrop-filter: blur(6px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.35);
        margin-bottom: 0.9rem;
    }

    .card {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 16px 18px;
        backdrop-filter: blur(4px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.35);
        margin-bottom: 0.75rem;
    }
                
    .card:hover {
        border-color: rgba(99,102,241,0.35);
        transform: translateY(-2px);
        transition: all 0.15s ease;
    }

    .mini-label {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-bottom: 0.2rem;
    }

    .big-value {
        font-size: 2rem;
        font-weight: 750;
        color: #f8fafc;
        line-height: 1.1;
    }

    .pill {
        display: inline-block;
        padding: 0.22rem 0.68rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.35rem;
        margin-right: 0.35rem;
    }

    .pill-green {
        background: #dcfce7;
        color: #15803d;
    }

    .pill-amber {
        background: #fef3c7;
        color: #b45309;
    }

    .pill-red {
        background: #fee2e2;
        color: #b91c1c;
    }

    .nav-divider {
        margin-top: 0.35rem;
        margin-bottom: 0.95rem;
        border-bottom: 1px solid #e5e7eb;
    }

    .soft-note {
        color: #6b7280;
        font-size: 0.92rem;
        margin-top: 0.45rem;
    }

    .landing-panel {
        background: linear-gradient(180deg, #0f172a 0%, #172554 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 28px 28px 24px 28px;
        box-shadow: 0 14px 35px rgba(15, 23, 42, 0.30);
        margin-bottom: 1.2rem;
    }

    .landing-brand {
        color: #f8fafc;
        font-size: 1.5rem;
        font-weight: 700;
    }

    .landing-top-note {
        color: #cbd5e1;
        font-size: 0.95rem;
        text-align: right;
        margin-top: 0.35rem;
    }

    .landing-title {
        color: #f8fafc;
        font-size: 3.3rem;
        line-height: 1.1;
        font-weight: 800;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .landing-text {
        color: #cbd5e1;
        font-size: 1.12rem;
        line-height: 1.75;
        margin-bottom: 1.1rem;
    }

    .landing-section-title {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 750;
        text-align: center;
        margin-top: 0.7rem;
        margin-bottom: 1.3rem;
    }

    .feature-card-dark {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 22px 18px;
        text-align: center;
        min-height: 220px;
    }

    .feature-icon {
        font-size: 2.1rem;
        margin-bottom: 0.6rem;
    }

    .feature-title {
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.55rem;
    }

    .feature-text {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.6;
    }

    .how-step-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 16px 18px;
        margin-bottom: 0.8rem;
    }

    .step-row {
        display: flex;
        gap: 14px;
        align-items: flex-start;
    }

    .step-badge {
        min-width: 44px;
        height: 44px;
        border-radius: 999px;
        background: #7c3aed;
        color: white;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
    }

    .step-title {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .step-text {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.55;
    }

    .landing-footer-box {
        border-top: 1px solid rgba(255,255,255,0.08);
        margin-top: 1.5rem;
        padding-top: 1.4rem;
    }

    .landing-footer-heading {
        color: #f8fafc;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .landing-footer-text {
        color: #cbd5e1;
        font-size: 0.98rem;
        line-height: 1.6;
    }
    .vega-embed {
        background: transparent;
        border-radius: 14px;
        padding: 6px;
    }
    div[data-testid="stDataFrame"] {
        background: rgba(30, 41, 59, 0.75);
        border-radius: 14px;
        padding: 4px;
    }
    div[data-testid="stButton"] > button {
        border-radius: 12px;
    }
    div[data-testid="stNumberInput"] input {
        text-align: left;
    }

    div[data-testid="stNumberInput"] {
        margin-bottom: 0.2rem;
    }
    div[data-testid="stRadio"] > div {
        gap: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)