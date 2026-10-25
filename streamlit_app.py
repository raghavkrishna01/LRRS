"""
LRRS — Loan Risk Reduction System
Streamlit web application entry point.

Run locally:   streamlit run streamlit_app.py
Run in Docker: docker compose up --build
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from app.config import APP_TITLE, APP_ICON
from app.ui.theme import inject_css, render_theme_toggle

# ── Theme initialisation (must happen before set_page_config CSS injection) ──
if "_theme_toggle" not in st.session_state:
    st.session_state["_theme_toggle"] = False

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject CSS immediately so the very first render is themed
inject_css()

# ── Page registry ─────────────────────────────────────────────────────────────
from app.pages.pg_dashboard import render as dashboard
from app.pages.pg_risk_analyzer import render as risk_analyzer
from app.pages.pg_optimizer import render as optimizer
from app.pages.pg_customer_tracker import render as customer_tracker
from app.pages.pg_early_warning import render as early_warning
from app.pages.pg_interventions import render as interventions
from app.pages.pg_methodology import render as methodology

# ── Navigation ────────────────────────────────────────────────────────────────
nav = st.navigation(
    {
        "Portfolio": [
            st.Page(dashboard,         title="Dashboard",              url_path="dashboard"),
        ],
        "Loan Assessment": [
            st.Page(risk_analyzer,     title="Risk Analyzer",          url_path="risk-analyzer"),
            st.Page(optimizer,         title="Loan Optimizer",         url_path="loan-optimizer"),
        ],
        "Customer Monitoring": [
            st.Page(customer_tracker,  title="Customer Tracker",       url_path="customer-tracker"),
            st.Page(early_warning,     title="Early Warning",          url_path="early-warning"),
            st.Page(interventions,     title="Interventions",          url_path="interventions"),
        ],
        "Documentation": [
            st.Page(methodology,       title="Methodology & SHAP",     url_path="methodology"),
        ],
    },
    position="sidebar",
)

# ── Sidebar brand card + theme toggle ─────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="'
        'background:#0F4C75;'
        'border-radius:6px;'
        'padding:0.7rem 1rem;'
        'margin:0 0 0.5rem;'
        '">'
        '<div style="font-size:0.95rem;font-weight:800;color:#FFFFFF;letter-spacing:-0.01em;">'
        f'{APP_ICON}&nbsp; LRRS'
        '</div>'
        '<div style="font-size:10px;color:rgba(255,255,255,0.60);font-weight:500;'
        'text-transform:uppercase;letter-spacing:0.09em;margin-top:2px;">'
        'Loan Risk Reduction System'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<hr style="border:none;border-top:1px solid var(--border);margin:0.3rem 0 0.5rem;">',
        unsafe_allow_html=True,
    )
    render_theme_toggle()
    st.markdown(
        '<hr style="border:none;border-top:1px solid var(--border);margin:0.4rem 0 0.3rem;">',
        unsafe_allow_html=True,
    )

# ── Run selected page ─────────────────────────────────────────────────────────
nav.run()
