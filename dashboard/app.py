import sys
import os
from pathlib import Path

# Bind project root and dashboard directory to sys.path immediately
BASE_DIR = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = Path(__file__).resolve().parent
for _p in [str(BASE_DIR), str(DASHBOARD_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
import streamlit.components.v1 as components
import io
import importlib
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tempfile
import time
import gc
import json
import urllib.request
import urllib.error

# Simulation engine import with safe fallback
try:
    import dashboard.simulation_2d as simulation_2d
    importlib.reload(simulation_2d)
    from dashboard.simulation_2d import get_simulation_html
except ImportError:
    import simulation_2d
    importlib.reload(simulation_2d)
    from simulation_2d import get_simulation_html

# Dynamic reload to ensure live updates without server restart
import database.db_manager
importlib.reload(database.db_manager)
from database.db_manager import DatabaseManager

import core.agent.signal_controller
importlib.reload(core.agent.signal_controller)
from core.agent.signal_controller import AdaptiveSignalController, SignalState

import core.detection.vision_engine
importlib.reload(core.detection.vision_engine)
from core.detection.vision_engine import VisionEngine

from config.settings import TIME_FILTERS, NUM_LANES, VIDEOS_DIR

# Page configuration
st.set_page_config(
    page_title="Smart Traffic Management System | Control Room",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== DYNAMIC MULTI-THEME ENGINE ====================
if "color_theme" not in st.session_state:
    st.session_state["color_theme"] = "Soft Sand"

THEMES = {
    "Soft Sand": {
        "name": "Soft Sand",
        "app_bg": "#F5F2EC",
        "sidebar_bg": "#EFECE6",
        "sidebar_border": "#D9D9D4",
        "nav_bg": "#E8DCCD",
        "nav_border": "#D8C2B3",
        "card_bg": "#F8F5F1",
        "card_border": "#D8C2B3",
        "white_card_bg": "#FFFFFF",
        "white_card_border": "#E8DCCD",
        "card_shadow": "0 6px 22px rgba(143, 133, 124, 0.08)",
        "text_primary": "#2D2520",
        "text_secondary": "#54473F",
        "text_muted": "#8F857C",
        "accent": "#A88672",
        "accent_text": "#FFFFFF",
        "btn_bg": "linear-gradient(135deg, #D1AF9D 0%, #A88672 100%)",
        "btn_hover": "linear-gradient(135deg, #A88672 0%, #8F6B56 100%)",
        "btn_border": "#C8B9A9",
        "badge_normal_bg": "#D9D9D4",
        "badge_normal_text": "#2D2520",
        "badge_normal_border": "#C8B9A9",
        "card1_bg": "linear-gradient(135deg, #D9D9D4 0%, #C8B9A9 100%)",
        "card1_border": "#B8A999",
        "card1_text": "#2D2520",
        "card2_bg": "linear-gradient(135deg, #F8F5F1 0%, #E8DCCD 100%)",
        "card2_border": "#D8C2B3",
        "card2_text": "#2D2520",
        "card3_bg": "linear-gradient(135deg, #D8C2B3 0%, #D1AF9D 100%)",
        "card3_border": "#C8B9A9",
        "card3_text": "#2D2520",
        "bar_fill": "linear-gradient(90deg, #D1AF9D 0%, #A88672 100%)",
        "bar_track": "rgba(143, 133, 124, 0.18)",
        "chart_bg": "#F8F5F1",
        "chart_font": "#54473F",
        "chart_grid": "rgba(168, 134, 114, 0.15)",
        "chart_line": "#A88672",
        "chart_fill": "rgba(209, 175, 157, 0.28)",
        "chart_bar_colors": [[0, "#D9D9D4"], [0.5, "#D1AF9D"], [1, "#A88672"]],
        "gauge_track": "#E8DCCD",
        "gauge_stop1": "#D1AF9D",
        "gauge_stop2": "#A88672",
        "pie_colors": ["#A88672", "#D1AF9D", "#C8B9A9", "#D9D9D4", "#E8DCCD"],
        "signal_housing": "#E8DCCD",
        "signal_border": "#D8C2B3"
    },
    "Silver & White": {
        "name": "Silver & White",
        "app_bg": "#F8F9FA",
        "sidebar_bg": "#E9ECEF",
        "sidebar_border": "#DEE2E6",
        "nav_bg": "#E2E6EA",
        "nav_border": "#CED4DA",
        "card_bg": "#FFFFFF",
        "card_border": "#D0D5DD",
        "white_card_bg": "#FFFFFF",
        "white_card_border": "#E2E8F0",
        "card_shadow": "0 6px 20px rgba(100, 116, 139, 0.08)",
        "text_primary": "#0F172A",
        "text_secondary": "#475569",
        "text_muted": "#64748B",
        "accent": "#475569",
        "accent_text": "#FFFFFF",
        "btn_bg": "linear-gradient(135deg, #64748B 0%, #334155 100%)",
        "btn_hover": "linear-gradient(135deg, #475569 0%, #1E293B 100%)",
        "btn_border": "#94A3B8",
        "badge_normal_bg": "#E2E8F0",
        "badge_normal_text": "#0F172A",
        "badge_normal_border": "#CBD5E1",
        "card1_bg": "linear-gradient(135deg, #E2E8F0 0%, #CBD5E1 100%)",
        "card1_border": "#94A3B8",
        "card1_text": "#0F172A",
        "card2_bg": "linear-gradient(135deg, #FFFFFF 0%, #E9ECEF 100%)",
        "card2_border": "#CED4DA",
        "card2_text": "#0F172A",
        "card3_bg": "linear-gradient(135deg, #CBD5E1 0%, #94A3B8 100%)",
        "card3_border": "#64748B",
        "card3_text": "#0F172A",
        "bar_fill": "linear-gradient(90deg, #94A3B8 0%, #475569 100%)",
        "bar_track": "rgba(100, 116, 139, 0.18)",
        "chart_bg": "#FFFFFF",
        "chart_font": "#475569",
        "chart_grid": "rgba(203, 213, 225, 0.6)",
        "chart_line": "#475569",
        "chart_fill": "rgba(148, 163, 184, 0.22)",
        "chart_bar_colors": [[0, "#E2E8F0"], [0.5, "#94A3B8"], [1, "#475569"]],
        "gauge_track": "#E2E8F0",
        "gauge_stop1": "#94A3B8",
        "gauge_stop2": "#475569",
        "pie_colors": ["#334155", "#475569", "#64748B", "#94A3B8", "#CBD5E1"],
        "signal_housing": "#E2E8F0",
        "signal_border": "#CBD5E1"
    },
    "Black & White": {
        "name": "Black & White",
        "app_bg": "#000000",
        "sidebar_bg": "#0A0A0A",
        "sidebar_border": "#262626",
        "nav_bg": "#141414",
        "nav_border": "#333333",
        "card_bg": "#111111",
        "card_border": "#333333",
        "white_card_bg": "#141414",
        "white_card_border": "#404040",
        "card_shadow": "none",
        "text_primary": "#FFFFFF",
        "text_secondary": "#D4D4D4",
        "text_muted": "#888888",
        "accent": "#FFFFFF",
        "accent_text": "#000000",
        "btn_bg": "#1F1F1F",
        "btn_hover": "#FFFFFF",
        "btn_border": "#555555",
        "badge_normal_bg": "#222222",
        "badge_normal_text": "#FFFFFF",
        "badge_normal_border": "#555555",
        "card1_bg": "#181818",
        "card1_border": "#444444",
        "card1_text": "#FFFFFF",
        "card2_bg": "#121212",
        "card2_border": "#333333",
        "card2_text": "#FFFFFF",
        "card3_bg": "#1F1F1F",
        "card3_border": "#555555",
        "card3_text": "#FFFFFF",
        "bar_fill": "#FFFFFF",
        "bar_track": "#333333",
        "chart_bg": "#0A0A0A",
        "chart_font": "#FFFFFF",
        "chart_grid": "rgba(255, 255, 255, 0.2)",
        "chart_line": "#FFFFFF",
        "chart_fill": "rgba(255, 255, 255, 0.08)",
        "chart_bar_colors": [[0, "#444444"], [0.5, "#888888"], [1, "#FFFFFF"]],
        "gauge_track": "#262626",
        "gauge_stop1": "#888888",
        "gauge_stop2": "#FFFFFF",
        "pie_colors": ["#FFFFFF", "#CCCCCC", "#999999", "#666666", "#333333"],
        "signal_housing": "#1A1A1A",
        "signal_border": "#333333"
    },
    "Nordic Civic Minimal": {
        "name": "Nordic Civic Minimal",
        "app_bg": "#FAF8FF",
        "sidebar_bg": "#F2F3FF",
        "sidebar_border": "#DAE2FD",
        "nav_bg": "#EAEDFF",
        "nav_border": "#C3C6D7",
        "card_bg": "#FFFFFF",
        "card_border": "#E2E7FF",
        "white_card_bg": "#FFFFFF",
        "white_card_border": "#DAE2FD",
        "card_shadow": "0 1px 3px 0 rgba(15, 23, 42, 0.04), 0 1px 2px -1px rgba(15, 23, 42, 0.02)",
        "text_primary": "#131B2E",
        "text_secondary": "#434655",
        "text_muted": "#737686",
        "accent": "#2563EB",
        "accent_text": "#FFFFFF",
        "btn_bg": "linear-gradient(135deg, #2563EB 0%, #004AC6 100%)",
        "btn_hover": "linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%)",
        "btn_border": "#2563EB",
        "badge_normal_bg": "#ECFDF5",
        "badge_normal_text": "#065F46",
        "badge_normal_border": "#A7F3D0",
        "card1_bg": "linear-gradient(135deg, #EFF6FF 0%, #DAE2FD 100%)",
        "card1_border": "#BFDBFE",
        "card1_text": "#131B2E",
        "card2_bg": "linear-gradient(135deg, #FFFFFF 0%, #F2F3FF 100%)",
        "card2_border": "#E2E7FF",
        "card2_text": "#131B2E",
        "card3_bg": "linear-gradient(135deg, #2563EB 0%, #004AC6 100%)",
        "card3_border": "#1D4ED8",
        "card3_text": "#FFFFFF",
        "bar_fill": "linear-gradient(90deg, #82F5C1 0%, #2563EB 100%)",
        "bar_track": "rgba(37, 99, 235, 0.12)",
        "chart_bg": "#FFFFFF",
        "chart_font": "#131B2E",
        "chart_grid": "rgba(115, 118, 134, 0.15)",
        "chart_line": "#2563EB",
        "chart_fill": "rgba(37, 99, 235, 0.10)",
        "chart_bar_colors": [[0, "#DAE2FD"], [0.5, "#82F5C1"], [1, "#2563EB"]],
        "gauge_track": "#EAEDFF",
        "gauge_stop1": "#82F5C1",
        "gauge_stop2": "#2563EB",
        "pie_colors": ["#2563EB", "#006C4A", "#AE0010", "#004AC6", "#82F5C1"],
        "signal_housing": "#EAEDFF",
        "signal_border": "#DAE2FD"
    }
}

active_theme_name = st.session_state.get("color_theme", "Soft Sand")
if active_theme_name not in THEMES:
    active_theme_name = "Soft Sand"
th = THEMES[active_theme_name]

# Dynamic CSS Injection based on chosen theme
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    /* 1. CLEAN STREAMLIT HEADER & PRESERVE SIDEBAR TOGGLE BUTTON (ORIGINAL NATURAL POSITION) */
    header[data-testid="stHeader"], .stAppHeader {{
        background: transparent !important;
        color: {th['text_primary']} !important;
        height: 2.8rem !important;
        z-index: 99 !important;
        pointer-events: none !important;
    }}
    header[data-testid="stHeader"] button, .stAppHeader button,
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button {{
        pointer-events: auto !important;
        color: {th['text_primary']} !important;
        background: {th['card_bg']} !important;
        border: 1px solid {th['card_border']} !important;
        border-radius: 8px !important;
        padding: 4px 8px !important;
        cursor: pointer !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06) !important;
    }}
    header[data-testid="stHeader"] button:hover, .stAppHeader button:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover,
    [data-testid="collapsedControl"] button:hover {{
        background: {th['nav_bg']} !important;
        border-color: {th['accent']} !important;
    }}
    div[data-testid="stDeployButton"], button[data-testid="stDeployButton"], .stDeployButton, #MainMenu, footer {{
        display: none !important;
        visibility: hidden !important;
    }}

    /* EXPANDED SIDEBAR COLLAPSE BUTTON */
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarHeader"] button {{
        background: {th['card_bg']} !important;
        border: 1px solid {th['card_border']} !important;
        color: {th['text_primary']} !important;
        border-radius: 8px !important;
        cursor: pointer !important;
    }}
    [data-testid="stSidebarCollapseButton"] button:hover,
    [data-testid="stSidebarHeader"] button:hover {{
        background: {th['nav_bg']} !important;
        border-color: {th['accent']} !important;
    }}

    /* PULL SIDEBAR CONTENT UP TO THE VERY TOP & MAKE COMPACT */
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
    section[data-testid="stSidebar"] > div:first-child,
    div[data-testid="stSidebarContent"] {{
        padding-top: 0 !important;
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
        padding-bottom: 0.5rem !important;
        display: flex !important;
        flex-direction: column !important;
        height: 100% !important;
        min-height: calc(100vh - 40px) !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div,
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] .stVerticalBlock {{
        display: flex !important;
        flex-direction: column !important;
        flex-grow: 1 !important;
        height: 100% !important;
        min-height: calc(100vh - 55px) !important;
    }}
    [data-testid="stSidebarHeader"] {{
        padding: 0 !important;
        margin: 0 !important;
        height: 28px !important;
        min-height: 28px !important;
    }}
    [data-testid="stSidebarHeader"] button {{
        margin: 0 !important;
        padding: 2px !important;
        transform: scale(0.85) !important;
    }}
    .stVerticalBlock > div:has(.sidebar-bottom-container),
    div:has(.sidebar-bottom-container),
    .sidebar-bottom-container {{
        margin-top: auto !important;
        padding-top: 1rem !important;
        padding-bottom: 0.25rem !important;
    }}
    .st-key-btn_bottom_contact_nav {{
        margin-bottom: 0.75rem !important;
    }}
    .st-key-btn_bottom_contact_nav button {{
        border-radius: 12px !important;
        padding: 0.55rem 0.85rem !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        letter-spacing: -0.2px !important;
        transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }}
    /* 2. OUTER CANVAS & DYNAMIC THEME */
    .stApp {{
        background: {th['app_bg']} !important;
        background-color: {th['app_bg']} !important;
        color: {th['text_primary']} !important;
        font-family: 'Hanken Grotesk', 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }}
    .block-container {{
        background: {th['app_bg']} !important;
        border-radius: 0px !important;
        box-shadow: none !important;
        padding: 1.2rem 2rem 2rem 2rem !important;
        max-width: 100% !important;
        margin: 0 !important;
        color: {th['text_primary']} !important;
    }}

    /* Typography */
    h1, h2, h3, h4, h5, h6 {{
        color: {th['text_primary']} !important;
    }}
    p, span, label {{
        color: {th['text_secondary']};
    }}

    /* Universal Divider */
    hr {{
        border: 0 !important;
        border-top: 1px solid {th['nav_border']} !important;
    }}

    /* 3. SIDEBAR NAVIGATION */
    section[data-testid="stSidebar"] {{
        background-color: {th['sidebar_bg']} !important;
        border-right: 1px solid {th['sidebar_border']} !important;
        box-shadow: 4px 0 25px rgba(0, 0, 0, 0.1) !important;
    }}
    section[data-testid="stSidebar"] .stRadio > label {{
        display: none !important;
    }}

    div[data-testid="stRadio"] > div {{
        gap: 6px !important;
    }}
    div[data-testid="stRadio"] [data-testid="stRadioOption"],
    div[data-testid="stRadio"] label {{
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 12px !important;
        padding: 7px 12px !important;
        margin: 2px 0 !important;
        white-space: nowrap !important;
        color: {th['text_muted']} !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }}
    div[data-testid="stRadio"] [data-testid="stRadioOption"] p,
    div[data-testid="stRadio"] label p {{
        color: {th['text_muted']} !important;
        margin: 0 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }}
    div[data-testid="stRadio"] [data-testid="stRadioOption"]:hover,
    div[data-testid="stRadio"] label:hover {{
        background: {th['nav_bg']} !important;
        border-color: {th['nav_border']} !important;
        transform: none !important;
    }}
    div[data-testid="stRadio"] [data-testid="stRadioOption"]:hover p,
    div[data-testid="stRadio"] label:hover p {{
        color: {th['text_primary']} !important;
    }}

    /* Active Highlight */
    div[data-testid="stRadio"] [data-testid="stRadioOption"][data-selected="true"],
    div[data-testid="stRadio"] label:has(input:checked) {{
        background: {th['accent']} !important;
        border-color: {th['accent']} !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2) !important;
        transform: none !important;
    }}
    div[data-testid="stRadio"] [data-testid="stRadioOption"][data-selected="true"] p,
    div[data-testid="stRadio"] [data-testid="stRadioOption"][data-selected="true"] span,
    div[data-testid="stRadio"] label:has(input:checked) p,
    div[data-testid="stRadio"] label:has(input:checked) span {{
        color: {th['accent_text']} !important;
        font-weight: 700 !important;
    }}
    div[data-testid="stRadioOption"] > div:first-child:not([data-testid="stMarkdownContainer"]) {{
        display: none !important;
    }}

    /* 4. AIBST TOP HEADER BAR */
    .aibst-header-bar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 1.1rem;
        margin-bottom: 1.3rem;
        border-bottom: 1px solid {th['nav_border']};
    }}
    .aibst-brand-wrap {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .aibst-logo-icon {{
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: {th['btn_bg']};
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
    }}
    .aibst-brand-title {{
        color: {th['text_primary']} !important;
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.3px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .sub-header-row {{
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        margin-bottom: 1.2rem;
    }}
    .breadcrumb-text {{
        color: {th['text_muted']} !important;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.2px;
        margin-bottom: 4px;
    }}
    .main-page-title {{
        color: {th['text_primary']} !important;
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.4px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .corridor-counts-pill {{
        display: flex;
        align-items: center;
        gap: 8px;
        background: {th['nav_bg']};
        border: 1px solid {th['nav_border']};
        border-radius: 14px;
        padding: 5px 8px;
    }}
    .count-box {{
        background: {th['card_bg']};
        border: 1px solid {th['card_border']};
        border-radius: 10px;
        padding: 5px 12px;
        display: flex;
        align-items: baseline;
        gap: 5px;
    }}
    .count-val {{
        color: {th['text_primary']} !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        font-weight: 700;
    }}
    .count-lbl {{
        color: {th['accent']} !important;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
    }}

    /* 5. CARD STYLES */
    .dark-card, .ctrl-panel {{
        background: {th['card_bg']} !important;
        border: 1px solid {th['card_border']} !important;
        border-radius: 20px !important;
        padding: 18px 22px;
        box-shadow: {th['card_shadow']} !important;
        color: {th['text_primary']} !important;
    }}
    .white-card {{
        background: {th['white_card_bg']} !important;
        border: 1px solid {th['white_card_border']} !important;
        border-radius: 20px !important;
        padding: 20px !important;
        box-shadow: {th['card_shadow']} !important;
        color: {th['text_primary']} !important;
    }}
    .kpi-card {{
        background: {th['card_bg']};
        border: 1px solid {th['card_border']};
        border-radius: 16px;
        padding: 1rem 1.25rem;
        box-shadow: {th['card_shadow']} !important;
    }}
    .kpi-label {{
        color: {th['text_muted']};
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .kpi-value {{
        color: {th['text_primary']};
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.2;
        margin: 4px 0;
        font-family: 'JetBrains Mono', monospace;
    }}
    .kpi-meta {{
        color: {th['text_secondary']};
        font-size: 0.72rem;
        font-weight: 500;
    }}

    .card-emerald-gradient {{
        background: {th['card1_bg']} !important;
        border: 1px solid {th['card1_border']} !important;
        border-radius: 20px;
        padding: 18px 20px;
        color: {th['card1_text']} !important;
        margin-bottom: 1rem;
    }}
    .emerald-info-pill {{
        width: 22px;
        height: 22px;
        border-radius: 50%;
        background: rgba(128, 128, 128, 0.2);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 800;
        color: {th['card1_text']};
    }}
    .emerald-metric-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        font-size: 0.78rem;
        color: {th['card1_text']};
    }}
    .emerald-metric-row:last-child {{
        border-bottom: none;
    }}

    .card-teal-glass {{
        background: {th['card2_bg']} !important;
        border: 1px solid {th['card2_border']} !important;
        border-radius: 20px;
        padding: 18px 20px;
        color: {th['card2_text']} !important;
        margin-bottom: 1rem;
    }}
    .density-bar-group {{
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin-top: 10px;
    }}
    .density-bar-item {{
        display: flex;
        flex-direction: column;
        gap: 4px;
    }}
    .density-label {{
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
        color: {th['text_secondary']};
        font-weight: 600;
    }}
    .bar-track {{
        height: 6px;
        background: {th['bar_track']} !important;
        border-radius: 4px;
        overflow: hidden;
    }}
    .bar-fill {{
        height: 100%;
        background: {th['bar_fill']} !important;
        border-radius: 4px;
    }}

    .card-blue-cyan-gradient {{
        background: {th['card3_bg']} !important;
        border: 1px solid {th['card3_border']} !important;
        border-radius: 20px;
        padding: 20px 22px;
        color: {th['card3_text']} !important;
    }}
    .aibst-examine-btn {{
        background: {th['accent']} !important;
        color: {th['accent_text']} !important;
        border-radius: 20px !important;
        font-weight: 800 !important;
        font-size: 0.82rem !important;
        border: none !important;
        padding: 8px 20px !important;
        cursor: pointer !important;
        transition: all 0.28s ease !important;
        display: inline-block !important;
    }}

    /* 6. BUTTONS & PILLS */
    button[data-testid^="stBaseButton"],
    button[data-testid^="baseButton"],
    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button,
    .stButton button,
    .stDownloadButton button {{
        background: {th['btn_bg']} !important;
        color: {'#000000' if active_theme_name == 'Black & White' and False else '#FFFFFF'} !important;
        border: 1px solid {th['btn_border']} !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        font-size: 0.84rem !important;
        padding: 0.48rem 0.95rem !important;
        transition: all 0.28s ease !important;
        cursor: pointer !important;
        white-space: nowrap !important;
    }}
    button[data-testid^="stBaseButton"]:hover,
    button[data-testid^="baseButton"]:hover,
    div[data-testid="stButton"] button:hover,
    div[data-testid="stDownloadButton"] button:hover,
    .stButton button:hover,
    .stDownloadButton button:hover {{
        background: {th['btn_hover']} !important;
        color: {'#000000' if active_theme_name == 'Black & White' else '#FFFFFF'} !important;
        transform: translateY(-2px) !important;
    }}

    /* Style st.pills */
    div[data-testid="stPills"] {{
        gap: 8px !important;
        background: {th['nav_bg']} !important;
        border: 1px solid {th['nav_border']} !important;
        border-radius: 16px !important;
        padding: 5px 8px !important;
    }}
    div[data-testid="stPills"] button {{
        background: {th['card_bg']} !important;
        border: 1px solid {th['card_border']} !important;
        border-radius: 12px !important;
        color: {th['text_secondary']} !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        padding: 7px 16px !important;
        transition: all 0.25s ease !important;
    }}
    div[data-testid="stPills"] button:hover {{
        color: {th['text_primary']} !important;
        border-color: {th['accent']} !important;
        transform: translateY(-2px) !important;
    }}
    div[data-testid="stPills"] button[aria-selected="true"],
    div[data-testid="stPills"] button[data-selected="true"],
    div[data-testid="stPills"] button[data-state="active"] {{
        background: {th['accent']} !important;
        color: {th['accent_text']} !important;
        border: 1px solid {th['accent']} !important;
        font-weight: 700 !important;
        transform: translateY(-1px) !important;
    }}

    /* 7. INPUTS, DATAFRAMES, BADGES */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {th['card_border']} !important;
        border-radius: 16px !important;
        background: {th['card_bg']} !important;
        overflow: hidden !important;
    }}
    div[data-baseweb="select"] {{
        cursor: pointer !important;
        pointer-events: auto !important;
    }}
    div[data-baseweb="select"] > div {{
        background-color: {th['card_bg']} !important;
        border: 1.5px solid {th['card_border']} !important;
        color: {th['text_primary']} !important;
        border-radius: 12px !important;
        cursor: pointer !important;
        box-shadow: {th['card_shadow']} !important;
        transition: all 0.2s ease !important;
    }}
    div[data-baseweb="select"] > div:hover {{
        border-color: {th['accent']} !important;
    }}
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    [data-testid="stSelectboxVirtualDropdown"] {{
        z-index: 99999999 !important;
        pointer-events: auto !important;
    }}
    div[data-baseweb="popover"] ul,
    div[data-baseweb="menu"] ul,
    ul[role="listbox"] {{
        background-color: {th['card_bg']} !important;
        border: 1.5px solid {th['card_border']} !important;
        border-radius: 12px !important;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.22) !important;
        padding: 6px !important;
    }}
    div[data-baseweb="popover"] li,
    div[data-baseweb="menu"] li,
    li[role="option"] {{
        color: {th['text_primary']} !important;
        background-color: transparent !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 9px 14px !important;
        border-radius: 8px !important;
        margin: 2px 0 !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
    }}
    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="menu"] li:hover,
    li[role="option"]:hover,
    li[aria-selected="true"] {{
        background-color: {th['nav_bg']} !important;
        color: {th['accent']} !important;
    }}
    .stTextInput input {{
        background-color: {th['card_bg']} !important;
        border: 1px solid {th['card_border']} !important;
        color: {th['text_primary']} !important;
        border-radius: 12px !important;
    }}
    .stTextInput input:focus {{
        border-color: {th['accent']} !important;
    }}
    .ctrl-panel-title {{
        color: {th['text_primary']} !important;
        font-size: 0.98rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .signal-housing {{
        background: {th['signal_housing']};
        border: 1px solid {th['signal_border']};
        border-radius: 12px;
        padding: 8px 12px;
        display: inline-flex;
        gap: 10px;
        align-items: center;
        justify-content: center;
    }}
    .signal-lamp {{
        width: 26px;
        height: 26px;
        border-radius: 50%;
        opacity: 0.22;
        transition: all 0.25s ease;
    }}
    .lamp-red {{ background: #DC2626; }}
    .lamp-amber {{ background: #D97706; }}
    .lamp-green {{ background: #16A34A; }}
    .lamp-red.active {{ opacity: 1.0; box-shadow: 0 0 14px rgba(220, 38, 38, 0.8), inset 0 0 4px #FFFFFF; }}
    .lamp-amber.active {{ opacity: 1.0; box-shadow: 0 0 14px rgba(217, 119, 6, 0.8), inset 0 0 4px #FFFFFF; }}
    .lamp-green.active {{ opacity: 1.0; box-shadow: 0 0 16px rgba(22, 163, 74, 0.8), inset 0 0 4px #FFFFFF; }}

    .badge-normal {{
        background: {th['badge_normal_bg']};
        color: {th['badge_normal_text']};
        border: 1px solid {th['badge_normal_border']};
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
    }}
    .silver-badge {{
        background: {th['nav_bg']};
        color: {th['text_secondary']};
        border: 1px solid {th['nav_border']};
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
    }}
    .badge-critical {{
        background: rgba(220, 38, 38, 0.15);
        color: #EF4444;
        border: 1px solid rgba(220, 38, 38, 0.35);
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
    }}
    .badge-warning {{
        background: rgba(217, 119, 6, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(217, 119, 6, 0.35);
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
    }}

    @media (max-width: 1200px) {{
        .sub-header-row {{
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 10px !important;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# ==================== DATA FETCHING ENGINE ====================
def fetch_safe_stats(db_manager, hours_window):
    """Fetches high-level metrics with guaranteed fallbacks."""
    if hasattr(db_manager, "get_summary_stats"):
        try:
            return db_manager.get_summary_stats(hours=hours_window)
        except Exception:
            pass

    vehicles = [dict(r) for r in db_manager.get_vehicles(hours=hours_window)] if hasattr(db_manager, "get_vehicles") else []
    violations = [dict(r) for r in db_manager.get_helmet_violations(hours=hours_window)] if hasattr(db_manager, "get_helmet_violations") else []
    accidents = [dict(r) for r in db_manager.get_accidents(hours=hours_window)] if hasattr(db_manager, "get_accidents") else []
    by_type = [dict(r) for r in db_manager.get_vehicle_count_by_type(hours=hours_window)] if hasattr(db_manager, "get_vehicle_count_by_type") else []
    by_lane = [dict(r) for r in db_manager.get_lane_distribution(hours=hours_window)] if hasattr(db_manager, "get_lane_distribution") else []

    return {
        "total_vehicles": len(vehicles),
        "total_violations": len(violations),
        "total_accidents": len(accidents),
        "vehicle_types": {row["vehicle_type"]: row["count"] for row in by_type},
        "lane_counts": {row["lane_id"]: row["count"] for row in by_lane},
        "hours_queried": hours_window
    }

# Initialize Database Instance
db = DatabaseManager()

# ==================== SYNCHRONIZED NAVIGATION ENGINE ====================
NAV_SECTIONS = [
    "Vehicle Detection & Counting",
    "Adaptive Traffic Light Signal",
    "Contact Us / Report a Problem"
]

# Shortened, single-line clean labels
NAV_DISPLAY = {
    "Vehicle Detection & Counting": "🚗 Vehicle Radar",
    "Adaptive Traffic Light Signal": "🚦 Adaptive Signals",
    "Contact Us / Report a Problem": "💬 Contact Us"
}

if "nav_section" not in st.session_state or st.session_state.get("nav_section") not in NAV_SECTIONS:
    st.session_state["nav_section"] = "Vehicle Detection & Counting"

# ==================== SIDEBAR NAVIGATION ====================
with st.sidebar:
    # Compact logo & header pulled to the very top
    st.markdown(f"""
    <div style="padding: 0 0 0.35rem 0; margin-top: -0.2rem;">
        <div style="display: flex; align-items: center; gap: 9px;">
            <div style="width: 30px; height: 30px; border-radius: 8px; background: {th['btn_bg']}; display: flex; align-items: center; justify-content: center; color: #FFFFFF; box-shadow: 0 3px 10px rgba(0, 0, 0, 0.12);">
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"></path>
                </svg>
            </div>
            <div>
                <h3 style="color: {th['text_primary']}; margin: 0; font-size: 1.02rem; font-weight: 800; line-height: 1.1; letter-spacing: -0.3px;">AIBST</h3>
                <div style="color: {th['accent']}; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.5px; line-height: 1;">TRAFFIC AI &bull; V2.5</div>
            </div>
        </div>
    </div>
    <hr style="border: 0; border-top: 1px solid {th['nav_border']}; margin: 0.25rem 0 0.5rem 0;">
    """, unsafe_allow_html=True)

    if "nav_section" not in st.session_state:
        st.session_state["nav_section"] = "Vehicle Detection & Counting"

    st.markdown(f"""
    <div style="color: {th['text_muted']}; font-size: 0.65rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.6px; margin: 0.1rem 0 0.35rem 0;">
        SYSTEM MODULES
    </div>
    """, unsafe_allow_html=True)

    # Primary core modules radio at top (single-line options)
    PRIMARY_OPTIONS = ["Vehicle Detection & Counting", "Adaptive Traffic Light Signal"]
    cur_nav = st.session_state.get("nav_section")
    cur_index = PRIMARY_OPTIONS.index(cur_nav) if cur_nav in PRIMARY_OPTIONS else None

    def _on_primary_radio_change():
        chosen = st.session_state.get("top_primary_nav_radio")
        if chosen:
            st.session_state["nav_section"] = chosen

    selected_top = st.radio(
        "Primary Navigation",
        PRIMARY_OPTIONS,
        index=cur_index,
        format_func=lambda x: NAV_DISPLAY.get(x, x),
        label_visibility="collapsed",
        key="top_primary_nav_radio",
        on_change=_on_primary_radio_change
    )

    # Bottom Contact Option: Anchor to very bottom of sidebar with no extra descriptive cards
    is_contact_selected = (st.session_state.get("nav_section") == "Contact Us / Report a Problem")

    st.markdown(f"""
    <div class="sidebar-bottom-container">
        <hr style="border: 0; border-top: 1px solid {th['nav_border']}; margin: 0 0 0.65rem 0;">
    </div>
    """, unsafe_allow_html=True)

    def _select_contact_page():
        st.session_state["nav_section"] = "Contact Us / Report a Problem"
        if "top_primary_nav_radio" in st.session_state:
            st.session_state["top_primary_nav_radio"] = None

    st.button(
        "💬 Contact Us",
        key="btn_bottom_contact_nav",
        use_container_width=True,
        type="primary" if is_contact_selected else "secondary",
        on_click=_select_contact_page
    )

    # Default audit window (24 hours)
    active_hours = 24

# Query active logs from database
metrics = fetch_safe_stats(db, active_hours)
vehicle_records = [dict(r) for r in db.get_vehicles(hours=active_hours)] if hasattr(db, "get_vehicles") else []
violations_records = [dict(r) for r in db.get_helmet_violations(hours=active_hours)] if hasattr(db, "get_helmet_violations") else []
accidents_records = [dict(r) for r in db.get_accidents(hours=active_hours)] if hasattr(db, "get_accidents") else []
signal_records = [dict(r) for r in db.get_signal_logs(hours=active_hours)] if hasattr(db, "get_signal_logs") else []

selected_nav = st.session_state["nav_section"]

# ==================== GLOBAL TOP HEADER BAR (FULLY INTERACTIVE) ====================
NAV_MAP = {
    "🚗 Live Radar": "Vehicle Detection & Counting",
    "⚡ Adaptive AI": "Adaptive Traffic Light Signal",
    "💬 Support": "Contact Us / Report a Problem"
}
REVERSE_NAV_MAP = {
    "Vehicle Detection & Counting": "🚗 Live Radar",
    "Adaptive Traffic Light Signal": "⚡ Adaptive AI",
    "Contact Us / Report a Problem": "💬 Support"
}

top_c1, top_c2, top_c3 = st.columns([64, 13, 23], vertical_alignment="center")

with top_c1:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; height: 38px;">
        <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: {'#10B981' if active_theme_name != 'Black & White' else '#FFFFFF'}; box-shadow: 0 0 6px rgba(16, 185, 129, 0.6);"></span>
        <span style="font-size: 0.78rem; font-weight: 700; color: {th['text_muted']}; letter-spacing: 0.6px; text-transform: uppercase;">AIBST TRAFFIC CONTROL SYSTEM</span>
    </div>
    """, unsafe_allow_html=True)

with top_c2:
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: flex-end; gap: 6px; height: 38px;">
        <span style="font-size: 0.88rem;">🎨</span>
        <span style="font-size: 0.82rem; font-weight: 700; color: {th['text_primary']}; white-space: nowrap;">Colour Theme:</span>
    </div>
    """, unsafe_allow_html=True)

with top_c3:
    theme_list = ["Soft Sand", "Silver & White", "Black & White", "Nordic Civic Minimal"]
    c_idx = theme_list.index(active_theme_name) if active_theme_name in theme_list else 0

    def _on_header_theme_change():
        sel = st.session_state.get("header_theme_selector")
        if sel and sel in THEMES:
            st.session_state["color_theme"] = sel

    st.selectbox(
        "Colour Theme Selector",
        theme_list,
        index=c_idx,
        label_visibility="collapsed",
        key="header_theme_selector",
        on_change=_on_header_theme_change
    )

st.markdown(f'<hr style="border: 0; border-top: 1px solid {th["nav_border"]}; margin: 0.4rem 0 1rem 0;">', unsafe_allow_html=True)

# Global uploads directory and fast media deletion helper
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def safe_delete_media_file(fpath):
    if not fpath:
        return True
    import gc
    gc.collect()
    if not os.path.exists(fpath):
        return True
    try:
        os.remove(fpath)
        return True
    except Exception:
        try:
            trash_path = fpath + f".trash_{int(time.time() * 1000)}"
            os.rename(fpath, trash_path)
            try:
                os.remove(trash_path)
            except Exception:
                pass
            return True
        except Exception:
            return False

# =====================================================================
# 1. DASHBOARD (EXECUTIVE OVERVIEW - AIBST EXACT SCREENSHOT LAYOUT)
# =====================================================================
if selected_nav == "Vehicle Detection & Counting":
    st.markdown(f"""
    <div class="ctrl-panel" style="margin-bottom: 1.2rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="ctrl-panel-title" style="margin-bottom: 0;">
                <span>🚗</span> Vehicle Detection & Traffic Flow Radar
            </div>
        </div>
    """, unsafe_allow_html=True)

    input_mode = st.radio(
        "Camera Feed Ingestion Channel",
        ["📹 Surveillance Video / Photo Upload", "🔴 Live Hardware Webcam / Camera Stream"],
        horizontal=True
    )

    v_col1, v_col2 = st.columns([1, 2])
    selected_video = None
    video_source_label = ""
    is_image = False
    run_analysis_clicked = False
    run_image_clicked = False

    # Ensure uploads directory exists permanently
    UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    def safe_delete_media_file(fpath):
        if not fpath:
            return True
        import gc
        gc.collect()
        if not os.path.exists(fpath):
            return True
        try:
            os.remove(fpath)
            return True
        except Exception:
            try:
                trash_path = fpath + f".trash_{int(time.time() * 1000)}"
                os.rename(fpath, trash_path)
                try:
                    os.remove(trash_path)
                except Exception:
                    pass
                return True
            except Exception:
                return False

    with v_col1:
        if input_mode == "📹 Surveillance Video / Photo Upload":
            # Discover all available local videos and photos
            available_media = {}
            
            # Clean leftover trash files and index user uploads
            if os.path.exists(UPLOAD_DIR):
                for f in sorted(os.listdir(UPLOAD_DIR), reverse=True):
                    f_path = os.path.join(UPLOAD_DIR, f)
                    if ".trash" in f:
                        try:
                            os.remove(f_path)
                        except Exception:
                            pass
                        continue
                    if os.path.isfile(f_path) and not f.startswith("."):
                        ext = os.path.splitext(f)[1].lower()
                        if ext in [".mp4", ".avi", ".mov", ".mkv", ".webm", ".jpg", ".jpeg", ".png", ".bmp"]:
                            available_media[f"📁 {f}"] = f_path

            # Dynamic uploader key version to allow instant clearing on delete
            if "uploader_key_version" not in st.session_state:
                st.session_state["uploader_key_version"] = 0

            uploader_widget_key = f"traffic_media_file_uploader_{st.session_state['uploader_key_version']}"

            # File uploader with broad format support (videos + photos)
            uploaded = st.file_uploader(
                "Upload Traffic Media (MP4, AVI, MOV, MKV, JPG, PNG)",
                type=["mp4", "avi", "mov", "mkv", "webm", "jpg", "jpeg", "png", "bmp"],
                label_visibility="collapsed",
                key=uploader_widget_key
            )
            if uploaded is not None:
                save_path = os.path.join(UPLOAD_DIR, uploaded.name)
                # Only write to disk once (prevents disk lag on every rerun)
                if not os.path.exists(save_path) or os.path.getsize(save_path) != uploaded.size:
                    with open(save_path, "wb") as f:
                        f.write(uploaded.getbuffer())
                st.session_state["uploaded_media_path"] = save_path
                st.session_state["uploaded_media_name"] = uploaded.name
                available_media[f"📁 {uploaded.name}"] = save_path

            # Media selection dropdown
            if available_media:
                media_keys = list(available_media.keys())
                # Default selection to last uploaded if available
                default_idx = 0
                if "uploaded_media_name" in st.session_state:
                    for idx, k in enumerate(media_keys):
                        if st.session_state["uploaded_media_name"] in k:
                            default_idx = idx
                            break
                default_idx = min(max(0, default_idx), len(media_keys) - 1)

                chosen_label = st.selectbox(
                    "Choose Active Traffic Media Asset",
                    media_keys,
                    index=default_idx,
                    help="Select any uploaded video/photo to analyze."
                )
                selected_video = available_media[chosen_label]
                video_source_label = chosen_label
                ext = os.path.splitext(selected_video)[1].lower()
                is_image = ext in [".jpg", ".jpeg", ".png", ".bmp"]

                # Check if this is a user-uploaded file (stored in data/uploads/)
                is_uploaded_file = os.path.abspath(selected_video).startswith(os.path.abspath(UPLOAD_DIR))

                ready_col, del_col = st.columns([65, 35], vertical_alignment="center")
                with ready_col:
                    st.success(f"✓ Ready: {video_source_label}")
                with del_col:
                    if is_uploaded_file:
                        if st.button("🗑️ Remove Asset", key="btn_remove_active_asset", use_container_width=True, help=f"Permanently delete {os.path.basename(selected_video)}"):
                            safe_delete_media_file(selected_video)
                            # Increment uploader key so file_uploader widget immediately resets to empty
                            st.session_state["uploader_key_version"] = st.session_state.get("uploader_key_version", 0) + 1
                            st.session_state.pop("uploaded_media_path", None)
                            st.session_state.pop("uploaded_media_name", None)
                            st.session_state.pop("last_uploaded_name", None)
                            st.session_state.pop("session_analysis_records", None)
                            st.session_state.pop("session_analysis_counts", None)
                            st.session_state.pop("video_congestion_timeline", None)
                            st.rerun()
                    else:
                        st.caption("🔒 Sample Protected")

                # Manage / Remove other uploaded media files
                uploaded_files_list = [f for f in sorted(os.listdir(UPLOAD_DIR), reverse=True) if os.path.isfile(os.path.join(UPLOAD_DIR, f)) and not f.startswith(".") and ".trash" not in f] if os.path.exists(UPLOAD_DIR) else []
                if uploaded_files_list:
                    with st.expander("⚙️ Manage / Delete Uploaded Media"):
                        st.markdown(f"<span style='font-size: 0.78rem; color: {th['text_muted']};'>Uploaded storage (<code>data/uploads/</code>):</span>", unsafe_allow_html=True)
                        for uf in uploaded_files_list:
                            uf_path = os.path.join(UPLOAD_DIR, uf)
                            uf_c1, uf_c2 = st.columns([70, 30], vertical_alignment="center")
                            with uf_c1:
                                st.markdown(f"<span style='font-size: 0.78rem;'>📁 {uf}</span>", unsafe_allow_html=True)
                            with uf_c2:
                                if st.button("🗑️ Delete", key=f"btn_del_uf_{uf}", use_container_width=True):
                                    safe_delete_media_file(uf_path)
                                    # If deleted file was the active or uploaded asset, reset uploader widget
                                    if st.session_state.get("uploaded_media_name") == uf or (uploaded is not None and uploaded.name == uf):
                                        st.session_state["uploader_key_version"] = st.session_state.get("uploader_key_version", 0) + 1
                                    st.session_state.pop("uploaded_media_path", None)
                                    st.session_state.pop("uploaded_media_name", None)
                                    st.session_state.pop("last_uploaded_name", None)
                                    st.session_state.pop("session_analysis_records", None)
                                    st.session_state.pop("session_analysis_counts", None)
                                    st.session_state.pop("video_congestion_timeline", None)
                                    st.rerun()

                        if len(uploaded_files_list) > 1:
                            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
                            if st.button("🗑️ Delete All Uploaded Files", key="btn_del_all_uploads", use_container_width=True):
                                for uf in uploaded_files_list:
                                    safe_delete_media_file(os.path.join(UPLOAD_DIR, uf))
                                st.session_state["uploader_key_version"] = st.session_state.get("uploader_key_version", 0) + 1
                                st.session_state.pop("uploaded_media_path", None)
                                st.session_state.pop("uploaded_media_name", None)
                                st.session_state.pop("last_uploaded_name", None)
                                st.session_state.pop("session_analysis_records", None)
                                st.session_state.pop("session_analysis_counts", None)
                                st.session_state.pop("video_congestion_timeline", None)
                                st.rerun()
            else:
                selected_video = None
                st.info("Upload an MP4 video or photo above to begin.")

        else:
            # LIVE HARDWARE WEBCAM / CAMERA STREAM CHANNEL
            selected_video = None
            st.markdown(f"""
            <div style="background: {th['card_bg']}; border: 1px solid {th['card_border']}; border-radius: 12px; padding: 14px; margin-bottom: 12px;">
                <div style="color: {th['accent']}; font-size: 0.84rem; font-weight: 700;">🔴 Live Hardware Sensor Hub</div>
                <div style="color: {th['text_secondary']}; font-size: 0.74rem; margin-top: 4px;">Capture live frames directly from your connected camera or webcam sensor.</div>
            </div>
            """, unsafe_allow_html=True)

            cam_snapshot = st.camera_input("📸 Capture Instant Live Traffic Snapshot")
            if cam_snapshot is not None:
                # Save snapshot to persistent uploads directory
                snap_filename = f"webcam_snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                snap_path = os.path.join(UPLOAD_DIR, snap_filename)
                with open(snap_path, "wb") as f:
                    f.write(cam_snapshot.getbuffer())
                selected_video = snap_path
                video_source_label = f"Live Snapshot: {snap_filename}"
                is_image = True
                run_image_clicked = True

            st.markdown("<hr style='border: 0; border-top: 1px solid #D8C2B3; margin: 0.8rem 0 0.8rem 0;'>", unsafe_allow_html=True)
            
            # External full-speed launcher
            if st.button("🚀 Launch 60 FPS Standalone Webcam Window", use_container_width=True):
                try:
                    import subprocess
                    subprocess.Popen([sys.executable, "main.py", "--webcam"], cwd=str(BASE_DIR))
                    st.success("Webcam AI window launched! Press 'q' inside the video window to stop.")
                except Exception as ex:
                    st.error(f"Launch error: {ex}")

        # Action Buttons when media is available
        if selected_video and os.path.exists(selected_video):
            st.markdown("<hr style='border: 0; border-top: 1px solid #D8C2B3; margin: 0.8rem 0 0.8rem 0;'>", unsafe_allow_html=True)
            if is_image:
                if not run_image_clicked:
                    run_image_clicked = st.button("⚡ Analyze Traffic Photo / Image", key="btn_run_image_analysis", use_container_width=True)
            else:
                run_analysis_clicked = st.button("⚡ Start AI Video Analysis", key="btn_run_yolo_analysis", use_container_width=True)
            
            # Reset database records button
            if st.button("🗑️ Reset All Traffic Records to Zero", key="btn_reset_records", use_container_width=True):
                try:
                    c = db._get_cursor()
                    c.execute("DELETE FROM vehicle_logs")
                    c.execute("DELETE FROM helmet_violations")
                    c.execute("DELETE FROM accident_alerts")
                    c.execute("DELETE FROM signal_logs")
                    db.connection.commit()
                    st.session_state.pop("session_analysis_records", None)
                    st.session_state.pop("session_analysis_counts", None)
                    st.session_state.pop("video_congestion_timeline", None)
                    timeline_cache_path = os.path.join(BASE_DIR, "data", "video_timeline.json")
                    if os.path.exists(timeline_cache_path):
                        try:
                            os.remove(timeline_cache_path)
                        except Exception:
                            pass
                    st.success("All vehicle records reset to 0!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Reset error: {ex}")

    with v_col2:
        if selected_video and os.path.exists(selected_video):
            if is_image:
                # ================= IMAGE / PHOTO PROCESSING =================
                if run_image_clicked:
                    try:
                        import cv2
                        import core.detection.vision_engine
                        importlib.reload(core.detection.vision_engine)
                        from core.detection.vision_engine import VisionEngine
                        try:
                            c = db._get_cursor()
                            c.execute("DELETE FROM vehicle_logs")
                            db.connection.commit()
                        except Exception:
                            pass
                        v_engine = VisionEngine(db_manager=db)
                        v_engine.reset()
                        sig_controller = AdaptiveSignalController(db_manager=db)

                        img_bgr = cv2.imread(selected_video)
                        if img_bgr is None:
                            st.error("Could not read image file.")
                        else:
                            with st.spinner("Analyzing image with Vision Engine..."):
                                annotated, stats = v_engine.process_frame(img_bgr, frame_idx=1, record_to_db=True)
                                # Update background signal controller
                                for lane_id, count in stats.get("lane_counts", {}).items():
                                    sig_controller.update_density(lane_id, count)
                                sig_controller.tick_second()

                            rgb_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            st.image(rgb_img, caption=f"⚡ AI Vision Detections: {video_source_label}", use_container_width=True)

                            raw_counts = stats.get("class_counts", {})
                            c_cars = raw_counts.get("Car", 0)
                            c_bikes = raw_counts.get("Motorcycle", 0)
                            c_buses = raw_counts.get("Bus", 0)
                            c_trucks = raw_counts.get("Truck", 0)
                            total_now = stats.get("total_count", c_cars + c_bikes + c_buses + c_trucks)

                            active_lane = sig_controller.current_green_lane
                            lane_names = ["North", "South", "East", "West"]
                            active_lane_name = lane_names[active_lane] if active_lane < len(lane_names) else f"Lane {active_lane+1}"

                            st.markdown(f"""
                            <div style="background: {th['card_bg']}; border: 1.5px solid {th['card_border']}; border-radius: 12px; padding: 10px 16px; margin-top: 8px; box-shadow: {th['card_shadow']};">
                                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                                    <span style="font-weight: 700; color: {th['text_primary']}; font-size: 0.85rem;">🔴 Detections:</span>
                                    <span style="background: {th['nav_bg']}; color: {th['text_primary']}; padding: 3px 10px; border-radius: 8px; font-weight: 700; font-size: 0.8rem;">🚗 Cars: {c_cars}</span>
                                    <span style="background: {th['nav_bg']}; color: {th['text_primary']}; padding: 3px 10px; border-radius: 8px; font-weight: 700; font-size: 0.8rem;">🏍️ Bikes: {c_bikes}</span>
                                    <span style="background: {th['nav_bg']}; color: {th['text_primary']}; padding: 3px 10px; border-radius: 8px; font-weight: 700; font-size: 0.8rem;">🚌 Buses: {c_buses}</span>
                                    <span style="background: {th['nav_bg']}; color: {th['text_primary']}; padding: 3px 10px; border-radius: 8px; font-weight: 700; font-size: 0.8rem;">🚛 Trucks: {c_trucks}</span>
                                    <span style="background: {th['accent']}; color: {th['accent_text']}; padding: 3px 12px; border-radius: 8px; font-weight: 800; font-size: 0.82rem;">Total: {total_now}</span>
                                </div>
                                <div style="margin-top: 8px; padding-top: 6px; border-top: 1px dashed {th['nav_border']}; display: flex; justify-content: space-between; align-items: center; font-size: 0.78rem;">
                                    <span>🚦 <strong>Background Signal Controller:</strong> Corridor {active_lane_name} is <span style="color: #22C55E; font-weight: 800;">GREEN</span></span>
                                    <span style="color: {th['text_muted']};">Traffic Records & Signals Logged to Database</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.session_state["session_analysis_counts"] = raw_counts
                            st.session_state["analysis_success_msg"] = f"✅ Photo analysis complete: Detected {total_now} vehicles! Records and signals logged."
                    except Exception as e:
                        st.error(f"Image analysis error: {e}")
                else:
                    st.image(selected_video, caption=f"📷 Image Source: {video_source_label} • Click 'Analyze Traffic Photo' on the left.", use_container_width=True)
            else:
                # ================= VIDEO PROCESSING =================
                if run_analysis_clicked:
                    prog = st.progress(0, text="Initializing Fast AI Video Engine...")
                    live_img_placeholder = st.empty()
                    live_counters_placeholder = st.empty()
                    live_ticker_placeholder = st.empty()

                    try:
                        import cv2
                        import core.detection.vision_engine
                        from core.detection.vision_engine import VisionEngine

                        # Cache engine in session state to avoid reloading neural weights into memory
                        if "cached_vision_engine" not in st.session_state:
                            st.session_state["cached_vision_engine"] = VisionEngine(db_manager=db)
                        try:
                            c = db._get_cursor()
                            c.execute("DELETE FROM vehicle_logs")
                            db.connection.commit()
                        except Exception:
                            pass
                        v_engine = st.session_state["cached_vision_engine"]
                        v_engine.reset()
                        sig_controller = AdaptiveSignalController(db_manager=db)

                        cap = cv2.VideoCapture(selected_video)

                        total_cap_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                        raw_fps = cap.get(cv2.CAP_PROP_FPS)
                        try:
                            fps = float(raw_fps)
                            if math.isnan(fps) or fps <= 0:
                                fps = 25.0
                        except Exception:
                            fps = 25.0

                        # Optimal real-time video playback settings (10-20s duration)
                        try:
                            import torch
                            has_gpu = torch.cuda.is_available()
                        except Exception:
                            has_gpu = False
                        frame_step = 1
                        loop_start_time = time.time()

                        # Process complete video length across its entire duration
                        target_total_frames = total_cap_frames if total_cap_frames > 0 else 1500
                        total_duration_sec = target_total_frames / fps if target_total_frames > 0 else 50.0

                        if total_duration_sec <= 30:
                            interval_sec = 0.5
                        elif total_duration_sec <= 90:
                            interval_sec = 1.0
                        else:
                            interval_sec = 2.0

                        frames_done = 0
                        processed_count = 0
                        last_ui_update = 0.0
                        cumulative_counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}
                        session_logged_vehicles = []
                        recent_detected_feed = []
                        timeline_buckets = {}

                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frames_done += 1
                            if frames_done > target_total_frames:
                                break

                            # Process frames smoothly (every frame on GPU, alternate on CPU)
                            if frame_step > 1 and frames_done % frame_step != 0 and frames_done != target_total_frames:
                                continue

                            processed_count += 1
                            pct = int((frames_done / target_total_frames) * 100) if target_total_frames > 0 else min(99, processed_count * 2)

                            # Cloud Downscaling: Resize HD/4K frame to 640px for 4x faster YOLO inference
                            h_orig, w_orig = frame.shape[:2]
                            if w_orig > 640:
                                scale = 640.0 / w_orig
                                frame = cv2.resize(frame, (640, int(h_orig * scale)), interpolation=cv2.INTER_AREA)

                            annotated, stats = v_engine.process_frame(frame, frame_idx=frames_done, record_to_db=True)

                            # Real-time speed governor: sync playback directly to recording FPS (1.0x Real-time)
                            target_elapsed = frames_done / max(1.0, fps)
                            actual_elapsed = time.time() - loop_start_time
                            sleep_needed = target_elapsed - actual_elapsed
                            if sleep_needed > 0.005:
                                time.sleep(min(sleep_needed, 0.045))
                            raw_counts = stats.get("class_counts", {})
                            for k, v in raw_counts.items():
                                cumulative_counts[k] = v

                            # Update background adaptive signal controller
                            for lane_id, count in stats.get("lane_counts", {}).items():
                                sig_controller.update_density(lane_id, count)
                            if processed_count % 5 == 0:
                                sig_controller.tick_second()

                            # Video timeline congestion calculation (traffic wave)
                            active_now = stats.get("active_vehicles", 0)
                            cur_sec = frames_done / fps
                            bucket_idx = round(cur_sec / interval_sec) * interval_sec
                            if bucket_idx not in timeline_buckets:
                                timeline_buckets[bucket_idx] = []
                            timeline_buckets[bucket_idx].append(active_now)

                            # Rate-limited compressed streaming: ensures smooth playback without connection drops
                            cur_time = time.time()
                            if cur_time - last_ui_update >= 0.12 or frames_done >= target_total_frames:
                                last_ui_update = cur_time
                                # Encode to JPEG for fast network transmission over tunnel
                                _, encoded_frame = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                                live_img_placeholder.image(
                                    encoded_frame.tobytes(),
                                    caption=f"⚡ LIVE AI VISION FEED: Frame #{frames_done}/{target_total_frames} ({pct}%) | Real-time Analysis Active",
                                    use_container_width=True
                                )

                                c_cars = cumulative_counts.get("Car", 0)
                                c_bikes = cumulative_counts.get("Motorcycle", 0)
                                c_buses = cumulative_counts.get("Bus", 0)
                                c_trucks = cumulative_counts.get("Truck", 0)
                                total_v_now = c_cars + c_bikes + c_buses + c_trucks

                                active_lane = sig_controller.current_green_lane
                                lane_names = ["North (Lane 1)", "South (Lane 2)", "East (Lane 3)", "West (Lane 4)"]
                                active_lane_name = lane_names[active_lane] if active_lane < len(lane_names) else f"Lane {active_lane+1}"
                                rem_sec = sig_controller.lanes[active_lane].remaining_time

                                live_counters_placeholder.markdown(f'''
                                <div style="background: {th['card_bg']}; border: 1.5px solid {th['card_border']}; border-radius: 12px; padding: 10px 16px; margin-top: 8px; box-shadow: {th['card_shadow']};">
                                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                                        <span style="font-weight: 700; color: {th['text_primary']}; font-size: 0.85rem;">🔴 Live Detections:</span>
                                        <span style="background: {th['nav_bg']}; color: {th['text_primary']}; padding: 3px 10px; border-radius: 8px; font-weight: 700; font-size: 0.8rem;">🚗 Cars: {c_cars}</span>
                                        <span style="background: {th['nav_bg']}; color: {th['text_primary']}; padding: 3px 10px; border-radius: 8px; font-weight: 700; font-size: 0.8rem;">🏍️ Bikes: {c_bikes}</span>
                                        <span style="background: {th['nav_bg']}; color: {th['text_primary']}; padding: 3px 10px; border-radius: 8px; font-weight: 700; font-size: 0.8rem;">🚌 Buses: {c_buses}</span>
                                        <span style="background: {th['nav_bg']}; color: {th['text_primary']}; padding: 3px 10px; border-radius: 8px; font-weight: 700; font-size: 0.8rem;">🚛 Trucks: {c_trucks}</span>
                                        <span style="background: {th['accent']}; color: {th['accent_text']}; padding: 3px 12px; border-radius: 8px; font-weight: 800; font-size: 0.82rem;">Total: {total_v_now}</span>
                                    </div>
                                    <div style="margin-top: 8px; padding-top: 6px; border-top: 1px dashed {th['nav_border']}; display: flex; justify-content: space-between; align-items: center; font-size: 0.78rem;">
                                        <span>🚦 <strong>Background Signal Controller:</strong> Corridor {active_lane_name} is <span style="color: #22C55E; font-weight: 800;">GREEN</span> ({rem_sec}s allocated)</span>
                                        <span style="color: {th['text_muted']};">Adaptive Webster Light Wave Active</span>
                                    </div>
                                </div>
                                ''', unsafe_allow_html=True)

                            if target_total_frames > 0:
                                prog.progress(min(1.0, frames_done / target_total_frames), text=f"Analyzing video: Frame {frames_done}/{target_total_frames} ({pct}%)...")
                        cap.release()
                        prog.empty()

                        # Build smooth video duration congestion curve data
                        video_timeline_points = []
                        for b_sec in sorted(timeline_buckets.keys()):
                            vals = timeline_buckets[b_sec]
                            m, s_rem = divmod(int(b_sec), 60)
                            if interval_sec < 1.0:
                                fraction = int((b_sec % 1.0) * 10)
                                time_label = f"{m:02d}:{s_rem:02d}.{fraction}s"
                            else:
                                time_label = f"{m:02d}:{s_rem:02d}"
                            peak_val = max(vals) if vals else 0
                            video_timeline_points.append({
                                "time_label": time_label,
                                "time_sec": b_sec,
                                "congestion": peak_val
                            })

                        st.session_state["video_congestion_timeline"] = video_timeline_points
                        timeline_cache_path = os.path.join(BASE_DIR, "data", "video_timeline.json")
                        try:
                            import json
                            with open(timeline_cache_path, "w", encoding="utf-8") as f:
                                json.dump(video_timeline_points, f, indent=2)
                        except Exception:
                            pass

                        # Keep final frame and detections persistently visible without abrupt rerun
                        _, final_encoded = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        live_img_placeholder.image(
                            final_encoded.tobytes(),
                            caption=f"✅ FINAL ANALYSIS RESULT: Processed {frames_done} frames successfully! Bounding boxes & signals recorded.",
                            use_container_width=True
                        )
                        st.session_state["session_analysis_counts"] = cumulative_counts
                        st.success(f"✅ AI Video Analysis Complete: Successfully processed {frames_done} frames! All counts and signal timings updated below.")

                    except Exception as e:
                        prog.empty()
                        st.error(f"Inference execution error: {e}")
                else:
                    st.video(selected_video)
                    st.caption(f"📹 Source: {video_source_label} • Click 'Start AI Video Analysis' on the left to analyze the entire video.")
        else:
            st.markdown(f"""
            <div style="background: {th['card_bg']}; border: 1px dashed {th['card_border']}; border-radius: 14px; height: 260px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                <span style="font-size: 2.2rem; color: {th['accent']};">📹</span>
                <span style="color: {th['text_primary']}; font-size: 0.88rem; margin-top: 8px; font-weight: 600;">No video loaded</span>
                <span style="color: {th['text_secondary']}; font-size: 0.76rem; margin-top: 4px;">Upload a video (MP4, AVI) or photo on the left to start AI analysis.</span>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if "analysis_success_msg" in st.session_state:
        st.success(st.session_state.pop("analysis_success_msg"))

    # Active records from DB
    current_db_records = [dict(r) for r in db.get_vehicles(hours=active_hours)] if hasattr(db, "get_vehicles") else []
    
    # 1. FLEET CLASSIFICATION METRICS (0 IF NO ANALYSIS PERFORMED YET)
    st.markdown(f"<h5 style='color: {th['text_primary']}; font-size: 0.88rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin: 1.4rem 0 0.8rem 0;'>Fleet Classification Metrics</h5>", unsafe_allow_html=True)
    c_cols = st.columns(4)

    # Calculate actual counts from current_db_records
    if current_db_records:
        df_now = pd.DataFrame(current_db_records)
        v_counts_map = df_now["vehicle_type"].value_counts().to_dict() if "vehicle_type" in df_now.columns else {}
    else:
        v_counts_map = {}

    classes = [("Cars", "Car"), ("Motorcycles", "Motorcycle"), ("Buses", "Bus"), ("Heavy Trucks", "Truck")]
    for idx, (label, key) in enumerate(classes):
        cnt = v_counts_map.get(key, 0)
        with c_cols[idx]:
            st.markdown(f"""
            <div class="kpi-card" style="padding: 0.9rem 1.1rem;">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{cnt}</div>
                <div class="kpi-meta">YOLO Class &bull; Conf &gt; 0.45</div>
            </div>
            """, unsafe_allow_html=True)

    # 2. TWO NEW CHARTS: VEHICLE CLASSIFICATION BREAKDOWN & TRAFFIC VOLUME OVER TIME
    st.markdown(f"<h5 style='color: {th['text_primary']}; font-size: 0.88rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin: 1.6rem 0 0.8rem 0;'>Traffic Intelligence & Distribution Analytics</h5>", unsafe_allow_html=True)
    g1, g2 = st.columns(2)

    with g1:
        st.markdown(f'<div class="dark-card" style="margin-bottom: 1.2rem;">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size: 0.92rem; font-weight: 700; color: {th['text_primary']}; margin-bottom: 2px;">
            🚗 Fleet Composition by Vehicle Type
        </div>
        <div style="font-size: 0.72rem; color: {th['text_muted']}; margin-bottom: 8px;">
            Proportional distribution of identified vehicle categories
        </div>
        """, unsafe_allow_html=True)

        if current_db_records:
            df_chart = pd.DataFrame(current_db_records)
            if "vehicle_type" in df_chart.columns and not df_chart["vehicle_type"].dropna().empty:
                type_counts = df_chart["vehicle_type"].value_counts().reset_index()
                type_counts.columns = ["Vehicle Type", "Count"]

                fig_type = px.pie(
                    type_counts,
                    names="Vehicle Type",
                    values="Count",
                    hole=0.48,
                    color_discrete_sequence=th['pie_colors']
                )
                fig_type.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=260,
                    showlegend=True,
                    legend=dict(font=dict(color=th['chart_font'], size=11), orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_type, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No vehicle data recorded yet. Upload and analyze a video above.")
        else:
            st.markdown(f"""
            <div style="height: 200px; display: flex; align-items: center; justify-content: center; flex-direction: column; color: {th['text_muted']};">
                <span style="font-size: 2rem;">📊</span>
                <span style="font-size: 0.85rem; margin-top: 6px;">No video analyzed yet.</span>
                <span style="font-size: 0.75rem;">Click 'Start AI Video Analysis' above to generate chart.</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with g2:
        st.markdown(f'<div class="dark-card" style="margin-bottom: 1.2rem;">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size: 0.92rem; font-weight: 700; color: {th['text_primary']}; margin-bottom: 2px;">
            📈 Dynamic Traffic Congestion Wave
        </div>
        <div style="font-size: 0.72rem; color: {th['text_muted']}; margin-bottom: 8px;">
            Real-time vehicle density & congestion curve across video duration
        </div>
        """, unsafe_allow_html=True)

        # Retrieve video duration congestion points
        curve_data = None
        if "video_congestion_timeline" in st.session_state and st.session_state["video_congestion_timeline"]:
            curve_data = st.session_state["video_congestion_timeline"]
        else:
            timeline_cache_path = os.path.join(BASE_DIR, "data", "video_timeline.json")
            if os.path.exists(timeline_cache_path):
                try:
                    import json
                    with open(timeline_cache_path, "r", encoding="utf-8") as f:
                        curve_data = json.load(f)
                except Exception:
                    curve_data = None

        # Fallback to current_db_records if cache is not present but records exist in DB
        if not curve_data and current_db_records:
            try:
                df_time = pd.DataFrame(current_db_records)
                if "timestamp" in df_time.columns and not df_time["timestamp"].dropna().empty:
                    df_time["dt"] = pd.to_datetime(df_time["timestamp"])
                    df_time = df_time.sort_values("dt")
                    t_min = df_time["dt"].min()
                    df_time["rel_sec"] = (df_time["dt"] - t_min).dt.total_seconds().clip(lower=0)
                    max_rel = df_time["rel_sec"].max()
                    step = max(1.0, max_rel / 15.0) if max_rel > 0 else 1.0
                    df_time["time_bin"] = (df_time["rel_sec"] // step) * step
                    bin_counts = df_time.groupby("time_bin").size().reset_index(name="count")
                    curve_data = []
                    for _, row in bin_counts.iterrows():
                        sec_val = int(row["time_bin"])
                        m, s = divmod(sec_val, 60)
                        curve_data.append({
                            "time_label": f"{m:02d}:{s:02d}",
                            "time_sec": sec_val,
                            "congestion": int(row["count"])
                        })
            except Exception:
                curve_data = None

        if current_db_records and curve_data:
            try:
                x_vals = [p["time_label"] for p in curve_data]
                y_vals = [p["congestion"] for p in curve_data]

                fig_wave = go.Figure()
                fig_wave.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines+markers',
                    name='Congestion Density',
                    line=dict(color=th['accent'], width=3.2, shape='spline', smoothing=1.3),
                    marker=dict(size=6, color=th['accent'], line=dict(color=th['app_bg'], width=1.5)),
                    fill='tozeroy',
                    fillcolor=th['chart_fill'],
                    hovertemplate='<b>Video Time: %{x}</b><br>Active Congestion: <b>%{y} vehicles</b><extra></extra>'
                ))

                max_cong = max(y_vals) if y_vals else 0
                peak_idx = y_vals.index(max_cong) if y_vals and max_cong > 0 else 0
                peak_time = x_vals[peak_idx] if x_vals else "00:00"

                fig_wave.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=th['chart_font'], family="Plus Jakarta Sans, sans-serif", size=10),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=260,
                    xaxis=dict(
                        showgrid=False,
                        showticklabels=False,
                        title=dict(text="Continuous Video Duration Timeline (Hover cursor for real-time density)", font=dict(color=th['text_muted'], size=10))
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor=th['chart_grid'],
                        tickfont=dict(color=th['text_muted'], size=10),
                        title=dict(text="Vehicle Congestion", font=dict(color=th['text_muted'], size=10)),
                        zeroline=False,
                        rangemode="tozero"
                    )
                )
                st.plotly_chart(fig_wave, use_container_width=True, config={"displayModeBar": False})

                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.74rem; color: {th['text_secondary']}; padding-top: 4px; border-top: 1px solid {th['card_border']};">
                    <span>📈 Peak Wave: <strong style="color: {th['text_primary']};">{max_cong} vehicles</strong> at <strong>{peak_time}</strong></span>
                    <span>⏱️ Video Span: <strong style="color: {th['text_primary']};">{x_vals[-1] if x_vals else '0s'}</strong></span>
                </div>
                """, unsafe_allow_html=True)

            except Exception as ex:
                st.info(f"Timeline wave rendering in progress ({ex}).")
        else:
            st.markdown(f"""
            <div style="height: 200px; display: flex; align-items: center; justify-content: center; flex-direction: column; color: {th['text_muted']};">
                <span style="font-size: 2rem;">📈</span>
                <span style="font-size: 0.85rem; margin-top: 6px;">No video analyzed yet.</span>
                <span style="font-size: 0.75rem;">Click 'Start AI Video Analysis' above to generate timeline curve.</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. RECENT VEHICLE TRANSITS LEDGER (WITH EXCEL, PDF & CSV DOWNLOAD OPTIONS)
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin: 1.4rem 0 0.6rem 0;">
        <h5 style="color: {th['text_primary']}; font-size: 0.88rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin: 0;">
            Recent Vehicle Transits Ledger
        </h5>
    </div>
    """, unsafe_allow_html=True)

    if current_db_records:
        df_v = pd.DataFrame(current_db_records)
        if "id" in df_v.columns:
            df_v = df_v.sort_values("id", ascending=True).reset_index(drop=True)

        # Generate clear sequential vehicle numbering (1st, 2nd, 3rd Vehicle...)
        ordinal_labels = []
        for i in range(1, len(df_v) + 1):
            if 11 <= (i % 100) <= 13:
                suffix = "th"
            else:
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
            ordinal_labels.append(f"Vehicle #{i:02d} ({i}{suffix} Vehicle)")

        def _fmt_conf(val):
            try:
                v = float(val)
                return f"{v*100:.1f}%" if v <= 1.0 else f"{v:.1f}%"
            except Exception:
                return "95.0%"

        display_df = pd.DataFrame({
            "Vehicle No.": ordinal_labels,
            "Vehicle Type": df_v["vehicle_type"] if "vehicle_type" in df_v.columns else "Car",
            "AI Confidence": df_v["confidence"].apply(_fmt_conf) if "confidence" in df_v.columns else ["95.0%"] * len(df_v)
        })
        st.dataframe(display_df, use_container_width=True, height=280)

        # DOWNLOAD TOOLBAR FOR VEHICLE TRANSITS LEDGER (EXCEL, PDF, CSV)
        st.markdown(f"""
        <div style="background: {th['card_bg']}; border: 1px solid {th['card_border']}; border-radius: 12px; padding: 12px 16px; margin-top: 10px; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px;">
                <div style="color: {th['text_primary']}; font-size: 0.82rem; font-weight: 700;">
                    📥 Download Vehicle Transits Ledger:
                </div>
                <div style="color: {th['text_muted']}; font-size: 0.72rem;">
                    Export Formats: Multi-sheet Excel (.xlsx), Certified PDF Document (.pdf), Raw CSV (.csv)
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        down_c1, down_c2, down_c3 = st.columns(3)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Excel (.xlsx) Download
        with down_c1:
            try:
                import io
                xl_buf = io.BytesIO()
                with pd.ExcelWriter(xl_buf, engine="openpyxl") as writer:
                    display_df.to_excel(writer, sheet_name="Vehicle_Transits", index=False)
                xl_data = xl_buf.getvalue()
                st.download_button(
                    label="📊 Download Excel (.xlsx)",
                    data=xl_data,
                    file_name=f"vehicle_transits_{timestamp_str}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="btn_down_transits_excel"
                )
            except Exception as ex_xl:
                st.error(f"Excel export error: {ex_xl}")

        # 2. PDF (.pdf) Download
        with down_c2:
            try:
                from fpdf import FPDF
                from fpdf.enums import XPos, YPos

                pdf = FPDF(orientation="P", unit="mm", format="A4")
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.add_page()

                # Header Title
                pdf.set_font("helvetica", "B", 16)
                pdf.set_text_color(45, 37, 32)
                pdf.cell(0, 10, "SMART TRAFFIC MANAGEMENT SYSTEM", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

                pdf.set_font("helvetica", "B", 11)
                pdf.set_text_color(168, 134, 114)
                pdf.cell(0, 6, "Recent Vehicle Transits Ledger - Audit Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

                pdf.set_font("helvetica", "", 8.5)
                pdf.set_text_color(100, 116, 139)
                pdf.cell(0, 5, f"Observation Records: {len(display_df)} | Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
                pdf.ln(4)

                # Column headers
                cols = list(display_df.columns)
                col_widths_map = {
                    "Vehicle No.": 70,
                    "Vehicle Type": 60,
                    "AI Confidence": 60
                }
                widths = [col_widths_map.get(c, 50) for c in cols]
                total_w = sum(widths)
                if total_w > 0:
                    scale = 190.0 / total_w
                    widths = [w * scale for w in widths]

                pdf.set_fill_color(232, 220, 205)
                pdf.set_text_color(45, 37, 32)
                pdf.set_font("helvetica", "B", 8.5)
                for i, col in enumerate(cols):
                    is_last = (i == len(cols) - 1)
                    pdf.cell(
                        widths[i], 7, str(col), border=1, fill=True,
                        new_x=XPos.LMARGIN if is_last else XPos.RIGHT,
                        new_y=YPos.NEXT if is_last else YPos.TOP,
                        align="C"
                    )

                pdf.set_font("helvetica", "", 8)
                pdf.set_text_color(30, 41, 59)
                fill_row = False
                for _, row in display_df.iterrows():
                    if fill_row:
                        pdf.set_fill_color(248, 245, 241)
                    else:
                        pdf.set_fill_color(255, 255, 255)
                    for i, col in enumerate(cols):
                        is_last = (i == len(cols) - 1)
                        val = str(row[col])
                        if col == "Timestamp" and len(val) > 19:
                            val = val[:19]
                        if col == "AI Confidence":
                            try:
                                val = f"{float(val)*100:.1f}%" if float(val) <= 1.0 else f"{float(val):.1f}%"
                            except Exception:
                                pass
                        pdf.cell(
                            widths[i], 6, val, border=1, fill=True,
                            new_x=XPos.LMARGIN if is_last else XPos.RIGHT,
                            new_y=YPos.NEXT if is_last else YPos.TOP,
                            align="C"
                        )
                    fill_row = not fill_row

                pdf_bytes = bytes(pdf.output())
                st.download_button(
                    label="📄 Download PDF (.pdf)",
                    data=pdf_bytes,
                    file_name=f"vehicle_transits_{timestamp_str}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="btn_down_transits_pdf"
                )
            except Exception as ex_pdf:
                st.error(f"PDF export error: {ex_pdf}")

        # 3. CSV (.csv) Download
        with down_c3:
            try:
                csv_bytes = display_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download CSV (.csv)",
                    data=csv_bytes,
                    file_name=f"vehicle_transits_{timestamp_str}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="btn_down_transits_csv"
                )
            except Exception as ex_csv:
                st.error(f"CSV export error: {ex_csv}")

    else:
        st.info("No transit records available yet. Upload and analyze a video above to populate ledger.")

# =====================================================================
# 3. ADAPTIVE TRAFFIC LIGHT SIGNAL
# =====================================================================
elif selected_nav == "Adaptive Traffic Light Signal":
    # =========================================================================
    # SMART TRAFFIC MANAGEMENT SYSTEM - ADAPTIVE TRAFFIC LIGHT SIGNAL
    # =========================================================================

    # 4-Way Intersection Real-Time Simulation Deck
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin: 0.9rem 0 0.6rem 0;">
        <div style="color: {th['text_primary']}; font-size: 0.92rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; display: flex; align-items: center; gap: 8px;">
            <span>🚦</span><span>4-Way Intersection Live Adaptive Simulation</span>
        </div>
        <span style="color: {th['accent'] if active_theme_name == 'Soft Sand' else '#00f0ff'}; font-size: 0.76rem; font-weight: 700;">
            Real-time Waiting Queue Indicators &bull; Auto Dynamic Clearance
        </span>
    </div>
    """, unsafe_allow_html=True)

    sim_code = get_simulation_html(active_theme_name)
    components.html(sim_code, height=1280, scrolling=True)

# =====================================================================
# 3. CONTACT US / REPORT A PROBLEM (DEDICATED SECTION)
# =====================================================================
elif selected_nav == "Contact Us / Report a Problem":
    # ---- EmailJS Configuration (using existing account credentials) ----
    EMAILJS_SERVICE_ID = "service_hjzsn55"
    EMAILJS_TEMPLATE_ID = "template_7i98nki"
    EMAILJS_PUBLIC_KEY = "AFcI5Trke2PpJ6-fe"

    st.markdown(f"""
    <div style="
        max-width: 820px;
        margin: 0 auto 1.2rem auto;
        text-align: center;
    ">
        <h2 style="
            color: {th['text_primary']};
            font-size: 1.65rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.3px;
        ">Need Help? Contact Us</h2>
    </div>
    """, unsafe_allow_html=True)

    # Contact form
    with st.form("contact_support_form", clear_on_submit=True):
        st.markdown(f"""
        <div style="
            background: {th['card_bg']};
            border: 1px solid {th['card_border']};
            border-radius: 16px;
            padding: 1.5rem 1.8rem 0.8rem 1.8rem;
            box-shadow: {th['card_shadow']};
            margin-bottom: 0.5rem;
        ">
            <div style="
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 1.2rem;
                padding-bottom: 0.8rem;
                border-bottom: 1px solid {th['card_border']};
            ">
                <span style="font-size: 1.1rem;">📋</span>
                <span style="
                    color: {th['text_primary']};
                    font-size: 0.85rem;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                ">Support Request Form</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        form_c1, form_c2 = st.columns(2)
        with form_c1:
            contact_name = st.text_input(
                "Full Name *",
                placeholder="Enter your full name",
                key="contact_name_input"
            )
        with form_c2:
            contact_email = st.text_input(
                "Email Address *",
                placeholder="name@example.com",
                key="contact_email_input"
            )

        form_c3, form_c4 = st.columns(2)
        with form_c3:
            contact_subject = st.text_input(
                "Subject *",
                placeholder="Brief subject of your message",
                key="contact_subject_input"
            )
        with form_c4:
            contact_category = st.selectbox(
                "Problem Category *",
                options=["Technical Issue", "Bug Report", "Suggestion", "Other"],
                key="contact_category_select"
            )

        contact_message = st.text_area(
            "Message / Describe Your Problem *",
            placeholder="Describe the issue or suggestion in detail...",
            height=140,
            key="contact_message_area"
        )

        submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
        with submit_col2:
            submitted = st.form_submit_button(
                "📨 Send Message",
                use_container_width=True,
                type="primary"
            )

    if submitted:
        # ---- Validation ----
        import re as _re
        email_regex = _re.compile(r'^[^\s@]+@[^\s@]+\.[^\s@]+$')
        errors = []

        if not contact_name or len(contact_name.strip()) < 2:
            errors.append("Full Name is required (at least 2 characters).")
        if not contact_email or not email_regex.match(contact_email.strip()):
            errors.append("A valid Email Address is required.")
        if not contact_subject or len(contact_subject.strip()) < 3:
            errors.append("Subject is required (at least 3 characters).")
        if not contact_message or len(contact_message.strip()) < 5:
            errors.append("Message is required (at least 5 characters).")

        if errors:
            for err in errors:
                st.error(f"⚠️ {err}")
        else:
            # ---- Prepare EmailJS payload ----
            template_params = {
                "name": contact_name.strip(),
                "from_name": contact_name.strip(),
                "user_name": contact_name.strip(),
                "email": contact_email.strip(),
                "from_email": contact_email.strip(),
                "reply_to": contact_email.strip(),
                "user_email": contact_email.strip(),
                "subject": f"[AIBST Dashboard] {contact_subject.strip()}",
                "title": contact_subject.strip(),
                "message": f"Category: {contact_category}\n\n{contact_message.strip()}",
                "message_html": f"<b>Category:</b> {contact_category}<br><br>{contact_message.strip()}"
            }

            payload = json.dumps({
                "service_id": EMAILJS_SERVICE_ID,
                "template_id": EMAILJS_TEMPLATE_ID,
                "user_id": EMAILJS_PUBLIC_KEY,
                "template_params": template_params
            }).encode("utf-8")

            headers_dict = {
                "Content-Type": "application/json",
                "Origin": "http://localhost:8501",
                "Referer": "http://localhost:8501/",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            try:
                req = urllib.request.Request(
                    "https://api.emailjs.com/api/v1.0/email/send",
                    data=payload,
                    headers=headers_dict,
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    resp_body = resp.read().decode("utf-8", errors="replace")

                if resp.status == 200:
                    st.success(f"✅ Thank you, **{contact_name.strip()}**! Your message has been sent successfully. We will respond to **{contact_email.strip()}** shortly.")
                else:
                    st.error(f"❌ Unexpected response (HTTP {resp.status}): {resp_body}")

            except urllib.error.HTTPError as http_err:
                error_body = http_err.read().decode("utf-8", errors="replace")
                if "Invalid grant" in error_body:
                    st.error("❌ Gmail connection needs to be refreshed in the EmailJS dashboard. Please try again later.")
                else:
                    st.error(f"❌ Failed to send message (HTTP {http_err.code}): {error_body}")
            except urllib.error.URLError as url_err:
                st.error(f"❌ Network error: Could not reach the email service. Please check your internet connection.")
            except Exception as ex:
                st.error(f"❌ An unexpected error occurred: {ex}")

# =====================================================================
# GLOBAL FOOTER BRANDING (Always visible at the bottom of the current view)
# =====================================================================
st.markdown(f"""
<div style="
    text-align: center;
    padding: 2.2rem 0 1.5rem 0;
    margin-top: 2.5rem;
    border-top: 1px solid {th['card_border']};
">
    <p style="
        color: {th['text_muted']};
        font-size: 0.72rem;
        margin: 0;
        letter-spacing: 0.3px;
    ">
        AIBST Traffic AI &bull; Smart Traffic Management System &bull; v2.5
        <br>
        <span style="opacity: 0.7;">Built with ❤️ by Soumyadip Pradhan</span>
    </p>
</div>
""", unsafe_allow_html=True)
