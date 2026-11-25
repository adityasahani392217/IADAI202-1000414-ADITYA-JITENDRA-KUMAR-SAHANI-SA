"""Shared formatting, CSS theme injection, and UI helper utilities for MediScope AI."""

from __future__ import annotations

import streamlit as st

from src.config import PALETTE


def format_currency(value: float) -> str:
    """Format a dollar value compactly or cleanly."""
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    return f"${value:,.0f}"


def format_number(value: float | int) -> str:
    """Format an integer count with thousands separators."""
    return f"{int(round(value)):,}"


def inject_custom_css() -> None:
    """Inject the LogiSight / MediScope AI design system CSS into Streamlit."""
    st.markdown(
        f"""
        <style>
        /* Main layout & typography */
        .stApp {{
            background-color: {PALETTE["bg_neutral"]};
            color: {PALETTE["text_primary"]};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {PALETTE["navy"]};
            color: {PALETTE["white"]};
        }}
        section[data-testid="stSidebar"] * {{
            color: {PALETTE["white"]};
        }}
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stMultiSelect label,
        section[data-testid="stSidebar"] .stSlider label {{
            color: {PALETTE["light_teal"]} !important;
            font-weight: 600;
            font-size: 0.85rem;
        }}
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
            background-color: #1A315E;
            border-color: #2D4A7F;
            color: {PALETTE["white"]};
        }}
        section[data-testid="stSidebar"] div[data-baseweb="select"] span {{
            color: {PALETTE["white"]} !important;
        }}
        section[data-testid="stSidebar"] .stButton > button {{
            background-color: {PALETTE["teal"]};
            color: {PALETTE["white"]};
            border: none;
            border-radius: 8px;
            font-weight: 600;
        }}

        /* Header banner */
        .mediscope-header {{
            background: {PALETTE["white"]};
            border-left: 5px solid {PALETTE["teal"]};
            border-radius: 10px;
            padding: 1.1rem 1.4rem;
            margin-bottom: 1.1rem;
            border-top: 1px solid {PALETTE["border"]};
            border-right: 1px solid {PALETTE["border"]};
            border-bottom: 1px solid {PALETTE["border"]};
        }}
        .mediscope-title {{
            color: {PALETTE["navy"]};
            font-size: 1.65rem;
            font-weight: 700;
            margin: 0;
            line-height: 1.2;
        }}
        .mediscope-subtitle {{
            color: {PALETTE["grey"]};
            font-size: 0.92rem;
            margin-top: 0.25rem;
            margin-bottom: 0;
        }}

        /* KPI cards */
        .kpi-card {{
            background: {PALETTE["white"]};
            border: 1px solid {PALETTE["border"]};
            border-top: 3px solid {PALETTE["teal"]};
            border-radius: 10px;
            padding: 0.9rem 1rem;
            box-shadow: 0 1px 2px rgba(16, 35, 75, 0.04);
        }}
        .kpi-card.navy-accent {{
            border-top-color: {PALETTE["navy"]};
        }}
        .kpi-label {{
            color: {PALETTE["grey"]};
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 0.25rem;
        }}
        .kpi-value {{
            color: {PALETTE["navy"]};
            font-size: 1.45rem;
            font-weight: 700;
            line-height: 1.2;
        }}
        .kpi-delta {{
            color: {PALETTE["teal_dark"]};
            font-size: 0.78rem;
            margin-top: 0.25rem;
        }}

        /* Insight box */
        .insight-box {{
            background-color: {PALETTE["light_teal"]};
            border-left: 4px solid {PALETTE["teal"]};
            border-radius: 8px;
            padding: 0.85rem 1.1rem;
            color: {PALETTE["navy"]};
            font-size: 0.92rem;
            margin: 0.6rem 0 1rem 0;
        }}

        /* Mode badge */
        .mode-badge {{
            display: inline-block;
            background: {PALETTE["light_teal"]};
            color: {PALETTE["teal_dark"]};
            border: 1px solid {PALETTE["teal"]};
            border-radius: 999px;
            padding: 0.15rem 0.65rem;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str, mode_label: str | None = None) -> None:
    """Render a consistent header card across dashboard pages."""
    badge_html = f'<span class="mode-badge">{mode_label}</span>' if mode_label else ""
    html = (
        '<div class="mediscope-header">'
        '<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">'
        f'<div><h1 class="mediscope-title">{title}</h1>'
        f'<p class="mediscope-subtitle">{subtitle}</p></div>'
        f"{badge_html}"
        "</div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, subtext: str = "", navy: bool = False) -> None:
    """Render a styled KPI metric card."""
    accent_cls = "navy-accent" if navy else ""
    html = (
        f'<div class="kpi-card {accent_cls}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-delta">{subtext}</div>'
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)
