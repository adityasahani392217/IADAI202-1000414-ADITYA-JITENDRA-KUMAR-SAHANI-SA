"""Shared formatting, CSS theme injection, branding, and UI helper utilities."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from src.config import PALETTE, PROJECT_ROOT

LOGO_SIDEBAR_PATH = PROJECT_ROOT / "assets" / "mediscope_logo.svg"
LOGO_ICON_PATH = PROJECT_ROOT / "assets" / "mediscope_icon.svg"


def _inline_icon_data_uri() -> str:
    """Return base64 data URI for the MediScope AI icon SVG."""
    if LOGO_ICON_PATH.exists():
        raw = LOGO_ICON_PATH.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/svg+xml;base64,{b64}"
    return ""


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
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }}
        section[data-testid="stSidebar"] {{
            background-color: {PALETTE["navy"]};
            color: {PALETTE["white"]};
        }}
        section[data-testid="stSidebar"] * {{
            color: {PALETTE["white"]};
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
            border-radius: 8px;
            margin: 1px 6px;
            transition: background-color 0.15s ease;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {{
            background-color: rgba(13, 148, 136, 0.25);
        }}
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stMultiSelect label,
        section[data-testid="stSidebar"] .stSlider label {{
            color: {PALETTE["light_teal"]} !important;
            font-weight: 600;
            font-size: 0.84rem;
        }}
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
            background-color: #1A315E;
            border-color: #2D4A7F;
            color: {PALETTE["white"]};
            border-radius: 8px;
        }}
        section[data-testid="stSidebar"] div[data-baseweb="select"] span {{
            color: {PALETTE["white"]} !important;
        }}
        section[data-testid="stSidebar"] .stButton > button,
        section[data-testid="stSidebar"] .stDownloadButton > button {{
            background-color: {PALETTE["teal"]};
            color: {PALETTE["white"]};
            border: none;
            border-radius: 8px;
            font-weight: 600;
            transition: background-color 0.15s ease;
        }}
        section[data-testid="stSidebar"] .stButton > button:hover,
        section[data-testid="stSidebar"] .stDownloadButton > button:hover {{
            background-color: {PALETTE["teal_dark"]};
        }}

        /* Header banner */
        .mediscope-header {{
            background: {PALETTE["white"]};
            border-left: 5px solid {PALETTE["teal"]};
            border-radius: 12px;
            padding: 1.05rem 1.35rem;
            margin-bottom: 1rem;
            border-top: 1px solid {PALETTE["border"]};
            border-right: 1px solid {PALETTE["border"]};
            border-bottom: 1px solid {PALETTE["border"]};
            box-shadow: 0 1px 3px rgba(16, 35, 75, 0.05);
        }}
        .mediscope-title {{
            color: {PALETTE["navy"]};
            font-size: 1.55rem;
            font-weight: 700;
            margin: 0;
            line-height: 1.25;
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
            border-radius: 12px;
            padding: 0.95rem 1.05rem;
            box-shadow: 0 1px 3px rgba(16, 35, 75, 0.05);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 10px rgba(16, 35, 75, 0.08);
        }}
        .kpi-card.navy-accent {{
            border-top-color: {PALETTE["navy"]};
        }}
        .kpi-label {{
            color: {PALETTE["grey"]};
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
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
            font-size: 0.79rem;
            margin-top: 0.28rem;
        }}

        /* Insight & guide boxes */
        .insight-box {{
            background-color: {PALETTE["light_teal"]};
            border-left: 4px solid {PALETTE["teal"]};
            border-radius: 10px;
            padding: 0.85rem 1.1rem;
            color: {PALETTE["navy"]};
            font-size: 0.92rem;
            margin: 0.65rem 0 1rem 0;
            line-height: 1.48;
        }}
        .guide-box {{
            background-color: {PALETTE["white"]};
            border: 1px solid {PALETTE["border"]};
            border-left: 4px solid {PALETTE["navy"]};
            border-radius: 10px;
            padding: 0.75rem 1.05rem;
            color: {PALETTE["text_primary"]};
            font-size: 0.89rem;
            margin: 0.4rem 0 0.9rem 0;
            line-height: 1.45;
        }}
        .filter-banner {{
            background-color: #FEF3C7;
            border: 1px solid #F59E0B;
            border-radius: 8px;
            padding: 0.5rem 0.9rem;
            color: #78350F;
            font-size: 0.86rem;
            font-weight: 500;
            margin-bottom: 0.85rem;
        }}

        /* Mode badge */
        .mode-badge {{
            display: inline-block;
            background: {PALETTE["light_teal"]};
            color: {PALETTE["teal_dark"]};
            border: 1px solid {PALETTE["teal"]};
            border-radius: 999px;
            padding: 0.2rem 0.75rem;
            font-size: 0.76rem;
            font-weight: 600;
            white-space: nowrap;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str, mode_label: str | None = None) -> None:
    """Render a consistent header card with the MediScope AI logo icon across pages."""
    icon_uri = _inline_icon_data_uri()
    icon_html = (
        f'<img src="{icon_uri}" alt="MediScope AI" style="width:40px;height:40px;border-radius:10px;flex-shrink:0;">'
        if icon_uri
        else ""
    )
    badge_html = f'<span class="mode-badge">{mode_label}</span>' if mode_label else ""
    html = (
        '<div class="mediscope-header">'
        '<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.75rem;">'
        '<div style="display:flex; align-items:center; gap:0.85rem;">'
        f"{icon_html}"
        f'<div><h1 class="mediscope-title">{title}</h1>'
        f'<p class="mediscope-subtitle">{subtitle}</p></div>'
        "</div>"
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


def render_plain_english_guide(heading: str, body: str) -> None:
    """Render a concise plain-English explanation box to make charts easy to understand."""
    html = (
        '<div class="guide-box">'
        f"<strong>{heading}:</strong> {body}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_active_filter_banner(summary_text: str | None) -> None:
    """Display an explicit banner with an inline SVG filter icon when global filters are active."""
    if summary_text:
        filter_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" '
            'fill="none" stroke="#92400E" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" '
            'style="vertical-align:-2px; margin-right:6px;">'
            '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon></svg>'
        )
        st.markdown(
            f'<div class="filter-banner">{filter_svg}<strong>Active Global Filters:</strong> {summary_text}</div>',
            unsafe_allow_html=True,
        )
