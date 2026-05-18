"""Reusable color card component for Streamlit."""
import streamlit as st
from typing import Dict


def render_color_card(color_info: Dict, show_palette: bool = True) -> None:
    """
    Render a premium color information card.

    Args:
        color_info: Dict with rgb, hsv, hex, color_name, color_family.
        show_palette: Show a colored swatch block.
    """
    r = color_info.get("rgb", {}).get("R", 0)
    g = color_info.get("rgb", {}).get("G", 0)
    b = color_info.get("rgb", {}).get("B", 0)
    hex_code = color_info.get("hex", "#000000")
    color_name = color_info.get("color_name", "Unknown")
    color_family = color_info.get("color_family", "")
    hsv = color_info.get("hsv", {})

    # Determine text color for contrast on swatch
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    text_color = "#000000" if luminance > 128 else "#FFFFFF"

    if show_palette:
        st.markdown(
            f"""
            <div style="
                background: {hex_code};
                border-radius: 14px;
                padding: 28px 20px;
                text-align: center;
                margin-bottom: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            ">
                <div style="color:{text_color}; font-size:1.3rem; font-weight:700; letter-spacing:2px;">
                    {hex_code}
                </div>
                <div style="color:{text_color}; font-size:1rem; opacity:0.85; margin-top:4px;">
                    {color_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 Red", r)
        st.metric("🟢 Green", g)
        st.metric("🔵 Blue", b)
    with col2:
        st.metric("🌈 Hue", hsv.get("H", "—"))
        st.metric("💧 Saturation", hsv.get("S", "—"))
        st.metric("☀️ Value", hsv.get("V", "—"))
    with col3:
        st.metric("🏷️ Name", color_name)
        st.metric("🗂️ Family", color_family or "—")
        st.metric("🔑 HEX", hex_code)
