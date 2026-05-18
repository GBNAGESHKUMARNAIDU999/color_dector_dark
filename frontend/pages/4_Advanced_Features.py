"""
Page 4: Advanced AI Features
Fashion matcher, interior design recommender, traffic light detector, fruit ripeness.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import streamlit as st
import numpy as np
import cv2

from frontend.components.sidebar import apply_global_styles, render_sidebar
from frontend.components.palette_chart import render_palette_swatches, render_color_bar
from frontend.components.color_card import render_color_card
from ml.color_recommender import fashion_color_matcher, interior_palette_recommender, generate_all_palettes
from cv_pipeline.preprocessor import bytes_to_image, preprocess_image
from cv_pipeline.color_detector import detect_colors
from ml.kmeans_clustering import extract_dominant_colors
from cv_pipeline.color_classifier import get_color_name

st.set_page_config(page_title="Advanced AI — ColorAI", page_icon="🤖", layout="wide")
apply_global_styles()
render_sidebar()

st.markdown("""
<h1 style="background:linear-gradient(135deg,#059669,#10b981);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;
font-size:2.4rem;font-weight:800;">🤖 Advanced AI Features</h1>
<p style="color:#8888bb;margin-top:-8px;">
AI-powered fashion color matching, interior design recommendations, and smart image analysis.
</p>""", unsafe_allow_html=True)

tab_fashion, tab_interior, tab_traffic, tab_fruit, tab_palette = st.tabs([
    "👗 Fashion Matcher",
    "🏠 Interior Designer",
    "🚦 Traffic Light Detector",
    "🍎 Fruit Ripeness",
    "🌈 Full Palette Studio",
])

# ── Tab 1: Fashion Matcher ────────────────────────────────────────────────────
with tab_fashion:
    st.markdown("### 👗 AI Fashion Color Matcher")
    st.markdown("Select your clothing color to get AI-powered outfit pairing recommendations.")

    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        st.markdown("**Choose Base Color**")
        r_f = st.slider("Red", 0, 255, 220, key="f_r")
        g_f = st.slider("Green", 0, 255, 50, key="f_g")
        b_f = st.slider("Blue", 0, 255, 80, key="f_b")

        swatch_html = f"""
        <div style="background:rgb({r_f},{g_f},{b_f});border-radius:12px;
        height:100px;margin:12px 0;box-shadow:0 4px 20px rgba(0,0,0,0.4);"></div>
        """
        st.markdown(swatch_html, unsafe_allow_html=True)

        if st.button("✨ Get Fashion Recommendations", use_container_width=True, key="fashion_btn"):
            st.session_state["fashion_result"] = fashion_color_matcher(r_f, g_f, b_f)

    with col_f2:
        if "fashion_result" in st.session_state:
            result = st.session_state["fashion_result"]
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04);border-radius:12px;padding:20px;margin-bottom:16px;">
                <span style="font-size:0.9rem;color:#aaaacc;">Detected Tone</span><br>
                <span style="font-size:1.8rem;font-weight:700;color:#10b981;">
                    {result['tone'].upper()}
                </span><br>
                <span style="color:#8888aa;font-size:0.85rem;margin-top:8px;display:block;">
                    {result['styling_tip']}
                </span>
            </div>
            """, unsafe_allow_html=True)

            for ptype, pcolors in result["recommended_palettes"].items():
                st.markdown(f"**{ptype.replace('_',' ').title()} Palette**")
                render_palette_swatches(pcolors)
                st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.info("👈 Select a color and click **Get Fashion Recommendations**")

# ── Tab 2: Interior Designer ─────────────────────────────────────────────────
with tab_interior:
    st.markdown("### 🏠 Interior Design Color Recommender")

    col_i1, col_i2 = st.columns([1, 2])
    with col_i1:
        r_i = st.slider("Red", 0, 255, 70, key="i_r")
        g_i = st.slider("Green", 0, 255, 130, key="i_g")
        b_i = st.slider("Blue", 0, 255, 180, key="i_b")
        mood = st.selectbox("Room Mood", ["calm", "energetic", "elegant", "bold", "balanced"])

        st.markdown(f"""
        <div style="background:rgb({r_i},{g_i},{b_i});border-radius:12px;
        height:80px;margin:10px 0;box-shadow:0 4px 20px rgba(0,0,0,0.4);"></div>
        """, unsafe_allow_html=True)

        if st.button("🏠 Generate Interior Palette", use_container_width=True, key="interior_btn"):
            st.session_state["interior_result"] = interior_palette_recommender(r_i, g_i, b_i, mood)

    with col_i2:
        if "interior_result" in st.session_state:
            res = st.session_state["interior_result"]
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04);border-radius:12px;padding:20px;margin-bottom:16px;">
                <b style="color:#10b981;">{res['mood'].upper()} MOOD</b> —
                <span style="color:#aaaacc;">{res['palette_type'].replace('_',' ').title()}</span><br>
                <span style="color:#8888aa;font-size:0.85rem;margin-top:8px;display:block;">
                    💡 {res['design_tip']}
                </span>
            </div>
            """, unsafe_allow_html=True)
            render_palette_swatches(res["palette"])
        else:
            st.info("👈 Pick a wall color and mood, then click **Generate Interior Palette**")

# ── Tab 3: Traffic Light Detector ────────────────────────────────────────────
with tab_traffic:
    st.markdown("### 🚦 Traffic Light Color Detector")
    st.markdown("Upload an image containing a traffic light to detect its state.")

    uploaded_tl = st.file_uploader("Upload traffic light image", type=["jpg","jpeg","png"], key="tl_upload")
    if uploaded_tl:
        img_bytes = uploaded_tl.read()
        img = bytes_to_image(img_bytes)
        if img is not None:
            processed = preprocess_image(img)
            col_tl1, col_tl2 = st.columns(2)

            with col_tl1:
                st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), caption="Input Image", use_container_width=True)

            with col_tl2:
                detections = detect_colors(processed, target_colors=["red", "yellow", "green"])
                if detections:
                    # Determine strongest signal
                    scores = {c: sum(d.pixel_count for d in dets) for c, dets in detections.items()}
                    dominant_signal = max(scores, key=scores.get)

                    STATES = {
                        "red":    ("🔴 RED — STOP",       "#FF3333", "⛔ Traffic light is RED. Vehicles must stop."),
                        "yellow": ("🟡 YELLOW — CAUTION", "#FFCC00", "⚠️ Traffic light is YELLOW. Prepare to stop."),
                        "green":  ("🟢 GREEN — GO",       "#33CC66", "✅ Traffic light is GREEN. Safe to proceed."),
                    }
                    label, color, msg = STATES.get(dominant_signal, ("Unknown", "#888", "Cannot determine state."))

                    st.markdown(f"""
                    <div style="background:{color}22;border:2px solid {color};border-radius:14px;
                    padding:28px;text-align:center;margin-top:20px;">
                        <div style="font-size:2.5rem;margin-bottom:8px;">{label}</div>
                        <div style="color:#cccccc;font-size:0.9rem;">{msg}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("**Detection Scores:**")
                    for c, score in scores.items():
                        st.metric(c.upper(), f"{score:,} px")
                else:
                    st.warning("No traffic light colors (red/yellow/green) detected in this image.")
    else:
        st.info("Upload an image of a traffic light to analyze its state.")

# ── Tab 4: Fruit Ripeness ─────────────────────────────────────────────────────
with tab_fruit:
    st.markdown("### 🍎 Fruit Ripeness Detector")
    st.markdown("Upload a photo of fruit to estimate ripeness based on color analysis.")

    uploaded_fruit = st.file_uploader("Upload fruit image", type=["jpg","jpeg","png"], key="fruit_upload")
    if uploaded_fruit:
        img_bytes = uploaded_fruit.read()
        img = bytes_to_image(img_bytes)
        if img is not None:
            processed = preprocess_image(img)
            col_fr1, col_fr2 = st.columns(2)

            with col_fr1:
                st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), caption="Fruit Image", use_container_width=True)

            with col_fr2:
                dominant = extract_dominant_colors(processed, k=3)
                render_color_bar(dominant)

                # Ripeness heuristic based on dominant hue
                primary = dominant[0] if dominant else None
                if primary:
                    h = primary["hsv"]["H"]
                    s = primary["hsv"]["S"]
                    v = primary["hsv"]["V"]

                    # Hue-based ripeness mapping (OpenCV scale: H 0-179)
                    if 0 <= h <= 8 or 165 <= h <= 179:
                        stage, color, desc = "🔴 Ripe / Overripe", "#FF4444", "Deep red coloring indicates peak or over-ripeness."
                    elif 9 <= h <= 18:
                        stage, color, desc = "🟠 Nearly Ripe", "#FF8800", "Orange-red tones suggest the fruit is nearly ripe."
                    elif 19 <= h <= 30:
                        stage, color, desc = "🟡 Semi-Ripe", "#FFCC00", "Yellow tones indicate partial ripeness."
                    elif 31 <= h <= 80:
                        stage, color, desc = "🟢 Unripe", "#33CC66", "Green coloring indicates the fruit is still unripe."
                    else:
                        stage, color, desc = "❓ Unknown", "#888888", "Cannot determine ripeness from color alone."

                    st.markdown(f"""
                    <div style="background:{color}22;border:2px solid {color};border-radius:14px;
                    padding:24px;text-align:center;margin-top:12px;">
                        <div style="font-size:1.8rem;margin-bottom:8px;">{stage}</div>
                        <div style="color:#cccccc;font-size:0.85rem;">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("**Dominant Colors:**")
                render_palette_swatches(dominant)
    else:
        st.info("Upload a photo of a fruit (apple, banana, orange, etc.) to analyze ripeness.")

# ── Tab 5: Full Palette Studio ────────────────────────────────────────────────
with tab_palette:
    st.markdown("### 🌈 Full Color Palette Studio")

    col_p1, col_p2 = st.columns([1, 2])
    with col_p1:
        r_p = st.slider("Red", 0, 255, 100, key="p_r")
        g_p = st.slider("Green", 0, 255, 60, key="p_g")
        b_p = st.slider("Blue", 0, 255, 220, key="p_b")

        name_info = get_color_name(r_p, g_p, b_p)
        st.markdown(f"""
        <div style="background:rgb({r_p},{g_p},{b_p});border-radius:12px;
        height:90px;margin:12px 0;box-shadow:0 4px 20px rgba(0,0,0,0.4);"></div>
        <div style="text-align:center;color:#aaaacc;font-size:0.9rem;">
            {name_info['color_name']} · #{r_p:02X}{g_p:02X}{b_p:02X}
        </div>
        """, unsafe_allow_html=True)

    with col_p2:
        palettes = generate_all_palettes(r_p, g_p, b_p)
        for pname, pcolors in palettes.items():
            st.markdown(f"**{pname.replace('_',' ').title()}**")
            render_palette_swatches(pcolors)
            st.markdown("<br>", unsafe_allow_html=True)
