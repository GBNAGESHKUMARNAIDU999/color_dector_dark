"""
Page 1: Image Color Detection
Upload images, click to detect colors, extract dominant palette.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

from frontend.components.sidebar import apply_global_styles, render_sidebar
from frontend.components.color_card import render_color_card
from frontend.components.palette_chart import (
    render_dominant_pie_chart,
    render_palette_swatches,
    render_color_bar,
)
from cv_pipeline.preprocessor import bytes_to_image, preprocess_image, image_to_bytes
from cv_pipeline.color_detector import detect_colors, draw_detections, get_pixel_color
from cv_pipeline.color_classifier import get_color_name
from cv_pipeline.color_spaces import get_all_formats
from ml.kmeans_clustering import extract_dominant_colors
from ml.color_similarity import find_similar_colors
from ml.color_recommender import generate_all_palettes

st.set_page_config(
    page_title="Image Detection — ColorAI",
    page_icon="🖼️",
    layout="wide",
)
apply_global_styles()
sidebar_cfg = render_sidebar()

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style="
        background:linear-gradient(135deg,#6c47ff,#a855f7);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        font-size:2.4rem;font-weight:800;
    ">🖼️ Image Color Detection</h1>
    <p style="color:#8888bb;margin-top:-8px;">
        Upload an image to detect colors, extract dominant palette, and explore color harmonies.
    </p>
    """,
    unsafe_allow_html=True,
)

# ── File Upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop your image here",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown(
        """
        <div style="
            background:rgba(108,71,255,0.06);border:2px dashed rgba(108,71,255,0.3);
            border-radius:16px;padding:60px;text-align:center;margin-top:20px;
        ">
            <div style="font-size:3rem;margin-bottom:12px;">📸</div>
            <div style="color:#8888bb;font-size:1rem;">
                Upload a JPG, PNG, BMP or WebP image to begin analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Load & Process Image ──────────────────────────────────────────────────────
image_bytes = uploaded.read()
image_bgr = bytes_to_image(image_bytes)
if image_bgr is None:
    st.error("Could not decode the image. Please try a different file.")
    st.stop()

processed = preprocess_image(image_bgr, resize=True, denoise=True)
h, w = processed.shape[:2]

# ── Layout: Image + Controls ──────────────────────────────────────────────────
col_img, col_info = st.columns([3, 2])

with col_img:
    st.markdown("#### 📌 Uploaded Image")

    # Run detection
    with st.spinner("Detecting colors..."):
        detections = detect_colors(
            processed,
            target_colors=sidebar_cfg["target_colors"],
        )
        annotated = draw_detections(
            processed,
            detections,
            draw_label=sidebar_cfg["show_confidence"],
        )

    # Display annotated image
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, use_container_width=True, caption="Detected Color Regions")

    total_detected = sum(len(v) for v in detections.values())
    st.success(f"✅ Found {total_detected} color region(s) across {len(detections)} color(s)")

with col_info:
    st.markdown("#### 🔍 Detected Colors")
    if detections:
        for color_name, det_list in detections.items():
            top = det_list[0]
            st.markdown(
                f"""
                <div style="
                    background:rgba(255,255,255,0.04);border-radius:10px;
                    padding:10px 14px;margin-bottom:8px;
                    border-left:4px solid {'#6c47ff'};
                ">
                    <b style="color:#e0e0ff;">{color_name.upper()}</b>
                    <span style="float:right;color:#a0a0cc;font-size:0.85rem;">
                        {top.confidence:.0%} confidence
                    </span><br>
                    <span style="color:#8888aa;font-size:0.8rem;">
                        {top.pixel_count:,} px · {top.percentage:.1f}% of image
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No colors detected. Try adjusting target colors in the sidebar.")

# ── Dominant Colors ───────────────────────────────────────────────────────────
st.divider()
st.markdown("### 🎨 Dominant Color Palette")

with st.spinner("Extracting dominant colors..."):
    dominant = extract_dominant_colors(processed, k=sidebar_cfg["k_clusters"])

render_color_bar(dominant)

tab1, tab2 = st.tabs(["🥧 Distribution Chart", "🎨 Palette Swatches"])
with tab1:
    render_dominant_pie_chart(dominant)
with tab2:
    render_palette_swatches(dominant)

# ── Dominant Color Details ────────────────────────────────────────────────────
st.markdown("#### Color Details")
cols = st.columns(min(len(dominant), 5))
for col, dc in zip(cols, dominant):
    with col:
        render_color_card(
            {
                "rgb": dc["rgb"],
                "hsv": dc["hsv"],
                "hex": dc["hex"],
                "color_name": dc["color_name"],
                "color_family": dc["color_family"],
            },
            show_palette=True,
        )
        st.caption(f"**Rank #{dc['rank']}** · {dc['percentage']}%")

# ── Color Picker (manual RGB) ─────────────────────────────────────────────────
st.divider()
st.markdown("### 🔬 Manual Color Explorer")
col_r, col_g, col_b = st.columns(3)
with col_r:
    r_val = st.slider("Red", 0, 255, 128, key="exp_r")
with col_g:
    g_val = st.slider("Green", 0, 255, 64, key="exp_g")
with col_b:
    b_val = st.slider("Blue", 0, 255, 200, key="exp_b")

formats = get_all_formats(r_val, g_val, b_val)
name_info = get_color_name(r_val, g_val, b_val)
render_color_card({**formats, **name_info}, show_palette=True)

# Similar colors
with st.expander("🔍 Similar Named Colors", expanded=False):
    similar = find_similar_colors(r_val, g_val, b_val, top_n=5)
    render_palette_swatches([
        {"rgb": s["rgb"], "hex": s["hex_code"], "color_name": s["color_name"],
         "color_family": s["color_family"]}
        for s in similar
    ])

# Color palettes
with st.expander("🌈 Color Harmony Palettes", expanded=False):
    palettes = generate_all_palettes(r_val, g_val, b_val)
    for ptype, pcolors in palettes.items():
        st.markdown(f"**{ptype.replace('_', ' ').title()}**")
        render_palette_swatches(pcolors)
        st.markdown("<br>", unsafe_allow_html=True)
