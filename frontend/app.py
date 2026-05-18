"""
ColorAI — Main Dashboard
Entry point for the Streamlit multi-page application.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from frontend.components.sidebar import apply_global_styles, render_sidebar

st.set_page_config(
    page_title="ColorAI — Color Detection System",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_styles()
render_sidebar()

# ── Hero Header ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center;padding:60px 0 40px;">
        <div style="font-size:4rem;margin-bottom:10px;">🎨</div>
        <h1 style="
            font-size:3.2rem;font-weight:800;margin:0;
            background:linear-gradient(135deg,#6c47ff 0%,#a855f7 50%,#06b6d4 100%);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            letter-spacing:-1px;
        ">ColorAI</h1>
        <p style="font-size:1.15rem;color:#8888bb;margin-top:10px;font-weight:300;">
            Production-grade AI Color Detection &amp; Tracking System
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Feature Cards ────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

cards = [
    ("🖼️", "Image Detection", "Upload images and detect colors with click-to-identify",
     "1_Image_Detection", "#6c47ff", "#8b5cf6"),
    ("📷", "Webcam Detection", "Real-time live color tracking from your webcam",
     "2_Webcam_Detection", "#0891b2", "#06b6d4"),
    ("🎬", "Video Analysis", "Frame-by-frame color timeline for video files",
     "3_Video_Analysis", "#be185d", "#ec4899"),
    ("🤖", "Advanced AI", "Fashion matcher, interior design & fruit ripeness",
     "4_Advanced_Features", "#059669", "#10b981"),
]

for col, (icon, title, desc, page, c1, c2) in zip([col1, col2, col3, col4], cards):
    with col:
        st.markdown(
            f"""
            <div style="
                background:linear-gradient(135deg,{c1}22,{c2}11);
                border:1px solid {c1}55;
                border-radius:16px;
                padding:28px 20px;
                text-align:center;
                cursor:pointer;
                transition:all 0.3s ease;
                height:200px;
                display:flex;flex-direction:column;align-items:center;justify-content:center;
            ">
                <div style="font-size:2.4rem;margin-bottom:10px;">{icon}</div>
                <div style="font-size:1.05rem;font-weight:700;color:#e0e0ff;margin-bottom:8px;">
                    {title}
                </div>
                <div style="font-size:0.78rem;color:#9999bb;line-height:1.5;">
                    {desc}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── System Metrics ────────────────────────────────────────────────────────────
st.markdown("### 📊 System Capabilities")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("🎨 Named Colors", "280+", "Full spectrum")
m2.metric("🔬 Color Spaces", "4", "RGB, HSV, HEX, LAB")
m3.metric("⚡ Realtime FPS", "30+", "Webcam stream")
m4.metric("🧠 ML Clusters", "Up to 12", "K-Means")
m5.metric("🎯 Detection Colors", "24+", "HSV palette")

st.divider()

# ── Tech Stack ────────────────────────────────────────────────────────────────
st.markdown("### 🛠️ Technology Stack")
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("""
    **CV & ML**
    - OpenCV 4.9 — Detection & Tracking
    - Scikit-learn — K-Means Clustering
    - SciPy — LAB Delta-E Similarity
    """)

with col_b:
    st.markdown("""
    **Backend**
    - FastAPI — REST API
    - Pydantic — Data Validation
    - Loguru — Structured Logging
    """)

with col_c:
    st.markdown("""
    **Frontend**
    - Streamlit — Interactive UI
    - Plotly — Data Visualization
    - streamlit-webrtc — Live Webcam
    """)
