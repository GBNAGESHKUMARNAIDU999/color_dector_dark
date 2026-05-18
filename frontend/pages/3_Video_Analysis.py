"""
Page 3: Video Color Analysis
Upload a video and get frame-by-frame color timeline analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import streamlit as st
import tempfile
import json
import pandas as pd

from frontend.components.sidebar import apply_global_styles, render_sidebar
from frontend.components.palette_chart import (
    render_dominant_pie_chart, render_palette_swatches,
    render_color_bar, render_color_timeline,
)
from backend.services.video_service import analyze_video

st.set_page_config(page_title="Video Analysis — ColorAI", page_icon="🎬", layout="wide")
apply_global_styles()
sidebar_cfg = render_sidebar()

st.markdown("""
<h1 style="background:linear-gradient(135deg,#be185d,#ec4899);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;
font-size:2.4rem;font-weight:800;">🎬 Video Color Analysis</h1>
<p style="color:#8888bb;margin-top:-8px;">
Upload a video to generate a color timeline, dominant scene palette, and frame-by-frame report.
</p>""", unsafe_allow_html=True)

uploaded_video = st.file_uploader("Upload video", type=["mp4","avi","mov","mkv","webm"], label_visibility="collapsed")

if uploaded_video is None:
    st.markdown("""<div style="background:rgba(190,24,93,0.06);border:2px dashed rgba(190,24,93,0.3);
    border-radius:16px;padding:60px;text-align:center;margin-top:20px;">
    <div style="font-size:3rem;margin-bottom:12px;">🎥</div>
    <div style="color:#8888bb;">Upload a video file to begin analysis</div></div>""", unsafe_allow_html=True)
    st.stop()

col_s1, col_s2 = st.columns(2)
with col_s1:
    sample_n = st.slider("Sample every N frames", 1, 30, 5)
with col_s2:
    k_vid = st.slider("Dominant colors per frame", 2, 10, 5)

if not st.button("🚀 Analyze Video", use_container_width=True):
    st.stop()

video_bytes = uploaded_video.read()
suffix = os.path.splitext(uploaded_video.name)[1]
with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
    tmp.write(video_bytes)
    tmp_path = tmp.name

with st.spinner("🎬 Analyzing video..."):
    try:
        result = analyze_video(
            video_path=tmp_path,
            target_colors=sidebar_cfg["target_colors"],
            sample_every_n_frames=sample_n,
            k_clusters=k_vid,
        )
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        os.remove(tmp_path)
        st.stop()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

st.success("✅ Analysis complete!")
m1, m2, m3, m4 = st.columns(4)
m1.metric("🎞️ Total Frames", f"{result['total_frames']:,}")
m2.metric("🔬 Processed", f"{result['processed_frames']:,}")
m3.metric("⏱️ Duration", f"{result['duration_sec']:.1f}s")
m4.metric("⚡ Analysis Time", f"{result['processing_time_ms']:.0f}ms")

st.divider()
st.markdown("### 🎨 Dominant Scene Colors")
if result["dominant_scene_colors"]:
    render_color_bar(result["dominant_scene_colors"])
    tab1, tab2 = st.tabs(["🥧 Distribution", "🎨 Swatches"])
    with tab1:
        render_dominant_pie_chart(result["dominant_scene_colors"])
    with tab2:
        render_palette_swatches(result["dominant_scene_colors"])

st.divider()
st.markdown("### 📈 Color Detection Timeline")
render_color_timeline(result["frame_results"])

st.divider()
st.markdown("### 📊 Frame-by-Frame Report")
if result["frame_results"]:
    rows = []
    for fr in result["frame_results"]:
        row = {"Frame": fr["frame_index"], "Timestamp (s)": fr["timestamp_sec"]}
        for color, count in fr["detections"].items():
            row[color.capitalize()] = count
        if fr.get("dominant"):
            row["Top Color"] = fr["dominant"][0].get("color_name", "—")
        rows.append(row)
    df = pd.DataFrame(rows).fillna(0)
    st.dataframe(df, use_container_width=True, height=350)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(),
                          "color_report.csv", "text/csv", use_container_width=True)
    with col_dl2:
        st.download_button("⬇️ Download JSON", json.dumps(result, indent=2, default=str).encode(),
                          "color_report.json", "application/json", use_container_width=True)
