"""
Page 2: Real-Time Webcam Color Detection
Live color detection via webcam using streamlit-webrtc or OpenCV fallback.
Clean, clutter-free detection with configurable sensitivity.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import streamlit as st
import numpy as np
import cv2
import time
from threading import Lock

from frontend.components.sidebar import apply_global_styles, render_sidebar

st.set_page_config(
    page_title="Webcam Detection — ColorAI",
    page_icon="📷",
    layout="wide",
)
apply_global_styles()
sidebar_cfg = render_sidebar()

st.markdown(
    """
    <h1 style="
        background:linear-gradient(135deg,#0891b2,#06b6d4);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        font-size:2.4rem;font-weight:800;
    ">📷 Real-Time Webcam Detection</h1>
    <p style="color:#8888bb;margin-top:-8px;">
        Live color detection and object tracking from your webcam feed.
    </p>
    """,
    unsafe_allow_html=True,
)

# Primary colors only — avoids noise from neutral/skin-tone matches
PRIMARY_COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "cyan"]
ALL_COLORS = [
    "red", "orange", "yellow", "lime", "green", "teal", "cyan",
    "sky blue", "blue", "navy", "purple", "magenta", "pink",
    "brown", "maroon", "olive", "gold", "turquoise",
    "gray", "white", "black",
]

# ── WebRTC-based detection ────────────────────────────────────────────────────
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    from cv_pipeline.color_detector import detect_colors, draw_detections
    from tracking.contour_tracker import ContourTracker

    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    class ColorDetectionProcessor(VideoProcessorBase):
        """WebRTC video processor — detects and tracks colors in real time."""

        def __init__(self):
            self._tracker = ContourTracker(
                max_disappeared=8,
                max_distance=100,
                trail_length=10,
                max_objects=10,
            )
            self._lock = Lock()
            self.target_colors = None
            self.show_bboxes = True
            self.show_trail = False
            self.enhance_colors = False
            self._fps_time = time.time()
            self._fps = 0.0

        def recv(self, frame):
            import av
            img = frame.to_ndarray(format="bgr24")

            # Resize for speed
            img = cv2.resize(img, (640, 480))

            # Light denoise only — no heavy processing
            processed = cv2.GaussianBlur(img, (5, 5), 0)

            # Optional mild saturation boost
            if self.enhance_colors:
                hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
                processed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            with self._lock:
                t_colors = self.target_colors
                show_bb = self.show_bboxes
                show_tr = self.show_trail

            # Update tracker with preprocessed frame
            active = self._tracker.update(processed, target_colors=t_colors)

            # Draw tracks on the ORIGINAL image (clean background)
            if show_bb:
                img = self._tracker.draw_tracks(img, show_trail=show_tr)

            # FPS counter
            now = time.time()
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(now - self._fps_time, 1e-6))
            self._fps_time = now

            cv2.putText(
                img, f"FPS: {self._fps:.1f}  Objects: {len(active)}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 2, cv2.LINE_AA,
            )

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # ── Settings column ───────────────────────────────────────────────────────
    col_cfg, col_stream = st.columns([1, 2])
    with col_cfg:
        st.markdown("#### ⚙️ Detection Settings")
        target_sel = st.multiselect(
            "Colors to track",
            options=ALL_COLORS,
            default=["red", "blue", "green"],
            key="wc_colors",
        )
        show_bbox_cb = st.checkbox("Show bounding boxes", value=True, key="wc_bbox")
        show_trail_cb = st.checkbox("Show movement trail", value=False, key="wc_trail")
        enhance_cb = st.checkbox("Boost color saturation", value=False, key="wc_enhance",
                                 help="Mild saturation boost for dim lighting")

        st.markdown("#### 📊 Live Stats")
        stats_placeholder = st.empty()

    with col_stream:
        st.markdown("#### 📹 Live Feed")
        ctx = webrtc_streamer(
            key="color-detection",
            video_processor_factory=ColorDetectionProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor:
            with ctx.video_processor._lock:
                ctx.video_processor.target_colors = target_sel or None
                ctx.video_processor.show_bboxes = show_bbox_cb
                ctx.video_processor.show_trail = show_trail_cb
                ctx.video_processor.enhance_colors = enhance_cb

            stats_placeholder.markdown(
                f"""
                <div style="background:rgba(255,255,255,0.05);border-radius:10px;padding:16px;">
                    <div style="color:#06b6d4;font-size:0.9rem;margin-bottom:8px;">
                        <b>🟢 Stream Active</b>
                    </div>
                    <div style="color:#aaaacc;font-size:0.82rem;">
                        Tracking {len(target_sel)} color(s)<br>
                        Trail: {'ON' if show_trail_cb else 'OFF'}<br>
                        Boxes: {'ON' if show_bbox_cb else 'OFF'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Usage guide
    st.divider()
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info("🎯 **Step 1** — Allow camera access in your browser")
    with col_b:
        st.info("🎨 **Step 2** — Select target colors in the settings panel")
    with col_c:
        st.info("⚡ **Step 3** — Watch real-time color detection and tracking!")

except ImportError:
    # Fallback: OpenCV local webcam (for environments without streamlit-webrtc)
    st.warning(
        "⚠️ **streamlit-webrtc** is not installed. Using local OpenCV webcam instead.\n\n"
        "Install it: `pip install streamlit-webrtc av`"
    )

    from cv_pipeline.color_detector import detect_colors, draw_detections

    st.markdown("#### 📹 Local Webcam Feed")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        run_btn = st.button("▶ Start Webcam", key="start_wc", use_container_width=True)
        stop_btn = st.button("⏹ Stop", key="stop_wc", use_container_width=True)
        frame_placeholder = st.empty()

    with col_right:
        target_sel = st.multiselect(
            "Colors to detect",
            options=ALL_COLORS,
            default=["red", "blue", "green"],
        )

    if run_btn:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("No webcam detected.")
        else:
            for _ in range(600):  # ~20s at 30fps
                ret, frame = cap.read()
                if not ret or stop_btn:
                    break
                frame = cv2.resize(frame, (640, 480))
                processed = cv2.GaussianBlur(frame, (5, 5), 0)
                detections = detect_colors(processed, target_colors=target_sel or None)
                annotated = draw_detections(frame, detections)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_rgb, use_container_width=True)
            cap.release()
