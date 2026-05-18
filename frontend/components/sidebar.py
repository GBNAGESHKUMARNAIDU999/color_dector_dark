"""Shared Streamlit sidebar configuration component."""
import streamlit as st
from utils.config import settings


DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0a0a1a !important;
    color: #e0e0ff !important;
}
.stApp { background: linear-gradient(135deg, #0a0a1a 0%, #0d0d2b 50%, #0a0a1a 100%); }

/* Cards */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 12px 16px !important;
    backdrop-filter: blur(10px);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6c47ff, #a855f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(108,71,255,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(108,71,255,0.6) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 2px dashed rgba(108,71,255,0.5) !important;
    border-radius: 14px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 8px !important;
    color: #aaaacc !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6c47ff, #a855f7) !important;
    color: white !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(10,10,30,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}

/* Selectbox + multiselect */
.stSelectbox > div, .stMultiSelect > div {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 8px !important;
}

/* Slider */
.stSlider > div > div { background: #6c47ff !important; }

/* Divider */
hr { border-color: rgba(255,255,255,0.1) !important; }
</style>
"""


def apply_global_styles() -> None:
    """Inject premium dark mode CSS into every page."""
    st.markdown(DARK_CSS, unsafe_allow_html=True)


def render_sidebar() -> dict:
    """
    Render the shared sidebar with global settings.

    Returns:
        Dict with sidebar settings: k_clusters, target_colors, show_confidence.
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;padding:20px 0 10px;">
                <div style="font-size:2.2rem;">🎨</div>
                <div style="font-size:1.2rem;font-weight:700;
                    background:linear-gradient(135deg,#6c47ff,#a855f7);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                    ColorAI
                </div>
                <div style="font-size:0.72rem;color:#8888aa;margin-top:2px;">
                    Color Detection & Tracking System
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("### ⚙️ Detection Settings")
        k_clusters = st.slider("Dominant Colors (K)", min_value=2, max_value=10, value=5, step=1)

        all_colors = [
            "red", "orange", "yellow", "lime", "green", "teal", "cyan",
            "sky blue", "blue", "navy", "purple", "magenta", "pink",
            "brown", "maroon", "olive", "gold", "coral", "beige",
            "lavender", "turquoise", "gray", "white", "black",
        ]
        target_colors = st.multiselect(
            "Target Colors",
            options=all_colors,
            default=[],
            placeholder="All colors",
        )

        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        show_bboxes = st.checkbox("Show Bounding Boxes", value=True)

        st.divider()
        st.markdown(
            """
            <div style="font-size:0.72rem;color:#666688;text-align:center;padding:8px 0;">
                🔬 Powered by OpenCV + K-Means<br>
                ⚡ Real-time Color Intelligence
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "k_clusters": k_clusters,
        "target_colors": target_colors or None,
        "show_confidence": show_confidence,
        "show_bboxes": show_bboxes,
    }
