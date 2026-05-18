"""
Global configuration settings for the Color Detection System.
All configurable parameters are centralized here for easy modification.
"""

from pydantic_settings import BaseSettings
from typing import Dict, List
import os


class Settings(BaseSettings):
    """Application-wide settings loaded from environment or defaults."""

    # --- API ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_VERSION: str = "v1"
    APP_TITLE: str = "Color Detection & Tracking API"
    APP_DESCRIPTION: str = "Production-grade AI Color Detection and Object Tracking System"
    DEBUG: bool = False

    # --- File Upload Limits ---
    MAX_IMAGE_SIZE_MB: int = 10
    MAX_VIDEO_SIZE_MB: int = 100
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"]
    ALLOWED_VIDEO_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

    # --- CV Pipeline ---
    DEFAULT_RESIZE_WIDTH: int = 800
    DEFAULT_KMEANS_CLUSTERS: int = 5
    MIN_CONTOUR_AREA: int = 2000         # high threshold — ignore noise, detect real objects only
    BLUR_KERNEL_SIZE: int = 5
    MORPH_KERNEL_SIZE: int = 7           # larger kernel — stronger noise cleanup
    MAX_DETECTIONS_PER_COLOR: int = 3    # cap bounding boxes per color to avoid clutter

    # --- Paths ---
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    COLORS_CSV_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "datasets", "colors.csv"
    )
    UPLOAD_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "uploads"
    )

    # --- Color HSV Ranges: {name: {lower:[H,S,V], upper:[H,S,V]}} ---
    # OpenCV HSV: H 0-179, S 0-255, V 0-255
    # ─────────────────────────────────────────────────────────────────────────
    # Carefully calibrated with NO GAPS between adjacent hue bands.
    # Saturation and value thresholds lowered for real-world webcam lighting.
    # ─────────────────────────────────────────────────────────────────────────
    COLOR_RANGES: Dict = {
        # ── Red (wraps around 0°/180° — two ranges) ──
        "red1":       {"lower": [0,   70,  50],  "upper": [10,  255, 255]},
        "red2":       {"lower": [165, 70,  50],  "upper": [179, 255, 255]},

        # ── Orange ──
        "orange":     {"lower": [11,  100, 100], "upper": [22,  255, 255]},

        # ── Yellow ──
        "yellow":     {"lower": [23,  50,  70],  "upper": [35,  255, 255]},

        # ── Yellow-Green / Lime ──
        "lime":       {"lower": [36,  50,  70],  "upper": [45,  255, 255]},

        # ── Green — wide range covering lime through forest green ──
        "green":      {"lower": [46,  50,  50],  "upper": [75,  255, 255]},

        # ── Teal / Dark Cyan ──
        "teal":       {"lower": [76,  40,  30],  "upper": [85,  255, 255]},

        # ── Cyan / Aqua ──
        "cyan":       {"lower": [86,  40,  40],  "upper": [100, 255, 255]},

        # ── Sky Blue / Light Blue ──
        "sky_blue":   {"lower": [101, 30,  60],  "upper": [112, 255, 255]},

        # ── Blue — covers medium blue through deep blue ──
        "blue":       {"lower": [113, 50,  30],  "upper": [130, 255, 255]},

        # ── Navy / Dark Blue ──
        "navy":       {"lower": [113, 50,  10],  "upper": [130, 255, 80]},

        # ── Purple / Violet ──
        "purple":     {"lower": [131, 30,  30],  "upper": [155, 255, 255]},

        # ── Magenta / Fuchsia ──
        "magenta":    {"lower": [156, 40,  50],  "upper": [164, 255, 255]},

        # ── Pink / Rose ──
        "pink":       {"lower": [145, 20,  100], "upper": [170, 255, 255]},

        # ── Brown — low saturation + low value orange/red hue ──
        "brown1":     {"lower": [5,   100, 30],  "upper": [25,  200, 130]},
        "brown2":     {"lower": [0,   100, 30],  "upper": [10,  200, 120]},

        # ── Maroon — dark red ──
        "maroon":     {"lower": [0,   100, 20],  "upper": [10,  255, 90]},

        # ── Olive — dark yellow-green ──
        "olive":      {"lower": [25,  50,  30],  "upper": [45,  200, 130]},

        # ── Gold / Amber ──
        "gold":       {"lower": [18,  150, 150], "upper": [30,  255, 255]},


        # ── Turquoise ──
        "turquoise":  {"lower": [80,  50,  100], "upper": [95,  255, 255]},

        # ── Gray — low saturation, medium value ──
        "gray":       {"lower": [0,   0,   80],  "upper": [179, 30,  160]},

        # ── White — very low saturation, very high value ──
        "white":      {"lower": [0,   0,   210], "upper": [179, 25,  255]},

        # ── Black — very low value ──
        "black":      {"lower": [0,   0,   0],   "upper": [179, 255, 30]},
    }

    # --- Streamlit ---
    STREAMLIT_PORT: int = 8501
    BACKEND_URL: str = "http://localhost:8000"

    class Config:
        env_file = ".env"
        case_sensitive = True


# Singleton instance
settings = Settings()
