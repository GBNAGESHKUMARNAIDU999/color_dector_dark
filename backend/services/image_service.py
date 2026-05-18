"""
Image processing service — orchestrates CV pipeline and ML engine for API use.
"""

import time
import cv2
import numpy as np
from typing import Dict, List, Optional

from cv_pipeline.preprocessor import preprocess_image, bytes_to_image
from cv_pipeline.color_detector import detect_colors, get_pixel_color
from cv_pipeline.color_classifier import get_color_name
from cv_pipeline.color_spaces import get_all_formats
from ml.kmeans_clustering import extract_dominant_colors
from ml.color_similarity import find_similar_colors
from ml.color_recommender import generate_all_palettes
from utils.logger import logger


def analyze_image(
    image_bytes: bytes,
    target_colors: Optional[List[str]] = None,
    k_clusters: int = 5,
    click_x: Optional[int] = None,
    click_y: Optional[int] = None,
) -> Dict:
    """
    Full image color analysis pipeline.

    Args:
        image_bytes: Raw uploaded image bytes.
        target_colors: List of color names to detect (None = all).
        k_clusters: Number of dominant colors to extract.
        click_x, click_y: Optional pixel coordinate for click-to-detect.

    Returns:
        Full analysis dict ready for serialization.
    """
    t_start = time.perf_counter()

    # 1. Decode
    image = bytes_to_image(image_bytes)
    if image is None:
        raise ValueError("Failed to decode image.")

    h, w = image.shape[:2]

    # 2. Preprocess
    processed = preprocess_image(image, resize=True, denoise=True)

    # 3. Detect colors
    detections = detect_colors(processed, target_colors=target_colors)
    detection_list = []
    for color_name, det_list in detections.items():
        for det in det_list:
            detection_list.append({
                "color_name":   det.color_name,
                "confidence":   det.confidence,
                "pixel_count":  det.pixel_count,
                "percentage":   det.percentage,
                "bounding_box": list(det.bounding_box),
                "center":       list(det.center),
            })

    # 4. Dominant colors via K-Means
    dominant = extract_dominant_colors(processed, k=k_clusters)

    # 5. Pixel click color info (optional)
    pixel_info = None
    if click_x is not None and click_y is not None:
        pixel_info = get_pixel_color(processed, click_x, click_y)
        # Enrich with color name
        r = pixel_info["rgb"]["R"]
        g = pixel_info["rgb"]["G"]
        b = pixel_info["rgb"]["B"]
        name_info = get_color_name(r, g, b)
        pixel_info["color_name"] = name_info["color_name"]
        pixel_info["color_family"] = name_info["color_family"]

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return {
        "image_size":       {"width": w, "height": h},
        "detected_colors":  detection_list,
        "dominant_colors":  dominant,
        "pixel_color":      pixel_info,
        "processing_time_ms": elapsed_ms,
    }


def get_color_info(r: int, g: int, b: int) -> Dict:
    """
    Return full color information for an RGB triplet.

    Args:
        r, g, b: Integer 0-255 color values.

    Returns:
        Dict with all color space values, name, similar colors, and palettes.
    """
    all_formats = get_all_formats(r, g, b)
    name_info = get_color_name(r, g, b)
    similar = find_similar_colors(r, g, b, top_n=5)
    palettes = generate_all_palettes(r, g, b)

    return {
        **all_formats,
        **name_info,
        "similar_colors": similar,
        "palettes": palettes,
    }
