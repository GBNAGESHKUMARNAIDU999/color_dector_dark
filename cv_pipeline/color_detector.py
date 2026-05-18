"""
HSV-range based color detector.
Detects named colors in images and video frames using configurable HSV ranges.
Supports 24+ color categories with robust wrap-around handling.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from utils.config import settings
from utils.logger import logger
from cv_pipeline.preprocessor import morphological_open, morphological_close


@dataclass
class DetectionResult:
    """Result for a single detected color region."""
    color_name: str
    confidence: float            # 0.0 – 1.0
    pixel_count: int
    percentage: float            # % of total image area
    bounding_box: Tuple[int, int, int, int]   # (x, y, w, h)
    center: Tuple[int, int]      # (cx, cy)
    mask: Optional[np.ndarray] = field(default=None, repr=False)


# ── Map variant keys to their display name ──────────────────────────────────
# Keys ending with digits (red1, red2, brown1, brown2) are merged into one.
_VARIANT_MAP = {
    "red1": "red",
    "red2": "red",
    "brown1": "brown",
    "brown2": "brown",
}


def _display_name(key: str) -> str:
    """Resolve a color range key to a display-friendly color name."""
    return _VARIANT_MAP.get(key, key.replace("_", " "))


def _merge_keys(key: str) -> List[str]:
    """Return all keys that merge into the same display name."""
    display = _display_name(key)
    return [
        k for k in settings.COLOR_RANGES
        if _display_name(k) == display
    ]


def _build_mask(
    hsv_image: np.ndarray,
    lower: List[int],
    upper: List[int],
) -> np.ndarray:
    """Create a binary mask for an HSV range."""
    lo = np.array(lower, dtype=np.uint8)
    hi = np.array(upper, dtype=np.uint8)
    return cv2.inRange(hsv_image, lo, hi)


def detect_colors(
    image: np.ndarray,
    target_colors: Optional[List[str]] = None,
    min_area: int = settings.MIN_CONTOUR_AREA,
    max_per_color: int = settings.MAX_DETECTIONS_PER_COLOR,
) -> Dict[str, List[DetectionResult]]:
    """
    Detect all specified colors in a BGR image using HSV masking.

    Args:
        image: Input BGR image (numpy array).
        target_colors: List of color names to detect (None = detect all).
        min_area: Minimum contour pixel area to consider a valid detection.
        max_per_color: Maximum detections to keep per color (largest first).

    Returns:
        Dict mapping color_name → list of DetectionResult objects.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    total_pixels = image.shape[0] * image.shape[1]
    results: Dict[str, List[DetectionResult]] = {}

    color_ranges = settings.COLOR_RANGES

    # Track which display names we've already processed
    processed_display_names = set()

    for color_key, ranges in color_ranges.items():
        display = _display_name(color_key)

        # Skip if already processed (variant was merged earlier)
        if display in processed_display_names:
            continue

        # Skip if not in the target list
        if target_colors and display not in target_colors:
            continue

        # Build merged mask for all variant keys
        variant_keys = _merge_keys(color_key)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for vk in variant_keys:
            vr = color_ranges[vk]
            partial = _build_mask(hsv, vr["lower"], vr["upper"])
            mask = cv2.bitwise_or(mask, partial)

        processed_display_names.add(display)

        # Clean mask
        mask = morphological_open(mask)
        mask = morphological_close(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[DetectionResult] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2

            # Confidence: based on contour fill ratio (area vs bounding box)
            bbox_area = max(w * h, 1)
            fill_ratio = min(area / bbox_area, 1.0)
            confidence = round(fill_ratio, 3)

            pct = round((area / total_pixels) * 100, 2)

            detections.append(
                DetectionResult(
                    color_name=display,
                    confidence=confidence,
                    pixel_count=int(area),
                    percentage=pct,
                    bounding_box=(x, y, w, h),
                    center=(cx, cy),
                    mask=mask,
                )
            )

        if detections:
            # Sort by pixel count descending, keep only top N
            detections.sort(key=lambda d: d.pixel_count, reverse=True)
            detections = detections[:max_per_color]
            results[display] = detections
            logger.debug(f"Detected '{display}': {len(detections)} region(s)")

    return results


def draw_detections(
    image: np.ndarray,
    results: Dict[str, List[DetectionResult]],
    draw_label: bool = True,
    draw_center: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on detected color regions.

    Args:
        image: BGR image to annotate.
        results: Output from detect_colors().
        draw_label: Whether to draw the color name label.
        draw_center: Whether to draw center crosshair.

    Returns:
        Annotated BGR image.
    """
    COLOR_MAP = {
        "red":        (0, 0, 255),
        "orange":     (0, 128, 255),
        "yellow":     (0, 255, 255),
        "lime":       (0, 255, 128),
        "green":      (0, 200, 0),
        "teal":       (128, 128, 0),
        "cyan":       (255, 255, 0),
        "sky blue":   (235, 206, 135),
        "blue":       (255, 0, 0),
        "navy":       (128, 0, 0),
        "purple":     (200, 0, 200),
        "magenta":    (255, 0, 255),
        "pink":       (180, 105, 255),
        "brown":      (42, 42, 165),
        "maroon":     (0, 0, 128),
        "olive":      (0, 128, 128),
        "gold":       (0, 215, 255),
        "coral":      (80, 127, 255),
        "beige":      (220, 245, 245),
        "lavender":   (250, 230, 230),
        "turquoise":  (208, 224, 64),
        "gray":       (128, 128, 128),
        "white":      (220, 220, 220),
        "black":      (60, 60, 60),
    }

    annotated = image.copy()

    for color_name, detections in results.items():
        box_color = COLOR_MAP.get(color_name, (100, 255, 100))
        for det in detections:
            x, y, w, h = det.bounding_box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)

            if draw_label:
                label = f"{color_name.upper()} {det.confidence:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x, y - th - 8), (x + tw + 4, y), box_color, -1)
                cv2.putText(
                    annotated, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
                )

            if draw_center:
                cx, cy = det.center
                cv2.drawMarker(
                    annotated, (cx, cy), box_color,
                    cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA,
                )

    return annotated


def get_pixel_color(image: np.ndarray, x: int, y: int) -> Dict:
    """
    Return color information for a single pixel at (x, y).

    Args:
        image: BGR image.
        x, y: Pixel coordinates.

    Returns:
        Dict with BGR, RGB, HSV, HEX values.
    """
    from cv_pipeline.color_spaces import bgr_to_rgb, rgb_to_hsv, rgb_to_hex

    h, w = image.shape[:2]
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))

    b, g, r = int(image[y, x, 0]), int(image[y, x, 1]), int(image[y, x, 2])
    rgb = bgr_to_rgb(b, g, r)
    hsv = rgb_to_hsv(*rgb)
    hex_code = rgb_to_hex(*rgb)

    return {
        "bgr": {"B": b, "G": g, "R": r},
        "rgb": {"R": rgb[0], "G": rgb[1], "B": rgb[2]},
        "hsv": {"H": hsv[0], "S": hsv[1], "V": hsv[2]},
        "hex": hex_code,
        "position": {"x": x, "y": y},
    }
