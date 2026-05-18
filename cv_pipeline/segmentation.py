"""
Image segmentation utilities.
Provides contour detection, region extraction, and bounding box generation.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from utils.config import settings
from utils.logger import logger


@dataclass
class Segment:
    """Represents a detected image segment / region."""
    contour: np.ndarray
    bounding_box: Tuple[int, int, int, int]   # (x, y, w, h)
    area: float
    center: Tuple[int, int]
    aspect_ratio: float
    extent: float         # ratio of contour area to bounding box area
    roi: Optional[np.ndarray] = None   # cropped region


def find_segments(
    mask: np.ndarray,
    min_area: int = settings.MIN_CONTOUR_AREA,
    max_segments: int = 20,
) -> List[Segment]:
    """
    Find all meaningful contour segments in a binary mask.

    Args:
        mask: Binary uint8 mask (foreground = 255).
        min_area: Minimum pixel area to keep.
        max_segments: Maximum number of segments to return (largest first).

    Returns:
        List of Segment objects sorted by area descending.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    segments: List[Segment] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2
        aspect_ratio = round(w / max(h, 1), 3)
        extent = round(area / max(w * h, 1), 3)

        segments.append(
            Segment(
                contour=cnt,
                bounding_box=(x, y, w, h),
                area=area,
                center=(cx, cy),
                aspect_ratio=aspect_ratio,
                extent=extent,
            )
        )

    # Sort by area descending, cap at max_segments
    segments.sort(key=lambda s: s.area, reverse=True)
    return segments[:max_segments]


def extract_roi(image: np.ndarray, segment: Segment, padding: int = 5) -> np.ndarray:
    """
    Extract the region of interest from an image given a Segment.

    Args:
        image: Source BGR image.
        segment: Detected segment.
        padding: Extra pixels to add around the bounding box.

    Returns:
        Cropped BGR image (ROI).
    """
    x, y, w, h = segment.bounding_box
    img_h, img_w = image.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)
    return image[y1:y2, x1:x2]


def draw_segments(
    image: np.ndarray,
    segments: List[Segment],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    fill_alpha: float = 0.15,
) -> np.ndarray:
    """
    Draw segment contours and bounding boxes on an image.

    Args:
        image: Source BGR image.
        segments: List of detected segments.
        color: BGR colour for drawing.
        thickness: Contour line thickness.
        fill_alpha: Opacity of the filled contour overlay (0 = no fill).

    Returns:
        Annotated image.
    """
    annotated = image.copy()
    overlay = image.copy()

    for i, seg in enumerate(segments):
        x, y, w, h = seg.bounding_box
        # Filled contour on overlay
        cv2.drawContours(overlay, [seg.contour], -1, color, -1)
        # Bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
        # Label
        label = f"#{i + 1}  {int(seg.area)}px"
        cv2.putText(
            annotated, label, (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

    if fill_alpha > 0:
        cv2.addWeighted(overlay, fill_alpha, annotated, 1 - fill_alpha, 0, annotated)

    return annotated


def create_color_mask(image: np.ndarray, color_name: str) -> np.ndarray:
    """
    Create a binary mask for a specific color name using configured HSV ranges.

    Args:
        image: BGR image.
        color_name: Color name matching a key in config.COLOR_RANGES.

    Returns:
        Binary uint8 mask.
    """
    from cv_pipeline.preprocessor import morphological_open, morphological_close
    from utils.config import settings

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ranges = settings.COLOR_RANGES

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for key, r in ranges.items():
        if key.rstrip("12") == color_name:
            lo = np.array(r["lower"], dtype=np.uint8)
            hi = np.array(r["upper"], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

    mask = morphological_open(mask)
    mask = morphological_close(mask)
    return mask


def segment_by_color(
    image: np.ndarray,
    color_name: str,
    min_area: int = settings.MIN_CONTOUR_AREA,
) -> Tuple[np.ndarray, List[Segment]]:
    """
    Full segmentation pipeline for a single color.

    Args:
        image: BGR image.
        color_name: Target color name.
        min_area: Minimum segment area.

    Returns:
        Tuple of (mask, segments).
    """
    mask = create_color_mask(image, color_name)
    segments = find_segments(mask, min_area=min_area)

    for seg in segments:
        seg.roi = extract_roi(image, seg)

    logger.debug(f"Segmented '{color_name}': {len(segments)} segment(s) found.")
    return mask, segments
