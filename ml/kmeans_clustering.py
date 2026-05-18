"""
K-Means clustering for dominant color extraction.
Extracts the N most prominent colors from any image with coverage percentages.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
import warnings

from utils.config import settings
from utils.logger import logger
from cv_pipeline.color_spaces import bgr_to_rgb, rgb_to_hex, rgb_to_hsv
from cv_pipeline.color_classifier import get_color_name_batch


def extract_dominant_colors(
    image: np.ndarray,
    k: int = settings.DEFAULT_KMEANS_CLUSTERS,
    sample_size: int = 10_000,
    exclude_white: bool = False,
    exclude_black: bool = False,
) -> List[Dict]:
    """
    Extract the K most dominant colors from an image using K-Means clustering.

    Args:
        image: BGR numpy array.
        k: Number of dominant colors to extract (default 5).
        sample_size: Max pixels to sample for speed (random sampling if exceeded).
        exclude_white: Skip near-white pixels before clustering.
        exclude_black: Skip near-black pixels before clustering.

    Returns:
        List of dicts sorted by percentage descending:
          {rgb, hex, hsv, color_name, color_family, percentage}
    """
    # Convert to RGB for human-readable outputs
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = rgb_image.reshape(-1, 3).astype(np.float32)

    # Optional filtering
    if exclude_white:
        pixels = pixels[~(np.all(pixels > 220, axis=1))]
    if exclude_black:
        pixels = pixels[~(np.all(pixels < 35, axis=1))]

    if len(pixels) == 0:
        logger.warning("No pixels remaining after filtering — skipping clustering.")
        return []

    # Random subsample for speed
    if len(pixels) > sample_size:
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        pixels = pixels[indices]

    # Clamp k to available pixels
    k = min(k, len(pixels))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        kmeans = KMeans(
            n_clusters=k,
            n_init=10,
            max_iter=300,
            random_state=42,
        )
        kmeans.fit(pixels)

    # Count pixels per cluster
    labels = kmeans.labels_
    counts = np.bincount(labels, minlength=k)
    total = counts.sum()

    centers = kmeans.cluster_centers_.astype(int)  # shape (k, 3) in RGB

    # Classify all cluster centers in one vectorized call
    rgb_list = [(int(c[0]), int(c[1]), int(c[2])) for c in centers]
    names = get_color_name_batch(rgb_list)

    results = []
    for i in range(k):
        r, g, b = int(centers[i][0]), int(centers[i][1]), int(centers[i][2])
        pct = round((counts[i] / total) * 100, 2)
        hsv = rgb_to_hsv(r, g, b)

        results.append({
            "rank":         i + 1,
            "rgb":          {"R": r, "G": g, "B": b},
            "hex":          rgb_to_hex(r, g, b),
            "hsv":          {"H": hsv[0], "S": hsv[1], "V": hsv[2]},
            "color_name":   names[i]["color_name"],
            "color_family": names[i]["color_family"],
            "percentage":   pct,
            "pixel_count":  int(counts[i]),
        })

    # Sort by percentage descending, re-rank
    results.sort(key=lambda x: x["percentage"], reverse=True)
    for rank, item in enumerate(results, 1):
        item["rank"] = rank

    logger.info(f"K-Means extracted {k} dominant colors.")
    return results


def build_palette_strip(
    dominant_colors: List[Dict],
    width: int = 600,
    height: int = 80,
) -> np.ndarray:
    """
    Generate a horizontal color palette strip from dominant color results.

    Args:
        dominant_colors: Output from extract_dominant_colors().
        width: Output image width in pixels.
        height: Output image height in pixels.

    Returns:
        BGR palette image as numpy array.
    """
    palette = np.zeros((height, width, 3), dtype=np.uint8)
    total_pct = sum(c["percentage"] for c in dominant_colors)

    x_start = 0
    for color in dominant_colors:
        r = color["rgb"]["R"]
        g = color["rgb"]["G"]
        b = color["rgb"]["B"]
        segment_width = int((color["percentage"] / max(total_pct, 1)) * width)
        x_end = min(x_start + segment_width, width)
        palette[:, x_start:x_end] = [b, g, r]   # BGR
        x_start = x_end

    return palette


def compute_color_histogram(
    image: np.ndarray,
    bins: int = 256,
    channel: str = "all",
) -> Dict[str, np.ndarray]:
    """
    Compute color histograms for R, G, B channels.

    Args:
        image: BGR image.
        bins: Number of histogram bins.
        channel: 'all', 'r', 'g', or 'b'.

    Returns:
        Dict mapping channel name → histogram array.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    histograms = {}
    channels = {"r": 0, "g": 1, "b": 2}

    for name, idx in channels.items():
        if channel == "all" or channel == name:
            hist = cv2.calcHist([rgb], [idx], None, [bins], [0, 256])
            histograms[name] = hist.flatten()

    return histograms


def find_optimal_k(image: np.ndarray, k_range: Tuple[int, int] = (2, 10)) -> int:
    """
    Find optimal K for K-Means using the elbow method (inertia).

    Args:
        image: BGR image.
        k_range: (min_k, max_k) range to evaluate.

    Returns:
        Optimal K value.
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = rgb_image.reshape(-1, 3).astype(np.float32)

    # Subsample
    if len(pixels) > 5000:
        indices = np.random.choice(len(pixels), 5000, replace=False)
        pixels = pixels[indices]

    inertias = []
    k_values = list(range(k_range[0], k_range[1] + 1))

    for k in k_values:
        km = KMeans(n_clusters=k, n_init=5, max_iter=100, random_state=42)
        km.fit(pixels)
        inertias.append(km.inertia_)

    # Elbow: find largest second derivative
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)

    if len(diffs2) == 0:
        return k_values[0]

    elbow_idx = int(np.argmax(np.abs(diffs2))) + 1
    optimal = k_values[elbow_idx]
    logger.info(f"Optimal K (elbow): {optimal}")
    return optimal
