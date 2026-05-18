"""
Video analysis service — processes uploaded videos frame-by-frame.
"""

import time
import cv2
import numpy as np
from typing import Dict, List, Optional

from cv_pipeline.preprocessor import preprocess_frame
from cv_pipeline.color_detector import detect_colors
from ml.kmeans_clustering import extract_dominant_colors
from utils.logger import logger


def analyze_video(
    video_path: str,
    target_colors: Optional[List[str]] = None,
    sample_every_n_frames: int = 5,
    k_clusters: int = 5,
) -> Dict:
    """
    Analyze a video file for color distribution across frames.

    Args:
        video_path: Path to the video file on disk.
        target_colors: Colors to detect (None = all).
        sample_every_n_frames: Process every Nth frame for speed.
        k_clusters: Dominant colors per frame.

    Returns:
        Full video analysis dict.
    """
    t_start = time.perf_counter()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = round(total_frames / fps, 2)

    frame_results = []
    all_dominant_colors: List[Dict] = []
    frame_idx = 0
    processed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every_n_frames == 0:
            frame = preprocess_frame(frame)
            timestamp_sec = round(frame_idx / fps, 3)

            # Color detections
            detections = detect_colors(frame, target_colors=target_colors)
            detection_summary = {
                name: len(dets) for name, dets in detections.items()
            }

            # Dominant colors
            dominant = extract_dominant_colors(frame, k=k_clusters)
            all_dominant_colors.extend(dominant)

            frame_results.append({
                "frame_index":    frame_idx,
                "timestamp_sec":  timestamp_sec,
                "detections":     detection_summary,
                "dominant":       dominant[:3],   # top 3 per frame
            })
            processed_count += 1

        frame_idx += 1

    cap.release()

    # Aggregate dominant colors across all frames
    scene_colors = _aggregate_dominant_colors(all_dominant_colors, top_n=k_clusters)

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info(
        f"Video analysis complete: {processed_count}/{total_frames} frames, "
        f"{elapsed_ms:.0f}ms"
    )

    return {
        "total_frames":         total_frames,
        "processed_frames":     processed_count,
        "fps":                  fps,
        "duration_sec":         duration_sec,
        "frame_results":        frame_results,
        "dominant_scene_colors": scene_colors,
        "processing_time_ms":   elapsed_ms,
    }


def _aggregate_dominant_colors(colors: List[Dict], top_n: int = 5) -> List[Dict]:
    """
    Merge dominant color lists from all frames into a ranked scene palette.

    Aggregates by hex code, summing percentages, then normalizes.

    Args:
        colors: Flattened list of dominant color dicts.
        top_n: Number of top scene colors to return.

    Returns:
        Top N aggregated colors sorted by total weight.
    """
    aggregated: Dict[str, Dict] = {}

    for color in colors:
        key = color["hex"]
        if key not in aggregated:
            aggregated[key] = {**color, "total_weight": 0.0}
        aggregated[key]["total_weight"] += color["percentage"]

    sorted_colors = sorted(
        aggregated.values(), key=lambda c: c["total_weight"], reverse=True
    )

    total = sum(c["total_weight"] for c in sorted_colors) or 1.0
    for i, c in enumerate(sorted_colors[:top_n], 1):
        c["scene_percentage"] = round(c["total_weight"] / total * 100, 2)
        c["rank"] = i

    return sorted_colors[:top_n]
