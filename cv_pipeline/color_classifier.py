"""
Color classifier using nearest-neighbor lookup on the colors.csv dataset.
Maps any RGB value to the closest named color using Euclidean distance in RGB space.
"""

import csv
import math
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.config import settings
from utils.logger import logger


@lru_cache(maxsize=1)
def _load_color_database() -> List[Dict]:
    """
    Load and cache the color database from colors.csv.

    Returns:
        List of dicts with keys: color_name, R, G, B, hex_code, color_family
    """
    path = settings.COLORS_CSV_PATH
    if not os.path.exists(path):
        logger.error(f"Colors CSV not found at: {path}")
        return []

    colors = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                colors.append({
                    "color_name":   row["color_name"].strip(),
                    "R":            int(row["R"]),
                    "G":            int(row["G"]),
                    "B":            int(row["B"]),
                    "hex_code":     row.get("hex_code", "").strip(),
                    "color_family": row.get("color_family", "").strip(),
                })
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipped malformed CSV row: {row} — {e}")

    logger.info(f"Loaded {len(colors)} named colors from database.")
    return colors


def get_color_name(r: int, g: int, b: int) -> Dict:
    """
    Find the closest named color to a given RGB value.

    Uses Euclidean distance in RGB space to find the nearest match
    from the 200+ color database.

    Args:
        r, g, b: Integer values 0-255.

    Returns:
        Dict with: color_name, hex_code, color_family, distance (lower = closer match)
    """
    database = _load_color_database()
    if not database:
        return {
            "color_name": "Unknown",
            "hex_code": "#000000",
            "color_family": "Unknown",
            "distance": 999,
        }

    best = None
    best_dist = float("inf")

    for entry in database:
        dr = r - entry["R"]
        dg = g - entry["G"]
        db = b - entry["B"]
        dist = math.sqrt(dr * dr + dg * dg + db * db)
        if dist < best_dist:
            best_dist = dist
            best = entry

    return {
        "color_name":   best["color_name"],
        "hex_code":     best["hex_code"],
        "color_family": best["color_family"],
        "distance":     round(best_dist, 2),
    }


def get_color_name_batch(rgb_list: List[Tuple[int, int, int]]) -> List[Dict]:
    """
    Classify a batch of RGB values efficiently using vectorized NumPy operations.

    Args:
        rgb_list: List of (R, G, B) tuples.

    Returns:
        List of classification dicts (same order as input).
    """
    database = _load_color_database()
    if not database:
        return [{"color_name": "Unknown", "hex_code": "#000000",
                 "color_family": "Unknown", "distance": 999}] * len(rgb_list)

    # Build database matrix: shape (N_colors, 3)
    db_array = np.array([[c["R"], c["G"], c["B"]] for c in database], dtype=np.float32)

    results = []
    for r, g, b in rgb_list:
        query = np.array([[r, g, b]], dtype=np.float32)
        dists = np.linalg.norm(db_array - query, axis=1)
        idx = int(np.argmin(dists))
        best = database[idx]
        results.append({
            "color_name":   best["color_name"],
            "hex_code":     best["hex_code"],
            "color_family": best["color_family"],
            "distance":     round(float(dists[idx]), 2),
        })
    return results


def get_color_families() -> List[str]:
    """Return all unique color family names in the database."""
    database = _load_color_database()
    return sorted(set(c["color_family"] for c in database if c["color_family"]))


def search_by_name(query: str) -> List[Dict]:
    """
    Search the color database by partial name match (case-insensitive).

    Args:
        query: Partial color name to search for.

    Returns:
        List of matching color entries.
    """
    database = _load_color_database()
    query_lower = query.lower()
    return [c for c in database if query_lower in c["color_name"].lower()]
