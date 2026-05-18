"""
Color similarity analysis using CIE Delta-E 2000 in LAB space.
Provides perceptually accurate color distance calculations.
"""

import math
import numpy as np
from typing import List, Tuple, Dict

from cv_pipeline.color_spaces import rgb_to_lab, rgb_to_hex, rgb_to_hsv
from cv_pipeline.color_classifier import _load_color_database
from utils.logger import logger


def delta_e_cie76(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
    """
    Compute CIE76 colour difference (simple Euclidean in LAB space).

    Args:
        lab1, lab2: (L, a, b) tuples.

    Returns:
        Colour difference value (lower = more similar).
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))


def delta_e_cie2000(
    lab1: Tuple[float, float, float],
    lab2: Tuple[float, float, float],
) -> float:
    """
    Compute CIE Delta-E 2000 — the most perceptually accurate color difference metric.

    Args:
        lab1, lab2: (L, a, b) tuples in CIE LAB space.

    Returns:
        Delta-E 2000 value (0 = identical, < 1 = imperceptible difference).
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    # Step 1: Compute C* and h*
    C1 = math.sqrt(a1 ** 2 + b1 ** 2)
    C2 = math.sqrt(a2 ** 2 + b2 ** 2)
    C_avg = (C1 + C2) / 2
    C_avg7 = C_avg ** 7
    G = 0.5 * (1 - math.sqrt(C_avg7 / (C_avg7 + 25 ** 7)))

    a1p = a1 * (1 + G)
    a2p = a2 * (1 + G)
    C1p = math.sqrt(a1p ** 2 + b1 ** 2)
    C2p = math.sqrt(a2p ** 2 + b2 ** 2)

    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360

    # Step 2: Delta L', Delta C', Delta H'
    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p - h1p > 180:
        dhp = h2p - h1p - 360
    else:
        dhp = h2p - h1p + 360

    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))

    # Step 3: Averages
    Lp_avg = (L1 + L2) / 2
    Cp_avg = (C1p + C2p) / 2

    if C1p * C2p == 0:
        hp_avg = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hp_avg = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        hp_avg = (h1p + h2p + 360) / 2
    else:
        hp_avg = (h1p + h2p - 360) / 2

    T = (
        1
        - 0.17 * math.cos(math.radians(hp_avg - 30))
        + 0.24 * math.cos(math.radians(2 * hp_avg))
        + 0.32 * math.cos(math.radians(3 * hp_avg + 6))
        - 0.20 * math.cos(math.radians(4 * hp_avg - 63))
    )

    Cp_avg7 = Cp_avg ** 7
    RC = 2 * math.sqrt(Cp_avg7 / (Cp_avg7 + 25 ** 7))
    SL = 1 + 0.015 * (Lp_avg - 50) ** 2 / math.sqrt(20 + (Lp_avg - 50) ** 2)
    SC = 1 + 0.045 * Cp_avg
    SH = 1 + 0.015 * Cp_avg * T
    d_theta = 30 * math.exp(-((hp_avg - 275) / 25) ** 2)
    RT = -math.sin(math.radians(2 * d_theta)) * RC

    # Final Delta-E 2000
    dE = math.sqrt(
        (dLp / SL) ** 2
        + (dCp / SC) ** 2
        + (dHp / SH) ** 2
        + RT * (dCp / SC) * (dHp / SH)
    )
    return round(dE, 4)


def color_similarity_rgb(
    r1: int, g1: int, b1: int,
    r2: int, g2: int, b2: int,
    method: str = "delta_e_2000",
) -> Dict:
    """
    Compute perceptual similarity between two RGB colours.

    Args:
        r1, g1, b1: First colour (0-255 each).
        r2, g2, b2: Second colour (0-255 each).
        method: 'delta_e_2000' or 'delta_e_76' or 'euclidean_rgb'.

    Returns:
        Dict with: distance, similarity_pct, method
    """
    if method == "euclidean_rgb":
        dist = math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)
        max_dist = math.sqrt(3 * 255 ** 2)
        similarity = round((1 - dist / max_dist) * 100, 2)
    else:
        lab1 = rgb_to_lab(r1, g1, b1)
        lab2 = rgb_to_lab(r2, g2, b2)
        if method == "delta_e_2000":
            dist = delta_e_cie2000(lab1, lab2)
        else:
            dist = delta_e_cie76(lab1, lab2)
        # Delta-E < 1 imperceptible; > 50 very different
        similarity = round(max(0, (1 - dist / 50)) * 100, 2)

    return {
        "distance": round(dist, 4),
        "similarity_pct": similarity,
        "method": method,
    }


def find_similar_colors(
    r: int, g: int, b: int,
    top_n: int = 5,
    method: str = "delta_e_2000",
) -> List[Dict]:
    """
    Find the N most similar named colors from the database.

    Args:
        r, g, b: Query colour.
        top_n: Number of results to return.
        method: Similarity method.

    Returns:
        List of dicts sorted by similarity (ascending distance).
    """
    database = _load_color_database()
    lab_query = rgb_to_lab(r, g, b)

    scores = []
    for entry in database:
        lab_entry = rgb_to_lab(entry["R"], entry["G"], entry["B"])
        if method == "delta_e_2000":
            dist = delta_e_cie2000(lab_query, lab_entry)
        else:
            dist = delta_e_cie76(lab_query, lab_entry)

        scores.append({
            "color_name":   entry["color_name"],
            "hex_code":     entry["hex_code"],
            "color_family": entry["color_family"],
            "rgb":          {"R": entry["R"], "G": entry["G"], "B": entry["B"]},
            "distance":     round(dist, 4),
            "similarity_pct": round(max(0, (1 - dist / 50)) * 100, 2),
        })

    scores.sort(key=lambda x: x["distance"])
    return scores[:top_n]
