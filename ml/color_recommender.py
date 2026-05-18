"""
Color recommendation engine.
Generates complementary, analogous, triadic, and split-complementary palettes.
Includes fashion and interior design color matchers.
"""

import colorsys
from typing import List, Dict, Tuple

from cv_pipeline.color_spaces import rgb_to_hex, rgb_to_hsv, hsv_to_rgb
from cv_pipeline.color_classifier import get_color_name_batch
from utils.logger import logger


# ──────────────────────────────────────────────────────────────────────────────
# Core palette generators (hue-wheel based)
# ──────────────────────────────────────────────────────────────────────────────

def _hsv_rotate(h_opencv: int, s: int, v: int, angle: float) -> Tuple[int, int, int]:
    """Rotate hue by angle degrees (OpenCV HSV scale: H 0-179)."""
    h_full = (h_opencv * 2 + angle) % 360          # Convert to 0-360
    h_ocv = int(h_full / 2) % 180                  # Back to OpenCV scale
    return hsv_to_rgb(h_ocv, s, v)


def _enrich(rgb_list: List[Tuple[int, int, int]]) -> List[Dict]:
    """Add hex, color_name, color_family to a list of RGB tuples."""
    names = get_color_name_batch(rgb_list)
    results = []
    for (r, g, b), info in zip(rgb_list, names):
        results.append({
            "rgb": {"R": r, "G": g, "B": b},
            "hex": rgb_to_hex(r, g, b),
            "color_name": info["color_name"],
            "color_family": info["color_family"],
        })
    return results


def complementary(r: int, g: int, b: int) -> List[Dict]:
    """Generate complementary color (180° opposite on hue wheel)."""
    h, s, v = rgb_to_hsv(r, g, b)
    comp = _hsv_rotate(h, s, v, 180)
    return _enrich([(r, g, b), comp])


def analogous(r: int, g: int, b: int, angle: float = 30.0) -> List[Dict]:
    """Generate 3-color analogous palette (±angle degrees)."""
    h, s, v = rgb_to_hsv(r, g, b)
    return _enrich([
        _hsv_rotate(h, s, v, -angle),
        (r, g, b),
        _hsv_rotate(h, s, v, angle),
    ])


def triadic(r: int, g: int, b: int) -> List[Dict]:
    """Generate triadic palette (3 colors, 120° apart)."""
    h, s, v = rgb_to_hsv(r, g, b)
    return _enrich([
        (r, g, b),
        _hsv_rotate(h, s, v, 120),
        _hsv_rotate(h, s, v, 240),
    ])


def split_complementary(r: int, g: int, b: int, angle: float = 30.0) -> List[Dict]:
    """Generate split-complementary palette (base + 2 colors near complement)."""
    h, s, v = rgb_to_hsv(r, g, b)
    return _enrich([
        (r, g, b),
        _hsv_rotate(h, s, v, 180 - angle),
        _hsv_rotate(h, s, v, 180 + angle),
    ])


def tetradic(r: int, g: int, b: int) -> List[Dict]:
    """Generate tetradic (square) palette — 4 colors, 90° apart."""
    h, s, v = rgb_to_hsv(r, g, b)
    return _enrich([
        (r, g, b),
        _hsv_rotate(h, s, v, 90),
        _hsv_rotate(h, s, v, 180),
        _hsv_rotate(h, s, v, 270),
    ])


def monochromatic(r: int, g: int, b: int, steps: int = 5) -> List[Dict]:
    """
    Generate monochromatic palette by varying lightness.

    Args:
        r, g, b: Base color.
        steps: Number of shades to generate.

    Returns:
        List of dicts from darkest to lightest.
    """
    h, s, v = rgb_to_hsv(r, g, b)
    results = []
    for i in range(steps):
        v_new = int(50 + (i / (steps - 1)) * 205)   # 50-255
        v_new = min(255, max(0, v_new))
        results.append(hsv_to_rgb(h, s, v_new))
    return _enrich(results)


# ──────────────────────────────────────────────────────────────────────────────
# Domain-specific recommenders
# ──────────────────────────────────────────────────────────────────────────────

# Fashion color pairing rules (skin tone → recommended palette type)
FASHION_RULES = {
    "warm":   ["analogous", "triadic"],
    "cool":   ["complementary", "split_complementary"],
    "neutral": ["monochromatic", "analogous"],
}

# Interior design mood → palette type mapping
INTERIOR_MOODS = {
    "calm":       "monochromatic",
    "energetic":  "triadic",
    "elegant":    "analogous",
    "bold":       "complementary",
    "balanced":   "split_complementary",
}


def fashion_color_matcher(r: int, g: int, b: int) -> Dict:
    """
    Suggest fashion color pairings for a given base color.

    Detects warm/cool/neutral tone and recommends appropriate palettes.

    Args:
        r, g, b: Base clothing/accessory color.

    Returns:
        Dict with tone classification and suggested palettes.
    """
    h, s, v = rgb_to_hsv(r, g, b)
    h_deg = h * 2   # convert to 0-360

    # Classify tone
    if s < 40:
        tone = "neutral"
    elif 0 <= h_deg < 60 or 330 <= h_deg <= 360:
        tone = "warm"
    elif 60 <= h_deg < 180:
        tone = "warm" if h_deg < 90 else "cool"
    else:
        tone = "cool"

    palettes = {}
    for palette_type in FASHION_RULES[tone]:
        fn = {
            "analogous": analogous,
            "triadic": triadic,
            "complementary": complementary,
            "split_complementary": split_complementary,
        }.get(palette_type)
        if fn:
            palettes[palette_type] = fn(r, g, b)

    return {
        "base_color": {
            "rgb": {"R": r, "G": g, "B": b},
            "hex": rgb_to_hex(r, g, b),
        },
        "tone": tone,
        "recommended_palettes": palettes,
        "styling_tip": _fashion_tip(tone),
    }


def _fashion_tip(tone: str) -> str:
    tips = {
        "warm": "Warm tones pair well with earth tones, gold accents, and terracotta.",
        "cool": "Cool tones complement silver, navy, and jewel tones beautifully.",
        "neutral": "Neutral tones are highly versatile — try monochromatic layering.",
    }
    return tips.get(tone, "")


def interior_palette_recommender(r: int, g: int, b: int, mood: str = "calm") -> Dict:
    """
    Recommend an interior design color palette based on mood.

    Args:
        r, g, b: Dominant wall or furniture color.
        mood: One of 'calm', 'energetic', 'elegant', 'bold', 'balanced'.

    Returns:
        Dict with mood, palette, and design tip.
    """
    mood = mood.lower()
    palette_type = INTERIOR_MOODS.get(mood, "monochromatic")

    fn_map = {
        "monochromatic": monochromatic,
        "triadic": triadic,
        "analogous": analogous,
        "complementary": complementary,
        "split_complementary": split_complementary,
    }
    fn = fn_map[palette_type]
    palette = fn(r, g, b)

    design_tips = {
        "calm":     "Use soft, muted shades and add natural textures like linen and wood.",
        "energetic":"Pair with bold accents, geometric patterns, and metallic finishes.",
        "elegant":  "Incorporate marble, brass, and deep jewel-tone accent pieces.",
        "bold":     "Let one statement wall dominate; keep furniture in neutral tones.",
        "balanced": "60-30-10 rule: 60% dominant, 30% secondary, 10% accent color.",
    }

    return {
        "base_color": {"rgb": {"R": r, "G": g, "B": b}, "hex": rgb_to_hex(r, g, b)},
        "mood": mood,
        "palette_type": palette_type,
        "palette": palette,
        "design_tip": design_tips.get(mood, ""),
    }


def generate_all_palettes(r: int, g: int, b: int) -> Dict:
    """
    Generate all palette types for a given base color.

    Args:
        r, g, b: Base color RGB values.

    Returns:
        Dict with all palette types as keys.
    """
    return {
        "complementary":       complementary(r, g, b),
        "analogous":           analogous(r, g, b),
        "triadic":             triadic(r, g, b),
        "split_complementary": split_complementary(r, g, b),
        "tetradic":            tetradic(r, g, b),
        "monochromatic":       monochromatic(r, g, b),
    }
