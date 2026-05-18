"""
Color space conversion utilities.
Supports BGR, RGB, HSV, HEX, and LAB color spaces with full docstrings.
"""

import cv2
import numpy as np
from typing import Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Type Aliases
# ──────────────────────────────────────────────────────────────────────────────
BGR = Tuple[int, int, int]   # (B, G, R) — OpenCV native
RGB = Tuple[int, int, int]   # (R, G, B)
HSV = Tuple[int, int, int]   # (H 0-179, S 0-255, V 0-255) — OpenCV scale
LAB = Tuple[float, float, float]
HEX = str                    # "#RRGGBB"


# ──────────────────────────────────────────────────────────────────────────────
# RGB ↔ BGR
# ──────────────────────────────────────────────────────────────────────────────

def rgb_to_bgr(r: int, g: int, b: int) -> BGR:
    """Convert RGB to OpenCV's BGR tuple."""
    return (b, g, r)


def bgr_to_rgb(b: int, g: int, r: int) -> RGB:
    """Convert OpenCV BGR tuple to RGB."""
    return (r, g, b)


# ──────────────────────────────────────────────────────────────────────────────
# RGB ↔ HEX
# ──────────────────────────────────────────────────────────────────────────────

def rgb_to_hex(r: int, g: int, b: int) -> HEX:
    """
    Convert an (R, G, B) triplet to a HEX colour string.

    Args:
        r, g, b: Integer values 0-255.

    Returns:
        Hex string in the form '#RRGGBB'.
    """
    r, g, b = int(r), int(g), int(b)
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_rgb(hex_color: HEX) -> RGB:
    """
    Convert a HEX colour string to an (R, G, B) triplet.

    Args:
        hex_color: Colour string like '#RRGGBB' or 'RRGGBB'.

    Returns:
        (R, G, B) tuple with values 0-255.

    Raises:
        ValueError: If the string is not a valid 6-digit hex colour.
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid HEX colour: #{hex_color}")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)


# ──────────────────────────────────────────────────────────────────────────────
# RGB ↔ HSV  (OpenCV scale: H 0-179, S 0-255, V 0-255)
# ──────────────────────────────────────────────────────────────────────────────

def rgb_to_hsv(r: int, g: int, b: int) -> HSV:
    """
    Convert RGB to HSV using OpenCV's scale (H: 0-179).

    Returns:
        (H, S, V) tuple.
    """
    pixel = np.uint8([[[b, g, r]]])           # OpenCV expects BGR
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
    h, s, v = int(hsv[0, 0, 0]), int(hsv[0, 0, 1]), int(hsv[0, 0, 2])
    return (h, s, v)


def hsv_to_rgb(h: int, s: int, v: int) -> RGB:
    """
    Convert HSV (OpenCV scale) back to RGB.

    Args:
        h: Hue 0-179
        s: Saturation 0-255
        v: Value 0-255

    Returns:
        (R, G, B) tuple.
    """
    pixel = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)
    b, g, r = int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])
    return (r, g, b)


# ──────────────────────────────────────────────────────────────────────────────
# RGB ↔ LAB
# ──────────────────────────────────────────────────────────────────────────────

def rgb_to_lab(r: int, g: int, b: int) -> LAB:
    """
    Convert RGB to CIE LAB colour space via OpenCV.

    LAB is perceptually uniform — ideal for colour similarity calculations.

    Returns:
        (L, a, b) floats.
    """
    pixel = np.uint8([[[b, g, r]]])
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2Lab)
    L = float(lab[0, 0, 0])
    a = float(lab[0, 0, 1]) - 128.0
    b_ = float(lab[0, 0, 2]) - 128.0
    return (L, a, b_)


def lab_to_rgb(L: float, a: float, b_: float) -> RGB:
    """
    Convert CIE LAB back to RGB.

    Args:
        L: Lightness 0-100
        a, b_: Chromaticity (approx -128 to +127)

    Returns:
        (R, G, B) tuple.
    """
    pixel = np.uint8([[[int(L), int(a + 128), int(b_ + 128)]]])
    bgr = cv2.cvtColor(pixel, cv2.COLOR_Lab2BGR)
    b, g, r = int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])
    return (r, g, b)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: all formats from a single RGB input
# ──────────────────────────────────────────────────────────────────────────────

def get_all_formats(r: int, g: int, b: int) -> dict:
    """
    Return all colour space representations for a given RGB value.

    Args:
        r, g, b: Integer values 0-255.

    Returns:
        dict with keys: rgb, bgr, hsv, hex, lab
    """
    return {
        "rgb": {"R": r, "G": g, "B": b},
        "bgr": {"B": b, "G": g, "R": r},
        "hsv": dict(zip(["H", "S", "V"], rgb_to_hsv(r, g, b))),
        "hex": rgb_to_hex(r, g, b),
        "lab": dict(zip(["L", "a", "b"], rgb_to_lab(r, g, b))),
    }
