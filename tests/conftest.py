"""Pytest configuration and shared fixtures."""
import sys
import os
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def solid_red_image():
    """100x100 pure red BGR image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = [0, 0, 255]   # BGR red
    return img


@pytest.fixture
def solid_green_image():
    """100x100 pure green BGR image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = [0, 255, 0]
    return img


@pytest.fixture
def solid_blue_image():
    """100x100 pure blue BGR image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = [255, 0, 0]
    return img


@pytest.fixture
def multi_color_image():
    """400x100 image with red, green, blue, yellow quarters."""
    img = np.zeros((100, 400, 3), dtype=np.uint8)
    img[:, :100]   = [0, 0, 255]     # Red
    img[:, 100:200] = [0, 255, 0]    # Green
    img[:, 200:300] = [255, 0, 0]    # Blue
    img[:, 300:]   = [0, 255, 255]   # Yellow
    return img


@pytest.fixture
def sample_image_bytes(solid_red_image):
    """Encoded JPEG bytes of a red image."""
    import cv2
    _, buf = cv2.imencode(".jpg", solid_red_image)
    return buf.tobytes()
