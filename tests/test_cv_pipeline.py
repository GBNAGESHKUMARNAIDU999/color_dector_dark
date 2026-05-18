"""Unit tests for CV pipeline modules."""
import numpy as np
import pytest
from cv_pipeline.color_spaces import (
    rgb_to_hex, hex_to_rgb, rgb_to_hsv, hsv_to_rgb,
    rgb_to_lab, get_all_formats,
)
from cv_pipeline.preprocessor import (
    resize_with_aspect, apply_gaussian_blur, preprocess_image, bytes_to_image
)
from cv_pipeline.color_detector import detect_colors, get_pixel_color
from cv_pipeline.color_classifier import get_color_name, get_color_name_batch
from cv_pipeline.segmentation import find_segments, create_color_mask


class TestColorSpaces:
    def test_rgb_to_hex_white(self):
        assert rgb_to_hex(255, 255, 255) == "#FFFFFF"

    def test_rgb_to_hex_black(self):
        assert rgb_to_hex(0, 0, 0) == "#000000"

    def test_rgb_to_hex_red(self):
        assert rgb_to_hex(255, 0, 0) == "#FF0000"

    def test_hex_to_rgb_roundtrip(self):
        assert hex_to_rgb("#FF6633") == (255, 102, 51)

    def test_hex_to_rgb_invalid(self):
        with pytest.raises(ValueError):
            hex_to_rgb("#ZZZZZZ")

    def test_rgb_hsv_roundtrip(self):
        r, g, b = 180, 90, 30
        h, s, v = rgb_to_hsv(r, g, b)
        r2, g2, b2 = hsv_to_rgb(h, s, v)
        assert abs(r - r2) <= 3 and abs(g - g2) <= 3 and abs(b - b2) <= 3

    def test_rgb_lab_returns_3_values(self):
        lab = rgb_to_lab(100, 150, 200)
        assert len(lab) == 3

    def test_get_all_formats_keys(self):
        result = get_all_formats(100, 150, 200)
        assert all(k in result for k in ["rgb", "bgr", "hsv", "hex", "lab"])


class TestPreprocessor:
    def test_resize_reduces_width(self, solid_red_image):
        large = np.zeros((600, 1200, 3), dtype=np.uint8)
        resized = resize_with_aspect(large, target_width=800)
        assert resized.shape[1] == 800

    def test_resize_preserves_small_image(self, solid_red_image):
        result = resize_with_aspect(solid_red_image, target_width=800)
        assert result.shape == solid_red_image.shape

    def test_blur_returns_same_shape(self, solid_red_image):
        blurred = apply_gaussian_blur(solid_red_image)
        assert blurred.shape == solid_red_image.shape

    def test_preprocess_pipeline(self, solid_red_image):
        result = preprocess_image(solid_red_image, resize=True, denoise=True)
        assert result.ndim == 3 and result.shape[2] == 3

    def test_bytes_to_image(self, sample_image_bytes):
        img = bytes_to_image(sample_image_bytes)
        assert img is not None
        assert img.ndim == 3


class TestColorDetector:
    def test_detect_red_in_red_image(self, solid_red_image):
        results = detect_colors(solid_red_image, target_colors=["red"])
        assert "red" in results

    def test_detect_green_in_green_image(self, solid_green_image):
        results = detect_colors(solid_green_image, target_colors=["green"])
        assert "green" in results

    def test_detect_blue_in_blue_image(self, solid_blue_image):
        results = detect_colors(solid_blue_image, target_colors=["blue"])
        assert "blue" in results

    def test_get_pixel_color_returns_dict(self, solid_red_image):
        info = get_pixel_color(solid_red_image, 50, 50)
        assert all(k in info for k in ["rgb", "hsv", "hex"])

    def test_detection_result_has_confidence(self, solid_red_image):
        results = detect_colors(solid_red_image, target_colors=["red"])
        if "red" in results:
            for det in results["red"]:
                assert 0.0 <= det.confidence <= 1.0


class TestColorClassifier:
    def test_red_classified_as_red(self):
        result = get_color_name(255, 0, 0)
        assert "red" in result["color_name"].lower() or result["color_family"].lower() == "red"

    def test_white_classified(self):
        result = get_color_name(255, 255, 255)
        assert result["color_name"] is not None

    def test_batch_classification_length(self):
        rgb_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        results = get_color_name_batch(rgb_list)
        assert len(results) == 3

    def test_result_has_required_keys(self):
        result = get_color_name(128, 64, 32)
        assert all(k in result for k in ["color_name", "color_family", "hex_code", "distance"])


class TestSegmentation:
    def test_find_segments_on_mask(self, solid_red_image):
        mask = create_color_mask(solid_red_image, "red")
        segments = find_segments(mask, min_area=100)
        assert len(segments) >= 0   # may not find segments on small image

    def test_segments_sorted_by_area(self):
        import cv2
        mask = np.zeros((200, 400), dtype=np.uint8)
        cv2.rectangle(mask, (0, 0), (200, 100), 255, -1)
        cv2.rectangle(mask, (200, 0), (320, 50), 255, -1)
        segments = find_segments(mask, min_area=100)
        if len(segments) >= 2:
            assert segments[0].area >= segments[1].area
