"""Unit tests for ML engine modules."""
import numpy as np
import pytest
from ml.kmeans_clustering import extract_dominant_colors, build_palette_strip
from ml.color_similarity import color_similarity_rgb, find_similar_colors, delta_e_cie2000
from ml.color_recommender import (
    complementary, analogous, triadic, split_complementary,
    monochromatic, fashion_color_matcher, interior_palette_recommender,
)


class TestKMeansClustering:
    def test_dominant_colors_count(self, solid_red_image):
        result = extract_dominant_colors(solid_red_image, k=3)
        assert len(result) == 3

    def test_percentages_sum_to_100(self, multi_color_image):
        result = extract_dominant_colors(multi_color_image, k=4)
        total = sum(c["percentage"] for c in result)
        assert abs(total - 100.0) < 1.0

    def test_result_keys(self, solid_red_image):
        result = extract_dominant_colors(solid_red_image, k=2)
        for item in result:
            assert all(k in item for k in ["rank", "rgb", "hex", "hsv", "color_name", "percentage"])

    def test_ranks_are_sequential(self, multi_color_image):
        result = extract_dominant_colors(multi_color_image, k=4)
        ranks = [c["rank"] for c in result]
        assert ranks == sorted(ranks)

    def test_palette_strip_shape(self, multi_color_image):
        dominant = extract_dominant_colors(multi_color_image, k=4)
        strip = build_palette_strip(dominant, width=600, height=80)
        assert strip.shape == (80, 600, 3)

    def test_k1_returns_single_color(self, solid_red_image):
        result = extract_dominant_colors(solid_red_image, k=1)
        assert len(result) == 1


class TestColorSimilarity:
    def test_identical_colors_zero_distance(self):
        result = color_similarity_rgb(128, 64, 32, 128, 64, 32)
        assert result["distance"] < 0.01

    def test_black_vs_white_max_distance(self):
        result = color_similarity_rgb(0, 0, 0, 255, 255, 255, method="euclidean_rgb")
        assert result["similarity_pct"] < 10

    def test_similarity_methods(self):
        for method in ["delta_e_2000", "delta_e_76", "euclidean_rgb"]:
            result = color_similarity_rgb(200, 100, 50, 210, 95, 55, method=method)
            assert "distance" in result and "similarity_pct" in result

    def test_find_similar_colors_count(self):
        result = find_similar_colors(255, 0, 0, top_n=5)
        assert len(result) == 5

    def test_find_similar_sorted_by_distance(self):
        result = find_similar_colors(0, 0, 255, top_n=5)
        dists = [r["distance"] for r in result]
        assert dists == sorted(dists)

    def test_delta_e_symmetry(self):
        lab1 = (50.0, 25.0, -30.0)
        lab2 = (60.0, -10.0, 20.0)
        assert abs(delta_e_cie2000(lab1, lab2) - delta_e_cie2000(lab2, lab1)) < 0.01


class TestColorRecommender:
    def test_complementary_returns_2_colors(self):
        result = complementary(255, 0, 0)
        assert len(result) == 2

    def test_analogous_returns_3_colors(self):
        result = analogous(0, 128, 255)
        assert len(result) == 3

    def test_triadic_returns_3_colors(self):
        result = triadic(100, 200, 50)
        assert len(result) == 3

    def test_monochromatic_steps(self):
        result = monochromatic(128, 64, 200, steps=5)
        assert len(result) == 5

    def test_palette_color_has_rgb_and_hex(self):
        result = complementary(200, 100, 50)
        for c in result:
            assert "rgb" in c and "hex" in c

    def test_fashion_matcher_keys(self):
        result = fashion_color_matcher(200, 50, 80)
        assert all(k in result for k in ["base_color", "tone", "recommended_palettes", "styling_tip"])

    def test_fashion_tone_valid(self):
        result = fashion_color_matcher(200, 50, 80)
        assert result["tone"] in ["warm", "cool", "neutral"]

    def test_interior_recommender_keys(self):
        result = interior_palette_recommender(70, 130, 180, "calm")
        assert all(k in result for k in ["mood", "palette_type", "palette", "design_tip"])

    def test_interior_all_moods(self):
        for mood in ["calm", "energetic", "elegant", "bold", "balanced"]:
            result = interior_palette_recommender(100, 100, 100, mood)
            assert result["mood"] == mood
