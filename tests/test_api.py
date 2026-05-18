"""FastAPI endpoint integration tests."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
import cv2
from httpx import AsyncClient, ASGITransport
from backend.main import app


@pytest.fixture
def red_image_bytes():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = [0, 0, 255]
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_root_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert "Color Detection" in response.json()["message"]


@pytest.mark.asyncio
async def test_image_analyze_endpoint(red_image_bytes):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/image/analyze",
            files={"file": ("test.jpg", red_image_bytes, "image/jpeg")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "dominant_colors" in data
    assert "image_size" in data


@pytest.mark.asyncio
async def test_color_info_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/image/color-info?r=255&g=0&b=0")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


@pytest.mark.asyncio
async def test_invalid_file_type_rejected():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/image/analyze",
            files={"file": ("malicious.exe", b"MZ\x90\x00", "application/octet-stream")},
        )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_color_info_out_of_range():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/image/color-info?r=999&g=0&b=0")
    assert response.status_code == 422   # Validation error
