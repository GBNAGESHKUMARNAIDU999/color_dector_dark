"""Pydantic schemas for API responses."""
from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class SuccessResponse(BaseModel):
    success: bool = True
    message: str = "OK"
    data: Any = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None


class ImageAnalysisResponse(BaseModel):
    success: bool = True
    image_size: Dict[str, int]          # {width, height}
    detected_colors: List[Dict]
    dominant_colors: List[Dict]
    pixel_color: Optional[Dict] = None
    processing_time_ms: float


class VideoAnalysisResponse(BaseModel):
    success: bool = True
    total_frames: int
    processed_frames: int
    fps: float
    duration_sec: float
    frame_results: List[Dict]
    dominant_scene_colors: List[Dict]
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    service: str = "Color Detection API"
