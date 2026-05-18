"""
FastAPI video analysis router.
Endpoints for frame-by-frame color timeline analysis.
"""

import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from typing import List, Optional

from backend.schemas.response_schema import VideoAnalysisResponse, SuccessResponse
from backend.services.video_service import analyze_video
from utils.validators import validate_video_file, validate_file_size, MAX_VIDEO_BYTES
from utils.logger import logger

router = APIRouter(prefix="/api/video", tags=["Video Analysis"])


@router.post("/analyze", response_model=VideoAnalysisResponse)
async def analyze_video_endpoint(
    file: UploadFile = File(..., description="Video file to analyze"),
    target_colors: Optional[List[str]] = Query(default=None),
    sample_every_n_frames: int = Query(default=5, ge=1, le=30),
    k_clusters: int = Query(default=5, ge=1, le=10),
):
    """
    Analyze a video file for color content:
    - Frame-by-frame color detection timeline
    - Dominant scene color palette
    - Per-frame timestamps and summaries
    """
    validate_video_file(file)
    content = await file.read()
    validate_file_size(content, MAX_VIDEO_BYTES, label="Video")

    logger.info(f"Analyzing video: {file.filename} ({len(content) / 1e6:.1f} MB)")

    # Save to temp file for OpenCV to read
    suffix = os.path.splitext(file.filename or ".mp4")[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = analyze_video(
            video_path=tmp_path,
            target_colors=target_colors,
            sample_every_n_frames=sample_every_n_frames,
            k_clusters=k_clusters,
        )
    except Exception as e:
        logger.error(f"Video analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {**result, "success": True}
