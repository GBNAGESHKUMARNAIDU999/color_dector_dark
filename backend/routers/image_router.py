"""
FastAPI image processing router.
Endpoints for image color detection, dominant extraction, and pixel analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional

from backend.schemas.response_schema import ImageAnalysisResponse, SuccessResponse
from backend.services.image_service import analyze_image, get_color_info
from utils.validators import validate_image_file, validate_file_size, MAX_IMAGE_BYTES
from utils.logger import logger

router = APIRouter(prefix="/api/image", tags=["Image Detection"])


@router.post("/analyze", response_model=ImageAnalysisResponse)
async def analyze_image_endpoint(
    file: UploadFile = File(..., description="Image file to analyze"),
    target_colors: Optional[List[str]] = Query(
        default=None,
        description="Colors to detect (e.g. red, blue). Leave empty for all."
    ),
    k_clusters: int = Query(default=5, ge=1, le=12, description="Dominant colors to extract"),
    click_x: Optional[int] = Query(default=None, description="X pixel for click-to-detect"),
    click_y: Optional[int] = Query(default=None, description="Y pixel for click-to-detect"),
):
    """
    Full color analysis of an uploaded image.

    Returns:
    - Detected color regions with bounding boxes + confidence
    - K-Means dominant color palette
    - Optional pixel-click color info
    """
    validate_image_file(file)
    content = await file.read()
    validate_file_size(content, MAX_IMAGE_BYTES, label="Image")

    logger.info(f"Analyzing image: {file.filename} ({len(content)} bytes)")

    try:
        result = analyze_image(
            image_bytes=content,
            target_colors=target_colors,
            k_clusters=k_clusters,
            click_x=click_x,
            click_y=click_y,
        )
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {**result, "success": True}


@router.get("/color-info")
async def color_info_endpoint(
    r: int = Query(..., ge=0, le=255, description="Red channel"),
    g: int = Query(..., ge=0, le=255, description="Green channel"),
    b: int = Query(..., ge=0, le=255, description="Blue channel"),
):
    """
    Return full color information for an RGB triplet:
    - All color space representations (RGB, HSV, HEX, LAB)
    - Nearest named color from database
    - Similar colors (Delta-E 2000)
    - All palette types
    """
    try:
        result = get_color_info(r, g, b)
        return SuccessResponse(data=result)
    except Exception as e:
        logger.error(f"Color info failed for ({r},{g},{b}): {e}")
        raise HTTPException(status_code=500, detail=str(e))
