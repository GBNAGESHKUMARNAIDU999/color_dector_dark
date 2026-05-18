"""
Input validation utilities for file uploads and API requests.
Provides security checks for file type, size, and content integrity.
"""

from pathlib import Path
from typing import Optional

from fastapi import UploadFile, HTTPException
from utils.config import settings
from utils.logger import logger

MAX_IMAGE_BYTES = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
MAX_VIDEO_BYTES = settings.MAX_VIDEO_SIZE_MB * 1024 * 1024

# Magic byte signatures for common image formats
IMAGE_SIGNATURES = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG": "png",
    b"BM": "bmp",
    b"GIF": "gif",
    b"RIFF": "webp",  # partial check; full check needs bytes 8-12
}


def _check_magic_bytes(content: bytes) -> Optional[str]:
    """Detect file type from magic bytes (first few bytes)."""
    for sig, fmt in IMAGE_SIGNATURES.items():
        if content[: len(sig)] == sig:
            return fmt
    return None


def validate_image_file(file: UploadFile) -> None:
    """
    Validate an uploaded image for extension and MIME type.

    Args:
        file: FastAPI UploadFile instance.

    Raises:
        HTTPException 400: If validation fails.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported extension '{ext}'. Allowed: {settings.ALLOWED_IMAGE_EXTENSIONS}",
        )
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid MIME type: {content_type}",
        )
    logger.debug(f"Image validated: {file.filename} [{content_type}]")


def validate_video_file(file: UploadFile) -> None:
    """
    Validate an uploaded video for extension.

    Args:
        file: FastAPI UploadFile instance.

    Raises:
        HTTPException 400: If validation fails.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported extension '{ext}'. Allowed: {settings.ALLOWED_VIDEO_EXTENSIONS}",
        )
    logger.debug(f"Video validated: {file.filename}")


def validate_file_size(content: bytes, max_bytes: int, label: str = "File") -> None:
    """
    Assert file content does not exceed the byte limit.

    Args:
        content: Raw file bytes.
        max_bytes: Maximum allowed size in bytes.
        label: Label used in the error message.

    Raises:
        HTTPException 413: If content exceeds max_bytes.
    """
    if len(content) > max_bytes:
        size_mb = len(content) / (1024 * 1024)
        limit_mb = max_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"{label} too large ({size_mb:.1f} MB). Limit: {limit_mb:.0f} MB",
        )


def validate_image_content(content: bytes) -> None:
    """
    Validate image magic bytes to prevent disguised file uploads.

    Args:
        content: Raw bytes of the uploaded file.

    Raises:
        HTTPException 400: If magic bytes don't match known image formats.
    """
    detected = _check_magic_bytes(content)
    if detected is None:
        raise HTTPException(
            status_code=400,
            detail="File content does not match a supported image format.",
        )
