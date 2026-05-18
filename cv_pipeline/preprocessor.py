"""
Image and frame preprocessing utilities.
Handles resize, denoising, morphological operations, and histogram equalization.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from utils.config import settings
from utils.logger import logger


def resize_with_aspect(
    image: np.ndarray,
    target_width: int = settings.DEFAULT_RESIZE_WIDTH,
) -> np.ndarray:
    """
    Resize an image to a target width while preserving aspect ratio.

    Args:
        image: Input BGR image as numpy array.
        target_width: Desired output width in pixels.

    Returns:
        Resized BGR image.
    """
    h, w = image.shape[:2]
    if w <= target_width:
        return image
    scale = target_width / w
    new_h = int(h * scale)
    resized = cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)
    logger.debug(f"Resized: {w}x{h} → {target_width}x{new_h}")
    return resized


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: int = settings.BLUR_KERNEL_SIZE,
) -> np.ndarray:
    """
    Apply Gaussian blur for noise reduction.

    Args:
        image: Input image.
        kernel_size: Odd kernel size (e.g., 5 for 5x5 kernel).

    Returns:
        Blurred image.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1   # ensure odd
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_bilateral_filter(image: np.ndarray) -> np.ndarray:
    """
    Apply bilateral filter — edges preserved, noise reduced.
    Better than Gaussian for colour segmentation tasks.

    Args:
        image: Input BGR image.

    Returns:
        Filtered image.
    """
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB space.
    Enhances local contrast without over-amplifying noise.

    Args:
        image: Input BGR image.

    Returns:
        Contrast-enhanced BGR image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


def morphological_open(
    mask: np.ndarray,
    kernel_size: int = settings.MORPH_KERNEL_SIZE,
) -> np.ndarray:
    """
    Morphological opening — removes small noise blobs from a binary mask.

    Args:
        mask: Binary mask (uint8).
        kernel_size: Square kernel side length.

    Returns:
        Cleaned binary mask.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)


def morphological_close(
    mask: np.ndarray,
    kernel_size: int = settings.MORPH_KERNEL_SIZE,
) -> np.ndarray:
    """
    Morphological closing — fills small holes in a binary mask.

    Args:
        mask: Binary mask (uint8).
        kernel_size: Square kernel side length.

    Returns:
        Filled binary mask.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Dilate a binary mask to expand detected regions."""
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(mask, kernel, iterations=iterations)


def preprocess_image(
    image: np.ndarray,
    resize: bool = True,
    denoise: bool = True,
    enhance: bool = False,
) -> np.ndarray:
    """
    Full preprocessing pipeline for an input image.

    Args:
        image: Raw BGR image.
        resize: Whether to resize to DEFAULT_RESIZE_WIDTH.
        denoise: Whether to apply bilateral filter.
        enhance: Whether to apply CLAHE contrast enhancement.

    Returns:
        Preprocessed BGR image ready for detection.
    """
    if resize:
        image = resize_with_aspect(image)
    if denoise:
        image = apply_bilateral_filter(image)
    if enhance:
        image = apply_clahe(image)
    return image


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Lightweight frame preprocessing for real-time webcam/video streams.
    Avoids heavy operations to maintain FPS.

    Args:
        frame: Raw BGR frame.

    Returns:
        Preprocessed frame.
    """
    frame = resize_with_aspect(frame, target_width=640)
    frame = apply_gaussian_blur(frame, kernel_size=5)
    return frame


def bytes_to_image(data: bytes) -> Optional[np.ndarray]:
    """
    Decode raw bytes into an OpenCV BGR image.

    Args:
        data: Raw image file bytes.

    Returns:
        BGR numpy array or None if decoding failed.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        logger.error("Failed to decode image from bytes.")
    return image


def image_to_bytes(image: np.ndarray, ext: str = ".jpg") -> bytes:
    """
    Encode an OpenCV image to bytes.

    Args:
        image: BGR numpy array.
        ext: File extension to encode as (e.g., '.jpg', '.png').

    Returns:
        Encoded image bytes.
    """
    success, buffer = cv2.imencode(ext, image)
    if not success:
        raise RuntimeError("Failed to encode image to bytes.")
    return buffer.tobytes()
