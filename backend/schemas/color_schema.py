"""Pydantic schemas for color data."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class RGBColor(BaseModel):
    R: int = Field(..., ge=0, le=255)
    G: int = Field(..., ge=0, le=255)
    B: int = Field(..., ge=0, le=255)


class HSVColor(BaseModel):
    H: int = Field(..., ge=0, le=179)
    S: int = Field(..., ge=0, le=255)
    V: int = Field(..., ge=0, le=255)


class LABColor(BaseModel):
    L: float
    a: float
    b: float


class ColorInfo(BaseModel):
    rgb: RGBColor
    hsv: HSVColor
    hex: str
    color_name: str
    color_family: str
    lab: Optional[LABColor] = None


class DominantColor(BaseModel):
    rank: int
    rgb: RGBColor
    hex: str
    hsv: HSVColor
    color_name: str
    color_family: str
    percentage: float
    pixel_count: int


class DetectionBox(BaseModel):
    color_name: str
    confidence: float
    pixel_count: int
    percentage: float
    bounding_box: List[int]   # [x, y, w, h]
    center: List[int]         # [cx, cy]


class ColorSimilarityResult(BaseModel):
    color_name: str
    hex_code: str
    color_family: str
    rgb: RGBColor
    distance: float
    similarity_pct: float


class PaletteColor(BaseModel):
    rgb: RGBColor
    hex: str
    color_name: str
    color_family: str


class ColorPalette(BaseModel):
    palette_type: str
    colors: List[PaletteColor]
