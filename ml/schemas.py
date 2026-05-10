from pydantic import BaseModel, Field
from typing import Optional


class RecommendedConfig(BaseModel):
    dpi: int
    tile_size: int
    overlap: float
    imgsz: int
    dinov2_window: int


class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class ScaleDetectionRequest(BaseModel):
    plan_id: str
    page_number: int = 1
    image_path: str = Field(..., description="Path to rendered page image (low DPI ok)")
    tenant_id: str


class ScaleDetectionResponse(BaseModel):
    plan_id: str
    page_number: int
    scale: str = Field(..., examples=["1:100"])
    scale_confidence: float = Field(..., ge=0.0, le=1.0)
    method: str = Field(
        ...,
        description="ocr_cartouche | graphic_bar | reverse_inference | fallback",
    )
    cartouche_region: Optional[BBox] = None
    legend_region: Optional[BBox] = None
    recommended: RecommendedConfig
    uncertainty_note: str
    warnings: list[str] = []


class ReverseInferenceHint(BaseModel):
    class_slug: str
    bbox: BBox
    known_physical_mm: float = Field(
        ..., description="Known physical size of the symbol in mm"
    )
    current_dpi: int
