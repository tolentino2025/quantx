import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from scale_detection import detect_scale, get_recommended_config
from yolo_inference import YOLOInferenceEngine, TileConfig, ModelRegistry
from schemas import (
    ScaleDetectionRequest,
    ScaleDetectionResponse,
    ReverseInferenceHint,
    BBox,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


_registry = ModelRegistry()
_yolo_engine = YOLOInferenceEngine(registry=_registry)


class YOLOInferenceRequest(BaseModel):
    plan_id: str
    page_number: int = 1
    image_path: str
    model_version: str
    tile_size: int = 1024
    overlap: float = 0.20
    imgsz: int = 1024
    conf: float = 0.25
    iou: float = 0.45
    legend_region: Optional[BBox] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("QuantX ML service starting up")
    yield
    log.info("QuantX ML service shutting down")


app = FastAPI(
    title="QuantX ML Service",
    description="SPCI symbol detection pipeline — YOLO, DINOv2, Scale Detection",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "quantx-ml"}


@app.post("/scale-detection", response_model=ScaleDetectionResponse)
def scale_detection(request: ScaleDetectionRequest) -> ScaleDetectionResponse:
    """
    Detect drawing scale from a rendered PDF page image.
    Tries OCR on cartouche, then graphic bar, then full-page OCR, then fallback.
    Also identifies the legend region to be masked before inference.
    """
    if not os.path.exists(request.image_path):
        raise HTTPException(
            status_code=404,
            detail=f"Image not found: {request.image_path}",
        )

    log.info(
        "Scale detection started | plan_id=%s page=%d",
        request.plan_id,
        request.page_number,
    )

    result = detect_scale(request.image_path)
    recommended = get_recommended_config(result.scale_denominator)

    log.info(
        "Scale detection done | plan_id=%s scale=%s confidence=%.2f method=%s",
        request.plan_id,
        result.scale,
        result.confidence,
        result.method,
    )

    return ScaleDetectionResponse(
        plan_id=request.plan_id,
        page_number=request.page_number,
        scale=result.scale,
        scale_confidence=result.confidence,
        method=result.method,
        cartouche_region=result.cartouche_region,
        legend_region=result.legend_region,
        recommended=recommended,
        uncertainty_note=result.uncertainty_note,
        warnings=result.warnings,
    )


@app.post("/scale-detection/reverse", response_model=ScaleDetectionResponse)
def scale_detection_reverse(
    request: ScaleDetectionRequest,
    hint: ReverseInferenceHint,
) -> ScaleDetectionResponse:
    """
    Estimate scale from a known symbol's physical size and its detected bbox.
    Used when OCR and graphic bar both fail but a reference detection exists.
    """
    bbox = hint.bbox
    bbox_px = max(bbox.x2 - bbox.x1, bbox.y2 - bbox.y1)
    if bbox_px <= 0:
        raise HTTPException(status_code=422, detail="bbox has zero size")

    pixels_per_mm = hint.current_dpi / 25.4
    denominator = bbox_px / (hint.known_physical_mm * pixels_per_mm)

    from scale_detection import snap_to_common_scale, _estimate_cartouche_region
    from PIL import Image as PILImage

    img = PILImage.open(request.image_path)
    width, height = img.size
    cartouche = _estimate_cartouche_region(width, height)

    snapped = snap_to_common_scale(int(denominator))
    recommended = get_recommended_config(snapped)

    warnings = []
    if abs(snapped - denominator) / snapped > 0.20:
        warnings.append(
            f"reverse_inference_high_uncertainty: estimated {denominator:.1f}, snapped to {snapped}"
        )

    return ScaleDetectionResponse(
        plan_id=request.plan_id,
        page_number=request.page_number,
        scale=f"1:{snapped}",
        scale_confidence=0.70,
        method="reverse_inference",
        cartouche_region=cartouche,
        legend_region=None,
        recommended=recommended,
        uncertainty_note=(
            f"Escala estimada por inferência reversa usando '{hint.class_slug}' "
            f"({hint.known_physical_mm}mm físicos, {bbox_px}px detectados a {hint.current_dpi}dpi). "
            f"Denominador bruto: {denominator:.1f}, ajustado para: {snapped}."
        ),
        warnings=warnings,
    )


@app.post("/yolo-inference")
def yolo_inference(request: YOLOInferenceRequest) -> dict:
    """
    Run YOLO inference on a rendered page image.
    Returns per-tile raw detections in tile-local coordinates.
    TileNMS (TypeScript) handles cross-tile deduplication downstream.
    """
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")

    log.info(
        "YOLO inference | plan_id=%s page=%d model=%s",
        request.plan_id, request.page_number, request.model_version,
    )

    try:
        result = _yolo_engine.run(
            plan_id=request.plan_id,
            page_number=request.page_number,
            image_path=request.image_path,
            model_version=request.model_version,
            tile_config=TileConfig(
                tile_size=request.tile_size,
                overlap=request.overlap,
                imgsz=request.imgsz,
                conf=request.conf,
                iou=request.iou,
            ),
            legend_region=request.legend_region,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return _yolo_engine.serialize_result(result)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    log.exception("Unhandled exception in ML service")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )
