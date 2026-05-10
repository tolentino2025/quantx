import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from scale_detection import detect_scale, get_recommended_config
from yolo_inference import YOLOInferenceEngine, TileConfig, ModelRegistry
from dinov2_inference import (
    DINOv2Embedder,
    FewShotMatcher,
    InMemoryEmbeddingStore,
    DetectionHint,
    similarity_tier,
)
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
_embedding_store = InMemoryEmbeddingStore()  # swap for PostgresEmbeddingStore in production


class EmbedRequest(BaseModel):
    tenant_id: str
    class_slug: str
    sample_id: Optional[str] = None
    image_path: str
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    dinov2_window: int = 128


class MatchRequest(BaseModel):
    tenant_id: str
    image_path: str
    detections: list[dict]   # list of DetectionHint dicts
    dinov2_window: int = 128
    top_k: int = 5


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


class ScaleDetectionReverseRequest(ScaleDetectionRequest, ReverseInferenceHint):
    """Combined body for the reverse-inference endpoint (ScaleDetectionRequest + ReverseInferenceHint)."""


@app.post("/scale-detection/reverse", response_model=ScaleDetectionResponse)
def scale_detection_reverse(request: ScaleDetectionReverseRequest) -> ScaleDetectionResponse:
    """
    Estimate scale from a known symbol's physical size and its detected bbox.
    Used when OCR and graphic bar both fail but a reference detection exists.
    """
    bbox = request.bbox
    bbox_px = max(bbox.x2 - bbox.x1, bbox.y2 - bbox.y1)
    if bbox_px <= 0:
        raise HTTPException(status_code=422, detail="bbox has zero size")

    pixels_per_mm = request.current_dpi / 25.4
    denominator = bbox_px / (request.known_physical_mm * pixels_per_mm)

    from scale_detection import snap_to_common_scale, _estimate_cartouche_region

    img = Image.open(request.image_path)
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
            f"Escala estimada por inferência reversa usando '{request.class_slug}' "
            f"({request.known_physical_mm}mm físicos, {bbox_px}px detectados a {request.current_dpi}dpi). "
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


@app.post("/dinov2/embed")
def dinov2_embed(request: EmbedRequest) -> dict:
    """
    Generate and persist a DINOv2 embedding for a new catalog sample.
    Called by Dataset Curation Agent after each human correction.
    """
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")

    try:
        embedder = DINOv2Embedder()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DINOv2 model unavailable: {exc}")

    matcher = FewShotMatcher(embedder=embedder, store=_embedding_store)
    image = Image.open(request.image_path).convert("RGB")
    crop = DINOv2Embedder.make_crop(
        image,
        request.bbox_x1, request.bbox_y1,
        request.bbox_x2, request.bbox_y2,
        request.dinov2_window,
    )
    record = matcher.embed_and_store(
        tenant_id=request.tenant_id,
        class_slug=request.class_slug,
        crop=crop,
        sample_id=request.sample_id,
    )

    log.info("Embedding stored | tenant=%s class=%s sample=%s", request.tenant_id, request.class_slug, record.sample_id)
    return {
        "sample_id": record.sample_id,
        "tenant_id": record.tenant_id,
        "class_slug": record.class_slug,
        "embedding_dim": len(record.embedding),
        "library_size": _embedding_store.count_by_tenant(request.tenant_id),
    }


@app.post("/dinov2/match")
def dinov2_match(request: MatchRequest) -> dict:
    """
    Match a batch of YOLO detections against the tenant's personal library.
    Returns similarity scores and agreement with YOLO classification.
    """
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")

    library_size = _embedding_store.count_by_tenant(request.tenant_id)
    if library_size == 0:
        return {
            "tenant_id": request.tenant_id,
            "library_size": 0,
            "matches": [],
            "warning": "tenant_library_empty",
        }

    try:
        embedder = DINOv2Embedder()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DINOv2 model unavailable: {exc}")

    hints = [
        DetectionHint(
            detection_id=d["detection_id"],
            tile_id=d["tile_id"],
            yolo_class_slug=d["class_slug"],
            bbox_x1=d["bbox_tile"]["x1"],
            bbox_y1=d["bbox_tile"]["y1"],
            bbox_x2=d["bbox_tile"]["x2"],
            bbox_y2=d["bbox_tile"]["y2"],
            yolo_confidence=d.get("confidence", 0.0),
        )
        for d in request.detections
    ]

    image = Image.open(request.image_path).convert("RGB")
    matcher = FewShotMatcher(embedder=embedder, store=_embedding_store)
    results = matcher.match_batch(
        request.tenant_id, image, hints,
        dinov2_window=request.dinov2_window,
        top_k=request.top_k,
    )

    matches = []
    for hint, match in zip(hints, results):
        if match is None:
            matches.append({
                "detection_id": hint.detection_id,
                "matched": False,
                "similarity": 0.0,
                "tier": "no_match",
            })
        else:
            matches.append({
                "detection_id": match.detection_id,
                "matched": True,
                "matched_class_slug": match.matched_class_slug,
                "yolo_class_slug": match.class_slug,
                "similarity": match.similarity,
                "tier": similarity_tier(match.similarity),
                "agreement": match.agreement,
                "top_k": [
                    {"class_slug": m.class_slug, "similarity": m.similarity}
                    for m in match.top_k
                ],
            })

    return {
        "tenant_id": request.tenant_id,
        "library_size": library_size,
        "matches": matches,
    }


@app.delete("/dinov2/library/{tenant_id}/{class_slug}")
def dinov2_delete_class(tenant_id: str, class_slug: str) -> dict:
    """Soft-delete all embeddings for a class (use after catalog soft-delete)."""
    deleted = _embedding_store.delete_by_class(tenant_id, class_slug)
    return {
        "tenant_id": tenant_id,
        "class_slug": class_slug,
        "deleted_count": deleted,
        "library_size": _embedding_store.count_by_tenant(tenant_id),
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    log.exception("Unhandled exception in ML service")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )
