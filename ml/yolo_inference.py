import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from schemas import BBox

log = logging.getLogger(__name__)

# ─── data classes ─────────────────────────────────────────────────────────────


@dataclass
class TileConfig:
    tile_size: int
    overlap: float
    imgsz: int
    conf: float = 0.25
    iou: float = 0.45


@dataclass
class TileInfo:
    tile_id: str
    origin_x: int
    origin_y: int
    width: int
    height: int


@dataclass
class RawDetection:
    detection_id: str
    tile_id: str
    class_id: int
    class_slug: str
    bbox_tile: BBox          # coords relative to tile
    confidence: float
    model_version: str


@dataclass
class TileResult:
    tile: TileInfo
    detections: list[RawDetection] = field(default_factory=list)


@dataclass
class InferenceResult:
    plan_id: str
    page_number: int
    model_version: str
    tile_results: list[TileResult]
    warnings: list[str] = field(default_factory=list)

    @property
    def total_detections(self) -> int:
        return sum(len(t.detections) for t in self.tile_results)


# ─── model registry ───────────────────────────────────────────────────────────


class ModelRegistry:
    """Loads and caches YOLO models. Validates classes against the DB."""

    def __init__(self, models_dir: str = "models"):
        self._models_dir = Path(models_dir)
        self._cache: dict[str, object] = {}
        self._class_maps: dict[str, dict[int, str]] = {}

    def load(self, model_version: str) -> tuple[object, dict[int, str]]:
        if model_version in self._cache:
            return self._cache[model_version], self._class_maps[model_version]

        model_path = self._models_dir / f"{model_version}.pt"
        classes_path = self._models_dir / f"{model_version}.classes.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if not classes_path.exists():
            raise FileNotFoundError(
                f"Classes file not found: {classes_path}. "
                "Run validate:model before activating a new model."
            )

        class_map = self._load_class_map(classes_path)

        # Lazy import — ultralytics only needed at inference time
        from ultralytics import YOLO  # type: ignore
        model = YOLO(str(model_path))
        self._validate_model_classes(model, class_map, model_version)

        self._cache[model_version] = model
        self._class_maps[model_version] = class_map

        log.info("Model loaded: %s (%d classes)", model_version, len(class_map))
        return model, class_map

    def _load_class_map(self, path: Path) -> dict[int, str]:
        with open(path) as f:
            data = json.load(f)
        # Expected format: {"0": "sprinkler-pendent-k57", "1": "extintor-pqs", ...}
        return {int(k): v for k, v in data.items()}

    def _validate_model_classes(
        self, model: object, class_map: dict[int, str], version: str
    ) -> None:
        model_names = getattr(model, "names", {})
        if len(model_names) != len(class_map):
            raise ValueError(
                f"Model {version} has {len(model_names)} classes "
                f"but classes file has {len(class_map)}. "
                "Update the classes file or use the correct model."
            )
        log.info("Model class count validated: %d classes", len(class_map))


# ─── tiler ────────────────────────────────────────────────────────────────────


class ImageTiler:
    """Splits a page image into overlapping tiles."""

    def tile(self, image: Image.Image, config: TileConfig) -> list[TileInfo]:
        w, h = image.size
        step = int(config.tile_size * (1 - config.overlap))
        tiles: list[TileInfo] = []

        y = 0
        while y < h:
            x = 0
            while x < w:
                tw = min(config.tile_size, w - x)
                th = min(config.tile_size, h - y)
                tiles.append(TileInfo(
                    tile_id=str(uuid4()),
                    origin_x=x,
                    origin_y=y,
                    width=tw,
                    height=th,
                ))
                x += step
            y += step

        log.debug("Tiled image %dx%d into %d tiles (size=%d, overlap=%.0f%%)",
                  w, h, len(tiles), config.tile_size, config.overlap * 100)
        return tiles

    def crop(self, image: Image.Image, tile: TileInfo) -> Image.Image:
        crop = image.crop((
            tile.origin_x,
            tile.origin_y,
            tile.origin_x + tile.width,
            tile.origin_y + tile.height,
        ))
        # Pad to square tile_size if the tile is at the image boundary
        if crop.size != (tile.width, tile.height):
            return crop
        return crop


# ─── inference engine ─────────────────────────────────────────────────────────


class YOLOInferenceEngine:
    """Runs YOLO inference per tile and collects raw detections."""

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self._registry = registry or ModelRegistry()
        self._tiler = ImageTiler()

    def run(
        self,
        plan_id: str,
        page_number: int,
        image_path: str,
        model_version: str,
        tile_config: TileConfig,
        legend_region: Optional[BBox] = None,
    ) -> InferenceResult:
        warnings: list[str] = []

        image = self._load_and_preprocess(image_path, legend_region, warnings)
        model, class_map = self._registry.load(model_version)
        tiles = self._tiler.tile(image, tile_config)

        log.info(
            "Inference started | plan_id=%s page=%d model=%s tiles=%d",
            plan_id, page_number, model_version, len(tiles),
        )

        tile_results: list[TileResult] = []
        for tile in tiles:
            crop = self._tiler.crop(image, tile)
            detections = self._infer_tile(crop, tile, model, class_map, tile_config, model_version)
            tile_results.append(TileResult(tile=tile, detections=detections))

        result = InferenceResult(
            plan_id=plan_id,
            page_number=page_number,
            model_version=model_version,
            tile_results=tile_results,
            warnings=warnings,
        )

        log.info(
            "Inference done | plan_id=%s total_detections=%d",
            plan_id, result.total_detections,
        )
        return result

    def _load_and_preprocess(
        self,
        image_path: str,
        legend_region: Optional[BBox],
        warnings: list[str],
    ) -> Image.Image:
        image = Image.open(image_path).convert("RGB")

        if legend_region:
            draw = ImageDraw.Draw(image)
            draw.rectangle(
                (legend_region.x1, legend_region.y1, legend_region.x2, legend_region.y2),
                fill=(255, 255, 255),
            )
            log.debug("Legend region masked: %s", legend_region)
        else:
            warnings.append("legend_region_not_provided: false positives from legend possible")

        return image

    def _infer_tile(
        self,
        crop: Image.Image,
        tile: TileInfo,
        model: object,
        class_map: dict[int, str],
        config: TileConfig,
        model_version: str,
    ) -> list[RawDetection]:
        # Resize crop to imgsz if needed
        infer_img = crop
        if max(crop.size) != config.imgsz:
            scale = config.imgsz / max(crop.size)
            new_w = int(crop.width * scale)
            new_h = int(crop.height * scale)
            infer_img = crop.resize((new_w, new_h), Image.LANCZOS)

        results = model.predict(  # type: ignore
            source=np.array(infer_img),
            imgsz=config.imgsz,
            conf=config.conf,
            iou=config.iou,
            verbose=False,
        )

        detections: list[RawDetection] = []
        scale_back = crop.width / infer_img.width

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_slug = class_map.get(class_id, f"unknown-class-{class_id}")
                confidence = float(box.conf.item())

                xyxy = box.xyxy[0].tolist()
                # Scale coordinates back to original crop size
                bbox_tile = BBox(
                    x1=int(xyxy[0] * scale_back),
                    y1=int(xyxy[1] * scale_back),
                    x2=int(xyxy[2] * scale_back),
                    y2=int(xyxy[3] * scale_back),
                )

                detections.append(RawDetection(
                    detection_id=str(uuid4()),
                    tile_id=tile.tile_id,
                    class_id=class_id,
                    class_slug=class_slug,
                    bbox_tile=bbox_tile,
                    confidence=confidence,
                    model_version=model_version,
                ))

        return detections

    def serialize_result(self, result: InferenceResult) -> dict:
        """Convert InferenceResult to JSON-serializable dict for the API response."""
        return {
            "plan_id": result.plan_id,
            "page_number": result.page_number,
            "model_version": result.model_version,
            "total_detections": result.total_detections,
            "warnings": result.warnings,
            "tile_results": [
                {
                    "tile": {
                        "tile_id": tr.tile.tile_id,
                        "origin_x": tr.tile.origin_x,
                        "origin_y": tr.tile.origin_y,
                        "width": tr.tile.width,
                        "height": tr.tile.height,
                    },
                    "detections": [
                        {
                            "detection_id": d.detection_id,
                            "tile_id": d.tile_id,
                            "class_slug": d.class_slug,
                            "bbox_tile": {
                                "x1": d.bbox_tile.x1,
                                "y1": d.bbox_tile.y1,
                                "x2": d.bbox_tile.x2,
                                "y2": d.bbox_tile.y2,
                            },
                            "confidence": d.confidence,
                            "source": "yolo",
                            "model_version": d.model_version,
                        }
                        for d in tr.detections
                    ],
                }
                for tr in result.tile_results
            ],
        }
