from unittest.mock import MagicMock, patch, PropertyMock
from uuid import UUID

import numpy as np
import pytest
from PIL import Image

from schemas import BBox
from yolo_inference import (
    TileConfig,
    TileInfo,
    ImageTiler,
    YOLOInferenceEngine,
    ModelRegistry,
    RawDetection,
    InferenceResult,
)


# ─── helpers ──────────────────────────────────────────────────────────────────


def make_image(width=2048, height=1448, color=(255, 255, 255)) -> Image.Image:
    return Image.new("RGB", (width, height), color=color)


def make_tile(origin_x=0, origin_y=0, size=1024) -> TileInfo:
    return TileInfo(
        tile_id="tile-test",
        origin_x=origin_x,
        origin_y=origin_y,
        width=size,
        height=size,
    )


def make_mock_box(x1=100, y1=100, x2=140, y2=140, conf=0.87, cls=0):
    box = MagicMock()
    box.xyxy = [MagicMock()]
    box.xyxy[0].tolist.return_value = [float(x1), float(y1), float(x2), float(y2)]
    box.conf.item.return_value = conf
    box.cls.item.return_value = cls
    return box


def make_mock_model(boxes: list) -> MagicMock:
    model = MagicMock()
    result = MagicMock()
    result.boxes = boxes
    model.predict.return_value = [result]
    return model


# ─── ImageTiler ───────────────────────────────────────────────────────────────


class TestImageTiler:
    def test_single_tile_for_small_image(self):
        tiler = ImageTiler()
        img = make_image(640, 480)
        config = TileConfig(tile_size=1024, overlap=0.20, imgsz=1024)
        tiles = tiler.tile(img, config)
        assert len(tiles) == 1

    def test_multiple_tiles_for_large_image(self):
        tiler = ImageTiler()
        img = make_image(2048, 2048)
        config = TileConfig(tile_size=1024, overlap=0.20, imgsz=1024)
        tiles = tiler.tile(img, config)
        # step = 1024 * 0.80 = 819
        # x: 0, 819, 1638 → 3 columns; same for y → 9 tiles
        assert len(tiles) == 9

    def test_tile_ids_are_unique(self):
        tiler = ImageTiler()
        img = make_image(2048, 2048)
        config = TileConfig(tile_size=1024, overlap=0.20, imgsz=1024)
        tiles = tiler.tile(img, config)
        ids = [t.tile_id for t in tiles]
        assert len(ids) == len(set(ids))

    def test_tiles_have_valid_uuids(self):
        tiler = ImageTiler()
        config = TileConfig(tile_size=1024, overlap=0.20, imgsz=1024)
        tiles = tiler.tile(make_image(1024, 1024), config)
        for tile in tiles:
            UUID(tile.tile_id)  # raises ValueError if not a valid UUID

    def test_crop_returns_correct_region(self):
        tiler = ImageTiler()
        # Create image with distinct colors per region
        img = Image.new("RGB", (2048, 1024), color=(200, 200, 200))
        tile = make_tile(origin_x=512, origin_y=0, size=512)
        crop = tiler.crop(img, tile)
        assert crop.size == (512, 512)

    def test_boundary_tile_clips_correctly(self):
        tiler = ImageTiler()
        img = make_image(1500, 1500)
        config = TileConfig(tile_size=1024, overlap=0.0, imgsz=1024)
        tiles = tiler.tile(img, config)
        boundary = [t for t in tiles if t.origin_x + t.width == 1500 or t.origin_y + t.height == 1500]
        assert len(boundary) > 0

    @pytest.mark.parametrize("overlap,expected_min_tiles", [
        (0.00, 4),  # 2x2 for 2048x2048 with step=1024
        (0.20, 9),  # 3x3 with step=819
        (0.25, 9),  # 3x3 with step=768
    ])
    def test_overlap_increases_tile_count(self, overlap, expected_min_tiles):
        tiler = ImageTiler()
        img = make_image(2048, 2048)
        config = TileConfig(tile_size=1024, overlap=overlap, imgsz=1024)
        tiles = tiler.tile(img, config)
        assert len(tiles) >= expected_min_tiles


# ─── YOLOInferenceEngine ──────────────────────────────────────────────────────


class TestYOLOInferenceEngine:
    CLASS_MAP = {0: "sprinkler-pendent-k57", 1: "extintor-pqs", 2: "hidrante-parede"}

    def _make_engine(self, model=None, class_map=None) -> YOLOInferenceEngine:
        registry = MagicMock(spec=ModelRegistry)
        registry.load.return_value = (
            model or make_mock_model([]),
            class_map or self.CLASS_MAP,
        )
        return YOLOInferenceEngine(registry=registry)

    def test_returns_inference_result(self, tmp_path):
        img = make_image(640, 480)
        img_path = str(tmp_path / "page.png")
        img.save(img_path)

        engine = self._make_engine()
        result = engine.run(
            plan_id="plan-001",
            page_number=1,
            image_path=img_path,
            model_version="yolov15-spci-2026.04",
            tile_config=TileConfig(tile_size=1024, overlap=0.20, imgsz=640),
        )

        assert isinstance(result, InferenceResult)
        assert result.plan_id == "plan-001"
        assert result.model_version == "yolov15-spci-2026.04"

    def test_detections_have_correct_class_slug(self, tmp_path):
        img = make_image(640, 480)
        img_path = str(tmp_path / "page.png")
        img.save(img_path)

        box = make_mock_box(cls=0, conf=0.88)
        engine = self._make_engine(model=make_mock_model([box]))

        result = engine.run(
            plan_id="plan-001",
            page_number=1,
            image_path=img_path,
            model_version="yolov15-spci-2026.04",
            tile_config=TileConfig(tile_size=1024, overlap=0.20, imgsz=640),
        )

        assert result.total_detections == 1
        d = result.tile_results[0].detections[0]
        assert d.class_slug == "sprinkler-pendent-k57"
        assert d.confidence == 0.88

    def test_unknown_class_id_gets_fallback_slug(self, tmp_path):
        img = make_image(640, 480)
        img_path = str(tmp_path / "page.png")
        img.save(img_path)

        box = make_mock_box(cls=99, conf=0.75)
        engine = self._make_engine(model=make_mock_model([box]))

        result = engine.run(
            plan_id="plan-001",
            page_number=1,
            image_path=img_path,
            model_version="yolov15-spci-2026.04",
            tile_config=TileConfig(tile_size=1024, overlap=0.20, imgsz=640),
        )

        d = result.tile_results[0].detections[0]
        assert d.class_slug == "unknown-class-99"

    def test_legend_region_masked_before_inference(self, tmp_path):
        # Create image with a red legend region
        img = Image.new("RGB", (640, 480), color=(255, 255, 255))
        # Make legend region distinctly colored
        for x in range(400, 640):
            for y in range(400, 480):
                img.putpixel((x, y), (255, 0, 0))
        img_path = str(tmp_path / "page.png")
        img.save(img_path)

        captured_arrays = []
        model = MagicMock()
        result_mock = MagicMock()
        result_mock.boxes = []
        def capture_predict(source, **kwargs):
            captured_arrays.append(source.copy())
            return [result_mock]
        model.predict.side_effect = capture_predict

        engine = self._make_engine(model=model)
        engine.run(
            plan_id="plan-001",
            page_number=1,
            image_path=img_path,
            model_version="yolov15-spci-2026.04",
            tile_config=TileConfig(tile_size=640, overlap=0.0, imgsz=640),
            legend_region=BBox(x1=400, y1=400, x2=640, y2=480),
        )

        # The legend region (red pixels at bottom-right) should be white in all tiles
        for arr in captured_arrays:
            # Check that no red pixels survived masking
            red_pixels = np.all(arr == [255, 0, 0], axis=-1)
            assert not red_pixels.any(), "Legend region should have been masked to white"

    def test_warning_added_when_no_legend_region(self, tmp_path):
        img = make_image(640, 480)
        img_path = str(tmp_path / "page.png")
        img.save(img_path)

        engine = self._make_engine()
        result = engine.run(
            plan_id="plan-001",
            page_number=1,
            image_path=img_path,
            model_version="yolov15-spci-2026.04",
            tile_config=TileConfig(tile_size=1024, overlap=0.20, imgsz=640),
            legend_region=None,
        )

        assert any("legend_region_not_provided" in w for w in result.warnings)

    def test_multiple_detections_across_tiles(self, tmp_path):
        img = make_image(2048, 1024)
        img_path = str(tmp_path / "page.png")
        img.save(img_path)

        box = make_mock_box(cls=0, conf=0.85)
        engine = self._make_engine(model=make_mock_model([box]))

        result = engine.run(
            plan_id="plan-001",
            page_number=1,
            image_path=img_path,
            model_version="yolov15-spci-2026.04",
            tile_config=TileConfig(tile_size=1024, overlap=0.20, imgsz=1024),
        )

        # Each tile returns 1 detection; with 2048x1024 and overlap 0.20:
        # x: step=819, tiles at 0, 819, 1638 → 3 columns
        # y: fits in 1 row → total 3 tiles, each with 1 detection
        assert len(result.tile_results) >= 2
        assert result.total_detections >= 2

    def test_serialize_result_structure(self, tmp_path):
        img = make_image(640, 480)
        img_path = str(tmp_path / "page.png")
        img.save(img_path)

        box = make_mock_box(cls=1, conf=0.79)
        engine = self._make_engine(model=make_mock_model([box]))

        result = engine.run(
            plan_id="plan-serialize",
            page_number=2,
            image_path=img_path,
            model_version="yolov15-spci-2026.04",
            tile_config=TileConfig(tile_size=1024, overlap=0.20, imgsz=640),
        )

        serialized = engine.serialize_result(result)
        assert serialized["plan_id"] == "plan-serialize"
        assert serialized["page_number"] == 2
        assert "tile_results" in serialized
        d = serialized["tile_results"][0]["detections"][0]
        assert d["source"] == "yolo"
        assert d["class_slug"] == "extintor-pqs"
        assert "bbox_tile" in d
        assert all(k in d["bbox_tile"] for k in ["x1", "y1", "x2", "y2"])


# ─── ModelRegistry ────────────────────────────────────────────────────────────


class TestModelRegistry:
    def test_raises_when_pt_missing(self, tmp_path):
        registry = ModelRegistry(models_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Model not found"):
            registry.load("yolov15-spci-2026.04")

    def test_raises_when_classes_file_missing(self, tmp_path):
        (tmp_path / "yolov15-spci-2026.04.pt").touch()
        registry = ModelRegistry(models_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Classes file not found"):
            registry.load("yolov15-spci-2026.04")

    def test_loads_class_map_correctly(self, tmp_path):
        (tmp_path / "model.pt").touch()
        classes = {"0": "sprinkler-pendent-k57", "1": "extintor-pqs"}
        (tmp_path / "model.classes.json").write_text(__import__("json").dumps(classes))

        with patch("yolo_inference.ModelRegistry._validate_model_classes"):
            with patch("yolo_inference.ModelRegistry.load") as mock_load:
                # Test the _load_class_map helper directly
                registry = ModelRegistry(models_dir=str(tmp_path))
                from pathlib import Path
                result = registry._load_class_map(tmp_path / "model.classes.json")

        assert result == {0: "sprinkler-pendent-k57", 1: "extintor-pqs"}
