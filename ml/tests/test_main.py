"""
Integration tests for the FastAPI ML service (main.py).
All heavy dependencies (YOLO model, DINOv2 model, PIL file I/O) are mocked
so tests run without GPU, without real image files, and without model weights.
"""
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ── app import ────────────────────────────────────────────────────────────────
# Patch heavy I/O before importing main so no real files are needed at import time
with patch("yolo_inference.ModelRegistry"), patch("yolo_inference.YOLOInferenceEngine"):
    from main import app

client = TestClient(app)


# ── helpers ───────────────────────────────────────────────────────────────────

def _fake_image_path(tmp_path, width=512, height=512):
    """Write a real PNG so os.path.exists() passes in the endpoint."""
    p = tmp_path / f"{uuid4()}.png"
    Image.new("RGB", (width, height), color=(240, 240, 240)).save(str(p))
    return str(p)


def _scale_request(image_path):
    return {
        "plan_id": str(uuid4()),
        "page_number": 1,
        "image_path": image_path,
        "tenant_id": "tenant-test",
    }


def _fake_scale_result():
    from scale_detection import DetectionResult
    return DetectionResult(
        scale="1:100",
        scale_denominator=100,
        confidence=0.92,
        method="ocr_cartouche",
        cartouche_region=None,
        legend_region=None,
        uncertainty_note="",
        warnings=[],
    )


def _fake_recommended():
    from schemas import RecommendedConfig
    return RecommendedConfig(dpi=300, tile_size=1024, overlap=0.2, imgsz=1024, dinov2_window=128)


# ── GET /health ───────────────────────────────────────────────────────────────


def test_health_returns_ok():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert res.json()["service"] == "quantx-ml"


# ── POST /scale-detection ─────────────────────────────────────────────────────


class TestScaleDetectionEndpoint:
    def test_returns_200_with_valid_image(self, tmp_path):
        img_path = _fake_image_path(tmp_path)
        with (
            patch("main.detect_scale", return_value=_fake_scale_result()),
            patch("main.get_recommended_config", return_value=_fake_recommended()),
        ):
            res = client.post("/scale-detection", json=_scale_request(img_path))

        assert res.status_code == 200
        body = res.json()
        assert body["scale"] == "1:100"
        assert body["scale_confidence"] == pytest.approx(0.92)
        assert body["method"] == "ocr_cartouche"

    def test_returns_404_when_image_missing(self):
        res = client.post("/scale-detection", json=_scale_request("/nonexistent/path.png"))
        assert res.status_code == 404

    def test_response_has_recommended_config(self, tmp_path):
        img_path = _fake_image_path(tmp_path)
        with (
            patch("main.detect_scale", return_value=_fake_scale_result()),
            patch("main.get_recommended_config", return_value=_fake_recommended()),
        ):
            res = client.post("/scale-detection", json=_scale_request(img_path))

        rec = res.json()["recommended"]
        assert rec["dpi"] == 300
        assert rec["tile_size"] == 1024
        assert rec["dinov2_window"] == 128

    def test_plan_id_echoed_in_response(self, tmp_path):
        img_path = _fake_image_path(tmp_path)
        req = _scale_request(img_path)
        with (
            patch("main.detect_scale", return_value=_fake_scale_result()),
            patch("main.get_recommended_config", return_value=_fake_recommended()),
        ):
            res = client.post("/scale-detection", json=req)

        assert res.json()["plan_id"] == req["plan_id"]


# ── POST /scale-detection/reverse ─────────────────────────────────────────────


class TestScaleDetectionReverse:
    def test_returns_scale_from_reverse_inference(self, tmp_path):
        img_path = _fake_image_path(tmp_path)
        payload = {
            "plan_id": str(uuid4()),
            "page_number": 1,
            "image_path": img_path,
            "tenant_id": "tenant-test",
            "class_slug": "sprinkler-pendent-k57",
            "bbox": {"x1": 100, "y1": 100, "x2": 140, "y2": 140},
            "known_physical_mm": 4.0,
            "current_dpi": 300,
        }
        res = client.post("/scale-detection/reverse", json=payload)
        assert res.status_code == 200
        assert res.json()["method"] == "reverse_inference"
        assert res.json()["scale"].startswith("1:")

    def test_returns_422_when_bbox_has_zero_size(self, tmp_path):
        img_path = _fake_image_path(tmp_path)
        payload = {
            "plan_id": str(uuid4()),
            "page_number": 1,
            "image_path": img_path,
            "tenant_id": "tenant-test",
            "class_slug": "sprinkler-pendent-k57",
            "bbox": {"x1": 100, "y1": 100, "x2": 100, "y2": 100},
            "known_physical_mm": 4.0,
            "current_dpi": 300,
        }
        res = client.post("/scale-detection/reverse", json=payload)
        assert res.status_code == 422


# ── POST /yolo-inference ──────────────────────────────────────────────────────


class TestYOLOInferenceEndpoint:
    def _make_yolo_result(self, plan_id, page_number):
        from yolo_inference import InferenceResult, TileResult, TileInfo
        tile = TileInfo(
            tile_id=str(uuid4()),
            origin_x=0, origin_y=0,
            width=1024, height=1024,
        )
        return InferenceResult(
            plan_id=plan_id,
            page_number=page_number,
            model_version="yolov15-spci-2026.04",
            tile_results=[TileResult(tile=tile, detections=[])],
            warnings=[],
        )

    def test_returns_200_with_valid_request(self, tmp_path):
        img_path = _fake_image_path(tmp_path)
        plan_id = str(uuid4())
        result = self._make_yolo_result(plan_id, 1)
        with patch("main._yolo_engine") as mock_engine:
            mock_engine.run.return_value = result
            mock_engine.serialize_result.return_value = {
                "plan_id": plan_id,
                "page_number": 1,
                "model_version": "yolov15-spci-2026.04",
                "total_detections": 0,
                "warnings": [],
                "tile_results": [],
            }
            res = client.post("/yolo-inference", json={
                "plan_id": plan_id,
                "page_number": 1,
                "image_path": img_path,
                "model_version": "yolov15-spci-2026.04",
            })

        assert res.status_code == 200
        assert res.json()["plan_id"] == plan_id

    def test_returns_404_when_image_missing(self):
        res = client.post("/yolo-inference", json={
            "plan_id": str(uuid4()),
            "page_number": 1,
            "image_path": "/nonexistent/image.png",
            "model_version": "yolov15-spci-2026.04",
        })
        assert res.status_code == 404

    def test_engine_called_with_tile_config(self, tmp_path):
        img_path = _fake_image_path(tmp_path)
        plan_id = str(uuid4())
        result = self._make_yolo_result(plan_id, 1)
        with patch("main._yolo_engine") as mock_engine:
            mock_engine.run.return_value = result
            mock_engine.serialize_result.return_value = {"plan_id": plan_id}
            client.post("/yolo-inference", json={
                "plan_id": plan_id,
                "page_number": 1,
                "image_path": img_path,
                "model_version": "yolov15-spci-2026.04",
                "tile_size": 512,
                "overlap": 0.25,
                "conf": 0.30,
            })
        call_kwargs = mock_engine.run.call_args
        tile_cfg = call_kwargs.kwargs["tile_config"]
        assert tile_cfg.tile_size == 512
        assert tile_cfg.conf == pytest.approx(0.30)


# ── POST /dinov2/embed ────────────────────────────────────────────────────────


class TestDINOv2EmbedEndpoint:
    def test_returns_503_when_embedder_unavailable(self, tmp_path):
        img_path = _fake_image_path(tmp_path)
        with patch("main.DINOv2Embedder", side_effect=RuntimeError("no GPU")):
            res = client.post("/dinov2/embed", json={
                "tenant_id": "t1",
                "class_slug": "sprinkler-pendent-k57",
                "image_path": img_path,
                "bbox_x1": 100, "bbox_y1": 100,
                "bbox_x2": 140, "bbox_y2": 140,
            })
        assert res.status_code == 503

    def test_returns_404_when_image_missing(self):
        res = client.post("/dinov2/embed", json={
            "tenant_id": "t1",
            "class_slug": "sprinkler-pendent-k57",
            "image_path": "/no/image.png",
            "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 64, "bbox_y2": 64,
        })
        assert res.status_code == 404

    def test_embed_stores_and_returns_metadata(self, tmp_path):
        from dinov2_inference import EmbeddingRecord
        img_path = _fake_image_path(tmp_path)
        sample_id = str(uuid4())
        fake_record = EmbeddingRecord(
            tenant_id="t1",
            class_slug="sprinkler-pendent-k57",
            sample_id=sample_id,
            embedding=np.zeros(768, dtype=np.float32),
        )
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.zeros(768, dtype=np.float32)
        mock_matcher = MagicMock()
        mock_matcher.embed_and_store.return_value = fake_record

        with (
            patch("main.DINOv2Embedder", return_value=mock_embedder),
            patch("main.FewShotMatcher", return_value=mock_matcher),
            patch("main._embedding_store") as mock_store,
        ):
            mock_store.count_by_tenant.return_value = 1
            res = client.post("/dinov2/embed", json={
                "tenant_id": "t1",
                "class_slug": "sprinkler-pendent-k57",
                "image_path": img_path,
                "bbox_x1": 100, "bbox_y1": 100,
                "bbox_x2": 140, "bbox_y2": 140,
                "sample_id": sample_id,
            })

        assert res.status_code == 200
        body = res.json()
        assert body["sample_id"] == sample_id
        assert body["class_slug"] == "sprinkler-pendent-k57"
        assert body["library_size"] == 1


# ── POST /dinov2/match ────────────────────────────────────────────────────────


class TestDINOv2MatchEndpoint:
    def _detection_dict(self, detection_id=None):
        return {
            "detection_id": detection_id or str(uuid4()),
            "tile_id": "tile-001",
            "class_slug": "sprinkler-pendent-k57",
            "bbox_tile": {"x1": 100, "y1": 100, "x2": 140, "y2": 140},
            "confidence": 0.87,
        }

    def test_returns_empty_when_library_empty(self, tmp_path):
        img_path = _fake_image_path(tmp_path)
        with patch("main._embedding_store") as mock_store:
            mock_store.count_by_tenant.return_value = 0
            res = client.post("/dinov2/match", json={
                "tenant_id": "t1",
                "image_path": img_path,
                "detections": [self._detection_dict()],
            })
        assert res.status_code == 200
        body = res.json()
        assert body["library_size"] == 0
        assert body["matches"] == []
        assert body["warning"] == "tenant_library_empty"

    def test_returns_404_when_image_missing(self):
        res = client.post("/dinov2/match", json={
            "tenant_id": "t1",
            "image_path": "/no/image.png",
            "detections": [],
        })
        assert res.status_code == 404

    def test_returns_503_when_embedder_unavailable(self, tmp_path):
        img_path = _fake_image_path(tmp_path)
        with (
            patch("main._embedding_store") as mock_store,
            patch("main.DINOv2Embedder", side_effect=RuntimeError("no GPU")),
        ):
            mock_store.count_by_tenant.return_value = 5
            res = client.post("/dinov2/match", json={
                "tenant_id": "t1",
                "image_path": img_path,
                "detections": [self._detection_dict()],
            })
        assert res.status_code == 503

    def test_match_returns_results_per_detection(self, tmp_path):
        from dinov2_inference import MatchResult, FewShotMatch
        img_path = _fake_image_path(tmp_path)
        det_id = str(uuid4())
        fake_match = FewShotMatch(
            detection_id=det_id,
            tile_id="tile-001",
            class_slug="sprinkler-pendent-k57",
            matched_class_slug="sprinkler-pendent-k57",
            similarity=0.95,
            agreement="agree",
            top_k=[],
        )
        mock_embedder = MagicMock()
        mock_matcher = MagicMock()
        mock_matcher.match_batch.return_value = [fake_match]

        with (
            patch("main._embedding_store") as mock_store,
            patch("main.DINOv2Embedder", return_value=mock_embedder),
            patch("main.FewShotMatcher", return_value=mock_matcher),
        ):
            mock_store.count_by_tenant.return_value = 5
            res = client.post("/dinov2/match", json={
                "tenant_id": "t1",
                "image_path": img_path,
                "detections": [self._detection_dict(det_id)],
            })

        assert res.status_code == 200
        matches = res.json()["matches"]
        assert len(matches) == 1
        assert matches[0]["matched"] is True
        assert matches[0]["similarity"] == pytest.approx(0.95)
        assert matches[0]["agreement"] == "agree"

    def test_unmatched_detection_gets_no_match_tier(self, tmp_path):
        img_path = _fake_image_path(tmp_path)
        mock_embedder = MagicMock()
        mock_matcher = MagicMock()
        mock_matcher.match_batch.return_value = [None]

        with (
            patch("main._embedding_store") as mock_store,
            patch("main.DINOv2Embedder", return_value=mock_embedder),
            patch("main.FewShotMatcher", return_value=mock_matcher),
        ):
            mock_store.count_by_tenant.return_value = 5
            res = client.post("/dinov2/match", json={
                "tenant_id": "t1",
                "image_path": img_path,
                "detections": [self._detection_dict()],
            })

        assert res.status_code == 200
        match = res.json()["matches"][0]
        assert match["matched"] is False
        assert match["tier"] == "no_match"


# ── DELETE /dinov2/library/{tenant_id}/{class_slug} ───────────────────────────


class TestDINOv2DeleteClass:
    def test_delete_returns_deleted_count(self):
        with patch("main._embedding_store") as mock_store:
            mock_store.delete_by_class.return_value = 3
            mock_store.count_by_tenant.return_value = 7
            res = client.delete("/dinov2/library/tenant-1/sprinkler-pendent-k57")

        assert res.status_code == 200
        body = res.json()
        assert body["deleted_count"] == 3
        assert body["library_size"] == 7
        assert body["class_slug"] == "sprinkler-pendent-k57"
        assert body["tenant_id"] == "tenant-1"

    def test_delete_returns_zero_when_class_absent(self):
        with patch("main._embedding_store") as mock_store:
            mock_store.delete_by_class.return_value = 0
            mock_store.count_by_tenant.return_value = 10
            res = client.delete("/dinov2/library/tenant-1/nonexistent-class")

        assert res.status_code == 200
        assert res.json()["deleted_count"] == 0


# ── POST /render-pdf ──────────────────────────────────────────────────────────


class TestRenderPDFEndpoint:
    def _fake_pages(self, n=3):
        return [Image.new("RGB", (2480, 3508), color=(255, 255, 255)) for _ in range(n)]

    def test_returns_404_when_pdf_missing(self, tmp_path):
        res = client.post("/render-pdf", json={
            "plan_id": str(uuid4()),
            "pdf_path": "/nonexistent/file.pdf",
            "output_dir": str(tmp_path),
        })
        assert res.status_code == 404

    def test_returns_page_count_and_paths(self, tmp_path):
        plan_id = str(uuid4())
        fake_pdf = tmp_path / "plan.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        with patch("main.convert_from_path", return_value=self._fake_pages(3)):
            res = client.post("/render-pdf", json={
                "plan_id": plan_id,
                "pdf_path": str(fake_pdf),
                "output_dir": str(tmp_path),
            })

        assert res.status_code == 200
        body = res.json()
        assert body["page_count"] == 3
        assert len(body["image_paths"]) == 3
        assert body["plan_id"] == plan_id

    def test_image_paths_are_sequential(self, tmp_path):
        fake_pdf = tmp_path / "plan.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        with patch("main.convert_from_path", return_value=self._fake_pages(2)):
            res = client.post("/render-pdf", json={
                "plan_id": "plan-test",
                "pdf_path": str(fake_pdf),
                "output_dir": str(tmp_path),
            })

        paths = res.json()["image_paths"]
        assert paths[0].endswith("page-001.png")
        assert paths[1].endswith("page-002.png")

    def test_returns_503_when_poppler_missing(self, tmp_path):
        from pdf2image.exceptions import PDFInfoNotInstalledError
        fake_pdf = tmp_path / "plan.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        with patch("main.convert_from_path", side_effect=PDFInfoNotInstalledError()):
            res = client.post("/render-pdf", json={
                "plan_id": str(uuid4()),
                "pdf_path": str(fake_pdf),
                "output_dir": str(tmp_path),
            })

        assert res.status_code == 503
        assert "poppler" in res.json()["detail"].lower()

    def test_returns_422_for_corrupt_pdf(self, tmp_path):
        from pdf2image.exceptions import PDFPageCountError
        fake_pdf = tmp_path / "plan.pdf"
        fake_pdf.write_bytes(b"not a pdf")

        with patch("main.convert_from_path", side_effect=PDFPageCountError("bad pdf")):
            res = client.post("/render-pdf", json={
                "plan_id": str(uuid4()),
                "pdf_path": str(fake_pdf),
                "output_dir": str(tmp_path),
            })

        assert res.status_code == 422
