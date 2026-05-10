import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from scale_detection import (
    _parse_scale_from_text,
    _estimate_cartouche_region,
    get_recommended_config,
    detect_scale,
    snap_to_common_scale,
    _SCALE_CONFIGS,
)


# ─── unit: text parsing ────────────────────────────────────────────────────────

class TestParseScaleFromText:
    def test_standard_format(self):
        assert _parse_scale_from_text("Escala: 1:100") == 100

    def test_slash_format(self):
        assert _parse_scale_from_text("ESC. 1/100") == 100

    def test_with_spaces(self):
        assert _parse_scale_from_text("1 : 100") == 100

    def test_lowercase(self):
        assert _parse_scale_from_text("escala 1:50") == 50

    def test_200_scale(self):
        assert _parse_scale_from_text("ESCALA 1:200") == 200

    def test_no_match_returns_none(self):
        assert _parse_scale_from_text("Rev. 01 - Data: 2026-05-09") is None

    def test_ignores_non_standard_denominators(self):
        # 300 is not in _COMMON_SCALES but is within 10% of... nothing
        # So it should return None
        assert _parse_scale_from_text("1:300") is None

    def test_snaps_within_10_percent(self):
        # 105 is within 10% of 100
        assert _parse_scale_from_text("1:105") == 100

    def test_multiline_text(self):
        text = "Cliente: Engenharia XYZ\nEscala: 1:100\nData: 2026-05"
        assert _parse_scale_from_text(text) == 100

    def test_text_with_noise(self):
        # Simulates OCR noise around the scale
        assert _parse_scale_from_text("Esc.a|a: 1:100 rev") == 100


# ─── unit: cartouche region ───────────────────────────────────────────────────

class TestEstimateCartoucheRegion:
    def test_a1_landscape_300dpi(self):
        width, height = 7433, 10512
        bbox = _estimate_cartouche_region(width, height)
        assert bbox.x1 == int(width * 0.72)
        assert bbox.y1 == int(height * 0.82)
        assert bbox.x2 == width
        assert bbox.y2 == height

    def test_region_is_valid(self):
        bbox = _estimate_cartouche_region(1024, 768)
        assert bbox.x1 < bbox.x2
        assert bbox.y1 < bbox.y2


# ─── unit: scale snapping ─────────────────────────────────────────────────────

class TestSnapToCommonScale:
    @pytest.mark.parametrize("input_val,expected", [
        (48, 50), (52, 50),
        (98, 100), (103, 100),
        (195, 200), (205, 200),
        (248, 250),
    ])
    def test_snaps_to_nearest(self, input_val, expected):
        assert snap_to_common_scale(input_val) == expected


# ─── unit: recommended config ─────────────────────────────────────────────────

class TestGetRecommendedConfig:
    @pytest.mark.parametrize("denom,expected_dpi,expected_tile", [
        (50, 200, 640),
        (100, 300, 1024),
        (200, 400, 1280),
    ])
    def test_known_scales(self, denom, expected_dpi, expected_tile):
        cfg = get_recommended_config(denom)
        assert cfg.dpi == expected_dpi
        assert cfg.tile_size == expected_tile

    def test_snaps_unknown_scale(self):
        cfg = get_recommended_config(130)
        # 130 is closer to 150 than to 100
        assert cfg.dpi == _SCALE_CONFIGS[150]["dpi"]

    def test_returns_all_fields(self):
        cfg = get_recommended_config(100)
        assert cfg.overlap > 0
        assert cfg.imgsz > 0
        assert cfg.dinov2_window > 0


# ─── integration: detect_scale with mocked OCR ───────────────────────────────

def _make_test_image(width=800, height=600) -> Image.Image:
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Write scale text in cartouche region (~bottom-right)
    draw.text((int(width * 0.75), int(height * 0.85)), "Escala: 1:100", fill=(0, 0, 0))
    return img


class TestDetectScale:
    def test_detects_scale_via_ocr_cartouche(self, tmp_path):
        img = _make_test_image()
        img_path = str(tmp_path / "page.png")
        img.save(img_path)

        with patch("scale_detection.pytesseract.image_to_string") as mock_ocr:
            mock_ocr.return_value = "Escala: 1:100"
            result = detect_scale(img_path)

        assert result.scale == "1:100"
        assert result.method == "ocr_cartouche"
        assert result.confidence == 0.90
        assert not result.warnings or "scale_not_detected" not in result.warnings[0]

    def test_falls_back_to_fullpage_when_cartouche_empty(self, tmp_path):
        img = _make_test_image()
        img_path = str(tmp_path / "page.png")
        img.save(img_path)

        call_count = 0

        def mock_ocr(img, config=""):
            nonlocal call_count
            call_count += 1
            # cartouche returns nothing, fullpage returns scale
            if "--psm 6" in config:
                return "Rev. 01"
            return "Escala 1:200"

        with patch("scale_detection.pytesseract.image_to_string", side_effect=mock_ocr):
            with patch("scale_detection._detect_via_graphic_bar", return_value=None):
                result = detect_scale(img_path)

        assert result.scale == "1:200"
        assert result.method == "ocr_fullpage"
        assert result.confidence == 0.70

    def test_uses_fallback_when_all_strategies_fail(self, tmp_path):
        img = _make_test_image()
        img_path = str(tmp_path / "page.png")
        img.save(img_path)

        with patch("scale_detection.pytesseract.image_to_string", return_value=""):
            with patch("scale_detection._detect_via_graphic_bar", return_value=None):
                result = detect_scale(img_path)

        assert result.scale == "1:100"
        assert result.method == "fallback"
        assert result.confidence == 0.30
        assert any("fallback" in w for w in result.warnings)

    def test_legend_region_detected(self, tmp_path):
        img = _make_test_image()
        img_path = str(tmp_path / "page.png")
        img.save(img_path)

        def mock_ocr(img, config=""):
            if "--psm 6" in config:
                return "Escala: 1:100"
            if "--psm 1" in config:
                return "LEGENDA\nSprinkler\nExtintor"
            return ""

        with patch("scale_detection.pytesseract.image_to_string", side_effect=mock_ocr):
            result = detect_scale(img_path)

        assert result.legend_region is not None

    def test_ocr_exception_adds_warning(self, tmp_path):
        img = _make_test_image()
        img_path = str(tmp_path / "page.png")
        img.save(img_path)

        with patch(
            "scale_detection.pytesseract.image_to_string",
            side_effect=Exception("tesseract not found"),
        ):
            with patch("scale_detection._detect_via_graphic_bar", return_value=None):
                result = detect_scale(img_path)

        assert result.method == "fallback"
        assert any("ocr" in w for w in result.warnings)
