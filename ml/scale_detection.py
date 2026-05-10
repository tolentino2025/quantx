import re
import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

from schemas import BBox, RecommendedConfig

log = logging.getLogger(__name__)

# Matches: "1:100", "ESC. 1:100", "Escala 1/100", "1 : 100", etc.
_SCALE_RE = re.compile(
    r"(?:esc(?:ala)?\.?\s*[=:]?\s*)?1\s*[:/]\s*(\d{2,4})",
    re.IGNORECASE,
)

# Common SPCI drawing scales in order of frequency
_COMMON_SCALES = [50, 75, 100, 150, 200, 250]

_SCALE_CONFIGS: dict[int, dict] = {
    50:  {"dpi": 200, "tile_size": 640,  "overlap": 0.15, "imgsz": 640,  "dinov2_window": 96},
    75:  {"dpi": 250, "tile_size": 768,  "overlap": 0.18, "imgsz": 768,  "dinov2_window": 112},
    100: {"dpi": 300, "tile_size": 1024, "overlap": 0.20, "imgsz": 1024, "dinov2_window": 128},
    150: {"dpi": 350, "tile_size": 1024, "overlap": 0.22, "imgsz": 1024, "dinov2_window": 128},
    200: {"dpi": 400, "tile_size": 1280, "overlap": 0.25, "imgsz": 1280, "dinov2_window": 160},
    250: {"dpi": 450, "tile_size": 1280, "overlap": 0.25, "imgsz": 1280, "dinov2_window": 160},
}

_LEGEND_KEYWORDS = ["LEGENDA", "LISTA DE SÍMBOLOS", "SIMBOLOGIA", "CONVENÇÕES"]


@dataclass
class DetectionResult:
    scale: str
    scale_denominator: int
    confidence: float
    method: str
    cartouche_region: Optional[BBox]
    legend_region: Optional[BBox]
    warnings: list[str]
    uncertainty_note: str


def detect_scale(image_path: str) -> DetectionResult:
    """
    Run all scale detection strategies in priority order.
    Returns the first successful result or fallback.
    """
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    warnings: list[str] = []

    cartouche_region = _estimate_cartouche_region(width, height)
    legend_region = _detect_legend_region(img, warnings)

    # Strategy 1: OCR on cartouche
    result = _detect_via_ocr(img, cartouche_region, warnings)
    if result:
        return DetectionResult(
            scale=f"1:{result}",
            scale_denominator=result,
            confidence=0.90,
            method="ocr_cartouche",
            cartouche_region=cartouche_region,
            legend_region=legend_region,
            warnings=warnings,
            uncertainty_note=f"Escala lida via OCR do cartouche: '1:{result}'. Confiança alta.",
        )

    # Strategy 2: Graphic scale bar
    result = _detect_via_graphic_bar(img, warnings)
    if result:
        return DetectionResult(
            scale=f"1:{result}",
            scale_denominator=result,
            confidence=0.80,
            method="graphic_bar",
            cartouche_region=cartouche_region,
            legend_region=legend_region,
            warnings=warnings,
            uncertainty_note=f"Escala estimada via barra gráfica: '1:{result}'. Confiança moderada.",
        )

    # Strategy 3: Full-page OCR fallback
    result = _detect_via_fullpage_ocr(img, warnings)
    if result:
        return DetectionResult(
            scale=f"1:{result}",
            scale_denominator=result,
            confidence=0.70,
            method="ocr_fullpage",
            cartouche_region=cartouche_region,
            legend_region=legend_region,
            warnings=warnings,
            uncertainty_note=f"Escala encontrada via OCR full-page: '1:{result}'. Confiança moderada-baixa.",
        )

    # Fallback
    warnings.append("scale_not_detected_using_fallback_1_100")
    return DetectionResult(
        scale="1:100",
        scale_denominator=100,
        confidence=0.30,
        method="fallback",
        cartouche_region=cartouche_region,
        legend_region=legend_region,
        warnings=warnings,
        uncertainty_note="Escala não detectada. Usando fallback 1:100. Verifique o cartouche manualmente.",
    )


def get_recommended_config(scale_denominator: int) -> RecommendedConfig:
    # Snap to nearest known scale
    nearest = min(_COMMON_SCALES, key=lambda s: abs(s - scale_denominator))
    cfg = _SCALE_CONFIGS[nearest]
    return RecommendedConfig(**cfg)


# ─── private helpers ──────────────────────────────────────────────────────────


def _estimate_cartouche_region(width: int, height: int) -> BBox:
    return BBox(
        x1=int(width * 0.72),
        y1=int(height * 0.82),
        x2=width,
        y2=height,
    )


def _detect_via_ocr(img: Image.Image, region: BBox, warnings: list[str]) -> Optional[int]:
    crop = img.crop((region.x1, region.y1, region.x2, region.y2))
    crop = _preprocess_for_ocr(crop)

    try:
        text = pytesseract.image_to_string(crop, config="--psm 6 --oem 1")
    except Exception as exc:
        warnings.append(f"ocr_cartouche_failed: {exc}")
        return None

    return _parse_scale_from_text(text)


def _detect_via_fullpage_ocr(img: Image.Image, warnings: list[str]) -> Optional[int]:
    small = img.resize((min(img.width, 2000), min(img.height, 2800)), Image.LANCZOS)
    preprocessed = _preprocess_for_ocr(small)

    try:
        text = pytesseract.image_to_string(preprocessed, config="--psm 1 --oem 1")
    except Exception as exc:
        warnings.append(f"ocr_fullpage_failed: {exc}")
        return None

    return _parse_scale_from_text(text)


def _detect_via_graphic_bar(img: Image.Image, warnings: list[str]) -> Optional[int]:
    """
    Detect a graphic scale bar: a thick horizontal line segment near the
    cartouche, accompanied by measurement text (e.g. "0 5 10m").
    Returns the estimated denominator or None.
    """
    width, height = img.size
    # Search in bottom portion of image
    search_region = img.crop((0, int(height * 0.75), width, height))
    arr = np.array(search_region.convert("L"))

    # Detect horizontal edges
    edges = cv2.Canny(arr, 50, 150)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=80,
        minLineLength=int(width * 0.03), maxLineGap=10,
    )
    if lines is None:
        return None

    horizontal_lines = [
        line[0] for line in lines
        if abs(line[0][3] - line[0][1]) < 5  # nearly horizontal
    ]
    if not horizontal_lines:
        return None

    # Find the longest horizontal line — likely the scale bar
    longest = max(horizontal_lines, key=lambda l: abs(l[2] - l[0]))
    bar_px = abs(longest[2] - longest[0])

    # OCR the text near the bar to find the measurement value
    lx1, ly1, lx2, ly2 = longest
    text_region = search_region.crop((
        max(0, min(lx1, lx2) - 20),
        max(0, ly1 - 40),
        min(width, max(lx1, lx2) + 20),
        min(search_region.height, ly2 + 40),
    ))

    try:
        text = pytesseract.image_to_string(text_region, config="--psm 7 --oem 1")
    except Exception:
        return None

    # Look for a distance value in meters (e.g. "10m", "5 m", "10")
    dist_match = re.search(r"(\d+(?:\.\d+)?)\s*m?\b", text)
    if not dist_match:
        return None

    distance_m = float(dist_match.group(1))
    if distance_m <= 0:
        return None

    # scale = bar_px / (distance_m * 1000mm * pixels_per_mm_at_current_dpi)
    # We assume render DPI ~ 96 for the low-DPI input image
    pixels_per_mm = 96 / 25.4
    denominator = bar_px / (distance_m * 1000 * pixels_per_mm)

    # Snap to nearest common scale
    snapped = min(_COMMON_SCALES, key=lambda s: abs(s - denominator))
    if abs(snapped - denominator) / snapped > 0.30:
        warnings.append(f"graphic_bar_scale_uncertain: estimated {denominator:.0f}, snapped to {snapped}")

    return snapped


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    gray = img.convert("L")
    arr = np.array(gray)
    # Increase contrast
    arr = cv2.equalizeHist(arr)
    # Threshold to binary
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)


def _parse_scale_from_text(text: str) -> Optional[int]:
    for match in _SCALE_RE.finditer(text):
        denominator = int(match.group(1))
        if denominator in _COMMON_SCALES:
            return denominator
        # Accept within ±10% of a known scale
        snapped = min(_COMMON_SCALES, key=lambda s: abs(s - denominator))
        if abs(snapped - denominator) / snapped <= 0.10:
            return snapped
    return None


def _detect_legend_region(img: Image.Image, warnings: list[str]) -> Optional[BBox]:
    """
    Find the legend block by looking for keyword text in candidate regions.
    Checks top-right and bottom-left quadrants, which are most common.
    """
    width, height = img.size
    candidates = [
        # top-right
        (int(width * 0.60), 0, width, int(height * 0.40)),
        # bottom-left
        (0, int(height * 0.60), int(width * 0.40), height),
        # bottom-right (less common)
        (int(width * 0.60), int(height * 0.60), width, height),
    ]

    for rx1, ry1, rx2, ry2 in candidates:
        crop = img.crop((rx1, ry1, rx2, ry2))
        preprocessed = _preprocess_for_ocr(crop)
        try:
            text = pytesseract.image_to_string(preprocessed, config="--psm 1 --oem 1")
        except Exception:
            continue

        for keyword in _LEGEND_KEYWORDS:
            if keyword in text.upper():
                log.info("Legend region found at (%d,%d,%d,%d)", rx1, ry1, rx2, ry2)
                return BBox(x1=rx1, y1=ry1, x2=rx2, y2=ry2)

    warnings.append("legend_region_not_found")
    return None


def snap_to_common_scale(denominator: int) -> int:
    return min(_COMMON_SCALES, key=lambda s: abs(s - denominator))
