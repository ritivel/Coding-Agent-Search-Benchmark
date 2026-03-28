"""
Dual OCR engine: Tesseract (classical) + TrOCR (transformer-based).

Strategy:
  1. Tesseract processes the full binarized page and returns bulk text.
  2. TrOCR processes individual text-line crops for higher handwriting accuracy.
  3. Results are merged — TrOCR lines replace Tesseract lines when confidence is
     higher, giving us the best of both worlds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import pytesseract
from PIL import Image

import config

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Container for OCR output."""
    text: str
    source: str  # "tesseract", "trocr", or "merged"
    confidence: float = 0.0
    line_results: list[dict] = field(default_factory=list)


# ── Tesseract ──────────────────────────────────────────────────────

def tesseract_ocr(image: np.ndarray) -> OCRResult:
    """Run Tesseract on a preprocessed numpy image."""
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

    custom_config = (
        f"--psm {config.TESSERACT_PSM} "
        f"-l {config.TESSERACT_LANG} "
        "--oem 3"
    )

    data = pytesseract.image_to_data(
        image, config=custom_config, output_type=pytesseract.Output.DICT
    )

    lines: list[dict] = []
    current_line: list[str] = []
    current_conf: list[float] = []
    prev_line_num = -1

    for i, word in enumerate(data["text"]):
        line_num = data["line_num"][i]
        conf = float(data["conf"][i])

        if line_num != prev_line_num and current_line:
            avg_conf = sum(current_conf) / len(current_conf) if current_conf else 0
            lines.append({
                "text": " ".join(current_line),
                "confidence": avg_conf,
                "source": "tesseract",
            })
            current_line = []
            current_conf = []

        word = word.strip()
        if word and conf > 0:
            current_line.append(word)
            current_conf.append(conf)
        prev_line_num = line_num

    if current_line:
        avg_conf = sum(current_conf) / len(current_conf) if current_conf else 0
        lines.append({
            "text": " ".join(current_line),
            "confidence": avg_conf,
            "source": "tesseract",
        })

    full_text = "\n".join(ln["text"] for ln in lines)
    avg_confidence = (
        sum(ln["confidence"] for ln in lines) / len(lines) if lines else 0
    )

    return OCRResult(
        text=full_text,
        source="tesseract",
        confidence=avg_confidence,
        line_results=lines,
    )


# ── TrOCR (Transformer-based) ─────────────────────────────────────

_trocr_processor = None
_trocr_model = None


def _load_trocr():
    """Lazy-load TrOCR model to avoid startup cost when not needed."""
    global _trocr_processor, _trocr_model
    if _trocr_processor is None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        logger.info("Loading TrOCR model: %s", config.TROCR_MODEL)
        _trocr_processor = TrOCRProcessor.from_pretrained(config.TROCR_MODEL)
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(config.TROCR_MODEL)
    return _trocr_processor, _trocr_model


def _segment_lines(image: np.ndarray) -> list[np.ndarray]:
    """
    Simple horizontal-projection based line segmentation.

    Projects pixel intensities along the horizontal axis, finds gaps between
    text lines, and returns a list of cropped line images.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horizontal_proj = np.sum(binary, axis=1)

    threshold = np.max(horizontal_proj) * 0.02
    in_line = False
    lines = []
    start = 0

    for i, val in enumerate(horizontal_proj):
        if val > threshold and not in_line:
            start = i
            in_line = True
        elif val <= threshold and in_line:
            if i - start > 10:
                lines.append(image[max(0, start - 5): i + 5, :])
            in_line = False

    if in_line:
        lines.append(image[start:, :])

    return lines


def trocr_ocr(pil_image: Image.Image) -> OCRResult:
    """Run TrOCR on a PIL image, processing line by line."""
    processor, model = _load_trocr()

    img_array = np.array(pil_image.convert("RGB"))
    line_crops = _segment_lines(img_array)

    if not line_crops:
        line_crops = [img_array]

    lines: list[dict] = []
    for crop in line_crops:
        pil_crop = Image.fromarray(crop).convert("RGB")
        pixel_values = processor(pil_crop, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values, max_length=256)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        text = text.strip()
        if text:
            lines.append({"text": text, "confidence": 0.85, "source": "trocr"})

    full_text = "\n".join(ln["text"] for ln in lines)
    avg_conf = sum(ln["confidence"] for ln in lines) / len(lines) if lines else 0

    return OCRResult(
        text=full_text,
        source="trocr",
        confidence=avg_conf,
        line_results=lines,
    )


# ── Merge Strategy ─────────────────────────────────────────────────

def merge_results(tesseract_result: OCRResult, trocr_result: OCRResult) -> OCRResult:
    """
    Merge Tesseract and TrOCR outputs.

    Prefer TrOCR line-level results when available because it handles
    handwriting better. Fall back to Tesseract for lines TrOCR missed.
    """
    if not trocr_result.line_results:
        return tesseract_result
    if not tesseract_result.line_results:
        return trocr_result

    trocr_lines = trocr_result.line_results
    tess_lines = tesseract_result.line_results

    merged_lines = []
    max_len = max(len(trocr_lines), len(tess_lines))

    for i in range(max_len):
        trocr_line = trocr_lines[i] if i < len(trocr_lines) else None
        tess_line = tess_lines[i] if i < len(tess_lines) else None

        if trocr_line and tess_line:
            if trocr_line["confidence"] >= tess_line["confidence"]:
                merged_lines.append({**trocr_line, "source": "trocr"})
            else:
                merged_lines.append({**tess_line, "source": "tesseract"})
        elif trocr_line:
            merged_lines.append(trocr_line)
        elif tess_line:
            merged_lines.append(tess_line)

    full_text = "\n".join(ln["text"] for ln in merged_lines)
    avg_conf = (
        sum(ln["confidence"] for ln in merged_lines) / len(merged_lines)
        if merged_lines else 0
    )

    return OCRResult(
        text=full_text,
        source="merged",
        confidence=avg_conf,
        line_results=merged_lines,
    )


# ── Public API ─────────────────────────────────────────────────────

def extract_text(
    image_path: Union[str, Path],
    preprocessed_cv: np.ndarray | None = None,
    preprocessed_pil: Image.Image | None = None,
) -> OCRResult:
    """
    Run OCR on a handwritten document image.

    If preprocessed images are not supplied, raw loading is used (preprocessing
    should be done beforehand for best results).
    """
    if preprocessed_cv is None:
        preprocessed_cv = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if preprocessed_pil is None:
        preprocessed_pil = Image.open(str(image_path)).convert("RGB")

    logger.info("Running Tesseract OCR...")
    tess_result = tesseract_ocr(preprocessed_cv)
    logger.info(
        "Tesseract: %d lines, avg confidence %.1f%%",
        len(tess_result.line_results),
        tess_result.confidence,
    )

    if config.USE_TROCR:
        logger.info("Running TrOCR (handwriting-optimized)...")
        trocr_result = trocr_ocr(preprocessed_pil)
        logger.info(
            "TrOCR: %d lines extracted", len(trocr_result.line_results)
        )
        return merge_results(tess_result, trocr_result)

    return tess_result
