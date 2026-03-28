"""Stage 2: Handwriting OCR using Surya — modern multilingual document OCR toolkit."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class LineOCRResult:
    text: str
    confidence: float
    bbox: Optional[tuple] = None


@dataclass
class DocumentOCRResult:
    source_path: str
    lines: List[LineOCRResult] = field(default_factory=list)
    full_text: str = ""
    average_confidence: float = 0.0
    language: str = "en"

    def __post_init__(self):
        if self.lines and not self.full_text:
            self.full_text = "\n".join(line.text for line in self.lines if line.text.strip())
        if self.lines:
            confidences = [l.confidence for l in self.lines if l.confidence > 0]
            self.average_confidence = sum(confidences) / len(confidences) if confidences else 0.0


class SuryaOCREngine:
    """Handwriting OCR using Surya — supports 90+ languages with layout awareness."""

    def __init__(self, languages: Optional[List[str]] = None):
        self.languages = languages or ["en"]
        self._predictor = None
        self._recognition = None
        self._detection = None

    def _load_models(self):
        if self._predictor is not None:
            return
        logger.info("Loading Surya OCR models (first call may download weights)...")
        try:
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor

            self._recognition = RecognitionPredictor()
            self._detection = DetectionPredictor()
            logger.info("Surya OCR models loaded successfully.")
        except ImportError:
            logger.warning(
                "Surya OCR not installed. Falling back to TrOCR. "
                "Install with: pip install surya-ocr"
            )
            self._use_fallback = True

    def recognize_lines(self, line_images: List[np.ndarray], source_path: str = "") -> DocumentOCRResult:
        self._load_models()

        if hasattr(self, "_use_fallback") and self._use_fallback:
            return self._fallback_trocr(line_images, source_path)

        return self._surya_recognize(line_images, source_path)

    def _surya_recognize(self, line_images: List[np.ndarray], source_path: str) -> DocumentOCRResult:
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor

        pil_images = []
        for img in line_images:
            if len(img.shape) == 2:
                pil_img = Image.fromarray(img).convert("RGB")
            else:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pil_images.append(pil_img)

        lines = []
        for idx, pil_img in enumerate(pil_images):
            try:
                predictions = self._recognition([pil_img], [self.languages])
                if predictions and len(predictions) > 0:
                    pred = predictions[0]
                    if hasattr(pred, "text_lines") and pred.text_lines:
                        for tl in pred.text_lines:
                            text = tl.text if hasattr(tl, "text") else str(tl)
                            confidence = tl.confidence if hasattr(tl, "confidence") else 0.8
                            lines.append(LineOCRResult(text=text, confidence=confidence))
                    elif hasattr(pred, "text"):
                        lines.append(LineOCRResult(text=pred.text, confidence=0.8))
                    else:
                        text = str(pred)
                        lines.append(LineOCRResult(text=text, confidence=0.7))
            except Exception as e:
                logger.warning(f"Surya OCR failed on line {idx}: {e}")
                lines.append(LineOCRResult(text="", confidence=0.0))

        return DocumentOCRResult(source_path=source_path, lines=lines)

    def recognize_full_page(self, image_path: str) -> DocumentOCRResult:
        self._load_models()

        if hasattr(self, "_use_fallback") and self._use_fallback:
            return self._fallback_trocr_page(image_path)

        pil_img = Image.open(image_path).convert("RGB")

        lines = []
        try:
            det_results = self._detection([pil_img])
            rec_results = self._recognition([pil_img], [self.languages])

            if rec_results and len(rec_results) > 0:
                pred = rec_results[0]
                if hasattr(pred, "text_lines"):
                    for tl in pred.text_lines:
                        text = tl.text if hasattr(tl, "text") else str(tl)
                        confidence = tl.confidence if hasattr(tl, "confidence") else 0.8
                        bbox = tuple(tl.bbox) if hasattr(tl, "bbox") else None
                        lines.append(LineOCRResult(text=text, confidence=confidence, bbox=bbox))
        except Exception as e:
            logger.error(f"Full-page Surya OCR failed: {e}")

        return DocumentOCRResult(source_path=image_path, lines=lines)

    def _fallback_trocr(self, line_images: List[np.ndarray], source_path: str) -> DocumentOCRResult:
        logger.info("Using TrOCR fallback for OCR...")
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch

            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            model = model.to(device)

            lines = []
            for img in line_images:
                if len(img.shape) == 2:
                    pil_img = Image.fromarray(img).convert("RGB")
                else:
                    pil_img = Image.fromarray(img)

                pixel_values = processor(pil_img, return_tensors="pt").pixel_values.to(device)
                with torch.no_grad():
                    generated = model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)

                text = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0]

                if generated.scores:
                    probs = [torch.softmax(s, dim=-1).max().item() for s in generated.scores]
                    confidence = sum(probs) / len(probs) if probs else 0.0
                else:
                    confidence = 0.5

                lines.append(LineOCRResult(text=text, confidence=confidence))

            return DocumentOCRResult(source_path=source_path, lines=lines)

        except Exception as e:
            logger.error(f"TrOCR fallback also failed: {e}")
            return DocumentOCRResult(source_path=source_path, lines=[])

    def _fallback_trocr_page(self, image_path: str) -> DocumentOCRResult:
        img = np.array(Image.open(image_path).convert("L"))
        return self._fallback_trocr([img], image_path)
