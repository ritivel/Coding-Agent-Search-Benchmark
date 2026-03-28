"""
Handwriting OCR module using Microsoft TrOCR (transformer-based).

TrOCR is a state-of-the-art model for optical character recognition
of handwritten text, combining a Vision Transformer encoder with a
text Transformer decoder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Structured result from OCR on a single line image."""

    text: str
    confidence: float
    line_index: int


@dataclass
class DocumentOCRResult:
    """Aggregated OCR result for an entire document image."""

    lines: list[OCRResult] = field(default_factory=list)
    raw_text: str = ""
    source_path: str = ""

    def to_plain_text(self) -> str:
        return "\n".join(line.text for line in self.lines if line.text.strip())


class HandwritingOCR:
    """
    Recognise handwritten text using Microsoft TrOCR.

    The model is loaded lazily on first call and cached for subsequent
    invocations to avoid re-downloading weights.
    """

    def __init__(
        self,
        model_name: str = "microsoft/trocr-large-handwritten",
        device: str | None = None,
        max_new_tokens: int = 256,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._processor: TrOCRProcessor | None = None
        self._model: VisionEncoderDecoderModel | None = None

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info("HandwritingOCR will use device=%s", self.device)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading TrOCR model: %s", self.model_name)
        self._processor = TrOCRProcessor.from_pretrained(self.model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        logger.info("TrOCR model loaded successfully on %s", self.device)

    def recognise_line(self, image: Image.Image, line_index: int = 0) -> OCRResult:
        """Run OCR on a single text-line image."""
        self._load_model()
        assert self._processor is not None and self._model is not None

        pixel_values = self._processor(image, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                max_new_tokens=self.max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )

        text = self._processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

        scores = outputs.scores
        if scores:
            probs = [torch.softmax(s, dim=-1).max().item() for s in scores]
            confidence = sum(probs) / len(probs)
        else:
            confidence = 0.0

        return OCRResult(text=text.strip(), confidence=confidence, line_index=line_index)

    def recognise_document(
        self,
        line_images: list[Image.Image],
        source_path: str = "",
        min_confidence: float = 0.0,
    ) -> DocumentOCRResult:
        """Run OCR on all line images from a preprocessed document."""
        results: list[OCRResult] = []
        for idx, img in enumerate(line_images):
            result = self.recognise_line(img, line_index=idx)
            logger.debug(
                "Line %d: conf=%.3f text=%r", idx, result.confidence, result.text[:80]
            )
            if result.confidence >= min_confidence:
                results.append(result)

        doc_result = DocumentOCRResult(
            lines=results,
            raw_text="\n".join(r.text for r in results),
            source_path=source_path,
        )
        logger.info(
            "Recognised %d lines from %s (%d chars)",
            len(results),
            source_path,
            len(doc_result.raw_text),
        )
        return doc_result
