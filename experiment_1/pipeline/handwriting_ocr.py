"""
Handwriting OCR using Microsoft TrOCR (transformer-based).

Enhanced over Experiment 0 with:
  - Optional batch inference for faster multi-line processing on GPU
  - Confidence-weighted line filtering with configurable threshold
  - Richer metadata per line (bounding box placeholder, word count)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

logger = logging.getLogger(__name__)


@dataclass
class OCRLineResult:
    """Structured result from OCR on a single line image."""

    text: str
    confidence: float
    line_index: int
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())


@dataclass
class DocumentOCRResult:
    """Aggregated OCR result for an entire document image."""

    lines: list[OCRLineResult] = field(default_factory=list)
    raw_text: str = ""
    source_path: str = ""
    avg_confidence: float = 0.0

    def to_plain_text(self) -> str:
        return "\n".join(line.text for line in self.lines if line.text.strip())

    def compute_stats(self) -> dict:
        if not self.lines:
            return {"line_count": 0, "word_count": 0, "avg_confidence": 0.0}
        confs = [l.confidence for l in self.lines]
        return {
            "line_count": len(self.lines),
            "word_count": sum(l.word_count for l in self.lines),
            "avg_confidence": sum(confs) / len(confs),
            "min_confidence": min(confs),
            "max_confidence": max(confs),
        }


class HandwritingOCR:
    """Recognise handwritten text using Microsoft TrOCR.

    The model is loaded lazily on first call and cached for subsequent
    invocations. Supports batch inference when a GPU is available.
    """

    def __init__(
        self,
        model_name: str = "microsoft/trocr-large-handwritten",
        device: str | None = None,
        max_new_tokens: int = 256,
        batch_size: int = 4,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
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
        logger.info("TrOCR model loaded on %s", self.device)

    def recognise_line(self, image: Image.Image, line_index: int = 0) -> OCRLineResult:
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

        return OCRLineResult(text=text.strip(), confidence=confidence, line_index=line_index)

    def recognise_batch(self, images: list[Image.Image], start_index: int = 0) -> list[OCRLineResult]:
        """Run OCR on a batch of line images simultaneously (GPU-optimised)."""
        self._load_model()
        assert self._processor is not None and self._model is not None

        if self.device == "cpu" or len(images) <= 1:
            return [self.recognise_line(img, start_index + i) for i, img in enumerate(images)]

        inputs = self._processor(images, return_tensors="pt", padding=True)
        pixel_values = inputs.pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                max_new_tokens=self.max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )

        texts = self._processor.batch_decode(outputs.sequences, skip_special_tokens=True)

        results = []
        for i, text in enumerate(texts):
            scores = outputs.scores
            if scores:
                probs = [torch.softmax(s, dim=-1)[i].max().item() for s in scores]
                confidence = sum(probs) / len(probs)
            else:
                confidence = 0.0
            results.append(OCRLineResult(
                text=text.strip(), confidence=confidence, line_index=start_index + i
            ))
        return results

    def recognise_document(
        self,
        line_images: list[Image.Image],
        source_path: str = "",
        min_confidence: float = 0.0,
    ) -> DocumentOCRResult:
        """Run OCR on all line images from a preprocessed document."""
        all_results: list[OCRLineResult] = []

        for batch_start in range(0, len(line_images), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(line_images))
            batch = line_images[batch_start:batch_end]

            try:
                batch_results = self.recognise_batch(batch, start_index=batch_start)
            except Exception:
                logger.warning("Batch inference failed — falling back to sequential")
                batch_results = [
                    self.recognise_line(img, batch_start + i)
                    for i, img in enumerate(batch)
                ]

            for result in batch_results:
                logger.debug(
                    "Line %d: conf=%.3f text=%r",
                    result.line_index, result.confidence, result.text[:80],
                )
                if result.confidence >= min_confidence:
                    all_results.append(result)

        doc_result = DocumentOCRResult(
            lines=all_results,
            raw_text="\n".join(r.text for r in all_results),
            source_path=source_path,
        )
        if all_results:
            doc_result.avg_confidence = sum(r.confidence for r in all_results) / len(all_results)

        logger.info(
            "Recognised %d lines from %s (%d chars, avg_conf=%.3f)",
            len(all_results), source_path, len(doc_result.raw_text), doc_result.avg_confidence,
        )
        return doc_result
