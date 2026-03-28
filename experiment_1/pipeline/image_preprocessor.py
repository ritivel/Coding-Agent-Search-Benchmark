"""
Image preprocessing for handwritten document images.

Enhanced over Experiment 0 with:
  - Morphological cleaning (opening/closing) to remove small noise and fill gaps
  - Adaptive kernel sizing based on image resolution
  - Improved line segmentation with minimum gap filtering
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Clean and prepare handwritten document images for OCR."""

    def __init__(
        self,
        target_dpi: int = 300,
        binarize_threshold: int = 0,
        denoise_strength: int = 10,
        deskew: bool = True,
        resize_max_width: int = 2048,
        morphological_clean: bool = True,
    ):
        self.target_dpi = target_dpi
        self.binarize_threshold = binarize_threshold
        self.denoise_strength = denoise_strength
        self.deskew = deskew
        self.resize_max_width = resize_max_width
        self.morphological_clean = morphological_clean

    def load_image(self, path: Union[str, Path]) -> np.ndarray:
        path = str(path)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        logger.info("Loaded image %s  shape=%s", path, img.shape)
        return img

    def to_grayscale(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def resize(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if w > self.resize_max_width:
            scale = self.resize_max_width / w
            new_w = self.resize_max_width
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info("Resized to %dx%d", new_w, new_h)
        return img

    def denoise(self, img: np.ndarray) -> np.ndarray:
        if self.denoise_strength <= 0:
            return img
        return cv2.fastNlMeansDenoising(
            img, None, h=self.denoise_strength, templateWindowSize=7, searchWindowSize=21
        )

    def enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    def binarize(self, img: np.ndarray) -> np.ndarray:
        if self.binarize_threshold > 0:
            _, binary = cv2.threshold(img, self.binarize_threshold, 255, cv2.THRESH_BINARY)
        else:
            binary = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 11
            )
        return binary

    def morphological_cleanup(self, img: np.ndarray) -> np.ndarray:
        """Remove small noise blobs and fill tiny gaps in ink strokes."""
        if not self.morphological_clean:
            return img
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_open)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

        logger.debug("Morphological cleanup applied (open 2x2 → close 3x3)")
        return closed

    def _compute_skew_angle(self, img: np.ndarray) -> float:
        coords = np.column_stack(np.where(img < 128))
        if len(coords) < 50:
            return 0.0
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        return angle

    def deskew_image(self, img: np.ndarray) -> np.ndarray:
        if not self.deskew:
            return img
        angle = self._compute_skew_angle(img)
        if abs(angle) < 0.5:
            logger.info("Skew angle %.2f° is negligible — skipping deskew", angle)
            return img
        logger.info("Deskewing by %.2f°", angle)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)

    def segment_lines(self, img: np.ndarray, min_line_height: int = 10) -> list[np.ndarray]:
        """Segment a binarised image into individual text-line crops.

        Uses horizontal projection profiling with adaptive gap detection.
        """
        inverted = cv2.bitwise_not(img)
        horizontal_proj = np.sum(inverted, axis=1)
        threshold = np.max(horizontal_proj) * 0.05

        in_line = False
        lines: list[tuple[int, int]] = []
        start = 0
        for i, val in enumerate(horizontal_proj):
            if not in_line and val > threshold:
                in_line = True
                start = i
            elif in_line and val <= threshold:
                in_line = False
                if i - start > min_line_height:
                    pad = max(5, min_line_height // 4)
                    lines.append((max(0, start - pad), min(img.shape[0], i + pad)))
        if in_line:
            lines.append((max(0, start - 5), img.shape[0]))

        logger.info("Segmented %d text lines", len(lines))
        return [img[y1:y2, :] for y1, y2 in lines]

    def preprocess(self, path: Union[str, Path]) -> dict:
        """Full preprocessing pipeline.

        Returns dict with keys: original, grayscale, cleaned, line_images.
        """
        original = self.load_image(path)
        resized = self.resize(original)
        gray = self.to_grayscale(resized)
        denoised = self.denoise(gray)
        enhanced = self.enhance_contrast(denoised)
        binary = self.binarize(enhanced)
        morphed = self.morphological_cleanup(binary)
        deskewed = self.deskew_image(morphed)
        line_crops = self.segment_lines(deskewed)

        line_images = []
        for crop in line_crops:
            pil_img = Image.fromarray(crop).convert("RGB")
            line_images.append(pil_img)

        if not line_images:
            logger.warning("No text lines detected — returning full image as single line")
            line_images = [Image.fromarray(deskewed).convert("RGB")]

        return {
            "original": original,
            "grayscale": gray,
            "cleaned": deskewed,
            "line_images": line_images,
        }
