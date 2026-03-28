"""Stage 1: Image Preprocessing — clean, binarise, deskew, and segment handwritten images."""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import logging

from ..config import IMAGE_PREPROCESSING

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedImage:
    original_path: str
    cleaned_image: np.ndarray
    line_images: List[np.ndarray] = field(default_factory=list)
    skew_angle: float = 0.0
    width: int = 0
    height: int = 0


class ImagePreprocessor:
    def __init__(
        self,
        max_width: Optional[int] = None,
        denoise_strength: Optional[int] = None,
        clahe_clip_limit: Optional[float] = None,
        clahe_grid_size: Optional[tuple] = None,
        adaptive_block_size: Optional[int] = None,
        adaptive_c: Optional[int] = None,
        deskew: Optional[bool] = None,
        binarize: Optional[bool] = None,
    ):
        c = IMAGE_PREPROCESSING
        self.max_width = max_width if max_width is not None else c["max_width"]
        self.denoise_strength = denoise_strength if denoise_strength is not None else c["denoise_strength"]
        self.clahe_clip_limit = clahe_clip_limit if clahe_clip_limit is not None else c["clahe_clip_limit"]
        self.clahe_grid_size = clahe_grid_size if clahe_grid_size is not None else c["clahe_grid_size"]
        self.adaptive_block_size = adaptive_block_size if adaptive_block_size is not None else c["adaptive_block_size"]
        self.adaptive_c = adaptive_c if adaptive_c is not None else c["adaptive_c"]
        self.do_deskew = deskew if deskew is not None else c["deskew"]
        self.do_binarize = binarize if binarize is not None else c["binarize"]

    def process(self, image_path: str) -> PreprocessedImage:
        logger.info(f"Preprocessing image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            try:
                with Image.open(Path(image_path)) as pil_img:
                    img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            except OSError as e:
                raise FileNotFoundError(f"Cannot read image: {image_path}") from e

        img = self._resize(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, self.denoise_strength, 7, 21)
        enhanced = self._enhance_contrast(denoised)

        skew_angle = 0.0
        if self.do_deskew:
            enhanced, skew_angle = self._deskew(enhanced)
            logger.info(f"  Deskew angle: {skew_angle:.2f}°")

        if self.do_binarize:
            binary = cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.adaptive_block_size,
                self.adaptive_c,
            )
        else:
            binary = enhanced

        line_images = self._segment_lines(binary)
        logger.info(f"  Detected {len(line_images)} text lines")

        return PreprocessedImage(
            original_path=image_path,
            cleaned_image=binary,
            line_images=line_images,
            skew_angle=skew_angle,
            width=binary.shape[1],
            height=binary.shape[0],
        )

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if w > self.max_width:
            scale = self.max_width / w
            img = cv2.resize(img, (self.max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        return img

    def _enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_grid_size,
        )
        return clahe.apply(gray)

    def _deskew(self, img: np.ndarray) -> tuple:
        coords = np.column_stack(np.where(img < 128))
        if len(coords) < 50:
            return img, 0.0
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 0.5:
            return img, angle
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
        return rotated, angle

    def _segment_lines(self, binary: np.ndarray) -> List[np.ndarray]:
        inv = cv2.bitwise_not(binary)
        h_proj = np.sum(inv, axis=1)
        threshold = np.max(h_proj) * 0.05 if np.max(h_proj) > 0 else 0

        in_line = False
        lines = []
        start = 0
        for i, val in enumerate(h_proj):
            if val > threshold and not in_line:
                start = i
                in_line = True
            elif val <= threshold and in_line:
                if i - start > 10:
                    lines.append((start, i))
                in_line = False
        if in_line and len(h_proj) - start > 10:
            lines.append((start, len(h_proj)))

        padding = 5
        line_images = []
        for start, end in lines:
            y1 = max(0, start - padding)
            y2 = min(binary.shape[0], end + padding)
            line_img = binary[y1:y2, :]
            if line_img.shape[0] > 5 and line_img.shape[1] > 5:
                line_images.append(line_img)

        if not line_images:
            line_images = [binary]

        return line_images
