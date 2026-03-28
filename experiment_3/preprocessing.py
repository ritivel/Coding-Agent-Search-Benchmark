"""
Image preprocessing module for handwritten document OCR.

Applies a series of transformations to improve OCR accuracy:
  - Grayscale conversion
  - Noise removal (Non-local Means Denoising)
  - Adaptive thresholding for binarization
  - Deskewing via minAreaRect rotation
  - Optional rescaling to target DPI
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union

import config


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk as a BGR numpy array."""
    path = str(image_path)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(image: np.ndarray, strength: int | None = None) -> np.ndarray:
    """Non-local means denoising — good for preserving handwriting strokes."""
    h = strength or config.PREPROCESSING_DENOISE_STRENGTH
    return cv2.fastNlMeansDenoising(image, None, h, 7, 21)


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian thresholding produces clean black-on-white binarization."""
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        config.PREPROCESSING_ADAPTIVE_BLOCK_SIZE,
        config.PREPROCESSING_ADAPTIVE_C,
    )


def deskew(image: np.ndarray) -> np.ndarray:
    """Straighten a tilted document using minAreaRect on detected contours."""
    coords = np.column_stack(np.where(image < 128))
    if len(coords) < 50:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def rescale(image: np.ndarray, target_height: int = 3000) -> np.ndarray:
    """Scale image so its height matches target_height (preserving aspect ratio)."""
    h, w = image.shape[:2]
    if h >= target_height:
        return image
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_CUBIC)


def preprocess(image_path: Union[str, Path]) -> tuple[np.ndarray, Image.Image]:
    """
    Full preprocessing pipeline.

    Returns:
        opencv_img: processed image as numpy array (for Tesseract)
        pil_img:    PIL Image in RGB (for TrOCR)
    """
    img = load_image(image_path)
    gray = to_grayscale(img)
    denoised = denoise(gray)
    deskewed = deskew(denoised)
    binary = adaptive_threshold(deskewed)
    scaled = rescale(binary)

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return scaled, pil_img
