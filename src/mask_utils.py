from __future__ import annotations

import cv2
import numpy as np


def clean_binary_mask(
    mask: np.ndarray,
    *,
    min_area: int = 5,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """Reduce noise in a binary mask using morphology and area filtering."""

    if mask is None or mask.size == 0:
        return mask

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels <= 1:
        return cleaned

    filtered = np.zeros_like(cleaned)
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered[labels == label_idx] = 255

    return filtered
