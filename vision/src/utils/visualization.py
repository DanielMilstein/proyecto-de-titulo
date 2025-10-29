from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

from src.schemas.detection import Detection


def draw_detections(
    image_bgr: np.ndarray,
    detections: Iterable[Detection],
    color: Tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Draw detections onto a BGR image and return a copy.

    - image_bgr: OpenCV image (H, W, 3) in BGR.
    - detections: iterable of Detection objects.
    - color: optional BGR color for all boxes; if None, color per class hash.
    """
    out = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox.to_xyxy()
        if color is None:
            # Simple deterministic color per class
            c = _color_for_label(det.label)
        else:
            c = color

        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        label = f"{det.label}:{det.score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(th + 4, y1)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 2, ty + baseline - 4), (0, 0, 0), -1)
        cv2.putText(out, label, (x1 + 1, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _color_for_label(label: str) -> Tuple[int, int, int]:
    # Map label to a BGR tuple deterministically
    h = abs(hash(label))
    return (h % 256, (h >> 8) % 256, (h >> 16) % 256)

