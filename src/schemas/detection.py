from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in center-coordinates (xc, yc, w, h).

    Coordinates are in pixel space unless noted otherwise.
    """

    xc: float
    yc: float
    w: float
    h: float

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        x1 = int(round(self.xc - self.w / 2.0))
        y1 = int(round(self.yc - self.h / 2.0))
        x2 = int(round(self.xc + self.w / 2.0))
        y2 = int(round(self.yc + self.h / 2.0))
        return x1, y1, x2, y2


@dataclass(frozen=True)
class Detection:
    """Single detection result.

    - label: class name
    - score: confidence score (0..1)
    - bbox: bounding box (center format)
    """

    label: str
    score: float
    bbox: BBox

