from __future__ import annotations

"""Interfaces (Protocols) for service components.

These enable clean dependency inversion and testability.
"""

from typing import Protocol, Tuple, Optional, List
import numpy as np

from src.schemas.detection import Detection


class FrameSource(Protocol):
    def latest(self) -> Tuple[Optional[np.ndarray], float]:
        """Return the most recent frame (BGR) and timestamp seconds.

        Frame may be None if not yet available.
        """
        ...


class DetectionSource(Protocol):
    def latest(self) -> Tuple[List[Detection], Optional[np.ndarray], float]:
        """Return latest detections, optional annotated frame, and timestamp seconds."""
        ...


class DetectionClient(Protocol):
    def health(self) -> bool: ...
    def detect_url(self, img_url: str) -> List[Detection]: ...


class NotificationSender(Protocol):
    def send(self, event: "NotificationEvent") -> None: ...


class NotificationEvent(Protocol):
    title: Optional[str]
    text: Optional[str]
    detections: List[Detection]
    image_bytes: Optional[bytes]
    timestamp: Optional[float]
