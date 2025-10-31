from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

from src.schemas.detection import Detection


@dataclass
class NotificationEvent:
    """Generic notification payload.

    - title: optional short title
    - text: optional message text
    - detections: optional detection list to include/format
    - image_bytes: optional JPEG/PNG bytes to attach
    - timestamp: optional event time (epoch seconds)
    """

    title: Optional[str] = None
    text: Optional[str] = None
    detections: List[Detection] = field(default_factory=list)
    image_bytes: Optional[bytes] = None
    timestamp: Optional[float] = None


class NotificationSender:
    """Base class for notification senders.

    Subclasses should implement `send`.
    """

    def send(self, event: NotificationEvent) -> None:  # pragma: no cover
        raise NotImplementedError

