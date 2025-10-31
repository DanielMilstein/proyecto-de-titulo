from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List

from src.schemas.detection import Detection


@dataclass
class NotificationPolicy:
    """Policy: notify after N consecutive frames with any detection >= threshold.

    Applies a cooldown after a send to avoid spamming.
    """

    score_threshold: float
    frames_required: int
    cooldown_seconds: float

    # runtime state
    _streak: int = 0
    _last_sent_at: float = 0.0

    def reset(self) -> None:
        self._streak = 0

    def _meets(self, detections: List[Detection]) -> bool:
        for d in detections:
            if d.score >= self.score_threshold:
                return True
        return False

    def should_notify(self, detections: List[Detection], now: float | None = None) -> bool:
        now = time.time() if now is None else now

        if not detections:
            self._streak = 0
            return False

        if self._meets(detections):
            self._streak += 1
        else:
            self._streak = 0

        if self._streak >= self.frames_required:
            if (now - self._last_sent_at) >= self.cooldown_seconds:
                self._last_sent_at = now
                self._streak = 0  # reset after sending
                return True
        return False

