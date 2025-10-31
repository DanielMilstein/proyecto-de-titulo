from __future__ import annotations

"""Detection worker that periodically queries ml_api for detections.

Single-responsibility: fetch detections for a given image URL and
store the latest detections and optional annotated frame.
"""

from dataclasses import dataclass
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from src.schemas.detection import Detection
from src.services.ml_api_client import MlApiClient
from src.utils.visualization import draw_detections


@dataclass(frozen=True)
class DetectionConfig:
    public_base: str
    interval: float = 0.5


class DetectionWorker:
    def __init__(self, client: MlApiClient, camera_latest, config: DetectionConfig):
        """
        Args:
            client: MlApiClient used to call ml_api.
            camera_latest: callable returning (frame: np.ndarray|None, ts: float)
            config: DetectionConfig with public_base and interval.
        """
        base = config.public_base.rstrip("/")
        self._client = client
        self._public_base = base
        self._interval = max(0.05, config.interval)
        self._camera_latest = camera_latest

        self._t: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_dets: List[Detection] = []
        self._last_annotated: Optional[np.ndarray] = None
        self._last_at: float = 0.0

    def start(self) -> None:
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, name="DetectionWorker", daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._t and self._t.is_alive():
            self._t.join(timeout=1.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            ts = int(time.time() * 1000)
            snapshot_url = f"{self._public_base}/snapshot.jpg?ts={ts}"
            try:
                dets = self._client.detect_url(snapshot_url)
                frame, _ = self._camera_latest()
                if frame is not None:
                    annotated = draw_detections(frame, dets)
                    with self._lock:
                        self._last_dets = dets
                        self._last_annotated = annotated
                        self._last_at = time.time()
            except Exception:
                # Ignore transient errors; keep last good result
                pass
            time.sleep(self._interval)

    def latest(self) -> Tuple[List[Detection], Optional[np.ndarray], float]:
        with self._lock:
            dets = list(self._last_dets)
            annotated = None if self._last_annotated is None else self._last_annotated.copy()
            at = self._last_at
        return dets, annotated, at

