from __future__ import annotations

"""Camera worker that reads from a UVC device in a background thread.

Single-responsibility: acquire frames and expose the most recent frame.
"""

from dataclasses import dataclass
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    device_index: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None


class CameraWorker:
    """Grabs frames from a UVC camera in a background thread.

    Tries to configure YUYV but always exposes frames as BGR numpy arrays.
    """

    def __init__(self, config: CameraConfig) -> None:
        self._cfg = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._t: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._last_at: float = 0.0

    def start(self) -> None:
        cfg = self._cfg
        self._cap = cv2.VideoCapture(cfg.device_index, cv2.CAP_ANY)
        # Try to force YUYV; ignore failures (OpenCV may not honor it)
        try:
            fourcc = cv2.VideoWriter_fourcc(*"YUYV")
            self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass

        if cfg.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.width))
        if cfg.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.height))
        if cfg.fps:
            self._cap.set(cv2.CAP_PROP_FPS, int(cfg.fps))

        if not self._cap or not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera {cfg.device_index}")

        self._stop.clear()
        self._t = threading.Thread(target=self._loop, name="CameraWorker", daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._t and self._t.is_alive():
            self._t.join(timeout=1.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._cap:
                break
            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._last_frame = frame
                    self._last_at = time.time()
            time.sleep(0.001)

    def latest(self) -> Tuple[Optional[np.ndarray], float]:
        with self._lock:
            if self._last_frame is None:
                return None, 0.0
            return self._last_frame.copy(), self._last_at

