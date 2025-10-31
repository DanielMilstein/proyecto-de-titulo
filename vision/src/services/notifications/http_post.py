from __future__ import annotations

import base64
import json
from typing import Dict, Optional

import requests

from src.services.notifications.base import NotificationEvent, NotificationSender


class HttpPostNotifier(NotificationSender):
    """Posts a JSON payload to a webhook endpoint.

    Schema:
      {
        "title": str|null,
        "text": str|null,
        "timestamp": float|null,
        "detections": [
          {"label": str, "score": float, "bbox": {"xc":...,"yc":...,"w":...,"h":...}}, ...
        ],
        "image_jpeg_base64": str|null
      }
    """

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 10.0) -> None:
        self._url = url
        self._headers = headers or {}
        self._timeout = timeout

    def send(self, event: NotificationEvent) -> None:
        payload = {
            "title": event.title,
            "text": event.text,
            "timestamp": event.timestamp,
            "detections": [
                {
                    "label": d.label,
                    "score": d.score,
                    "bbox": {"xc": d.bbox.xc, "yc": d.bbox.yc, "w": d.bbox.w, "h": d.bbox.h},
                }
                for d in event.detections
            ],
            "image_jpeg_base64": base64.b64encode(event.image_bytes).decode("ascii") if event.image_bytes else None,
        }
        r = requests.post(self._url, json=payload, headers=self._headers, timeout=self._timeout)
        r.raise_for_status()

