from __future__ import annotations

import os
from typing import List, Sequence

import requests

from src.schemas.detection import Detection, BBox


class MlApiClient:
    """HTTP client for the `ml_api` service.

    Responsibilities:
    - Build and send requests to ml_api
    - Map JSON responses into rich, typed Python objects

    Usage:
        client = MlApiClient(base_url="http://localhost:3333", token=os.getenv("ML_API_TOKEN"))
        dets = client.detect_url("https://example.com/image.jpg")
    """

    def __init__(self, base_url: str, token: str | None = None, timeout: float = 10.0) -> None:
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        self._base_url = base_url
        self._token = token
        self._timeout = timeout

    @property
    def base_url(self) -> str:
        return self._base_url

    def _headers(self) -> dict:
        headers: dict[str, str] = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def health(self) -> bool:
        url = f"{self._base_url}/hc/"
        r = requests.get(url, timeout=self._timeout)
        r.raise_for_status()
        return r.text.strip().lower() == "ok"

    def detect_url(self, img_url: str) -> List[Detection]:
        """Run inference for an image available at a public URL.

        Returns a list of Detection objects. The server returns detections in the
        format: [label, score, [xc, yc, w, h]].
        """
        url = f"{self._base_url}/p/"
        r = requests.get(url, params={"img": img_url}, headers=self._headers(), timeout=self._timeout)
        r.raise_for_status()
        data = r.json()
        return self._parse_detections(data.get("detections", []))

    @staticmethod
    def _parse_detections(raw: Sequence[Sequence]) -> List[Detection]:
        detections: List[Detection] = []
        for item in raw:
            # Expected: [label, score, [xc, yc, w, h]]
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                continue
            label = str(item[0])
            try:
                score = float(item[1])
                bbox_vals = item[2]
                bbox = BBox(float(bbox_vals[0]), float(bbox_vals[1]), float(bbox_vals[2]), float(bbox_vals[3]))
            except Exception:
                continue
            detections.append(Detection(label=label, score=score, bbox=bbox))
        return detections

