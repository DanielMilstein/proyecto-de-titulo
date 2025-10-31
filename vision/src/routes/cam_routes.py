from __future__ import annotations

"""Camera + detection routes as an APIRouter."""

from dataclasses import asdict
from typing import Optional, Tuple

import cv2
import numpy as np
from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from src.schemas.detection import Detection
from src.services.interfaces import FrameSource, DetectionSource, DetectionClient


def _encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    ok, data = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return data.tobytes()


def create_cam_router(
    client: DetectionClient,
    frame_source: FrameSource,
    detection_source: DetectionSource,
) -> APIRouter:
    router = APIRouter()

    @router.get("/hc/")
    def health() -> JSONResponse:
        try:
            ok = client.health()
        except Exception:
            ok = False
        frame, ts = frame_source.latest()
        return JSONResponse({"ml_api": ok, "frame": frame is not None, "ts": ts})

    @router.get("/snapshot.jpg")
    def snapshot() -> Response:
        frame, _ = frame_source.latest()
        if frame is None:
            return Response(status_code=503)
        data = _encode_jpeg(frame)
        return Response(content=data, media_type="image/jpeg")

    @router.get("/detections")
    def detections() -> JSONResponse:
        dets, _, at = detection_source.latest()
        return JSONResponse({
            "timestamp": at,
            "detections": [
                dict(label=d.label, score=d.score, bbox=asdict(d.bbox)) for d in dets
            ],
        })

    @router.get("/stream.mjpg")
    def stream() -> StreamingResponse:
        boundary = "frame"

        def gen():
            import time as _t
            while True:
                _dets, annotated, _ = detection_source.latest()
                if annotated is None:
                    frame, _ = frame_source.latest()
                    if frame is None:
                        _t.sleep(0.05)
                        continue
                    img = frame
                else:
                    img = annotated
                try:
                    data = _encode_jpeg(img, quality=80)
                except Exception:
                    _t.sleep(0.01)
                    continue
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    + b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(data)}\r\n\r\n".encode()
                    + data
                    + b"\r\n"
                )
                _t.sleep(0.03)

        return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")

    @router.get("/")
    def index() -> HTMLResponse:
        html = """
        <html>
          <head><title>Camera Detections</title></head>
          <body style="margin:0;background:#000;">
            <img src="/stream.mjpg" style="width:100%;max-width:100%;" />
          </body>
        </html>
        """
        return HTMLResponse(html)

    return router

