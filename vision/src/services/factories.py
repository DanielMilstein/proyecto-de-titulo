from __future__ import annotations

"""Factories to assemble service components (workers and FastAPI app)."""

from typing import Callable, Optional, Tuple

import numpy as np

from src.services.camera_worker import CameraConfig, CameraWorker
from src.services.detection_worker import DetectionConfig, DetectionWorker
from src.services.ml_api_client import MlApiClient
from src.services.interfaces import FrameSource, DetectionSource, DetectionClient
from src.routes.cam_routes import create_cam_router

# We import FastAPI and helpers locally in create_app to avoid hard deps during unit tests


def create_camera_worker(cfg: CameraConfig, start: bool = True) -> CameraWorker:
    cam = CameraWorker(cfg)
    if start:
        cam.start()
    return cam


def create_detection_worker(
    client: MlApiClient,
    camera_latest: Callable[[], Tuple[Optional[np.ndarray], float]],
    cfg: DetectionConfig,
    start: bool = True,
) -> DetectionWorker:
    det = DetectionWorker(client=client, camera_latest=camera_latest, config=cfg)
    if start:
        det.start()
    return det


def create_app(
    client: DetectionClient,
    frame_source: FrameSource,
    detection_source: DetectionSource,
):
    from fastapi import FastAPI

    app = FastAPI(title="Camera Detection Service")
    app.include_router(create_cam_router(client, frame_source, detection_source))
    return app
