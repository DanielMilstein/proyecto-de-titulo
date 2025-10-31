from __future__ import annotations

"""Dependency container that wires config, workers, client, and FastAPI app."""

from dataclasses import dataclass

from src.config import Config
from src.services.camera_worker import CameraConfig
from src.services.detection_worker import DetectionConfig
from src.services.factories import create_camera_worker, create_detection_worker, create_app
from src.services.ml_api_client import MlApiClient


@dataclass
class Container:
    config: Config

    def __post_init__(self) -> None:
        # Build dependencies
        self.client = MlApiClient(base_url=self.config.api_url, token=self.config.api_token)
        self.camera = create_camera_worker(
            CameraConfig(
                device_index=self.config.camera_index,
                width=self.config.width,
                height=self.config.height,
                fps=self.config.fps,
            ),
            start=True,
        )
        self.detector = create_detection_worker(
            client=self.client,
            camera_latest=self.camera.latest,
            cfg=DetectionConfig(public_base=self.config.public_base, interval=self.config.detect_interval),
            start=True,
        )

        self.app = create_app(client=self.client, frame_source=self.camera, detection_source=self.detector)

        @self.app.on_event("shutdown")
        def _shutdown():
            try:
                self.detector.stop()
            finally:
                self.camera.stop()
