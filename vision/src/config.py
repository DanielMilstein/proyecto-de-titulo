from __future__ import annotations

"""Application configuration and defaults.

Reads environment variables and provides a typed configuration object.
"""

from dataclasses import dataclass
import os
from typing import Optional


@dataclass(frozen=True)
class Config:
    # Camera
    camera_index: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None

    # ml_api
    api_url: str = os.getenv("ML_API_BASE_URL", "http://localhost:3333")
    api_token: Optional[str] = os.getenv("ML_API_TOKEN")

    # Service
    public_base: str = os.getenv("PUBLIC_BASE", "http://localhost:8080")
    detect_interval: float = 0.5
    host: str = "0.0.0.0"
    port: int = 8080

