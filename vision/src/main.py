from __future__ import annotations

"""CLI entrypoint for the camera detection service."""

import argparse
from typing import Optional, List
import os

import uvicorn

from src.config import Config
from src.container import Container


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="UVC camera streamer with ml_api detections (MJPEG)")
    p.add_argument("--camera", type=int, default=None, help="Camera device index (default: 0)")
    p.add_argument("--width", type=int, default=None, help="Capture width")
    p.add_argument("--height", type=int, default=None, help="Capture height")
    p.add_argument("--fps", type=int, default=None, help="Capture FPS hint")
    p.add_argument("--api-url", type=str, default=None, help="ml_api base URL")
    p.add_argument("--token", type=str, default=None, help="Bearer token for ml_api")
    p.add_argument("--public-base", type=str, default=None, help="Base URL where ml_api can reach this service")
    p.add_argument("--detect-interval", type=float, default=None, help="Seconds between detection requests")
    p.add_argument("--host", type=str, default=None)
    p.add_argument("--port", type=int, default=None)
    return p.parse_args(argv)


def build_config_from_env_and_args(args) -> Config:
    # Start with env defaults from Config dataclass; overlay CLI overrides if provided
    cfg = Config()
    kwargs = dict(
        camera_index=cfg.camera_index if args.camera is None else args.camera,
        width=cfg.width if args.width is None else args.width,
        height=cfg.height if args.height is None else args.height,
        fps=cfg.fps if args.fps is None else args.fps,
        api_url=cfg.api_url if args.api_url is None else args.api_url,
        api_token=cfg.api_token if args.token is None else args.token,
        public_base=cfg.public_base if args.public_base is None else args.public_base,
        detect_interval=cfg.detect_interval if args.detect_interval is None else args.detect_interval,
        host=cfg.host if args.host is None else args.host,
        port=cfg.port if args.port is None else args.port,
    )
    return Config(**kwargs)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = build_config_from_env_and_args(args)
    container = Container(cfg)
    uvicorn.run(container.app, host=cfg.host, port=cfg.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

