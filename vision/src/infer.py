from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests

from src.services.ml_api_client import MlApiClient
from src.utils.visualization import draw_detections


def _read_image_bgr_from_url(url: str, timeout: float = 10.0) -> np.ndarray:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    data = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode image from URL: {url}")
    return img


def _read_image_bgr_from_path(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image from path: {path}")
    return img


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Query ml_api, draw detections, and display the image.")
    parser.add_argument("--img-url", type=str, required=True, help="Public URL to the image to analyze.")
    parser.add_argument("--display", action="store_true", help="Show the image in a window (cv2.imshow).")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Minimum confidence to show detections (0..1 or 0..100). Default: 0.30 (30%).",
    )
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save the annotated image.")
    parser.add_argument("--api-url", type=str, default=os.getenv("ML_API_BASE_URL", "http://localhost:3333"), help="Base URL for ml_api.")
    parser.add_argument("--token", type=str, default=os.getenv("ML_API_TOKEN"), help="Bearer token for ml_api, if required.")
    args = parser.parse_args(argv)

    client = MlApiClient(base_url=args.api_url, token=args.token)

    # Fetch detections
    detections = client.detect_url(args.img_url)

    # Normalize threshold (allow percent input like 60 for 60%)
    thr = args.threshold
    if thr > 1.0:
        thr = thr / 100.0

    # Apply threshold filter
    detections = [d for d in detections if d.score >= thr]

    # Load image for visualization (from URL)
    image_bgr = _read_image_bgr_from_url(args.img_url)

    # Draw and export
    vis = draw_detections(image_bgr, detections)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save), vis)
        print(f"Saved: {args.save}")

    if args.display:
        cv2.imshow("Detections", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Print a compact textual summary
    for det in detections:
        x1, y1, x2, y2 = det.bbox.to_xyxy()
        print(f"{det.label}\t{det.score:.2f}\t({x1},{y1},{x2},{y2})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
