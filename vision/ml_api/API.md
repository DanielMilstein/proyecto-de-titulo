# ml_api HTTP API

Base URL: `http://localhost:3333` (via `docker compose up -d`).

## Authentication
- Optional Bearer token. If `ML_API_TOKEN` is set, every request must include:
  - Header: `Authorization: Bearer <ML_API_TOKEN>`

## Endpoints

### GET `/hc/`
- Purpose: Health check; verifies the model loaded.
- Response: `200 OK` with plain text `ok` (or `error` if model failed to load).
- Example:
  - `curl -s http://localhost:3333/hc/`

### GET `/p/`
- Purpose: Run inference on an external image URL.
- Query params:
  - `img` (required): Absolute URL to an image (jpg/png/bmp/tiff).
- Auth: Requires `Authorization` header if `ML_API_TOKEN` is set.
- Response: `200 OK` JSON: `{ "detections": [...] }`
- Detection item format: `[label: string, score: number, bbox: [xc, yc, w, h]]`
  - Coordinates are pixel values relative to the original image.
  - `xc, yc` = center point; `w, h` = width and height.
- Examples:
  - Without auth:
    - `curl "http://localhost:3333/p/?img=https://example.com/image.jpg"`
  - With auth:
    - `curl -H "Authorization: Bearer $ML_API_TOKEN" "http://localhost:3333/p/?img=https://example.com/image.jpg"`
  - Sample response:
```
{
  "detections": [
    ["spaghetti", 0.91, [512.4, 380.2, 130.7, 96.4]],
    ["error", 0.37, [290.0, 420.5, 80.3, 60.1]]
  ]
}
```

## Error Responses
- `401 Unauthorized`: Missing/invalid token when `ML_API_TOKEN` is configured.
- `400 Bad Request`: Failed to fetch or decode the image URL.
- `422 Unprocessable Entity`: Missing or invalid query params (e.g., no `img`).
- `5xx`: Unexpected server/model errors; check logs and Sentry (if `SENTRY_DSN` configured).

## Notes & Tips
- Supported formats: server attempts to decode via OpenCV (`.jpg/.jpeg/.png/.bmp/.tif/.tiff`).
- Threshold/NMS: Internal defaults are applied in the model backend; scores reflect class confidence after NMS.
- Observability: set `SENTRY_DSN` to capture errors; health check is `GET /hc/`.
- Networking: the server downloads the image from `img` URL; ensure it is publicly reachable from the container.

## Minimal Python Client
```py
import os, requests
url = "http://localhost:3333/p/"
params = {"img": "https://example.com/image.jpg"}
headers = {}
if os.getenv("ML_API_TOKEN"):
    headers["Authorization"] = f"Bearer {os.environ['ML_API_TOKEN']}"
r = requests.get(url, params=params, headers=headers, timeout=10)
r.raise_for_status()
print(r.json()["detections"])  # [[label, score, [xc, yc, w, h]], ...]
```

