# Módulo de Visión para Proyecto de Título

## Camera Streaming Service (UVC → ml_api → MJPEG)

A lightweight FastAPI service is included to capture frames from a UVC webcam, send them to `ml_api` for inference, and expose a real‑time annotated stream.

- Código: `src/services/cam_service.py`
- Requisitos: OpenCV (para cámara y JPEG), FastAPI y uvicorn (ya cubiertos por `requirements.txt`).

### Endpoints
- `GET /hc/`: estado (ml_api y cámara).
- `GET /snapshot.jpg`: último frame sin anotar (usado por ml_api).
- `GET /detections`: último resultado de detección (JSON).
- `GET /stream.mjpg`: stream MJPEG con cajas dibujadas (tiempo real).
- `GET /`: vista HTML mínima que muestra el stream.

### Cómo ejecutar
1) Inicia `ml_api` (Docker o local):
- Docker: `docker compose up -d` (expone `http://localhost:3333`).

2) Lanza el servicio de cámara (en otra terminal):

- Si `ml_api` corre localmente:
```
python src/services/cam_service.py \
  --api-url http://localhost:3333 \
  --public-base http://localhost:8080 \
  --port 8080
```

- Si `ml_api` corre en Docker: usa `host.docker.internal` para que el contenedor acceda al snapshot.
```
python src/services/cam_service.py \
  --api-url http://localhost:3333 \
  --public-base http://host.docker.internal:8080 \
  --port 8080
```

Luego abre el stream en el navegador: `http://localhost:8080/` (o `http://localhost:8080/stream.mjpg`).

### Flags útiles
- `--camera 0`: índice del dispositivo UVC (por defecto 0).
- `--width/--height/--fps`: sugerencias de captura.
- `--detect-interval 0.5`: segundos entre peticiones a `ml_api`.
- `--token $ML_API_TOKEN`: si `ml_api` requiere Bearer token.

### Variables de entorno soportadas
- `ML_API_BASE_URL` y `ML_API_TOKEN`: base URL y token para `ml_api`.
- `PUBLIC_BASE`: URL pública desde la perspectiva de `ml_api` (por ejemplo `http://host.docker.internal:8080`).

Nota: el servicio intenta solicitar YUYV al dispositivo, pero expone y procesa frames en BGR mediante OpenCV.
