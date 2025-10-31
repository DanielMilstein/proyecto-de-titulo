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

### Cómo ejecutar (CLI)
1) Inicia `ml_api` (Docker o local): `docker compose up -d ml_api` (expone `http://localhost:3333`).

2) Lanza el servicio de cámara localmente (entrada canónica):
```
python src/main.py \
  --api-url http://localhost:3333 \
  --public-base http://localhost:8080 \
  --port 8080
```

Si `ml_api` corre dentro de Docker y el servicio de cámara corre en el host, usa:
```
python src/main.py \
  --api-url http://localhost:3333 \
  --public-base http://host.docker.internal:8080 \
  --port 8080
```

Abre el stream en el navegador: `http://localhost:8080/` (o `http://localhost:8080/stream.mjpg`).

También se mantiene compatibilidad con: `python src/services/cam_service.py`.

### Flags útiles
- `--camera 0`: índice del dispositivo UVC (por defecto 0).
- `--width/--height/--fps`: sugerencias de captura.
- `--detect-interval 0.5`: segundos entre peticiones a `ml_api`.
- `--token $ML_API_TOKEN`: si `ml_api` requiere Bearer token.

### Variables de entorno soportadas
- `ML_API_BASE_URL` y `ML_API_TOKEN`: base URL y token para `ml_api`.
- `PUBLIC_BASE`: URL pública desde la perspectiva de `ml_api` (por ejemplo `http://host.docker.internal:8080`).

Nota: el servicio intenta solicitar YUYV al dispositivo, pero expone y procesa frames en BGR mediante OpenCV.

### Ejecutar con Docker Compose (ambos contenedores)

Se incluye `Dockerfile.cam` para el servicio FastAPI. Puedes levantar ambos servicios vía compose:

```
docker compose up -d
```

- `ml_api` expone `http://localhost:3333`
- `camera_app` expone `http://localhost:8080`

Notas sobre cámara UVC dentro de Docker:
- En Linux, pasa el dispositivo al contenedor (descomenta en `docker-compose.yaml`):
  - `devices: - /dev/video0:/dev/video0` y, si es necesario, `privileged: true`.
- En macOS/Windows, el acceso directo a la cámara desde Docker no está soportado; ejecuta el servicio de cámara en el host.

Variables útiles en Compose:
- `ML_API_BASE_URL=http://ml_api:3333` (acceso interno a ml_api)
- `PUBLIC_BASE=http://camera_app:8080` (acceso interno desde ml_api al snapshot)
