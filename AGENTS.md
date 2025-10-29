# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Python package scaffolding (`models/`, `schemas/`, `services/`, `utils/`). Keep new domain logic here.
- `ml_api/`: Flask-based inference service (gunicorn/wsgi). Model files live in `ml_api/model/`; core helpers under `ml_api/lib/`.
- `imagenes/`: Sample input images. `outputs/` and `out/` store rendered detections.
- `obico-server/`: Upstream server code (vendor). Avoid modifying unless you know the impact.
- Root: `docker-compose.yaml`, `requirements.txt`, ad‑hoc scripts `test_obico_model.py`, `test_hf.py`.

## Build, Test, and Development Commands
- Install deps (local venv): `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Run ML API (Docker): `docker compose up -d` (exposes `http://localhost:3333`)
- Run ML API (local): `cd ml_api && gunicorn --bind 0.0.0.0:3333 --workers 1 wsgi`
- Quick ONNX evaluation: `python test_obico_model.py --model path/to/model.onnx --source imagenes --save-dir out --num-classes 1 --anchors-preset voc`
- YOLOv5 HF demo: `python test_hf.py` (downloads weights; requires network access)

## Coding Style & Naming Conventions
- Python: 4‑space indentation, PEP 8; prefer type hints and docstrings.
- Names: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Files: keep new modules under `src/`; API-only helpers in `ml_api/lib/`.

## Testing Guidelines
- Current tests are script-based. Keep inputs under `imagenes/` and write results to `outputs/` or `out/`.
- Add small, deterministic samples for new features. When adding formal tests later, use `pytest` and name files `test_*.py`.
- Measure runtime where relevant; print per-image timings as in `test_obico_model.py`.

## Commit & Pull Request Guidelines
- Use clear, imperative commit messages (e.g., `feat: add ONNX decoder`, `fix: handle empty detections`). Group logical changes.
- PRs must include: purpose, summary of changes, run commands, and before/after evidence (logs or screenshots of detections).
- Link related issues. Exclude large binaries; reference external storage instead.

## Security & Configuration Tips
- Auth: `ML_API_TOKEN` enables `Authorization: Bearer <token>` checks in `ml_api`.
- Observability: set `SENTRY_DSN` to capture errors.
- Health: service check at `GET /hc/` (port `3333`).
- Keep secrets in env vars or `.env` (not committed). Large models should not live in the repo.

