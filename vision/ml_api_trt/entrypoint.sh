#!/bin/bash
set -e

MODEL_DIR=/app/model
ENGINE_FILE="$MODEL_DIR/model.engine"
ONNX_FILE="$MODEL_DIR/model.onnx"
CFG_FILE="$MODEL_DIR/model.cfg"
WEIGHTS_FILE="$MODEL_DIR/model.weights"


if [ -f "$ONNX_FILE" ]; then
    /usr/src/tensorrt/bin/trtexec \
        --onnx="$ONNX_FILE" \
        --saveEngine="$ENGINE_FILE" \
        --fp16

fi

echo "[entrypoint] Starting Obico ML API..."
cd /app
exec gunicorn --bind 0.0.0.0:3333 --workers 2 --timeout 120 wsgi:application
