#!/bin/bash
set -e


MODEL_DIR=/app/model
ENGINE_FILE=$MODEL_DIR/model.engine
ONNX_FILE=$MODEL_DIR/model.onnx
CFG_FILE=$MODEL_DIR/model.cfg
WEIGHTS_FILE=$MODEL_DIR/model.weights

if [ -f "$ENGINE_FILE" ]; then
	echo "[entrypoint] Found TensorRT engine, skipping build."
else
	if [ -f "$ONNX_FILE" ]; then 
		echo "[entrypoint] Building TensorRT engine from ONNX..."
		/usr/src/tensorrt/bin/trtexec \
		--onnx=$ONNX_FILE \
		--saveEngine=$ENGINE_FILE \
		--workspace=2048 \
		--fp16

	else
		echo "[entrypoint] Converting darknet to ONNX..."
		python3 /app/ml

echo "[entrypoint] Starting Obico ML API..."
cd /app
# the original runs via gunicorn; we can do the same
exec gunicorn \
    --bind 0.0.0.0:3335 \
    --workers 2 \
    --timeout 120 \
    wsgi:application