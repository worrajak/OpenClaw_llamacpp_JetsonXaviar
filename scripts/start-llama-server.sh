#!/usr/bin/env bash
# Start llama.cpp's OpenAI-compatible server with settings tuned for
# OpenClaw on a Jetson Xavier (CUDA, ~7.5 GB VRAM).
#
# Edit the two paths below for your environment.
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
MODEL_PATH="${MODEL_PATH:-$LLAMA_CPP_DIR/models/qwen2.5-3b-instruct-q4_k_m.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CTX="${CTX:-32768}"

if [[ ! -x "$LLAMA_CPP_DIR/build/bin/llama-server" ]]; then
    echo "llama-server not found at $LLAMA_CPP_DIR/build/bin/llama-server" >&2
    echo "Build llama.cpp with -DGGML_CUDA=ON first." >&2
    exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Model not found: $MODEL_PATH" >&2
    exit 1
fi

exec "$LLAMA_CPP_DIR/build/bin/llama-server" \
    -m  "$MODEL_PATH" \
    --host "$HOST" --port "$PORT" \
    -ngl 99 \
    -c "$CTX" \
    -fa on
