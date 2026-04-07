#!/usr/bin/env bash
# Start llama.cpp's OpenAI-compatible server with settings tuned for
# OpenClaw on a Jetson Xavier (CUDA, ~7.5 GB VRAM).
#
# Edit the two paths below for your environment.
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
MODEL_PATH="${MODEL_PATH:-$LLAMA_CPP_DIR/models/qwen2.5-3b-instruct-q4_k_m.gguf}"
HOST="${HOST:-127.0.0.1}"   # set HOST=0.0.0.0 to expose to the LAN
PORT="${PORT:-8080}"
CTX="${CTX:-32768}"
API_KEY="${LLAMA_API_KEY:-}" # required when HOST != 127.0.0.1

if [[ ! -x "$LLAMA_CPP_DIR/build/bin/llama-server" ]]; then
    echo "llama-server not found at $LLAMA_CPP_DIR/build/bin/llama-server" >&2
    echo "Build llama.cpp with -DGGML_CUDA=ON first." >&2
    exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Model not found: $MODEL_PATH" >&2
    exit 1
fi

args=(
    -m  "$MODEL_PATH"
    --host "$HOST" --port "$PORT"
    -ngl 99
    -c "$CTX"
    -fa on
)

if [[ "$HOST" != "127.0.0.1" && "$HOST" != "localhost" ]]; then
    if [[ -z "$API_KEY" ]]; then
        echo "HOST=$HOST exposes the server to the network." >&2
        echo "Refusing to start without LLAMA_API_KEY set." >&2
        echo "Generate one with:  openssl rand -hex 24" >&2
        exit 1
    fi
    args+=( --api-key "$API_KEY" )
fi

exec "$LLAMA_CPP_DIR/build/bin/llama-server" "${args[@]}"
