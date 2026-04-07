# OpenClaw + llama.cpp on Jetson Xavier (CUDA)

Run [OpenClaw](https://openclaw.ai) — a local AI assistant — fully offline on
an NVIDIA Jetson Xavier (or Xavier NX) using a CUDA-accelerated `llama-server`
from [llama.cpp](https://github.com/ggml-org/llama.cpp) as the model backend.

This repo documents the working setup and ships the exact OpenClaw config
needed to wire it to a local llama.cpp OpenAI-compatible endpoint.

> Verified on: Jetson Xavier (8 GB class, 7.5 GB usable VRAM), JetPack
> Linux 5.10 (Tegra), aarch64, Node.js v22, OpenClaw `2026.4.5`,
> llama.cpp build `8672 (25eec6f32)`, model
> `qwen2.5-3b-instruct-q4_k_m.gguf`.

---

## Why this setup

OpenClaw expects an LLM provider with **at least a 16k context window** (its
system prompt + skill manifests already consume ~16k tokens). It speaks the
OpenAI Chat Completions API, so any `llama-server` instance with enough
context works as a drop-in provider — no cloud API keys, no data leaving the
device.

Xavier's 7.5 GB VRAM constrains model choice to roughly a **3B–4B parameter**
model at Q4 quantization with a 32k context (using flash attention to keep the
KV cache small enough to fit).

---

## Prerequisites

1. **Jetson Xavier / Xavier NX** with JetPack and a working CUDA toolchain.
2. **llama.cpp built with CUDA**:
   ```bash
   git clone https://github.com/ggml-org/llama.cpp
   cd llama.cpp
   cmake -B build -DGGML_CUDA=ON
   cmake --build build -j --config Release
   ```
   Verify: `./build/bin/llama-server --version` should print `ggml_cuda_init: found 1 CUDA devices`.
3. **A GGUF model** that fits in VRAM. Recommended:
   [Qwen2.5-3B-Instruct Q4_K_M](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF)
   (≈ 2 GB on disk, 32k native context).
4. **Node.js ≥ 20** and npm.

---

## Step 1 — Install OpenClaw

```bash
npm install -g openclaw
openclaw --version
```

If you previously had a broken/partial install (e.g. an empty
`node_modules/@buape/carbon` directory) the CLI will throw
`Cannot find module '@buape/carbon'`. Fix it by reinstalling cleanly:

```bash
npm uninstall -g openclaw
npm cache verify
npm install -g openclaw
```

---

## Step 2 — Start `llama-server`

Run llama.cpp's OpenAI-compatible server with **at least 32k context** and
flash attention enabled (essential to fit the KV cache on Xavier):

```bash
cd /path/to/llama.cpp
./build/bin/llama-server \
    -m models/qwen2.5-3b-instruct-q4_k_m.gguf \
    --host 127.0.0.1 --port 8080 \
    -ngl 99 \
    -c 32768 \
    -fa on
```

Sanity check the endpoint:

```bash
curl -s http://127.0.0.1:8080/v1/models | head -c 200
curl -s http://127.0.0.1:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"qwen2.5-3b-instruct-q4_k_m.gguf",
          "messages":[{"role":"user","content":"say hi"}]}'
```

A `systemd` unit (see [`systemd/llama-server.service`](systemd/llama-server.service))
is provided if you want it to run on boot.

---

## Step 3 — Wire OpenClaw to llama.cpp

Apply the bundled provider config in one shot:

```bash
openclaw config set --batch-file openclaw-llamacpp.batch.json
openclaw models set llamacpp/qwen2.5-3b-instruct-q4_k_m.gguf
```

This writes the following into `~/.openclaw/openclaw.json`:

- A new provider `llamacpp` pointing at `http://127.0.0.1:8080/v1`, using the
  `openai-completions` adapter and a dummy API key (llama-server doesn't
  enforce auth unless you start it with `--api-key`).
- A model entry with **`contextWindow: 32768`** and
  **`compat.maxTokensField: max_tokens`** (llama.cpp follows the legacy
  OpenAI field name).
- `models.mode: merge` so OpenClaw's built-in model catalog still loads.
- `agents.defaults.memorySearch.enabled: false` — disables semantic memory,
  which would otherwise demand an embedding provider key.

See [`openclaw-llamacpp.batch.json`](openclaw-llamacpp.batch.json) for the
exact payload.

---

## Step 4 — Smoke test

```bash
# Reset session state if you've previously had a failed run wedged here:
echo '{}' > ~/.openclaw/agents/main/sessions/sessions.json

openclaw agent --local --agent main \
    --message "Reply with exactly the word PONG and nothing else."
```

Expected output:

```
PONG
```

You're done. Drop the `--local` flag once you run `openclaw onboard` to
install the Gateway service (interactive — needs a real TTY).

---

## Gotchas hit during this setup

- **Empty `@buape/carbon` from a partial npm install** — symptom is
  `Cannot find module '@buape/carbon'`. Fix: clean reinstall (see Step 1).
- **`ctx=4096 (min=16000)`** — OpenClaw refuses any provider with a context
  window under 16 000 tokens. Both `llama-server -c` *and* the
  `contextWindow` field in OpenClaw's config must be ≥ 16 000. We use 32 768.
- **`request (16671 tokens) exceeds the available context size (16384 tokens)`** —
  same root cause: the system prompt alone is ~16 k. Use 32 k everywhere.
- **Failed agent runs are sticky** — OpenClaw persists session state in
  `~/.openclaw/agents/main/sessions/sessions.json` and will replay the same
  failing run id forever. Wipe that file (`echo '{}' > …`) to recover.
- **`models set <bare-id>` may pick the wrong provider** — OpenClaw resolved
  the bare id under the built-in `openai/` provider. Always pass the
  fully-qualified id: `llamacpp/qwen2.5-3b-instruct-q4_k_m.gguf`.
- **`-fa` syntax** — newer llama.cpp requires
  `-fa on` (or `auto`/`off`); the bare flag form is rejected.

---

## Files in this repo

| File | Purpose |
|---|---|
| `README.md` | This document. |
| `openclaw-llamacpp.batch.json` | OpenClaw `config set --batch-file` payload that registers the llama.cpp provider and disables embedding-dependent features. |
| `scripts/start-llama-server.sh` | Convenience wrapper that launches `llama-server` with the recommended flags. Edit `MODEL_PATH` and `LLAMA_CPP_DIR` for your install. |
| `systemd/llama-server.service` | Optional `systemd` unit to auto-start `llama-server` at boot. |

---

## Maintainer

Worrajak Muangjai · <worrajak@gmail.com>

Issues and PRs welcome.
