#!/usr/bin/env bash
#
# Usage: srun --gres=gpu:1 --partition=besteffort ./launch.sh [infer_ollama.py args...]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OLLAMA_PORT=$((11434 + SLURM_JOB_ID % 1000))
OLLAMA_HOST="127.0.0.1"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:32b}"
OLLAMA_LOG="${OLLAMA_LOG:-ollama_${SLURM_JOB_ID}.log}"

export OLLAMA_HOST="${OLLAMA_HOST}:${OLLAMA_PORT}"

echo "[launch] Starting ollama server on port ${OLLAMA_PORT}..."
ollama serve > "${OLLAMA_LOG}" 2>&1 &
OLLAMA_PID=$!
trap "echo '[launch] Stopping ollama...'; kill ${OLLAMA_PID} 2>/dev/null || true" EXIT

echo "[launch] Waiting for ollama to be ready..."
MAX_WAIT=120
WAITED=0
while ! curl -s "http://${OLLAMA_HOST}/api/version" > /dev/null 2>&1; do
    sleep 1
    WAITED=$((WAITED + 1))
    if [[ ${WAITED} -ge ${MAX_WAIT} ]]; then
        echo "[launch] ERROR: ollama did not start within ${MAX_WAIT}s" >&2
        exit 1
    fi
done
echo "[launch] ollama ready after ${WAITED}s"

echo "[launch] Pulling model ${OLLAMA_MODEL}..."
ollama pull "${OLLAMA_MODEL}" || echo "[launch] Model may already be cached"

echo "[launch] Running infer_ollama.py..."
python -m src.infer_ollama \
    --host "${OLLAMA_HOST%%:*}" \
    --port "${OLLAMA_PORT}" \
    --model "${OLLAMA_MODEL}" \
    "$@"

echo "[launch] Done"
