#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8006}"
CONDA_ENV="${CONDA_ENV:-syntheseus-full-mic}"

export MAX_WORKERS=1
export USE_GPU=false

# 关闭 OpenMP/MKL 线程池，避免 fork 后子进程死锁
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "Starting Retro Synthesis Search API on port $PORT ..."
echo "Using conda env: $CONDA_ENV"
conda run -n "$CONDA_ENV" --no-capture-output uvicorn server.server:app --host 0.0.0.0 --port "$PORT" --timeout-keep-alive 1800
