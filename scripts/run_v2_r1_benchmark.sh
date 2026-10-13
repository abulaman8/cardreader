#!/bin/bash
set -euo pipefail
cd /home/pilot/Desktop/cardreader

echo "[$(date -Iseconds)] Step 1: Starting inference on frozen benchmark games (50 val + 50 test)..."
./venv/bin/python -m src.v2.infer --ckpt runs/v2_r1/best.pt --stride 10

echo "[$(date -Iseconds)] Step 2: Running benchmark evaluation script..."
./venv/bin/python benchmark/benchmark.py --run v2_r1 \
    --val-pred runs/v2_r1/bench/calibrated/val \
    --test-pred runs/v2_r1/bench/calibrated/test

echo "[$(date -Iseconds)] Benchmark finished successfully."
