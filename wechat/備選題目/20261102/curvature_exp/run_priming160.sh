#!/usr/bin/env bash
set -euo pipefail
cd /home/lmxxf/work/interp-curvature-exp
export PYTHONUNBUFFERED=1
venv/bin/python run_priming160.py --model Qwen3-0.6B
venv/bin/python run_priming160.py --model Qwen3.8-27B-FP8
