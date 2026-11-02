#!/usr/bin/env bash
set -euo pipefail
cd /home/lmxxf/work/interp-curvature-exp
export PYTHONUNBUFFERED=1
venv/bin/python run_curvature_experiments.py --model Qwen3-0.6B
sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches"
venv/bin/python run_curvature_experiments.py --model Qwen3.8-27B-FP8
venv/bin/python summarize.py
