#!/usr/bin/env bash
set -euo pipefail
cd /home/lmxxf/work/interp-curvature-exp/basepost
export PYTHONUNBUFFERED=1
../venv/bin/python prepare_protocol.py
../venv/bin/python run_basepost.py --model Qwen3.5-35B-A3B-Base
../venv/bin/python run_basepost.py --model Qwen3.5-35B-A3B
../venv/bin/python summarize_basepost.py
