#!/bin/bash
set -e  # stop on first error

# ── Config ────────────────────────────────────────────────────────
CONDA_ENV="graphs"          # <- change this
CONFIG="configs/run2.yaml"   # <- change this
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"  # always relative to script location

# ── Activate conda ────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ── Install package ───────────────────────────────────────────────
cd "$REPO_ROOT"
pip install -e . --quiet

# ── Run ───────────────────────────────────────────────────────────
echo "Running simulation with config: $CONFIG"
#python scripts/qldynamics_simulations.py --config "$CONFIG"
python $1 --config "$CONFIG"
#python scripts/postprocessing.py --config "$CONFIG"
echo "Done."
