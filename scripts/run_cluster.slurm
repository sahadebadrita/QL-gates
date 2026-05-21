#!/bin/bash
#SBATCH --job-name=qlgates_sim
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=0-0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=ds2325@princeton.edu
#SBATCH --partition=debug        # <- change to your cluster's partition name

# ── Config ────────────────────────────────────────────────────────
VENV_PATH="/home/ds2325/QL-gates/graphs"     # <- change this
CONFIG=${1:-configs/run1.yaml}  # pass as argument, or falls back to default
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Activate venv ─────────────────────────────────────────────────
source "$VENV_PATH/bin/activate"

# ── Install package ───────────────────────────────────────────────
cd "$REPO_ROOT"
pip install -e . --quiet

# ── Make sure output dirs exist ───────────────────────────────────
mkdir -p logs results

# ── Run ───────────────────────────────────────────────────────────
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Config     : $CONFIG"
echo "Start time : $(date)"

python scripts/run_sim.py --config "$CONFIG"

echo "End time   : $(date)"