#!/bin/bash
# ============================================================
# run_on_hpc.sh  —  RAGNAR evaluation job for hpc.itu.dk
#
# Submit:   sbatch run_on_hpc.sh
# Or run:   bash run_on_hpc.sh
# ============================================================

#SBATCH --job-name=ragnar_eval
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

mkdir -p logs

echo "============================================================"
echo "  RAGNAR HPC Evaluation"
echo "  Job  : ${SLURM_JOB_ID:-local}"
echo "  Node : $(hostname)"
echo "  Start: $(date)"
echo "============================================================"

# ── Activate environment ─────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ragnar          # or: source venv/bin/activate

# ── Run evaluation ───────────────────────────────────────────────────
# The Python script handles all hardware monitoring internally.
# Pass --compare to run both reranker configs and print a diff table.

python run_hpc_evaluation.py \
  --dataset     squad \
  --qa_limit    200   \
  --wiki_limit  5000  \
  --hw_interval 2     \
  --compare

echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
