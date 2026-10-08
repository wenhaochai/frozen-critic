#!/bin/bash
#SBATCH --job-name=fc-gdntest
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/frozen-critic/logs/slurm/%x-%j.out
# Smoke test: flash_qla vs fla Triton gradients for the GDN kernel (both modes, same inputs).
set -euo pipefail
R=/scratch/gpfs/GROUP/USER/project/frozen-critic; O=$R/results/gdn_backend; mkdir -p $O
S=$R/images/miles-20260930.sif; X="apptainer exec --nv --contain --cleanenv --bind /scratch/gpfs/GROUP/USER --env HOME=/tmp"
echo "== dispatch (flash_qla)"; $X $S python3 $R/tests/gdn_backend_grads.py $O/dispatch.pt
echo "== fla triton";          $X --env FLA_DISABLE_BACKEND_DISPATCH=1 $S python3 $R/tests/gdn_backend_grads.py $O/fla.pt
$X $S python3 - <<PY
import torch
a, b = torch.load("$O/dispatch.pt"), torch.load("$O/fla.pt")
for call in range(3):
    print("call", call, {n: round(((a[call][n] - b[call][n]).abs().max() / (b[call][n].abs().max() + 1e-6)).item(), 4) for n in b[call]})
for n in "qkvgb":
    print("dispatch call0 vs call2", n, round((a[0][n] - a[2][n]).abs().max().item(), 6), "| fla call0 vs call2", round((b[0][n] - b[2][n]).abs().max().item(), 6))
PY
