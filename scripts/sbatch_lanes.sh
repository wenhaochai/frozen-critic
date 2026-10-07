#!/bin/bash
# Packed Phase 1 job (owner's packing rules, 2026-10-07): one lane per run, 4 GPUs each, run side by side. Each lane
# runs its own smoke copy first (ledger kind other) and trains only if the smoke passes (kind main). Every task goes
# through tools/packed_task.sh, which adds a wall limit and writes logs/packed/$SLURM_JOB_ID.tsv for the record.
#   LANES="smoke_run:main_run [smoke_run:main_run]" from the env (set by scripts/submit_phase1.sh)
#SBATCH --account=pli
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/frozen-critic/logs/slurm/%x-%j.out
set -euo pipefail
ROOT=/scratch/gpfs/GROUP/USER/project/frozen-critic
PT=/scratch/gpfs/GROUP/USER/tools/packed_task.sh
export PACKED_LEDGER_DIR=$ROOT/logs/packed
: "${LANES:?}" "${MAIN_TIMEOUT:?}"
IFS=, read -ra ALL <<< "${CUDA_VISIBLE_DEVICES:?}"
read -ra LANE_SPECS <<< "$LANES"
echo "[job] $(date -u +%FT%TZ) node=$(hostname) job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES lanes=$LANES"
lane() {
  local i=$1 smoke=${2%%:*} main=${2##*:}
  local devs; devs=$(IFS=,; echo "${ALL[*]:$((i * 4)):4}")
  # a smoke copy must start from nothing: a checkpoint left by an earlier job would make it "resume" and pass vacuously
  [[ $smoke == smokecopy_* ]] && rm -rf -- "$ROOT/runs/$smoke/ckpt" "$ROOT/runs/$smoke/rollouts"
  if "$PT" "smoke-$main" - other 4 1h -- "$ROOT/scripts/run_task.sh" "$smoke" "$devs" "$i"; then
    "$PT" "$main" "$main" main 4 "$MAIN_TIMEOUT" -- "$ROOT/scripts/run_task.sh" "$main" "$devs" "$i"
  else
    echo "[job] lane $i: smoke $smoke failed, $main not started"; return 1
  fi
}
pids=()
# lane 1 starts 5 min later: each miles instance scans up from port 20000 for free ports, and two scanning at once can
# pick the same ones before SGLang binds them
for i in "${!LANE_SPECS[@]}"; do (sleep $((i * 300)); lane "$i" "${LANE_SPECS[$i]}") & pids+=($!); done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[job] $(date -u +%FT%TZ) done rc=$rc"
exit $rc
