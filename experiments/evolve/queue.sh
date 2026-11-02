#!/usr/bin/env bash
# GPU job queue for evolution rounds: one worker per GPU pulls jobs from a shared file.
#
#   experiments/evolve/queue.sh JOBS OUT_DIR [GPU ...]        (default GPUs: 0 1)
#
# JOBS has one job per line, "name|command" (blank lines and lines starting with # are skipped).
# Each job runs with CUDA_VISIBLE_DEVICES set to its worker's GPU; its console output is teed to
# OUT_DIR/<name>.log. Progress goes to OUT_DIR/queue.status ("start"/"end ... rc=" lines, then
# "all done"). Jobs can be appended to the live queue file OUT_DIR/queue.pending while it runs;
# the workers pick them up when they free. Run it in the background and wait on queue.status.
set -u
JOBS=${1:?jobs file}; OUT=${2:?output dir}; shift 2
GPUS=("$@"); [ ${#GPUS[@]} -eq 0 ] && GPUS=(0 1)
cd "$(dirname "$0")/../.." && source .venv/bin/activate
mkdir -p "$OUT"
Q="$OUT/queue.pending"; STATUS="$OUT/queue.status"
rg -v '^\s*(#|$)' "$JOBS" > "$Q"
: > "$STATUS"
run() {  # gpu name command
  echo "$(date +%T) start gpu$1 $2" >> "$STATUS"
  CUDA_VISIBLE_DEVICES=$1 bash -c "$3" 2>&1 | tee "$OUT/$2.log" > /dev/null
  echo "$(date +%T) end gpu$1 $2 rc=${PIPESTATUS[0]}" >> "$STATUS"
}
pop() { flock "$Q.lock" bash -c "head -n1 '$Q'; sed -i 1d '$Q'"; }
worker() { while line=$(pop) && [ -n "$line" ]; do run "$1" "${line%%|*}" "${line#*|}"; done; }
for g in "${GPUS[@]}"; do worker "$g" & sleep 2; done
wait
echo "$(date +%T) all done" >> "$STATUS"
