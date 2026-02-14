#!/usr/bin/env bash
set -euo pipefail

run_with_timeout() {
  local dur="$1"
  local log="$2"
  shift 2

  set +e
  timeout -s INT "$dur" "$@" > "$log" 2>&1
  local rc=$?
  set -e

  # Accept timeout-driven stop as normal for fixed-duration evaluation.
  # 124: timeout reached. 130: interrupted by SIGINT handling path.
  if [[ "$rc" -ne 0 && "$rc" -ne 124 && "$rc" -ne 130 ]]; then
    echo "[ERROR] run failed rc=$rc log=$log"
    return "$rc"
  fi
  return 0
}

carrier="${1:-515.142857M}"
uri="${2:-ip:192.168.10.90}"
dur_sec="${3:-300}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

mkdir -p logs

ts1="$(date +%Y%m%d_%H%M%S)"
log1="logs/m1_run1_${carrier//./}_${ts1}.log"
echo "[run1] $log1"
run_with_timeout "$dur_sec" "$log1" \
  julia --project=. ./examples/isdbt_demod.jl \
    -c "$carrier" \
    -i "$uri" \
    --diag \
    --no-const \
    --seq-trace \
    --seq-trace-log-interval 200

sleep 2

ts2="$(date +%Y%m%d_%H%M%S)"
log2="logs/m1_run2_${carrier//./}_${ts2}.log"
echo "[run2] $log2"
run_with_timeout "$dur_sec" "$log2" \
  julia --project=. ./examples/isdbt_demod.jl \
    -c "$carrier" \
    -i "$uri" \
    --diag \
    --no-const \
    --seq-trace \
    --seq-trace-log-interval 200

echo "[check] $log1 $log2"
bash scripts/check_milestone1.sh "$log1" "$log2"
