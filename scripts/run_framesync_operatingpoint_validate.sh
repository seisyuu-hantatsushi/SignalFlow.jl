#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <carrier_freq> <uri> <duration_sec> [snr_db_list...]"
  echo "Example: $0 515.142857M ip:192.168.10.90 300 12 6 0"
  exit 1
fi

carrier="$1"
uri="$2"
duration="$3"
shift 3

if [[ $# -eq 0 ]]; then
  snr_list=(12 6 0)
else
  snr_list=("$@")
fi

unlock_th="${FRAMESYNC_UNLOCK_THRESHOLD:-0.38}"
unlock_confirm="${FRAMESYNC_UNLOCK_CONFIRM:-20}"

mkdir -p logs
ts="$(date +%Y%m%d_%H%M%S)"
run_logs=()

for snr in "${snr_list[@]}"; do
  th_tag="${unlock_th/./p}"
  snr_tag="${snr//./p}"
  log="logs/eval_framesync_op_awgn${snr_tag}_th${th_tag}_uc${unlock_confirm}_$(echo "$carrier" | tr -d '.').log"
  log="${log%.log}_${ts}.log"
  run_logs+=("$log")

  echo "[run snr=${snr}dB th=${unlock_th} uc=${unlock_confirm}] ${log}"
  timeout -s INT "${duration}" julia --project=. ./examples/isdbt_demod.jl \
    -c "${carrier}" \
    -i "${uri}" \
    --diag \
    --no-const \
    --seq-trace \
    --seq-trace-log-interval 200 \
    --pilot-temporal-alpha 0.2 \
    --awgn-snr-db "${snr}" \
    --awgn-log-interval 10 \
    --framesync-unlock-threshold "${unlock_th}" \
    --framesync-unlock-confirm "${unlock_confirm}" \
    > "${log}" 2>&1 || true
done

echo "[check] ${run_logs[*]}"
bash scripts/check_framesync_operatingpoint.sh "${run_logs[@]}"
