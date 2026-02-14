#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <carrier_freq> <uri> <duration_sec> [snr_db_list...]"
  echo "Example: $0 515.142857M ip:192.168.10.90 300 12 10 8 6"
  exit 1
fi

carrier="$1"
uri="$2"
duration="$3"
shift 3

if [[ $# -eq 0 ]]; then
  snr_list=(12 10 8 6)
else
  snr_list=("$@")
fi

mkdir -p logs
ts="$(date +%Y%m%d_%H%M%S)"

for snr in "${snr_list[@]}"; do
  log="logs/eval_framesync_lowsnr_awgn${snr//./p}_$(echo "$carrier" | tr -d '.').log"
  log="${log%.log}_${ts}.log"
  echo "[run snr=${snr}dB] ${log}"
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
    > "${log}" 2>&1 || true
done

echo "[done] generated logs:"
ls -1t logs/eval_framesync_lowsnr_awgn*_"${ts}".log
