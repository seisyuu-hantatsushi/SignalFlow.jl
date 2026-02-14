#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <carrier_freq> <uri> <duration_sec> [awgn_snr_db]"
  echo "Example: $0 515.142857M ip:192.168.10.90 300 -6"
  exit 1
fi

carrier="$1"
uri="$2"
duration="$3"
awgn_snr="${4:--6}"

thresholds=(0.25 0.30 0.35)
mkdir -p logs
ts="$(date +%Y%m%d_%H%M%S)"

for th in "${thresholds[@]}"; do
  th_tag="${th/./p}"
  log="logs/eval_framesync_unlockth_${th_tag}_awgn${awgn_snr//./p}_$(echo "$carrier" | tr -d '.').log"
  log="${log%.log}_${ts}.log"
  echo "[run unlock_th=${th} awgn=${awgn_snr}dB] ${log}"
  timeout -s INT "${duration}" julia --project=. ./examples/isdbt_demod.jl \
    -c "${carrier}" \
    -i "${uri}" \
    --diag \
    --no-const \
    --seq-trace \
    --seq-trace-log-interval 200 \
    --pilot-temporal-alpha 0.2 \
    --awgn-snr-db "${awgn_snr}" \
    --awgn-log-interval 10 \
    --framesync-unlock-threshold "${th}" \
    > "${log}" 2>&1 || true
done

echo "[done] generated logs:"
ls -1t logs/eval_framesync_unlockth_*_"${ts}".log
