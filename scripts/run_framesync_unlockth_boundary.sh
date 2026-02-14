#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <carrier_freq> <uri> <duration_sec> [awgn_snr_db] [unlock_confirm]"
  echo "Example: $0 515.142857M ip:192.168.10.90 300 -6 20"
  exit 1
fi

carrier="$1"
uri="$2"
duration="$3"
awgn_snr="${4:--6}"
unlock_confirm="${5:-20}"

thresholds=(0.36 0.38 0.40)
mkdir -p logs
ts="$(date +%Y%m%d_%H%M%S)"

for th in "${thresholds[@]}"; do
  th_tag="${th/./p}"
  uc_tag="${unlock_confirm}"
  log="logs/eval_framesync_unlockth_${th_tag}_uc${uc_tag}_awgn${awgn_snr//./p}_$(echo "$carrier" | tr -d '.').log"
  log="${log%.log}_${ts}.log"
  echo "[run unlock_th=${th} unlock_confirm=${unlock_confirm} awgn=${awgn_snr}dB] ${log}"
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
    --framesync-unlock-confirm "${unlock_confirm}" \
    > "${log}" 2>&1 || true
done

echo "[done] generated logs:"
ls -1t logs/eval_framesync_unlockth_*_uc${unlock_confirm}_awgn${awgn_snr//./p}_*_"${ts}".log
