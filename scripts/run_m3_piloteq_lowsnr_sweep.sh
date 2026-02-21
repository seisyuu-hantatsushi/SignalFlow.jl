#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <carrier_freq> <uri> <duration_sec> [snr_db_list...]"
  echo "Example: $0 515.142857M ip:192.168.10.90 300 6 0 -2"
  echo "Env: PILOTEQ_ALPHAS=\"0.1 0.2 0.3\" FRAMESYNC_UNLOCK_THRESHOLD=0.25 FRAMESYNC_UNLOCK_CONFIRM=20"
  echo "Env: M3_ENABLE_EVM=1 M3_EVM_MOD=qpsk M3_EVM_LOG_INTERVAL=10"
  exit 1
fi

carrier="$1"
uri="$2"
duration="$3"
shift 3

if [[ $# -eq 0 ]]; then
  snr_list=(6 0 -2)
else
  snr_list=("$@")
fi

alpha_list_str="${PILOTEQ_ALPHAS:-0.1 0.2 0.3}"
# shellcheck disable=SC2206
alpha_list=(${alpha_list_str})

unlock_th="${FRAMESYNC_UNLOCK_THRESHOLD:-0.25}"
unlock_confirm="${FRAMESYNC_UNLOCK_CONFIRM:-20}"
enable_evm="${M3_ENABLE_EVM:-0}"
evm_mod="${M3_EVM_MOD:-qpsk}"
evm_log_interval="${M3_EVM_LOG_INTERVAL:-10}"

mkdir -p logs
ts="$(date +%Y%m%d_%H%M%S)"
run_logs=()

for snr in "${snr_list[@]}"; do
  snr_tag="${snr//./p}"
  for alpha in "${alpha_list[@]}"; do
    alpha_tag="${alpha//./p}"
    log="logs/eval_m3_piloteq_awgn${snr_tag}_alpha${alpha_tag}_th${unlock_th//./p}_uc${unlock_confirm}_$(echo "$carrier" | tr -d '.').log"
    log="${log%.log}_${ts}.log"
    run_logs+=("$log")

    echo "[run snr=${snr}dB alpha=${alpha} th=${unlock_th} uc=${unlock_confirm}] ${log}"
    cmd=(
      julia --project=. ./examples/isdbt_demod.jl
      -c "${carrier}"
      -i "${uri}"
      --diag
      --no-const
      --seq-trace
      --seq-trace-log-interval 200
      --pilot-temporal-alpha "${alpha}"
      --awgn-snr-db "${snr}"
      --awgn-log-interval 10
      --framesync-unlock-threshold "${unlock_th}"
      --framesync-unlock-confirm "${unlock_confirm}"
    )
    if [[ "${enable_evm}" == "1" ]]; then
      cmd+=(--evm --evm-mod "${evm_mod}" --evm-log-interval "${evm_log_interval}")
    fi
    timeout -s INT "${duration}" "${cmd[@]}" \
      > "${log}" 2>&1 || true
  done
done

echo "[done] generated logs:"
printf '%s\n' "${run_logs[@]}"

echo "[quick-summary]"
for f in "${run_logs[@]}"; do
  lock=$(rg -c "ISDBTFrameSync: lock corr=" "$f" || true)
  unlock=$(rg -c "ISDBTFrameSync: unlock corr=" "$f" || true)
  forced=$(rg -c "forced_resync" "$f" || true)
  outlier=$(rg -c "outlier_resync" "$f" || true)
  shutdown=$(rg -c "Shutdown complete\\." "$f" || true)
  fft_line=$(rg "FFTBlock input stats:" "$f" | tail -n 1 || true)
  sink_fail="-"
  if [[ -n "$fft_line" ]]; then
    sink_fail=$(sed -n 's/.*sink_fail=\([0-9]\+\).*/\1/p' <<<"$fft_line")
    sink_fail=${sink_fail:--}
  fi
  echo "[summary] $f : lock=${lock:-0} unlock=${unlock:-0} forced=${forced:-0} outlier=${outlier:-0} sink_fail=${sink_fail} shutdown=${shutdown:-0}"
done
