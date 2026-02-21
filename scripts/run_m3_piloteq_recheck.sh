#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <carrier_freq> <uri> <duration_sec> [snr_db] [repeats]"
  echo "Example: $0 515.142857M ip:192.168.10.90 600 -2 3"
  echo "Env: PILOTEQ_ALPHAS=\"0.1 0.3\" M3_ENABLE_EVM=1 M3_EVM_MOD=qpsk M3_EVM_LOG_INTERVAL=10"
  exit 1
fi

carrier="$1"
uri="$2"
duration="$3"
snr_db="${4:--2}"
repeats="${5:-3}"

alpha_list_str="${PILOTEQ_ALPHAS:-0.1 0.3}"
# shellcheck disable=SC2206
alpha_list=(${alpha_list_str})

unlock_th="${FRAMESYNC_UNLOCK_THRESHOLD:-0.25}"
unlock_confirm="${FRAMESYNC_UNLOCK_CONFIRM:-20}"
enable_evm="${M3_ENABLE_EVM:-1}"
evm_mod="${M3_EVM_MOD:-qpsk}"
evm_log_interval="${M3_EVM_LOG_INTERVAL:-10}"

mkdir -p logs
ts="$(date +%Y%m%d_%H%M%S)"
run_logs=()

for alpha in "${alpha_list[@]}"; do
  alpha_tag="${alpha//./p}"
  for r in $(seq 1 "$repeats"); do
    snr_tag="${snr_db//./p}"
    log="logs/eval_m3_piloteq_recheck_awgn${snr_tag}_alpha${alpha_tag}_r${r}_th${unlock_th//./p}_uc${unlock_confirm}_$(echo "$carrier" | tr -d '.').log"
    log="${log%.log}_${ts}.log"
    run_logs+=("$log")

    echo "[run alpha=${alpha} rep=${r}/${repeats} snr=${snr_db}dB] $log"
    cmd=(
      julia --project=. ./examples/isdbt_demod.jl
      -c "$carrier"
      -i "$uri"
      --diag
      --no-const
      --seq-trace
      --seq-trace-log-interval 200
      --pilot-temporal-alpha "$alpha"
      --awgn-snr-db "$snr_db"
      --awgn-log-interval 10
      --framesync-unlock-threshold "$unlock_th"
      --framesync-unlock-confirm "$unlock_confirm"
    )
    if [[ "$enable_evm" == "1" ]]; then
      cmd+=(--evm --evm-mod "$evm_mod" --evm-log-interval "$evm_log_interval")
    fi

    timeout -s INT "$duration" "${cmd[@]}" > "$log" 2>&1 || true
  done
done

echo "[done] generated logs:"
printf '%s\n' "${run_logs[@]}"

echo "[analyze]"
python3 scripts/analyze_m3_piloteq_logs.py "${run_logs[@]}"
