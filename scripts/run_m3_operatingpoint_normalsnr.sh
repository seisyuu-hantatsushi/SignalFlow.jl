#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <carrier_freq> <uri> <duration_sec> [snr_list...]"
  echo "Example: $0 515.142857M ip:192.168.10.90 600 12 6"
  echo "Env: PROFILE=relaxed|strict (default: relaxed)"
  echo "Env: FORCE_SLOPE_EPS=0 FORCE_CPE_EPS_DEG=0"
  echo "Env: REPEATS=2"
  echo "Env: IMPAIR_* / SYM_IMPAIR_* are forwarded if non-zero"
  exit 1
fi

carrier="$1"
uri="$2"
duration="$3"
shift 3

if [[ $# -gt 0 ]]; then
  snr_list=("$@")
else
  snr_list=(12 6)
fi

profile="${PROFILE:-relaxed}"
alpha="${PILOTEQ_ALPHA:-0.1}"
unlock_th="${FRAMESYNC_UNLOCK_THRESHOLD:-0.25}"
unlock_confirm="${FRAMESYNC_UNLOCK_CONFIRM:-20}"
repeats="${REPEATS:-2}"
force_slope_eps="${FORCE_SLOPE_EPS:-0}"
force_cpe_eps_deg="${FORCE_CPE_EPS_DEG:-0}"

if [[ "$profile" == "strict" ]]; then
  slope_ratio="${STRICT_SLOPE_MIN_USED_RATIO:-0.65}"
  cpe_on="${STRICT_CPE_MIN_UPDATE_CONF:-0.30}"
  cpe_off="${STRICT_CPE_MIN_UPDATE_CONF_OFF:-0.20}"
  slope_min_slope_step="${STRICT_SLOPE_MIN_SLOPE_STEP:-7.5e-5}"
  slope_min_intercept_step_deg="${STRICT_SLOPE_MIN_INTERCEPT_STEP_DEG:-0.4}"
  cpe_min_phase_step_deg="${STRICT_CPE_MIN_PHASE_STEP_DEG:-0.4}"
else
  slope_ratio="${RELAXED_SLOPE_MIN_USED_RATIO:-0.45}"
  cpe_on="${RELAXED_CPE_MIN_UPDATE_CONF:-0.15}"
  cpe_off="${RELAXED_CPE_MIN_UPDATE_CONF_OFF:-0.10}"
  slope_min_slope_step="${RELAXED_SLOPE_MIN_SLOPE_STEP:-1e-5}"
  slope_min_intercept_step_deg="${RELAXED_SLOPE_MIN_INTERCEPT_STEP_DEG:-0.1}"
  cpe_min_phase_step_deg="${RELAXED_CPE_MIN_PHASE_STEP_DEG:-0.1}"
fi

impair_cfo_hz="${IMPAIR_CFO_HZ:-0}"
impair_phase_jump_deg="${IMPAIR_PHASE_JUMP_DEG:-0}"
impair_phase_jump_interval_frames="${IMPAIR_PHASE_JUMP_INTERVAL_FRAMES:-0}"
impair_log_interval="${IMPAIR_LOG_INTERVAL:-10}"
sym_impair_cfo_hz="${SYM_IMPAIR_CFO_HZ:-0}"
sym_impair_phase_jump_deg="${SYM_IMPAIR_PHASE_JUMP_DEG:-0}"
sym_impair_phase_jump_interval_frames="${SYM_IMPAIR_PHASE_JUMP_INTERVAL_FRAMES:-0}"
sym_impair_slope_rad_per_bin="${SYM_IMPAIR_SLOPE_RAD_PER_BIN:-0}"
sym_impair_log_interval="${SYM_IMPAIR_LOG_INTERVAL:-10}"

mkdir -p logs
ts="$(date +%Y%m%d_%H%M%S)"
run_logs=()

run_one() {
  local snr_db="$1"
  local rep="$2"
  local snr_tag="${snr_db//./p}"
  local alpha_tag="${alpha//./p}"
  local log="logs/eval_m3_op_${profile}_awgn${snr_tag}_alpha${alpha_tag}_r${rep}_sf${force_slope_eps//./p}_cf${force_cpe_eps_deg//./p}_th${unlock_th//./p}_uc${unlock_confirm}_$(echo "$carrier" | tr -d '.').log"
  log="${log%.log}_${ts}.log"
  run_logs+=("$log")

  echo "[run snr=${snr_db}dB r=${rep}/${repeats}] $log"
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
    --slope-min-used-ratio "$slope_ratio"
    --slope-min-slope-step "$slope_min_slope_step"
    --slope-min-intercept-step-deg "$slope_min_intercept_step_deg"
    --cpe-min-update-conf "$cpe_on"
    --cpe-min-update-conf-off "$cpe_off"
    --cpe-min-phase-step-deg "$cpe_min_phase_step_deg"
    --slope-force-update-eps "$force_slope_eps"
    --cpe-force-update-eps-deg "$force_cpe_eps_deg"
    --evm --evm-mod qpsk --evm-log-interval 10
  )

  if [[ "$impair_cfo_hz" != "0" || "$impair_phase_jump_deg" != "0" ]]; then
    cmd+=(
      --impair-cfo-hz "$impair_cfo_hz"
      --impair-phase-jump-deg "$impair_phase_jump_deg"
      --impair-phase-jump-interval-frames "$impair_phase_jump_interval_frames"
      --impair-log-interval "$impair_log_interval"
    )
  fi
  if [[ "$sym_impair_cfo_hz" != "0" || "$sym_impair_phase_jump_deg" != "0" || "$sym_impair_slope_rad_per_bin" != "0" ]]; then
    cmd+=(
      --sym-impair-cfo-hz "$sym_impair_cfo_hz"
      --sym-impair-phase-jump-deg "$sym_impair_phase_jump_deg"
      --sym-impair-phase-jump-interval-frames "$sym_impair_phase_jump_interval_frames"
      --sym-impair-slope-rad-per-bin "$sym_impair_slope_rad_per_bin"
      --sym-impair-log-interval "$sym_impair_log_interval"
    )
  fi

  timeout -s INT "$duration" "${cmd[@]}" > "$log" 2>&1 || true
}

for snr_db in "${snr_list[@]}"; do
  for r in $(seq 1 "$repeats"); do
    run_one "$snr_db" "$r"
  done
done

echo "[done] generated logs:"
printf '%s\n' "${run_logs[@]}"

echo "[analyze]"
python3 scripts/analyze_m3_piloteq_logs.py "${run_logs[@]}"
