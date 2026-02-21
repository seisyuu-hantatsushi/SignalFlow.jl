#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <carrier_freq> <uri> <duration_sec> [snr_db] [repeats]"
  echo "Example: $0 515.142857M ip:192.168.10.90 600 -4 3"
  echo "Env: IMPAIR_CFO_HZ=120 IMPAIR_PHASE_JUMP_DEG=12 IMPAIR_PHASE_JUMP_INTERVAL_FRAMES=8 IMPAIR_LOG_INTERVAL=10"
  echo "Env: SYM_IMPAIR_CFO_HZ=0 SYM_IMPAIR_PHASE_JUMP_DEG=0 SYM_IMPAIR_PHASE_JUMP_INTERVAL_FRAMES=0 SYM_IMPAIR_SLOPE_RAD_PER_BIN=0 SYM_IMPAIR_LOG_INTERVAL=10"
  echo "Env: STRICT_SLOPE_MIN_SLOPE_STEP=7.5e-5 STRICT_SLOPE_MIN_INTERCEPT_STEP_DEG=0.4 STRICT_CPE_MIN_PHASE_STEP_DEG=0.4"
  echo "Env: RELAXED_SLOPE_MIN_SLOPE_STEP=1e-5 RELAXED_SLOPE_MIN_INTERCEPT_STEP_DEG=0.1 RELAXED_CPE_MIN_PHASE_STEP_DEG=0.1"
  echo "Env: STRICT_SLOPE_FORCE_UPDATE_EPS=0 STRICT_CPE_FORCE_UPDATE_EPS_DEG=0"
  echo "Env: RELAXED_SLOPE_FORCE_UPDATE_EPS=0 RELAXED_CPE_FORCE_UPDATE_EPS_DEG=0"
  exit 1
fi

carrier="$1"
uri="$2"
duration="$3"
snr_db="${4:--4}"
repeats="${5:-1}"

alpha="${PILOTEQ_ALPHA:-0.1}"
unlock_th="${FRAMESYNC_UNLOCK_THRESHOLD:-0.25}"
unlock_confirm="${FRAMESYNC_UNLOCK_CONFIRM:-20}"

# strict: current baseline-like gate settings
strict_slope_ratio="${STRICT_SLOPE_MIN_USED_RATIO:-0.65}"
strict_cpe_on="${STRICT_CPE_MIN_UPDATE_CONF:-0.30}"
strict_cpe_off="${STRICT_CPE_MIN_UPDATE_CONF_OFF:-0.20}"
strict_slope_min_slope_step="${STRICT_SLOPE_MIN_SLOPE_STEP:-7.5e-5}"
strict_slope_min_intercept_step_deg="${STRICT_SLOPE_MIN_INTERCEPT_STEP_DEG:-0.4}"
strict_cpe_min_phase_step_deg="${STRICT_CPE_MIN_PHASE_STEP_DEG:-0.4}"
strict_slope_force_update_eps="${STRICT_SLOPE_FORCE_UPDATE_EPS:-0}"
strict_cpe_force_update_eps_deg="${STRICT_CPE_FORCE_UPDATE_EPS_DEG:-0}"

# relaxed: easier update gating to exercise adaptation path
relaxed_slope_ratio="${RELAXED_SLOPE_MIN_USED_RATIO:-0.45}"
relaxed_cpe_on="${RELAXED_CPE_MIN_UPDATE_CONF:-0.15}"
relaxed_cpe_off="${RELAXED_CPE_MIN_UPDATE_CONF_OFF:-0.10}"
relaxed_slope_min_slope_step="${RELAXED_SLOPE_MIN_SLOPE_STEP:-1e-5}"
relaxed_slope_min_intercept_step_deg="${RELAXED_SLOPE_MIN_INTERCEPT_STEP_DEG:-0.1}"
relaxed_cpe_min_phase_step_deg="${RELAXED_CPE_MIN_PHASE_STEP_DEG:-0.1}"
relaxed_slope_force_update_eps="${RELAXED_SLOPE_FORCE_UPDATE_EPS:-0}"
relaxed_cpe_force_update_eps_deg="${RELAXED_CPE_FORCE_UPDATE_EPS_DEG:-0}"

# impairment knobs (AWGN-external disturbance)
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
  local profile="$1"
  local slope_ratio="$2"
  local cpe_on="$3"
  local cpe_off="$4"
  local slope_min_slope_step="$5"
  local slope_min_intercept_step_deg="$6"
  local cpe_min_phase_step_deg="$7"
  local slope_force_update_eps="$8"
  local cpe_force_update_eps_deg="$9"
  local rep="${10}"
  local snr_tag="${snr_db//./p}"
  local alpha_tag="${alpha//./p}"
  local log="logs/eval_m3_adapt_${profile}_awgn${snr_tag}_alpha${alpha_tag}_r${rep}_th${unlock_th//./p}_uc${unlock_confirm}_$(echo "$carrier" | tr -d '.').log"
  log="${log%.log}_${ts}.log"
  run_logs+=("$log")

  echo "[run ${profile} r=${rep}/${repeats}] $log"
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
    --slope-force-update-eps "$slope_force_update_eps"
    --cpe-force-update-eps-deg "$cpe_force_update_eps_deg"
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

for r in $(seq 1 "$repeats"); do
  run_one strict "$strict_slope_ratio" "$strict_cpe_on" "$strict_cpe_off" \
    "$strict_slope_min_slope_step" "$strict_slope_min_intercept_step_deg" "$strict_cpe_min_phase_step_deg" \
    "$strict_slope_force_update_eps" "$strict_cpe_force_update_eps_deg" "$r"
  run_one relaxed "$relaxed_slope_ratio" "$relaxed_cpe_on" "$relaxed_cpe_off" \
    "$relaxed_slope_min_slope_step" "$relaxed_slope_min_intercept_step_deg" "$relaxed_cpe_min_phase_step_deg" \
    "$relaxed_slope_force_update_eps" "$relaxed_cpe_force_update_eps_deg" "$r"
done

echo "[done] generated logs:"
printf '%s\n' "${run_logs[@]}"

echo "[analyze]"
python3 scripts/analyze_m3_piloteq_logs.py "${run_logs[@]}"
