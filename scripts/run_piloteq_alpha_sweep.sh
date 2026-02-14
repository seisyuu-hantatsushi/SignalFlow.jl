#!/usr/bin/env bash
set -euo pipefail

carrier="${1:-515.142857M}"
uri="${2:-ip:192.168.10.90}"
dur_sec="${3:-300}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
mkdir -p logs

run_one() {
  local alpha="$1"
  local tag="${alpha//./}"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local log="logs/eval_piloteq_alpha${tag}_${carrier//./}_${ts}.log"
  echo "[run alpha=${alpha}] $log" >&2

  set +e
  timeout -s INT "$dur_sec" \
    julia --project=. ./examples/isdbt_demod.jl \
      -c "$carrier" \
      -i "$uri" \
      --diag \
      --no-const \
      --seq-trace \
      --seq-trace-log-interval 200 \
      --pilot-temporal-alpha "$alpha" \
      > "$log" 2>&1
  local rc=$?
  set -e

  if [[ "$rc" -ne 0 && "$rc" -ne 124 && "$rc" -ne 130 ]]; then
    echo "[ERROR] alpha=${alpha} rc=$rc log=$log" >&2
    return "$rc"
  fi
  RUN_LOG="$log"
}

summarize_one() {
  local log="$1"
  local shutdown
  shutdown="$(rg -n "Shutdown complete\\." "$log" 2>/dev/null | wc -l)"
  local seqbad
  seqbad="$(rg -n "SeqTrace\\[ISDBTFrameSync\\] in_mismatch|ISDBTFrameSync: seq_probe where=dequeue" "$log" 2>/dev/null | wc -l)"
  local hstats
  hstats="$(rg -o "mean\\|H\\|=[0-9.]+" "$log" 2>/dev/null | sed 's/.*=//' | awk 'BEGIN{n=0}{v=$1+0;sum+=v;if(n==0||v<min)min=v;if(n==0||v>max)max=v;n++}END{if(n>0)printf(\"n=%d mean=%.4f min=%.4f max=%.4f\",n,sum/n,min,max);else printf(\"n=0\")}')"
  local rstats
  rstats="$(rg -o "residual_rms_deg=[0-9.]+" "$log" 2>/dev/null | sed 's/.*=//' | awk 'BEGIN{n=0}{v=$1+0;sum+=v;if(n==0||v<min)min=v;if(n==0||v>max)max=v;n++}END{if(n>0)printf(\"n=%d mean=%.2f min=%.2f max=%.2f\",n,sum/n,min,max);else printf(\"n=0\")}')"
  echo "[summary] $log"
  echo "  shutdown=$shutdown seq_bad=$seqbad mean|H|{$hstats} residual_rms_deg{$rstats}"
}

logs=()
for a in 0.1 0.2 0.3; do
  run_one "$a"
  logs+=("$RUN_LOG")
  sleep 2
done

for log in "${logs[@]}"; do
  summarize_one "$log"
done
