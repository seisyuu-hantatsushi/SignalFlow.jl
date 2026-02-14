#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <log1> <log2>"
  exit 2
fi

check_one() {
  local f="$1"
  local ok=1

  if [[ ! -f "$f" ]]; then
    echo "[FAIL] $f : file not found"
    return 1
  fi

  local shutdown_count mismatch_count seqprobe_count sink_fail
  shutdown_count="$(rg -n "Shutdown complete\\." "$f" | wc -l)"
  mismatch_count="$(rg -n "SeqTrace\\[ISDBTFrameSync\\] in_mismatch" "$f" | wc -l)"
  seqprobe_count="$(rg -n "ISDBTFrameSync: seq_probe where=dequeue" "$f" | wc -l)"
  sink_fail="$(sed -n 's/.*FFTBlock input stats:.*sink_fail=\([0-9]\+\).*/\1/p' "$f" | tail -n 1)"
  sink_fail="${sink_fail:-NA}"

  if [[ "$shutdown_count" -lt 1 ]]; then
    echo "[FAIL] $f : no graceful shutdown marker"
    ok=0
  fi
  if [[ "$mismatch_count" != "0" ]]; then
    echo "[FAIL] $f : ISDBTFrameSync in_mismatch=$mismatch_count"
    ok=0
  fi
  if [[ "$seqprobe_count" != "0" ]]; then
    echo "[FAIL] $f : ISDBTFrameSync dequeue seq_probe=$seqprobe_count"
    ok=0
  fi
  if [[ "$sink_fail" != "0" ]]; then
    echo "[FAIL] $f : FFTBlock sink_fail=$sink_fail"
    ok=0
  fi

  if [[ "$ok" -eq 1 ]]; then
    echo "[PASS] $f : shutdown=1 mismatch=0 seqprobe=0 sink_fail=0"
  fi
  return $((1 - ok))
}

all_ok=1
for f in "$@"; do
  if ! check_one "$f"; then
    all_ok=0
  fi
done

if [[ "$all_ok" -eq 1 ]]; then
  echo "MILESTONE-1: PASS (all logs satisfy criteria)"
  exit 0
fi

echo "MILESTONE-1: FAIL"
exit 1
