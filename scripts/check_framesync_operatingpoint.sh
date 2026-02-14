#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <log1> [log2 ...]"
  exit 1
fi

pass_all=1
for f in "$@"; do
  if [[ ! -f "$f" ]]; then
    echo "[FAIL] missing log: $f"
    pass_all=0
    continue
  fi

  # Distinguish environment/startup failures from FrameSync quality failures.
  startup_err=$(rg -n "iio_context\\* null pointer|AssertionError: iio_context\\* null pointer" "$f" || true)
  started=$(rg -c "start rxTask\\(ComplexF32\\)" "$f" || true)
  started=${started:-0}
  if [[ -n "$startup_err" || "$started" -eq 0 ]]; then
    reason="startup_failed"
    if [[ -n "$startup_err" ]]; then
      reason="iio_context_null_pointer"
    fi
    echo "[INVALID] $f : reason=$reason"
    pass_all=0
    continue
  fi

  lock=$(rg -c "ISDBTFrameSync: lock corr=" "$f" || true)
  unlock=$(rg -c "ISDBTFrameSync: unlock corr=" "$f" || true)
  forced=$(rg -c "forced_resync" "$f" || true)
  outlier=$(rg -c "outlier_resync" "$f" || true)
  shutdown=$(rg -c "Shutdown complete\." "$f" || true)

  lock=${lock:-0}
  unlock=${unlock:-0}
  forced=${forced:-0}
  outlier=${outlier:-0}
  shutdown=${shutdown:-0}

  fft_line=$(rg "FFTBlock input stats:" "$f" | tail -n 1 || true)
  sink_fail=999999
  if [[ -n "$fft_line" ]]; then
    sink_fail=$(sed -n 's/.*sink_fail=\([0-9]\+\).*/\1/p' <<<"$fft_line")
    sink_fail=${sink_fail:-999999}
  fi

  ok=1
  [[ "$lock" -ge 1 ]] || ok=0
  [[ "$unlock" -eq 0 ]] || ok=0
  [[ "$forced" -eq 0 ]] || ok=0
  [[ "$outlier" -eq 0 ]] || ok=0
  [[ "$shutdown" -ge 1 ]] || ok=0
  [[ "$sink_fail" -eq 0 ]] || ok=0

  if [[ "$ok" -eq 1 ]]; then
    echo "[PASS] $f : lock=$lock unlock=$unlock forced=$forced outlier=$outlier sink_fail=$sink_fail shutdown=$shutdown"
  else
    echo "[FAIL] $f : lock=$lock unlock=$unlock forced=$forced outlier=$outlier sink_fail=$sink_fail shutdown=$shutdown"
    pass_all=0
  fi
done

if [[ "$pass_all" -eq 1 ]]; then
  echo "OPERATING-POINT: PASS"
  exit 0
fi

echo "OPERATING-POINT: FAIL"
exit 1
