#!/usr/bin/env bash
set -euo pipefail

# Run the ISDB-T demod command in fixed-duration chunks and save logs per run.
#
# Example:
#   scripts/run_demod_5min.sh \
#     -c 515.142857M \
#     -i ip:192.168.10.90 \
#     -n 3 \
#     -d 5m \
#     -o demod_batch

RUNS=3
DURATION="5m"
OUT_PREFIX="demod_batch"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: run_demod_5min.sh -c <carrier> -i <uri> [options] [-- <extra args>]

Required:
  -c <carrier>      Carrier frequency (e.g. 515.142857M)
  -i <uri>          SDR URI (e.g. ip:192.168.10.90)

Options:
  -n <runs>         Number of runs (default: 3)
  -d <duration>     Duration per run for timeout (default: 5m)
  -o <prefix>       Output log prefix (default: demod_batch)
  -h                Show this help

Any arguments after '--' are passed through to examples/isdbt_demod.jl.
EOF
}

CARRIER=""
URI=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c)
      CARRIER="${2:-}"
      shift 2
      ;;
    -i)
      URI="${2:-}"
      shift 2
      ;;
    -n)
      RUNS="${2:-}"
      shift 2
      ;;
    -d)
      DURATION="${2:-}"
      shift 2
      ;;
    -o)
      OUT_PREFIX="${2:-}"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${CARRIER}" || -z "${URI}" ]]; then
  echo "Error: -c and -i are required." >&2
  usage
  exit 1
fi

if ! command -v timeout >/dev/null 2>&1; then
  echo "Error: 'timeout' command is required." >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"

for ((i=1; i<=RUNS; i++)); do
  LOG="${OUT_PREFIX}_${STAMP}_run$(printf "%02d" "${i}").log"
  echo "[run ${i}/${RUNS}] writing ${LOG}"
  set +e
  timeout --signal=INT "${DURATION}" \
    julia -t auto --project=./ examples/isdbt_demod.jl \
      -c "${CARRIER}" -i "${URI}" "${EXTRA_ARGS[@]}" \
      > "${LOG}" 2>&1
  RC=$?
  set -e

  # timeout returns 124 when time limit is reached; treat as success for chunked capture.
  if [[ ${RC} -ne 0 && ${RC} -ne 124 ]]; then
    echo "Run ${i} failed with exit code ${RC}. See ${LOG}" >&2
    exit ${RC}
  fi
done

echo "Completed ${RUNS} runs."
