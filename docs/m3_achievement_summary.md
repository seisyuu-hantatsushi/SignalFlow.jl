# M3 Achievement Summary

## Scope
- Goal: validate low-SNR adaptivity behavior after M1/M2, then choose a practical debug operating point with minimal EVM penalty.

## What Was Implemented
- Added impairment-based stress workflows (`IMPAIR_*`, `SYM_IMPAIR_*`) for adaptivity probing.
- Added force-update controls for phase correction path:
  - `--slope-force-update-eps`
  - `--cpe-force-update-eps-deg`
- Completed CPE runtime force-update behavior and force counters in logs.
- Added evaluation scripts:
  - `scripts/run_m3_adaptivity_probe.sh`
  - `scripts/run_m3_force_ab.sh`
  - `scripts/run_m3_force_taper.sh`
  - `scripts/run_m3_operatingpoint_normalsnr.sh`

## Key Results
- Adaptivity activation was confirmed:
  - Without force: `phase_up/cpe_up` stayed `0/0` under stress.
  - With force: updates became non-zero as expected.
- Factor isolation result:
  - `slope-force` increased EVM more than `cpe-force`.
  - `cpe-only` achieved update activation with smaller penalty.
- Normal-SNR confirmation (`12dB`, `6dB`) showed:
  - Stability maintained (`lock=1`, `unlock=0`, `sink_fail=0`).
  - Quality difference between `cpe_force=0` and `0.005` was small.

## Operating Policy (Provisional)
- Quality-first default:
  - `slope_force_update_eps=0`
  - `cpe_force_update_eps_deg=0`
- Debug/probing mode (to force CPE update observability):
  - `slope_force_update_eps=0`
  - `cpe_force_update_eps_deg=0.005`

## Remaining Items to Close M3
- Optional: periodic A/B recheck (`cf=0` vs `cf=0.005`) when RF condition changes.
- Proceed to next acceptance phase (constellation/TMCC criteria) using the policy above.
