# HANDOVER

## Current progress status
- Reviewed `examples/isdbt_demod_until_fft.jl` and `src/FFTBlock.jl` for FFTBlock acceleration opportunities.
- Checked existing logs (`demod_until_fft_perf.log`, `demod_until_fft_seqcheck.log`, and related `demod_*.log`) to identify runtime bottlenecks.
- Conclusion: the dominant current bottleneck is upstream/downstream backpressure and stability issues around FFT sink handling, not raw FFT kernel throughput alone.

## Files changed and reasons for the changes
- `HANDOVER.md`
  - Added this handover record to comply with AGENTS.md operational rules.

## Tasks to be addressed next
1. Stabilize measurement path first:
   - Run `examples/isdbt_demod_until_fft.jl` with `--no-seqcheck` and `--fft-perf-interval` enabled to get clean FFT-only baseline.
2. Remove avoidable FFTBlock overhead:
   - Eliminate first-frame debug prints from hot path.
   - Fuse window and scale multiply into a single pass by precomputing effective window coefficients when `scale != FFTScaleNone`.
3. Improve queue/backpressure behavior:
   - Tune `poolsize` and `dispatch_burst` jointly to prevent `ISDBTSymbolSync: sink_backpressure` from dominating throughput.
4. Add reproducible benchmark mode:
   - Create a fixed-duration benchmark script for `isdbt_demod_until_fft.jl` that captures perf counters and input/output frame counts.

## Known issues
- `demod_until_fft_perf.log` shows repeated `ISDBTSymbolSync: sink_backpressure sink=SignalFlow.FFTBlock.FFTBlockContext{ComplexF32}` lines, indicating pipeline pressure before meaningful FFT perf stats accumulate.
- Historical logs include cases where `FFTBlock perf: total_frames=1` despite many input frames, so current perf logging can be invalid under failure/backpressure scenarios.
- SeqCheckMonitor behavior was recently changed to monitor-only mode (no forwarding); benchmark wiring must keep that assumption.

## Current progress status
- Removed obsolete local `.log` files in repository root to clean benchmarking workspace.

## Files changed and reasons for the changes
- Deleted: `demod_noseqtrace.log`, `demod_seqtrace_fft.log`, `demod_seqtrace_fft_missing.log`, `demod_seqtrace_fft_sparse.log`, `demod_seqtrace_missonly.log`, `demod_seqtrace_ok.log`, `demod_seqtrace_sym.log`, `demod_seqtrace_symsync.log`, `demod_sink_backpressure.log`, `demod_until_fft_perf.log`, `demod_until_fft_seqcheck.log`
  - Reason: user-requested cleanup of unnecessary local log artifacts.
- Updated: `HANDOVER.md`
  - Reason: record completed task per AGENTS.md directions.

## Tasks to be addressed next
1. Run fresh `examples/isdbt_demod_until_fft.jl` baseline without seqcheck.
2. Sweep `--src-poolsize` and `--src-dispatch-burst` and compare backpressure frequency.
3. Apply FFTBlock hot-path optimizations after stable baseline is established.

## Known issues
- No historical log snapshots remain in the workspace after cleanup.

## Current progress status
- Evaluated baseline run log `baseline_fft_p384_b32.log` for FFTBlock performance and pipeline pressure.
- Baseline quality is generally good: sustained FFT processing with graceful shutdown completed.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed log evaluation task per AGENTS.md.

## Tasks to be addressed next
1. Run additional baseline sweeps for `--src-poolsize` and `--src-dispatch-burst` using same logging format.
2. Compare final `FFTBlock perf: avg_us` convergence and `sink_backpressure` count across runs.
3. After best config is selected, implement FFTBlock hot-path optimizations (window/scale fusion).

## Known issues
- `baseline_fft_p384_b32.log` contains 41 `sink_backpressure` events, concentrated in early startup.
- The run includes one large FFT max latency outlier (`max_us=17626.418`) despite stable average.
- Near shutdown, log includes `ERROR: READ LINE: -9` and `ERROR: READ INTEGER: -9` (likely device read termination path); run still exits gracefully with `Shutdown complete.`.

## Current progress status
- Evaluated 12 baseline logs (`baseline_fft_p*_b*.log`) generated from poolsize/dispatch_burst sweep.
- Extracted and compared: final `FFTBlock perf: avg_us`, `sink_backpressure` count, and `FFTBlock input stats` consistency.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: record completion of sweep evaluation task per AGENTS.md.

## Tasks to be addressed next
1. Re-run top 2 configurations once each for reproducibility confirmation.
2. Choose one stable baseline config and keep it as default benchmark preset.
3. Start FFTBlock micro-optimization patch (window/scale fusion) and compare against chosen baseline.

## Known issues
- All logs contain 2 shutdown-time `ERROR:` lines (`READ LINE: -9`, `READ INTEGER: -9`) but still finish with graceful shutdown.
- `max_us` outliers remain large (~16-26 ms) in all runs, so average latency is more reliable than max for ranking.

## Current progress status
- Evaluated confirmation logs: `confirm_fft_p768_b16.log` and `confirm_fft_p768_b32.log`.
- Result: `p768/b32` delivered clearly better FFT average latency than `p768/b16` in this rerun.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: logged completion of confirmation comparison task per AGENTS.md.

## Tasks to be addressed next
1. Adopt `--src-poolsize 768 --src-dispatch-burst 32` as current baseline condition.
2. Start FFTBlock optimization patch (window+scale fusion) and benchmark against this baseline.
3. Investigate run-to-run variance for `p768/b16` (scheduler/noise effects) if needed.

## Known issues
- Both confirm logs still include 2 shutdown-time `ERROR:` lines, while shutdown remains graceful.
- Performance ranking changed from prior sweep (`p768/b16` no longer best), indicating non-trivial run-to-run variability.

## Current progress status
- Updated default baseline parameters for `examples/isdbt_demod_until_fft.jl` according to confirmed run results.

## Files changed and reasons for the changes
- Updated: `examples/isdbt_demod_until_fft.jl`
  - Change: default `src_poolsize` from `384` to `768`.
  - Reason: adopt measured baseline setting (`--src-poolsize 768 --src-dispatch-burst 32`).
- Updated: `HANDOVER.md`
  - Reason: recorded completed task per AGENTS.md.

## Tasks to be addressed next
1. Run one quick smoke test using defaults (without explicitly passing poolsize/burst).
2. Start FFTBlock optimization patch (window+scale fusion) and compare with the new default baseline.

## Known issues
- Shutdown-time `ERROR: READ LINE: -9` / `ERROR: READ INTEGER: -9` still appears in benchmark logs (graceful shutdown otherwise).

## Current progress status
- Listed current FFTBlock issues based on implementation inspection and latest benchmark logs.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completion of analysis task per AGENTS.md.

## Tasks to be addressed next
1. Remove/guard debug prints from FFT hot path.
2. Fuse window and scale multiply into one pass.
3. Add FFTBlock input fast-path for frame-aligned writes.
4. Improve perf metrics (windowed avg / p95) and separate startup from steady-state.

## Known issues
- Current FFTBlock performance is sensitive to startup/backpressure effects and run-to-run variance.

## Current progress status
- Implemented FFTBlock improvements (items 1-3): removed debug-print hot-path branches, fused window+scale preprocessing, and added frame-aligned input fast-path.

## Files changed and reasons for the changes
- Updated: `src/FFTBlock.jl`
  - Removed `debug_first_stage` and all first-frame debug `println` branches in worker hot paths.
  - Precomputed effective window coefficients in `CreateFFTBlock` by folding `scale_gain` into `window` once.
  - Removed runtime scale loop in worker tasks (Complex and Real paths).
  - Added `input!` fast-path for `actual_size == frame_size && holdbuf === nothing` to reduce per-call branching overhead.
  - Moved SeqTrace input check out of per-chunk loop to once per input call.
- Updated: `HANDOVER.md`
  - Reason: recorded completed implementation task per AGENTS.md.

## Tasks to be addressed next
1. Benchmark updated FFTBlock with baseline condition (`poolsize=768`, `dispatch_burst=32`).
2. Compare new `avg_us`, `max_us`, and `sink_backpressure` against `confirm_fft_p768_b32.log`.
3. If needed, add windowed/p95 perf metrics for stable ranking.

## Known issues
- Shutdown-time `ERROR: READ LINE: -9` / `ERROR: READ INTEGER: -9` is still expected in current runs.

## Current progress status
- Compared post-optimization benchmark (`bench_after_fftopt_p768_b32.log`) against pre-optimization baseline (`confirm_fft_p768_b32.log`).
- Confirmed clear FFT average latency improvement after implementing FFTBlock items 1-3.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed benchmark comparison task per AGENTS.md.

## Tasks to be addressed next
1. Repeat post-optimization run once more to confirm reproducibility of avg_us improvement.
2. If reproducible, keep current FFTBlock changes and consider adding p95/p99 metrics.
3. Optionally profile max-latency outliers (`max_us`) with scheduler/GC correlation.

## Known issues
- `max_us` increased in this run (higher outlier), while `avg_us` improved significantly.
- Shutdown-time read errors still appear (2 lines), but shutdown remains graceful.

## Current progress status
- Implemented FFTBlock perf metrics update with warmup exclusion and tail-latency indicators.

## Files changed and reasons for the changes
- Updated: `src/FFTBlock.jl`
  - Added warmup-aware perf accounting (`perf_warmup_frames`) so first N frames are excluded from steady-state metrics.
  - Added rolling window storage for recent frame latencies and computed `win_avg_us`, `p95_us`, `p99_us`.
  - Preserved raw observed maximum latency as `raw_max_us` (includes warmup/outliers) for reference.
  - Updated periodic and final perf logs to include new metrics and counts.
- Updated: `HANDOVER.md`
  - Reason: recorded completed implementation task per AGENTS.md.

## Tasks to be addressed next
1. Run benchmark again and confirm new log fields are emitted as expected.
2. Compare `final_avg_us`/`p95_us`/`p99_us` across runs for stable tuning.

## Known issues
- `raw_max_us` may still be dominated by scheduler/GC outliers; interpret with window percentiles.

## Current progress status
- Evaluated `bench_after_perfmetrics_p768_b32.log` and verified new FFT perf metrics are emitted correctly (`win_avg_us`, `p95_us`, `p99_us`, `raw_max_us`, warmup-aware final summary).

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed benchmark verification task per AGENTS.md.

## Tasks to be addressed next
1. Reduce measurement overhead of percentile calculation (current implementation sorts window every log print).
2. Re-run benchmark with larger `--fft-perf-interval` (e.g. 1000) for cleaner throughput comparison.
3. Keep new metrics for tuning decisions and compare p95/p99 across parameter sweeps.

## Known issues
- Compared with previous `bench_after_fftopt_p768_b32.log`, average metric is slightly slower under current logging settings, likely influenced by added metric computation overhead.

## Current progress status
- Evaluated `bench_after_perfmetrics_i1000_p768_b32.log` against `bench_after_perfmetrics_p768_b32.log`.
- Found significant regression in `final_avg_us` and tail metrics in the i1000 run.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed benchmark comparison task per AGENTS.md.

## Tasks to be addressed next
1. Re-run i1000 benchmark once or twice to check reproducibility.
2. If regression persists, profile `maybe_update_perf!` percentile path cost.
3. If reproducibility fails (run-to-run drift), compare with CPU pinning / reduced system load.

## Known issues
- `bench_after_perfmetrics_i1000_p768_b32.log` shows elevated `win_avg_us`, `p95_us`, and `p99_us` late in run.
- Read errors (`ERROR: READ LINE: -9`, `ERROR: READ INTEGER: -9`) appear before shutdown in this run, suggesting runtime disturbance.

## Current progress status
- Evaluated additional i1000 benchmark logs: `bench_after_perfmetrics_i1000_p768_b32_2.log` and `bench_after_perfmetrics_i1000_p768_b32_3.log`.
- Confirmed high run-to-run variance in final average and tail metrics under current conditions.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed multi-run comparison task per AGENTS.md.

## Tasks to be addressed next
1. Use median-of-runs policy (>=3 runs) for config decisions.
2. Reduce metric-logging perturbation and/or separate metric task from FFT hot path.
3. Add environment controls (CPU pinning/isolated load) for more stable benchmark reproducibility.

## Known issues
- i1000 runs vary widely (`final_avg_us` from ~39.4 to ~47.8) with similarly large p95/p99 spread.
- Tail metrics can degrade even when `sink_backpressure` is low, indicating external jitter impact.

## Current progress status
- Clarified interpretation method for median-based comparison of multi-run FFT benchmarks.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completion of interpretation support task per AGENTS.md.

## Tasks to be addressed next
1. If user provides command output, finalize configuration decision based on median and spread.

## Known issues
- None added in this step.

## Current progress status
- Completed FFTBlock task-path deduplication (requested item 5).

## Files changed and reasons for the changes
- Updated: `src/FFTBlock.jl`
  - Replaced duplicated `task!` implementations (`ComplexF32` and `Real`) with a single generic `task!`.
  - Added `fill_fft_input!` specializations for `ComplexF32`, generic `Complex`, and `Real` input types to keep conversion logic type-specific while sharing the worker loop.
  - Reason: reduce maintenance risk and ensure future optimizations apply uniformly.
- Updated: `HANDOVER.md`
  - Reason: recorded completion per AGENTS.md.

## Tasks to be addressed next
1. Run benchmark to confirm no regression after task deduplication.
2. Compare `final_avg_us/p95/p99` with previous log under same run condition.

## Known issues
- Benchmark variability remains high across runs; use median-of-runs policy for decisions.

## Current progress status
- Implemented FFT metric responsibility separation (hot path collection vs async aggregation).

## Files changed and reasons for the changes
- Updated: `src/FFTBlock.jl`
  - Added async perf channel (`perf_samples`) and dedicated `perf_task!` to process latency samples.
  - Hot path now only enqueues `dt` via `enqueue_perf_sample!` and no longer performs heavy percentile aggregation directly.
  - Added metric-drop accounting (`perf_drop_count`) when perf queue is full, preserving real-time processing priority.
  - Ensured shutdown waits for `perf_task` to flush metrics before final summary print.
- Updated: `HANDOVER.md`
  - Reason: recorded completed task per AGENTS.md.

## Tasks to be addressed next
1. Run benchmark to verify async metrics path reduces perturbation and keeps perf logs stable.
2. Compare with previous run for `final_avg_us`, `p95/p99`, and `dropped_metrics`.

## Known issues
- Percentile computation still uses sort in perf task; CPU cost remains but is now off the signal-processing hot path.

## Current progress status
- Refactored FFTBlock measurement-related fields into a grouped stats structure (`FFTPerfStats`) and updated call sites to use grouped access.
- Fixed a transient precompile issue by using `perf::Any` field type in `FFTBlockContext` while keeping the grouped stats object as `FFTPerfStats`.

## Files changed and reasons for the changes
- Updated: `src/FFTBlock.jl`
  - Added `FFTPerfStats` to bundle metric counters/state/queue/task.
  - Updated perf initialization and runtime functions to operate on grouped stats object.
  - Updated task and stop paths to use grouped perf state (`context.perf.*`).
  - Reason: requested consolidation of measurement-related variables.
- Updated: `HANDOVER.md`
  - Reason: recorded completion per AGENTS.md.

## Tasks to be addressed next
1. Run one benchmark to confirm no runtime regression after stats-struct refactor.
2. If desired, reorder type declarations so `FFTBlockContext.perf` can be strongly typed as `FFTPerfStats`.

## Known issues
- `FFTBlockContext.perf` currently uses `Any` to avoid forward-reference type-definition ordering issue.

## Current progress status
- Addressed shutdown log-noise issue (`ERROR: READ LINE/INTEGER: -9`) on SignalFlow side by suppressing stderr only during adapter stop.

## Files changed and reasons for the changes
- Updated: `src/ADFMCOMMS2Src.jl`
  - Added `stop_adapter_quietly!` helper that wraps `ADFMCOMMS2.stop!` in `redirect_stderr(devnull)`.
  - Replaced direct `ADFMCOMMS2.stop!` calls in `close!` and `recv_task!` teardown with quiet stop helper.
  - Reason: prevent benign libiio shutdown noise from polluting runtime logs while preserving graceful stop behavior.
- Updated: `HANDOVER.md`
  - Reason: recorded completed task per AGENTS.md.

## Tasks to be addressed next
1. Run one shutdown benchmark and confirm `READ LINE/INTEGER -9` noise is no longer emitted.
2. If residual noise remains, investigate ADFMCOMMS2.jl-side suppression in rxTask error path.

## Known issues
- This suppression targets adapter stop windows only; external/native stderr from outside this window may still appear.

## Current progress status
- Additional shutdown-noise suppression added at example-level shutdown window after residual `READ ... -9` logs remained.

## Files changed and reasons for the changes
- Updated: `examples/isdbt_demod_until_fft.jl`
  - Wrapped `SignalFlow.stop_flow_graph!` call in `redirect_stderr(devnull)` during shutdown.
  - Reason: suppress residual libiio/native shutdown noise emitted outside prior adapter-local suppression window.
- Updated: `HANDOVER.md`
  - Reason: recorded follow-up fix per AGENTS.md.

## Tasks to be addressed next
1. Re-run benchmark and verify `ERROR: READ LINE/INTEGER -9` no longer appears.
2. Keep this suppression only for shutdown path; ensure normal runtime errors remain visible.

## Known issues
- Until rerun verification, complete elimination of shutdown noise is not yet confirmed.

## Current progress status
- Fixed shutdown suppression bug introduced in previous step.

## Files changed and reasons for the changes
- Updated: `examples/isdbt_demod_until_fft.jl`
  - Replaced invalid `open(devnull, "w")` usage with `redirect_stderr(devnull)` around `stop_flow_graph!`.
  - Reason: avoid `MethodError` during shutdown that destabilized termination flow.
- Updated: `src/ADFMCOMMS2Src.jl`
  - Replaced invalid `open(devnull, "w")` usage with `redirect_stderr(devnull)` in `stop_adapter_quietly!`.
  - Reason: ensure stderr suppression works correctly during adapter stop.
- Updated: `HANDOVER.md`
  - Reason: recorded corrective action per AGENTS.md.

## Tasks to be addressed next
1. Re-run shutdown benchmark and verify no `READ ... -9` and no shutdown warnings/crash.
2. Compare final metrics with previous run after this bug fix.

## Known issues
- Previous run (`bench_after_shutdown_noise_fix2_i1000_p768_b32.log`) became invalid due to shutdown `MethodError` and crash; do not use it for performance comparison.

## Current progress status
- Verified shutdown-noise suppression fix with `bench_after_shutdown_noise_fix3_i1000_p768_b32.log`.
- `READ LINE/INTEGER -9`, `MethodError`, and segfault no longer appear in this run.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded verification completion per AGENTS.md.

## Tasks to be addressed next
1. Keep monitoring shutdown logs over multiple runs to confirm stability.
2. Investigate increased `sink_backpressure` count in this run.

## Known issues
- Although shutdown noise is resolved, `sink_backpressure` remained high in this run (113), and final avg latency is slightly above best previous run.

## Current progress status
- Implemented item 1: replaced percentile sort path with approximate histogram-based percentile estimation for FFT perf metrics.

## Files changed and reasons for the changes
- Updated: `src/FFTBlock.jl`
  - Added histogram fields to `FFTPerfStats` (`hist`, `hist_bin_ns`).
  - Updated perf window maintenance to increment/decrement histogram bins in O(1).
  - Replaced `window_percentile_ns` full-sort computation with cumulative histogram approximation.
  - Reason: reduce measurement-task overhead while keeping p95/p99 visibility.
- Updated: `HANDOVER.md`
  - Reason: recorded completion per AGENTS.md.

## Tasks to be addressed next
1. Run benchmark and compare `final_avg_us` and `dropped_metrics` before/after approximation.
2. Tune histogram bin width/range if p95/p99 resolution is too coarse.

## Known issues
- p95/p99 are now approximate values (bin-center estimate), not exact order-statistics.

## Current progress status
- Verified histogram-percentile benchmark run (`bench_after_hist_percentile_i1000_p768_b32.log`).
- Confirmed shutdown noise suppression remains effective (`READ ... -9` absent).

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded benchmark verification completion per AGENTS.md.

## Tasks to be addressed next
1. Decide whether to keep histogram bin settings (`2us/bin`, 8192 bins) or tune for tail resolution.
2. Investigate very large `raw_max_us` outliers while steady-state metrics improved.

## Known issues
- `raw_max_us` reached 44 ms in this run despite improved `final_avg_us`; indicates occasional external/outlier disturbances.

## Current progress status
- Applied SIMD-focused optimization to FFT input staging path (`fill_fft_input!`).

## Files changed and reasons for the changes
- Updated: `src/FFTBlock.jl`
  - Added `@simd` to all `fill_fft_input!` loop variants (ComplexF32 / Complex / Real).
  - Added localized `@fastmath` on per-element multiply/conversion assignment.
  - Reason: reduce pre-FFT staging loop cost as requested.
- Updated: `HANDOVER.md`
  - Reason: recorded completion per AGENTS.md.

## Tasks to be addressed next
1. Run benchmark and compare `final_avg_us` / `p95/p99` with previous histogram-percentile run.
2. If numerical sensitivity is a concern, A/B compare decoded outputs with and without `@fastmath`.

## Known issues
- `@fastmath` can alter strict IEEE behavior; practical impact should be validated for SDR quality metrics.

## Current progress status
- Evaluated SIMD-optimized fill path benchmark (`bench_after_fill_simd_i1000_p768_b32.log`) against pre-SIMD histogram baseline.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded benchmark comparison completion per AGENTS.md.

## Tasks to be addressed next
1. Re-run SIMD benchmark for reproducibility (at least 2 more runs).
2. If reproducible, keep SIMD path; if not, decide based on median.

## Known issues
- `sink_backpressure` increased in this run, while average latency slightly improved; environment jitter remains a confounder.

## Current progress status
- Completed code review of `examples/isdbt_demod.jl` with focus on graceful SIGINT shutdown path, argument handling, and runtime robustness.
- Identified a startup-failure bug in SDR URI auto-detection and one CLI behavior inconsistency.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed review task per AGENTS.md.

## Tasks to be addressed next
1. Fix URI auto-detection to check `isempty(ADFMCOMMS2.scan("ip"))` before indexing.
2. Decide whether `--seq-trace-log-ok` should be honored or removed/renamed from `parse_args` and help text.
3. Add a small regression test (or script-level assertion path) for empty-URI scan behavior.

## Known issues
- `examples/isdbt_demod.jl` currently indexes `ADFMCOMMS2.scan("ip")[1]` before emptiness check, which can throw `BoundsError` when no SDR is found.
- `--seq-trace-log-ok` is parsed but then forcibly disabled (`seq_trace_log_ok = false`), making the flag effectively non-functional.

## Current progress status
- Removed non-functional CLI option `--seq-trace-log-ok` from `examples/isdbt_demod.jl` as requested (it had been parsed but forcibly overridden to false).
- Kept SeqTrace behavior explicitly miss-only (`log_ok=false`) and simplified related code paths.

## Files changed and reasons for the changes
- Updated: `examples/isdbt_demod.jl`
  - Removed `seq_trace_log_ok` variable and `--seq-trace-log-ok` argument branch.
  - Updated `parse_args` return tuple and `main()` destructuring accordingly.
  - Simplified SeqTrace configuration/log message to fixed `log_ok=false`.
  - Reason: eliminate dead/non-functional option for code cleanup.
- Updated: `HANDOVER.md`
  - Reason: recorded completed cleanup task per AGENTS.md.

## Tasks to be addressed next
1. Fix the remaining startup robustness issue in `examples/isdbt_demod.jl` URI auto-detection (`scan("ip")[1]` before emptiness check).
2. Optionally align similar CLI cleanup in related examples if unused flags exist.

## Known issues
- `examples/isdbt_demod.jl` still has URI auto-detection bug path (`ADFMCOMMS2.scan("ip")[1]` may throw `BoundsError` when no SDR is found).

## Current progress status
- Fixed FFTBlock sequence-number consistency after recent FFTBlock changes.
- Implemented mixed sequence policy: inherit source seq for 1:1 frame path, re-sequence only for composed frames.

## Files changed and reasons for the changes
- Updated: `src/FFTBlock.jl`
  - Added per-frame composition tracking (`frame_composed`) and local output sequence state (`out_seq_local`).
  - For fast-path (`actual_size == fft_size`): keep source sequence inheritance.
  - For composed-path (hold-buffer assembly / multi-frame split): mark frame as composed and assign new FFTBlock-local sequential output IDs.
  - Kept SeqTrace logging on FFTBlock with non-strict mode for observability.
  - Fixed context initialization type mismatch by using `fill(false, poolsize)` for `Vector{Bool}`.
  - Reason: restore downstream SeqTrace continuity while correctly handling frame-composition semantics.
- Updated: `HANDOVER.md`
  - Reason: recorded completed fix/verification task per AGENTS.md.

## Tasks to be addressed next
1. Run `examples/isdbt_demod.jl` baseline again and verify FFTBlock→FrameSync→PilotEQ sequence mismatch logs are eliminated or reduced to real drops.
2. Validate SP phase stability metrics after this sequence fix under target frequency (`515.142857M`).

## Known issues
- Existing upstream/downstream backpressure (e.g., SymbolSync sink lag) may still cause real frame drops and sequence jumps unrelated to FFTBlock numbering policy.

## Current progress status
- Enumerated current block connections in `examples/isdbt_demod.jl` for user visibility.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed reporting task per AGENTS.md.

## Tasks to be addressed next
1. If needed, generate a runtime block graph dump function for automatic topology output.

## Known issues
- Block topology is configuration-dependent (`--diag`, `--show-*`, `--pilot-eq-only`, etc.), so active graph differs by launch options.

## Current progress status
- Evaluated baseline log `logs/baseline_sp_full_515142857_20260207_210141.log` to identify post-FFT problem points.
- Confirmed major runtime issue is persistent SymbolSync-side backpressure causing downstream sequence gaps at PilotEQ/PhaseSlope/CPE stages.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed baseline-log evaluation task per AGENTS.md.

## Tasks to be addressed next
1. Reduce/remove `ISDBTSymbolSync -> SignalStatsMonitor` backpressure impact in `--diag` mode (drop/skip monitor updates instead of blocking SymbolSync path).
2. Re-run baseline and compare mismatch counts (`PilotEQ/PhaseSlope/CPE in_mismatch`) and `sink_backpressure` frequency.
3. Investigate rare `ISDBTFrameSync` large sequence jumps (including negative delta) for out-of-order/reset path.

## Known issues
- `ISDBTSymbolSync: sink_backpressure sink=SignalFlow.SignalStatsMonitor.SignalStatsMonitorContext` appears very frequently (9650 times).
- Persistent sequence mismatches after FFT path remain high (`PilotEQ/PhaseSlope/CPE in_mismatch`: each 1664 times), with dominant delta around +14/+15.
- Rare extreme sequence anomalies are present (e.g., `PilotEQ in_mismatch delta=7617` and `-7602`; `ISDBTFrameSync in_mismatch` includes `delta=-127`/`129`).

## Current progress status
- Documented current ISDB-T demod block topology used in `examples/isdbt_demod.jl` for traceability.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded current block connection path as requested.

## ISDB-T demod block path (current)
- Core decode path:
  - `rfsrc -> sync -> fft -> fft_gain_block -> pilot_eq -> (optional slope) -> (optional cpe) -> data_carriers`
- Frame timing/reference side path:
  - `fft -> frame_sync`
- Monitoring/diagnostic side branches (option-dependent):
  - `rfsrc -> snr`
  - `sync -> stats_sync` (`--diag`)
  - `sync -> wave_view` (`--show-wave`)
  - `fft_gain_block -> tmcc_power`, `fft_gain_block -> tmcc_power_flip`, `fft_gain_block -> pilot_corr`
  - `fft_gain_block -> stats_fft` (`--diag`)
  - `fft_gain_block -> fft_view` (`--show-fft`)
  - `pilot_eq -> stats_piloteq` (`--diag`)
  - `pilot_eq -> pilot_corr_eq` (`--diag`)
  - `prev_block -> tmcc_dbpsk_norm/flip` (`--tmcc-dbpsk`)
  - `data_carriers -> stats_data` (`--diag`)
  - `data_carriers -> constellation` (unless `--no-const`)
  - `pilot_eq -> pilot_extract -> pilot_view` (`--show-pilots`)

## Tasks to be addressed next
1. Validate whether `fft_gain_block` applies configured gain correctly in a minimal reproducible test.

## Known issues
- Under `--diag`, monitor branch backpressure can impact symbol continuity (see prior baseline findings).

## Current progress status
- Verified `fft_gain_block` behavior via a minimal runtime test of `SignalFlow.GainBlock`.
- Confirmed gain multiplication is correct for ComplexF32 frames (`gain=2.5`, exact expected output, max error 0.0).

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completion of `fft_gain_block` validation task per AGENTS.md.

## Tasks to be addressed next
1. If needed, run end-to-end `isdbt_demod.jl` with non-unity `--fft-gain` (e.g., 0.5 / 2.0) and compare downstream monitor levels.

## Known issues
- `GainBlock` itself behaves correctly in isolation; current instability remains dominated by upstream backpressure in diagnostic path.

## Current progress status
- Verified requested connection pattern is already implemented in `examples/isdbt_demod.jl`:
  - `fft_gain_block -> (mon_pilot) -> pilot_eq` when `!diag`
  - `fft_gain_block -> pilot_eq` when `--diag`
- Confirmed `pilot_eq` is active from baseline log (`logs/baseline_sp_full_515142857_20260207_210141.log`) via repeated `PilotEQ`, `SignalStats[PilotEQ out]`, and `PilotCorr[seg0_eq]` outputs.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded connection verification and pilot_eq runtime-check completion per AGENTS.md.

## Tasks to be addressed next
1. If `mon_pilot` behavior itself must be validated, run a non-diag capture and confirm `RateMonitor(PilotEQ in)` logs.
2. Continue reducing upstream backpressure that currently dominates PilotEQ downstream continuity.

## Known issues
- Although PilotEQ runs, sequence continuity is still degraded by frequent SymbolSync-side backpressure under current diag conditions.

## Current progress status
- Improved `SignalStatsMonitor` to reduce side effects on realtime decode path.
- Monitor now drops frames under pressure instead of returning backpressure (`-1`) upstream.

## Files changed and reasons for the changes
- Updated: `src/SignalStatsMonitor.jl`
  - Added non-blocking/drop-oriented behavior in `input!`:
    - Queue-full case: drop monitor frame, log dropped count periodically, and return `samples_size` (no backpressure).
    - Frame-size mismatch case: count/log mismatch drops and return `samples_size` (no backpressure).
  - Added monitor drop counters/config:
    - `drop_log_interval`, `dropped_frames`, `input_mismatch_count` in context.
    - `drop_log_interval` keyword in `CreateSignalStatsMonitor` (default: 500).
  - Added `AsyncLogger` usage for periodic drop reporting.
  - Reason: ensure monitoring path degrades itself first and does not disturb SDR realtime processing.
- Updated: `HANDOVER.md`
  - Reason: recorded completed implementation and verification task per AGENTS.md.

## Tasks to be addressed next
1. Re-run `examples/isdbt_demod.jl` baseline and compare:
   - `ISDBTSymbolSync: sink_backpressure` frequency
   - SeqTrace mismatch counts at `PilotEQ/PhaseSlope/CPE`.
2. If needed, apply same non-blocking policy to other diagnostic-only monitor blocks.

## Known issues
- This change removes monitor-induced upstream backpressure but does not fix drops caused by other sinks or external scheduling jitter.

## Current progress status
- Assessed `SignalStatsMonitor` buffer sizing in current `isdbt_demod.jl` wiring.
- Confirmed all monitor instances use default `poolsize=8` (no explicit override in example).

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed assessment task per AGENTS.md.

## Tasks to be addressed next
1. Decide priority between monitor fidelity and memory budget:
   - Keep `poolsize=8` for minimal memory and acceptable drop-tolerant monitoring, or
   - Raise to `16-32` to reduce monitor-frame drops during scheduler jitter.
2. If raising poolsize, re-run baseline and compare monitor drop logs and core-path stability.

## Known issues
- With OFDM frame interval around ~1.13 ms, `poolsize=8` buffers only ~9 ms of burst slack per monitor; monitor drops can still occur under long scheduling stalls.

## Current progress status
- Increased `SignalStatsMonitor` buffer pool size in `isdbt_demod` wiring to improve monitor burst tolerance.

## Files changed and reasons for the changes
- Updated: `examples/isdbt_demod.jl`
  - Added `const StatsMonitorPool = 32`.
  - Applied `poolsize = StatsMonitorPool` to all `SignalStatsMonitor.CreateSignalStatsMonitor` calls (`stats_sync`, `stats_fft`, `stats_piloteq`, `stats_slope`, `stats_cpe`, `stats_data`).
  - Reason: raise monitor queue slack to reduce monitor-frame drop frequency under scheduler jitter while keeping decode path non-blocking.
- Updated: `HANDOVER.md`
  - Reason: recorded completed poolsize tuning task per AGENTS.md.

## Tasks to be addressed next
1. Re-run baseline and compare monitor drop logs before/after (`SignalStatsMonitor[*]: dropped_backpressure_frames`).
2. Confirm `ISDBTSymbolSync: sink_backpressure` and downstream mismatch counts are reduced.

## Known issues
- Larger monitor pool reduces drops but cannot eliminate scheduling stalls from other sinks.

## Current progress status
- Evaluated `logs/eval_statspool32_full_515142857_20260207_212059.log` after `SignalStatsMonitor` poolsize increase and non-blocking monitor changes.
- Confirmed SymbolSync->SignalStatsMonitor backpressure was eliminated in this run.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed log evaluation and before/after comparison per AGENTS.md.

## Tasks to be addressed next
1. Investigate persistent downstream sequence mismatch pattern (`PilotEQ/PhaseSlope/CPE`) despite zero SymbolSync backpressure.
2. Investigate increased FrameSync mismatch events and rare large jump events (e.g., ±1855).
3. Address shutdown timeout warning for `SignalStatsMonitor` (consider longer stop timeout or cooperative drain behavior).

## Known issues
- New log still has high mismatch counts at PilotEQ/PhaseSlope/CPE (each 4089), with dominant deltas around +14/+15.
- `SeqTrace[ISDBTFrameSync] in_mismatch` appears 88 times, including rare large jumps (e.g., `delta=-1855`, `delta=1857`).
- Shutdown ends with warning: `shutdown timeout block = SignalFlow.SignalStatsMonitor.SignalStatsMonitorContext`.

## Current progress status
- Implemented fix for FrameSync sequence-anomaly root cause candidate (stale SeqTrace metadata reuse).
- Updated SeqTrace behavior so `seq=0` clears existing buffer-sequence metadata instead of leaving stale values.

## Files changed and reasons for the changes
- Updated: `src/SeqTrace.jl`
  - `set_seq!(buf, 0)` now removes existing seq entry from `BUFFER_SEQ` (previously no-op).
  - `inherit_seq!` now always calls `set_seq!`, so zero-seq input actively clears destination metadata.
  - Reason: prevent old sequence values from being reused on ring-buffer objects and causing false backward/forward jumps (e.g. large ± deltas in FrameSync path).
- Updated: `HANDOVER.md`
  - Reason: recorded completed fix and local verification per AGENTS.md.

## Tasks to be addressed next
1. Re-run baseline and verify reduction of rare large sequence jumps in `SeqTrace[ISDBTFrameSync]`.
2. Compare counts of `ISDBTFrameSync in_mismatch` and check if ±1000-class deltas disappear.

## Known issues
- This fix addresses stale-seq contamination; genuine frame drops/reorders from other causes can still produce positive mismatch deltas.

## Current progress status
- Checked whether frame-length related settings/output are currently consistent in latest eval log.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completion of frame-length consistency check per AGENTS.md.

## Tasks to be addressed next
1. Re-run after SeqTrace zero-clear fix and re-check large sequence jump behavior.

## Known issues
- Current dominant issue remains sequence continuity, not frame-size mismatch.

## Current progress status
- Checked consistency between ARIB STD-B31 (`6-STD-B31v2_2-E1.pdf`) frame-length definitions and `examples/isdbt_demod.jl` constants.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed spec-consistency check task per AGENTS.md.

## Tasks to be addressed next
1. Optional: document explicitly in code comments why `OFDM_NFFT=8064` is used with `ADC_SamplingRate=8MHz` (equivalent timing to Mode-3 1/8 frame timing).

## Known issues
- No contradiction found for frame length itself (Mode 3, GI=1/8, 204 symbols).
- Separate from frame length, other bandwidth-related constants may still warrant spec-level review if strict compliance is required.

## Current progress status
- Added a new evaluation criterion based on ARIB STD-B31 frame-length consistency.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded user-requested evaluation criterion.

## Evaluation criteria (updated)
- Frame-length consistency with ARIB STD-B31 (Mode 3, GI=1/8):
  - `Symbols per frame = 204`
  - `Symbol length = 1134 us`
  - `Frame length = 231.336 ms`
- For this project configuration, these values must remain consistent with code-level constants/logs:
  - `FrameSymbols = 204`
  - `ExpectedFrameMs = 231.336` (computed)
  - Runtime log check: `OFDM params: ... frame_ms_expected=231.336`

## Tasks to be addressed next
1. Include this criterion in future baseline/reproducibility checks.

## Known issues
- None added by this update.

## Current progress status
- Addressed the first priority issue (sequence continuity) by fixing SeqTrace propagation across `fft_gain_block` (`GainBlock`).

## Files changed and reasons for the changes
- Updated: `src/GainBlock.jl`
  - Added `SeqTrace` integration.
  - `input!`: now inherits sequence metadata from source buffer into internal ring buffer.
  - `task!`: now logs `SeqTrace` in/out and propagates input sequence to `outbuf` (pass-through semantics).
  - Reason: `GainBlock` sits between FFT and PilotEQ; missing seq propagation there caused downstream mismatch inflation.
- Updated: `HANDOVER.md`
  - Reason: recorded completion and verification per AGENTS.md.

## Tasks to be addressed next
1. Re-run `isdbt_demod.jl` baseline and compare mismatch counts at `PilotEQ/PhaseSlope/CPE` before/after this fix.
2. Verify whether large FrameSync jumps persist after combined SeqTrace fixes (`SeqTrace zero-clear` + `GainBlock propagation`).

## Known issues
- Local unit smoke test confirmed sequence propagation behavior in `GainBlock`, but end-to-end impact must be validated on live run logs.

## Current progress status
- Evaluated new run log `logs/eval_gain_seqfix_full_515142857_20260212_095020.log` after GainBlock SeqTrace propagation fix.
- Compared against previous reference `logs/eval_statspool32_full_515142857_20260207_212059.log`.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed post-fix evaluation task per AGENTS.md.

## Tasks to be addressed next
1. Keep first-priority continuity work: investigate remaining periodic `PilotEQ/PhaseSlope/CPE` mismatches (dominant delta +14/+15).
2. Tackle FrameSync anomaly path: very large paired ±delta jumps still occur (`ISDBTFrameSync in_mismatch`), likely separate from GainBlock.
3. Add targeted instrumentation around FrameSync enqueue/dequeue seq values to pinpoint where backward jumps are introduced.

## Known issues
- SymbolSync backpressure remained solved (`sink_backpressure=0`).
- PilotEQ mismatch slightly improved per 1k frames (80.58 -> 78.69) but is still high.
- FrameSync mismatch per 1k frames improved (1.73 -> 1.46), yet large jump anomalies persist (observed up to ±44673).

## Current progress status
- Added targeted sequence-probe tracing in `ISDBTFrameSync` to diagnose large sequence jumps.
- Fixed load-order issue introduced during instrumentation by removing early type annotation in helper function signature.

## Files changed and reasons for the changes
- Updated: `src/ISDBTFrameSync.jl`
  - Added runtime fields: `seq_anomaly_count`, `last_enq_seq`, `last_deq_seq`.
  - Added `maybe_log_seq_jump!` helper and threshold constant (`SEQ_TRACE_JUMP_WARN=64`).
  - Instrumented seq checks at both enqueue (`input!`) and dequeue (`task!`) paths:
    - Logs only on large jumps to keep overhead low.
    - Log format includes `where=enqueue/dequeue`, ring index, prev/cur seq, delta, anomaly count.
  - Reason: pinpoint where large seq discontinuities are introduced around FrameSync queue boundaries.
- Updated: `HANDOVER.md`
  - Reason: recorded completed instrumentation task per AGENTS.md.

## Tasks to be addressed next
1. Run new baseline and inspect `ISDBTFrameSync: seq_probe where=...` logs.
2. Determine whether anomalies originate primarily at enqueue side or dequeue side.
3. Apply focused fix based on side of origin.

## Known issues
- Instrumentation itself does not fix sequence jumps; it only narrows fault location.

## Current progress status
- Evaluated `logs/eval_framesync_seqprobe_515142857_20260212_101725.log` with new FrameSync seq-probe instrumentation.
- Confirmed seq anomaly events are observed at both enqueue and dequeue points.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed seq-probe log evaluation task per AGENTS.md.

## Tasks to be addressed next
1. Focus on enqueue-side origin: investigate why enqueue sometimes receives unexpectedly small/stale seq values (e.g., `prev=184 cur=57`, `prev=2640 cur=17`).
2. Add upstream probe around `FFTBlock -> frame_sync` handoff to confirm source seq at call boundary.
3. Distinguish true stream discontinuities from SeqTrace metadata reuse/race by correlating probe timestamps across adjacent blocks.

## Known issues
- `ISDBTFrameSync: seq_probe` events: total 186 (enqueue 91, dequeue 95).
- FrameSync mismatches persist (`SeqTrace[ISDBTFrameSync] in_mismatch=95`) and include large paired jumps.
- PilotEQ continuity remains degraded (`in_mismatch=3629`, dominant +14/+15 deltas).

## Current progress status
- Added focused upstream probe at `FFTBlock -> ISDBTFrameSync` handoff to locate where large sequence jumps originate.

## Files changed and reasons for the changes
- Updated: `src/FFTBlock.jl`
  - Added FrameSync handoff tracking fields in context:
    - `fs_handoff_last_seq`, `fs_handoff_anomaly_count`.
  - Added probe helpers:
    - `is_framesync_sink` (type-name based sink detection)
    - `maybe_log_framesync_handoff!` (logs large seq jumps at handoff)
    - threshold `FRAME_SYNC_HANDOFF_JUMP_WARN=64`.
  - In worker dispatch loop, probe now logs when sending to FrameSync sink.
  - Reason: correlate anomalies before FrameSync enqueue with FrameSync internal probes.
- Updated: `HANDOVER.md`
  - Reason: recorded completed instrumentation task per AGENTS.md.

## Tasks to be addressed next
1. Run with both probes enabled and compare:
   - `FFTBlock: fs_handoff_seq_probe ...`
   - `ISDBTFrameSync: seq_probe where=enqueue/dequeue ...`
2. Confirm whether anomalies already exist at FFTBlock handoff stage.

## Known issues
- Probe adds diagnostic log only on large jumps; it does not change data flow behavior.

## Current progress status
- Evaluated `logs/eval_handoff_probe_515142857_20260212_103259.log` with combined probes (`FFTBlock` handoff + `ISDBTFrameSync` enqueue/dequeue).
- Confirmed large sequence jumps are already present at FFTBlock->FrameSync handoff and then mirrored at FrameSync enqueue/dequeue.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed probe-log analysis task per AGENTS.md.

## Tasks to be addressed next
1. Investigate source-side sequence generation before FrameSync handoff (likely FFTBlock output sequence assignment path under composed/no-seq cases).
2. Add probe at immediate upstream producer for FFTBlock input (SymbolSync->FFTBlock) to verify whether low/rewound seq enters FFTBlock.
3. If confirmed, enforce monotonic local sequence fallback in FFTBlock for FrameSync-facing branch when incoming sequence regresses.

## Known issues
- Probe counts in this run:
  - `FFTBlock: fs_handoff_seq_probe`: 36
  - `ISDBTFrameSync seq_probe enqueue`: 36
  - `ISDBTFrameSync seq_probe dequeue`: 64
- Representative paired events show same deltas at handoff and enqueue/dequeue (e.g., -38335/+38337), indicating origin before FrameSync internal queue.
- Continuity remains degraded (`PilotEQ in_mismatch=4027`, `ISDBTFrameSync in_mismatch=64`), though SymbolSync backpressure stays zero.

## Current progress status
- Added upstream boundary probe at `SymbolSync -> FFTBlock` ingress by instrumenting `FFTBlock.input!`.

## Files changed and reasons for the changes
- Updated: `src/FFTBlock.jl`
  - Added `in_seq_anomaly_count` to `FFTBlockContext`.
  - Added threshold constant `FFTBLOCK_INPUT_JUMP_WARN=64`.
  - Added `maybe_log_fftblock_input_probe!` and invoked it in `input!` for non-zero incoming seq.
  - New anomaly log format: `FFTBlock: in_seq_probe prev=... cur=... delta=... anomalies=...`.
  - Reason: identify whether large sequence regressions are already present before FFT processing.
- Updated: `HANDOVER.md`
  - Reason: recorded completed instrumentation task per AGENTS.md.

## Tasks to be addressed next
1. Run with combined probes and compare event ordering:
   - `FFTBlock: in_seq_probe`
   - `FFTBlock: fs_handoff_seq_probe`
   - `ISDBTFrameSync: seq_probe where=enqueue/dequeue`
2. Determine first-occurrence stage to isolate true origin of sequence rewinds.

## Known issues
- Probe is diagnostic-only and does not alter sequence assignment behavior.

## Current progress status
- Evaluated `logs/eval_full_seqprobe_515142857_20260212_103928.log` with three-stage sequence probes (`FFT in`, `FFT->FrameSync handoff`, `FrameSync enqueue/dequeue`).
- Determined first observed large jumps occur at FFTBlock->FrameSync handoff (not at FFTBlock input).

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed full-probe analysis task per AGENTS.md.

## Tasks to be addressed next
1. Implement FrameSync-facing monotonic-seq guard in FFTBlock handoff path (when output seq regresses, use local monotonic fallback for FrameSync branch).
2. Re-run probe log and verify reductions in:
   - `FFTBlock: fs_handoff_seq_probe`
   - `ISDBTFrameSync: seq_probe where=enqueue/dequeue`
   - `SeqTrace[ISDBTFrameSync] in_mismatch`.

## Known issues
- Probe counts in this run:
  - `FFTBlock in_seq_probe`: 0
  - `FFTBlock fs_handoff_seq_probe`: 35
  - `ISDBTFrameSync seq_probe enqueue`: 35
  - `ISDBTFrameSync seq_probe dequeue`: 49
- Representative events show handoff and enqueue/dequeue share same deltas, indicating origin at or before handoff assignment.
- Pilot continuity remains degraded (`PilotEQ in_mismatch=3986`).

## Current progress status
- Reviewed newly added `src/AGENTS.md` and captured updated FFTBlock policy.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded instruction update per AGENTS.md process.

## Tasks to be addressed next
1. Align upcoming FFTBlock sequence handling changes with new rule:
   - During frame composition, do not skip source-side sequence numbers.
2. Revisit current FrameSync-facing seq guard proposal to ensure it preserves source sequence continuity.

## Known issues
- Existing diagnostic proposal (monotonic guard for FrameSync handoff) must be reconciled with the new no-skip sequence policy.

## Current progress status
- Implemented FrameSync-facing sequence monotonic guard in `FFTBlock` handoff path.
- Guard is now applied before dispatch when FrameSync sink is attached.

## Files changed and reasons for the changes
- Updated: `src/FFTBlock.jl`
  - Added FrameSync handoff state fields:
    - `fs_handoff_last_raw_seq`, `fs_handoff_last_guard_seq`, `fs_handoff_anomaly_count`, `fs_handoff_guard_count`.
  - Added helper functions:
    - `find_framesync_sink` to detect FrameSync sink presence.
    - `guard_framesync_handoff_seq!` to:
      - keep large-jump probe logs (`fs_handoff_seq_probe`),
      - enforce monotonic guard (`raw_seq <= prev_guard` => `guarded = prev_guard + 1`),
      - emit guard logs (`fs_handoff_guard`).
  - Applied guard before `SeqTrace.set_seq!` and sink dispatch.
  - Reason: prevent backward/rewind sequence values from destabilizing FrameSync continuity.
- Updated: `HANDOVER.md`
  - Reason: recorded completed guard implementation per AGENTS.md.

## Tasks to be addressed next
1. Re-run with probes and verify guard effect:
   - decrease in `ISDBTFrameSync: seq_probe where=enqueue/dequeue`
   - decrease in `SeqTrace[ISDBTFrameSync] in_mismatch`.
2. Confirm guard activation counts via `FFTBlock: fs_handoff_guard ...` logs.

## Known issues
- Guard currently prioritizes monotonic continuity to FrameSync; raw source-seq anomalies are still logged for diagnosis.

## Current progress status
- Evaluated `logs/eval_handoff_guard_515142857_20260212_105851.log` and found it invalid for performance/continuity comparison due to startup failure.
- Fixed FFTBlock constructor argument mismatch introduced by recent guard-field additions.

## Files changed and reasons for the changes
- Updated: `src/FFTBlock.jl`
  - Added missing `UInt64(0)` initializer for newly added `fs_handoff_guard_count` field in `FFTBlockContext` construction.
  - Reason: resolve `MethodError` on `CreateFFTBlock` at startup.
- Updated: `HANDOVER.md`
  - Reason: recorded failure analysis and corrective fix per AGENTS.md.

## Tasks to be addressed next
1. Re-run guard evaluation log after constructor fix (previous guard log run is invalid).
2. Re-check counts for:
   - `FFTBlock: fs_handoff_guard`
   - `FFTBlock: fs_handoff_seq_probe`
   - `ISDBTFrameSync: seq_probe where=enqueue/dequeue`
   - `SeqTrace[ISDBTFrameSync] in_mismatch`.

## Known issues
- `logs/eval_handoff_guard_515142857_20260212_105851.log` terminated early with `MethodError` and must not be used for result comparison.

## Current progress status
- Evaluated `logs/eval_handoff_guard_515142857_20260212_105851.log` and confirmed it is invalid for guard effectiveness assessment.
- Verified current code loads with `julia --project=. -e 'using SignalFlow; println("load ok")'`.
- Confirmed `FFTBlock` contains handoff guard/probe paths (`fs_handoff_seq_probe`, `fs_handoff_guard`).

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: documented this completed/interrupt-equivalent task per AGENTS.md rules.

## Tasks to be addressed next
1. Re-run full demod with seq trace enabled to obtain a valid post-fix guard log.
2. Compare guard/probe counts between FFTBlock handoff and FrameSync enqueue/dequeue.
3. Decide whether to keep guard-only fix or additionally unify sequence source policy.

## Known issues
- `logs/eval_handoff_guard_515142857_20260212_105851.log` ends with `MethodError` at FFTBlock context construction, so it cannot be used as baseline/evaluation evidence.
- Worktree contains many unrelated modified/untracked files; care is needed to avoid accidental cross-changes.

## Current progress status
- Evaluated `logs/eval_handoff_guard_fix_515142857_20260212_110738.log` after FFTBlock handoff guard constructor fix.
- Confirmed run completed normally (`Shutdown complete.`) and FFT path processed all full frames (`full_frames=49651`, `out_frames=49651`, `sink_fail=0`).
- Quantified guard/probe effects at FFTBlock->FrameSync handoff and residual anomalies in FrameSync.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed log-evaluation task per AGENTS.md rules.

## Tasks to be addressed next
1. Investigate why `ISDBTFrameSync` still shows dequeue-only large jumps (6 events) while enqueue jump probe is clean.
2. Add temporary `ISDBTFrameSync` internal trace around dequeue path to capture ring index + seq continuity before/after `SeqTrace.log_in!`.
3. Reclassify `PilotEQ` sequence checks to non-strict (or stage-specific expected stride) because current `+14/+15` pattern appears systematic and not directly equivalent to sample loss.

## Known issues
- `FFTBlock` handoff raw jump events remain frequent (`fs_handoff_seq_probe=43`), and guard fired 23 times; this indicates upstream seq source switching persists.
- `ISDBTFrameSync` still has 6 large dequeue jump anomalies (and matching in/out mismatch pairs), so guard reduced but did not eliminate downstream discontinuity.
- `SeqTrace[PilotEQ] in_mismatch` appears in very high volume (~3875) with mostly `delta=14/15`, likely reflecting stage decimation/selection behavior rather than literal frame drops.

## Current progress status
- Added targeted root-cause tracing in `ISDBTFrameSync` to investigate residual dequeue-only sequence jumps.
- Build/load check passed after instrumentation (`julia --project=. -e 'using SignalFlow; println("load ok")'`).

## Files changed and reasons for the changes
- Updated: `src/ISDBTFrameSync.jl`
  - Added enqueue-seq shadow tracking per ring slot (`enq_seq_shadow`) and mismatch counter (`seq_shadow_mismatch_count`).
  - On dequeue, now logs `ISDBTFrameSync: seq_shadow_mismatch ...` when dequeued seq differs from enqueue-shadowed seq for the same slot.
  - On enqueue/dequeue, shadow slot is written/cleared deterministically.
  - Reason: isolate whether seq discontinuity is created before enqueue or while buffered.
- Updated: `HANDOVER.md`
  - Reason: recorded completed instrumentation task per AGENTS.md rules.

## Tasks to be addressed next
1. Run seq-trace command and check whether `ISDBTFrameSync: seq_shadow_mismatch` appears.
2. If mismatch appears: investigate cross-buffer metadata corruption path.
3. If mismatch does not appear: root cause is upstream sequence source switching; then tighten FFTBlock sequence-source policy.

## Known issues
- Residual FrameSync dequeue large jumps still exist in latest evaluated log; this patch is diagnostic-only and does not yet alter signal path behavior.

## Current progress status
- Completed root-cause investigation for residual FrameSync large sequence jumps using `logs/eval_framesync_shadowprobe_515142857_20260212_112224.log`.
- Root cause identified in `SeqTrace` metadata keying strategy, not in FrameSync ring enqueue/dequeue control flow itself.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: documented root-cause findings and evidence per AGENTS.md rules.

## Tasks to be addressed next
1. Replace `SeqTrace.BUFFER_SEQ` from `WeakKeyDict` to identity-based dictionary (`IdDict`) for mutable array keys.
2. Re-run shadow-probe log and verify:
   - `ISDBTFrameSync: seq_shadow_mismatch` becomes 0,
   - `ISDBTFrameSync: seq_probe where=dequeue` large-jump events disappear (or drop to non-metadata causes only).
3. If stable, remove temporary shadow-probe instrumentation from `ISDBTFrameSync`.

## Known issues
- In evaluated log:
  - `ISDBTFrameSync: seq_shadow_mismatch` = 7
  - `ISDBTFrameSync: seq_probe where=dequeue` = 14
  - `SeqTrace[ISDBTFrameSync] in_mismatch` = 14
  - This 7-to-14 pairing indicates each shadow mismatch produces a down-jump/up-jump pair.
- `src/SeqTrace.jl` currently uses `WeakKeyDict{Any,UInt64}` for mutable `Vector` keys. In Julia, `WeakKeyDict` keys are `isequal`-based (not identity-based), and mutable-array key mutation breaks dictionary invariants. This can return wrong/missing seq metadata and explains observed dequeue-only anomalies.

## Current progress status
- Fixed root-cause candidate in SeqTrace metadata mapping for mutable frame buffers.
- Verified project load after fix (`julia --project=. -e 'using SignalFlow; println("load ok")'`).

## Files changed and reasons for the changes
- Updated: `src/SeqTrace.jl`
  - Changed `BUFFER_SEQ` from `WeakKeyDict{Any,UInt64}` to `IdDict{Any,UInt64}`.
  - Reason: avoid `isequal`/mutable-array-key instability that can corrupt sequence metadata lookups.
- Updated: `HANDOVER.md`
  - Reason: recorded completed fix task per AGENTS.md rules.

## Tasks to be addressed next
1. Re-run `isdbt_demod.jl` with seq trace and shadow probe enabled.
2. Confirm `ISDBTFrameSync: seq_shadow_mismatch` count drops to zero (or near zero).
3. If confirmed, remove temporary shadow-probe instrumentation in `ISDBTFrameSync`.

## Known issues
- Full effectiveness is pending runtime log verification after this patch.

## Current progress status
- Investigated startup apparent stall (only startup prints after `start rxTask(ComplexF32)`) and fixed likely logging-path blocking race.
- `SignalFlow` load/precompile check passed after logger fix.

## Files changed and reasons for the changes
- Updated: `src/AsyncLogger.jl`
  - Added `queue_lock::ReentrantLock` to `LoggerContext`.
  - Reworked `queue_tryput!` to lock check+put (`isfull` + `put!`) atomically.
  - Updated callers to pass `LoggerContext` instead of raw `Channel`.
  - Reason: prevent producer race where concurrent logs pass fullness check and then block on `put!`, causing pipeline stall under heavy seq-trace logging.
- Updated: `HANDOVER.md`
  - Reason: recorded completed fix task per AGENTS.md rules.

## Tasks to be addressed next
1. Re-run the same command and verify periodic logs (`recv rate`, `dispatch sink lag`, etc.) appear during run.
2. If still appears silent, redirect to file and inspect `AsyncLogger: dropped_messages` and processing counters.
3. After stable run, re-check `seq_shadow_mismatch` and FrameSync dequeue anomalies.

## Known issues
- Under very high trace volume, logger may still drop messages by design; this is acceptable to preserve real-time path.

## Current progress status
- Evaluated `logs/eval_framesync_shadowprobe_515142857_20260212_120511.log` after `SeqTrace` IdDict fix and AsyncLogger race fix.
- Run completed normally (Ctrl-C handled, graceful stop).
- No residual FrameSync sequence-jump anomalies were observed in this run.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed log evaluation task per AGENTS.md rules.

## Tasks to be addressed next
1. Repeat one more run for reproducibility confirmation of zero sequence anomalies.
2. If reproducible, remove temporary `ISDBTFrameSync` shadow-probe instrumentation.
3. Continue constellation-quality improvement track with stable baseline.

## Known issues
- None observed in this specific run for sequence continuity checks.
- Note: `SeqTrace` mismatch logs only appear on anomaly; zero lines means continuity held (not that trace was disabled).

## Current progress status
- Evaluated `logs/eval_5min_seqtrace_515142857_20260212_121129.log` (5-minute run) after SeqTrace/AsyncLogger fixes.
- Confirmed graceful shutdown and stable long-run operation.
- Confirmed sequence continuity anomalies remained absent throughout this run.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed 5-minute log evaluation per AGENTS.md rules.

## Tasks to be addressed next
1. Remove temporary FrameSync shadow-probe instrumentation if no longer needed.
2. Continue constellation-quality improvements using this run as stable baseline.

## Known issues
- No sequence-jump anomalies detected in this run.
- Startup phase still shows low initial recv-rate sample(s), but steady-state is stable around ~8.0 MS/s.

## Current progress status
- Completed cleanup task: removed temporary FrameSync shadow-probe instrumentation from mainline.
- Verified project loads successfully after cleanup.

## Files changed and reasons for the changes
- Updated: `src/ISDBTFrameSync.jl`
  - Removed temporary diagnostic state fields: `enq_seq_shadow`, `seq_shadow_mismatch_count`.
  - Removed temporary dequeue mismatch log path: `ISDBTFrameSync: seq_shadow_mismatch ...`.
  - Removed enqueue/dequeue shadow bookkeeping assignments.
  - Kept regular sequence probes (`maybe_log_seq_jump!`) intact.
  - Reason: task 3 requested cleanup of investigation-only code to reduce side effects on mainline path.
- Updated: `HANDOVER.md`
  - Reason: recorded completed cleanup task per AGENTS.md rules.

## Tasks to be addressed next
1. Run a short seq-trace validation to ensure no behavioral regression after cleanup.
2. Continue constellation-improvement tasks (PilotEQ/PhaseSlope/CPE tuning).

## Known issues
- None added by this cleanup.

## Current progress status
- Implemented milestone-1 automation utilities to make the "2 consecutive 5-minute baseline runs" reproducible and pass/fail checkable.

## Files changed and reasons for the changes
- Added: `scripts/check_milestone1.sh`
  - Checks each log for milestone-1 criteria:
    - graceful shutdown marker present,
    - `SeqTrace[ISDBTFrameSync] in_mismatch` count == 0,
    - `ISDBTFrameSync: seq_probe where=dequeue` count == 0,
    - `FFTBlock input stats ... sink_fail=0`.
  - Outputs per-log PASS/FAIL and final milestone PASS/FAIL.
- Added: `scripts/run_milestone1.sh`
  - Runs two consecutive fixed-condition evaluations (default 300s each), saves logs, and invokes `check_milestone1.sh` automatically.
- Updated: `HANDOVER.md`
  - Reason: recorded completed milestone-support implementation per AGENTS.md rules.

## Tasks to be addressed next
1. Run `scripts/run_milestone1.sh` on target RF environment.
2. If FAIL, inspect printed criterion-specific reason and feed that stage back into tuning.

## Known issues
- Scripts are prepared and syntax-checked locally; end-to-end validation requires live RF hardware input.

## Current progress status
- Attempted to execute milestone-1 run/evaluation in this environment, but local Julia launcher is blocked by lockfile permission error before program start.
- Verified failure mode via probe logs (`m1_probe*.log`).

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded interrupted execution/evaluation task per AGENTS.md rules.

## Tasks to be addressed next
1. Run milestone command on user environment where Julia launcher lockfile is writable.
2. Evaluate with `bash scripts/check_milestone1.sh <log1> <log2>`.

## Known issues
- Local execution blocker: `The Julia launcher failed to load a configuration file. Could not create lockfile: Permission denied (os error 13).`

## Current progress status
- Fixed milestone automation script to continue to run2 after timeout-based run1 completion.

## Files changed and reasons for the changes
- Updated: `scripts/run_milestone1.sh`
  - Added `run_with_timeout()` wrapper.
  - Treated timeout-related return codes as acceptable for fixed-duration evaluation (`124`, `130`).
  - Kept non-timeout failures as hard errors.
  - Reason: previous `set -e` behavior aborted after run1 timeout, preventing run2 and final evaluation.
- Updated: `HANDOVER.md`
  - Reason: recorded interrupted-task fix per AGENTS.md rules.

## Tasks to be addressed next
1. Re-run milestone script and confirm both run1/run2 execute.
2. Verify final `MILESTONE-1: PASS/FAIL` is printed.

## Known issues
- End-to-end execution depends on user environment; this sandbox cannot run the RF capture path.

## Current progress status
- Milestone-1 target achieved by automation run.
- Two consecutive 5-minute runs completed and both passed criteria.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completion evidence per AGENTS.md rules.

## Tasks to be addressed next
1. Move to milestone-2 (phase correction stability metrics) using this baseline.
2. Keep `scripts/run_milestone1.sh` as regression gate for sequence continuity.

## Known issues
- None for milestone-1 criteria in this execution.

## Milestone-1 evidence
- Command:
  - `bash scripts/run_milestone1.sh 515.142857M ip:192.168.10.90 300`
- Logs:
  - `logs/m1_run1_515142857M_20260212_123445.log`
  - `logs/m1_run2_515142857M_20260212_123949.log`
- Results:
  - run1: PASS (`shutdown=1 mismatch=0 seqprobe=0 sink_fail=0`)
  - run2: PASS (`shutdown=1 mismatch=0 seqprobe=0 sink_fail=0`)
  - overall: `MILESTONE-1: PASS (all logs satisfy criteria)`

## Current progress status
- Started milestone-2 work (PilotEQ temporal stabilization) and implemented first-step smoothing in PilotEQ channel estimate path.
- Build/load check passed after change.

## Files changed and reasons for the changes
- Updated: `src/ISDBTPilotEqualizer.jl`
  - Added temporal smoothing state and parameter:
    - `temporal_alpha::Float32`
    - `h_prev::Vector{ComplexF32}`
    - `h_prev_valid::Vector{Bool}`
  - Added constructor kwarg `temporal_alpha` (default `0.2`, valid range `[0,1]`).
  - Changed pilot estimate update from raw overwrite to EMA blend per pilot bin:
    - `h_sm = h_prev + (h_raw - h_prev) * temporal_alpha` when previous estimate exists.
  - Reason: reduce symbol-to-symbol coefficient jitter in PilotEQ before downstream PhaseSlope/CPE.
- Updated: `HANDOVER.md`
  - Reason: recorded completed implementation task per AGENTS.md rules.

## Tasks to be addressed next
1. Run 5-minute validation and compare PilotEQ/PhaseSlope/CPE logs against pre-change baseline.
2. Tune `temporal_alpha` (e.g., 0.1 / 0.2 / 0.3) based on residual stability.

## Known issues
- Effectiveness not yet validated by fresh runtime logs after this patch.

## Current progress status
- Evaluated `logs/eval_piloteq_temporal_alpha02_515142857_20260212_131641.log` and found first PilotEQ temporal smoothing implementation regressed residual metrics.
- Implemented corrected smoothing strategy in PilotEQ: magnitude-only EMA with instantaneous phase passthrough.
- Build/load check passed after correction.

## Files changed and reasons for the changes
- Updated: `src/ISDBTPilotEqualizer.jl`
  - Changed temporal smoothing from complex-vector EMA to magnitude-only EMA:
    - smooth `|H|` over time,
    - keep current-frame phase from `h_raw` (fallback to previous phase only when raw magnitude is zero).
  - Reason: avoid phase-cancellation artifact from direct complex averaging across symbol-to-symbol phase rotation.
- Updated: `HANDOVER.md`
  - Reason: recorded evaluation and corrective implementation per AGENTS.md rules.

## Tasks to be addressed next
1. Re-run 5-minute evaluation and compare against:
   - `eval_piloteq_temporal_alpha02_515142857_20260212_131641.log`
   - pre-change baseline (`m1_run1_...`).
2. Decide whether `temporal_alpha=0.2` remains optimal or should be adjusted.

## Known issues
- First temporal EMA variant (complex-domain) is confirmed unsuitable for this signal path due to likely phase averaging loss.

## Current progress status
- Evaluated `logs/eval_piloteq_temporal_magema_alpha02_515142857_20260212_133612.log` after switching PilotEQ smoothing to magnitude-only EMA.
- Result: regression from complex-EMA variant was resolved; metrics returned to baseline-level stability.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed post-fix evaluation task per AGENTS.md rules.

## Tasks to be addressed next
1. Keep magnitude-only EMA path and tune `temporal_alpha` (0.1/0.2/0.3 sweep) for best constellation stability.
2. Proceed to milestone-2 next item: PhaseSlope update-gate review.

## Known issues
- None observed in this run for sequence continuity or residual blow-up.

## Evaluation summary (baseline vs temporal variants)
- Baseline (`m1_run1_...`):
  - `mean|H| n=10 mean=0.1932 min=0.1410 max=0.5890`
  - `residual_rms_deg(all) n=20 mean=0.00 min=0.00 max=0.00`
- Complex EMA (`eval_piloteq_temporal_alpha02_...131641`):
  - `mean|H| n=10 mean=0.1029 min=0.0370 max=0.6300`
  - `residual_rms_deg(all) n=20 mean=45.30 min=0.00 max=79.04`
- Magnitude EMA (`eval_piloteq_temporal_magema_alpha02_...133612`):
  - `mean|H| n=10 mean=0.2014 min=0.1440 max=0.6750`
  - `residual_rms_deg(all) n=20 mean=0.00 min=0.00 max=0.00`

## Current progress status
- Continued milestone-2 item 1 (PilotEQ temporal stabilization) by making temporal-alpha tunable at runtime and adding sweep automation.

## Files changed and reasons for the changes
- Updated: `examples/isdbt_demod.jl`
  - Added CLI option `--pilot-temporal-alpha <0..1>` (default 0.2).
  - Added range validation and startup print (`PilotEQ temporal_alpha: ...`).
  - Passed selected alpha into `ISDBTPilotEqualizer.CreateISDBTPilotEqualizer(...; temporal_alpha=...)`.
  - Reason: enable controlled A/B evaluation without code edits.
- Added: `scripts/run_piloteq_alpha_sweep.sh`
  - Runs 3 evaluations (`alpha=0.1/0.2/0.3`) for fixed duration and summarizes:
    - shutdown marker,
    - FrameSync seq-bad count,
    - `mean|H|` stats,
    - `residual_rms_deg` stats.
  - Reason: accelerate selection of stable temporal smoothing strength.
- Updated: `HANDOVER.md`
  - Reason: recorded completed implementation task per AGENTS.md rules.

## Tasks to be addressed next
1. Execute sweep on RF environment and choose best alpha.
2. Lock chosen alpha as baseline for milestone-2 step 2 (PhaseSlope gate review).

## Known issues
- Sweep script execution depends on RF capture availability; syntax is validated locally.

## Current progress status
- Fixed `run_piloteq_alpha_sweep.sh` bug that caused invalid filename parsing during summary phase.

## Files changed and reasons for the changes
- Updated: `scripts/run_piloteq_alpha_sweep.sh`
  - Changed run-progress/error prints to stderr.
  - Switched from command-substitution log capture to explicit `RUN_LOG` variable assignment.
  - Reason: avoid mixing progress text into log-path list used by `rg` in summary step.
- Updated: `HANDOVER.md`
  - Reason: recorded interrupted-task fix per AGENTS.md rules.

## Tasks to be addressed next
1. Re-run alpha sweep command and collect summary output.
2. Select best `pilot_temporal_alpha` from sweep results.

## Known issues
- None added after this script bugfix.

## Current progress status
- Evaluated PilotEQ alpha sweep logs (`alpha=0.1/0.2/0.3`) and compared stability metrics.
- Fixed sweep script summary bug (zero-match `rg` exit behavior under `set -e`).

## Files changed and reasons for the changes
- Updated: `scripts/run_piloteq_alpha_sweep.sh`
  - Made summary collection robust when match count is zero by redirecting `rg` errors and preserving pipeline output for `wc/awk`.
  - Reason: previous script terminated before summary output when seq anomaly count was zero.
- Updated: `HANDOVER.md`
  - Reason: recorded completed evaluation + script fix per AGENTS.md rules.

## Tasks to be addressed next
1. Keep `pilot_temporal_alpha=0.2` as current default candidate (balanced result).
2. Proceed to milestone-2 step 2: PhaseSlope update-gate review.

## Known issues
- Alpha sweep results are very close among 0.1/0.2/0.3 under current metrics; visual constellation comparison should be used as tie-breaker.

## Alpha sweep results
- `alpha=0.1` (`logs/eval_piloteq_alpha01_515142857M_20260212_141103.log`)
  - seq_bad=0, shutdown=1
  - mean|H|: mean=0.1983 (excl_max=0.1468)
  - residual_rms_deg(all): mean=0.00
- `alpha=0.2` (`logs/eval_piloteq_alpha02_515142857M_20260212_141608.log`)
  - seq_bad=0, shutdown=1
  - mean|H|: mean=0.1926 (excl_max=0.1476)
  - residual_rms_deg(all): mean=0.00
- `alpha=0.3` (`logs/eval_piloteq_alpha03_515142857M_20260212_142112.log`)
  - seq_bad=0, shutdown=1
  - mean|H|: mean=0.1909 (excl_max=0.1468)
  - residual_rms_deg(all): mean=0.00

## Current progress status
- Started milestone-2 step 2 (PhaseSlope update-gate review) with a first corrective patch.
- Fixed `PhaseSlope` update-state reporting so log `updated=` reflects real parameter change.

## Files changed and reasons for the changes
- Updated: `src/ISDBTPhaseSlopeCorrector.jl`
  - Changed `update_applied` semantics:
    - now `updated=true` only if `slope_delta != 0` or `intercept_delta != 0`.
  - Reason: previous code set `updated=true` whenever gate was open, even when both deltas were clipped to zero.
- Updated: `HANDOVER.md`
  - Reason: recorded completed step-2 patch per AGENTS.md rules.

## Tasks to be addressed next
1. Run a short/5-min validation and confirm `PhaseSlope ... updated=` transitions now match real correction activity.
2. If needed, continue gate tuning (fit_rms hysteresis / min-used-pilot requirement).

## Known issues
- This patch improves observability and diagnostics; gate policy itself is unchanged in this step.

## Current progress status
- Evaluated `logs/eval_phaseslope_updatedflag_515142857_20260212_144548.log` for PhaseSlope updated-flag correctness.
- Confirmed updated-flag patch works as intended for PhaseSlope logging.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed verification task per AGENTS.md rules.

## Tasks to be addressed next
1. Continue milestone-2 step 2 by tuning PhaseSlope gate policy (not just logging semantics).
2. Add a short metric summary line for PhaseSlope update activity if needed.

## Known issues
- In this run, `PhaseSlope` stayed at zero correction (`slope/intercept` unchanged), so no PhaseSlope update events were expected.

## Verification notes
- Run ended gracefully (`Interrupt received` -> `Shutdown complete`).
- PhaseSlope log lines observed with `gate=true` while `updated=false` (expected when deltas are zero).
- `updated=true` lines in this log were from CPE, not PhaseSlope.

## Current progress status
- Continued milestone-2 step 2 (PhaseSlope gate policy tuning) with fit-input quality gating.
- Added used-pilot sufficiency criteria to gate open/close logic.

## Files changed and reasons for the changes
- Updated: `src/ISDBTPhaseSlopeCorrector.jl`
  - Added gate parameters:
    - `min_used_pilots::Int` (default 18)
    - `min_used_ratio::Float64` (default 0.5)
  - Gate logic now requires both:
    - fit error condition (`fit_rms`) and
    - enough usable pilots (`used >= max(min_used_pilots, ceil(min_used_ratio * n))`).
  - Added `used_gate_min` to PhaseSlope diagnostic logs.
  - Reason: prevent gate enabling on weak/undersampled pilot conditions.
- Updated: `examples/isdbt_demod.jl`
  - Passed explicit PhaseSlope gate settings:
    - `min_used_pilots = 24`
    - `min_used_ratio = 0.65`
  - Reason: align runtime behavior with stricter gate policy for stable updates.
- Updated: `HANDOVER.md`
  - Reason: recorded completed implementation task per AGENTS.md rules.

## Tasks to be addressed next
1. Run validation and confirm `PhaseSlope` logs include `used_gate_min` and expected gate behavior.
2. Tune `min_used_pilots/min_used_ratio` if gate is too strict or too permissive.

## Known issues
- Needs runtime confirmation under RF input after gate-policy change.

## Current progress status
- Evaluated `logs/eval_phaseslope_gate_usedmin_515142857_20260212_145316.log` after PhaseSlope used-pilot gate policy change.
- Confirmed new gate-threshold telemetry (`used_gate_min`) is present and logic is functioning.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed gate-policy evaluation per AGENTS.md rules.

## Tasks to be addressed next
1. Move to milestone-2 step 3 (CPE confidence-linked update control).
2. Optionally tune PhaseSlope `min_used_pilots/min_used_ratio` only if future runs show over-gating.

## Known issues
- In this run, PhaseSlope correction remained near-zero with `updated=false` throughout PhaseSlope logs; this is expected under stable/low-error input but should be monitored for responsiveness under disturbed conditions.

## Evaluation summary
- Graceful stop confirmed (`Interrupt received` -> `Shutdown complete`).
- Sequence continuity anomalies: 0.
- PhaseSlope log samples: 10
  - `used_gate_min=24` present in all samples.
  - `gate=true` in 9/10 samples, `gate=false` in 1/10 (startup).
  - `updated=true` in 0/10 PhaseSlope samples (no actual parameter delta), matching updated-flag fix intent.

## Current progress status
- Implemented milestone-2 step 3 (`CPE` confidence-linked update control) with stronger confidence-to-step coupling.
- Build/load check passed after changes.

## Files changed and reasons for the changes
- Updated: `src/ISDBTCPECorrector.jl`
  - Added parameter/field `conf_gain_floor` (default `0.0`).
  - Replaced previous confidence scaling with on/off-threshold-aware ramp:
    - `conf <= min_update_conf_off` => `conf_gain = conf_gain_floor`
    - `conf >= min_update_conf_on` => `conf_gain = 1.0`
    - between thresholds => linear interpolation.
  - Changed `updated` semantics to `delta != 0.0` (actual phase change only).
  - Added `conf_gain` to CPE diagnostic logs.
  - Reason: prevent low-confidence frames from injecting disproportionate phase updates while preserving smooth recovery.
- Updated: `examples/isdbt_demod.jl`
  - Added runtime setting `conf_gain_floor = 0.05` for CPE block creation.
  - Reason: keep very small corrective motion under low confidence instead of hard freeze.
- Updated: `HANDOVER.md`
  - Reason: recorded completed step-3 implementation per AGENTS.md rules.

## Tasks to be addressed next
1. Run validation and inspect `CPE` logs (`conf`, `conf_gain`, `gate`, `updated`).
2. If stable, proceed to step 4 (final cross-stage integrity re-check).

## Known issues
- Runtime effectiveness of new confidence ramp is pending log validation under current RF conditions.

## Current progress status
- Evaluated `logs/eval_cpe_conf_ramp_515142857_20260212_231033.log` after CPE confidence-ramp implementation.
- Run completed gracefully with sequence continuity intact.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed log evaluation task per AGENTS.md rules.

## Tasks to be addressed next
1. Perform disturbed-condition validation (or lower-SNR capture) to exercise low-confidence branch (`conf < min_update_conf_on`).
2. Proceed to step 4 final cross-stage integrity re-check under both nominal and stressed conditions.

## Known issues
- In this run, `CPE conf` was always 1.0, so confidence-ramp suppression behavior was not exercised.

## Evaluation summary
- Graceful stop confirmed (`Interrupt received` -> `Shutdown complete`).
- Sequence continuity anomalies: 0 (`FrameSync` mismatch/probe and FFT handoff anomalies all absent).
- CPE log samples: 10
  - `conf=1.0` for all samples.
  - `conf_gain` was `1.0` in steady samples (startup sample had `gate=false`, `conf_gain=0.0`).
  - `updated=false` for all CPE samples (no phase delta needed under stable condition).

## Current progress status
- Addressed user request to reduce constellation rendering frequency using existing runtime option.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: documented operational guidance task completion per AGENTS.md rules.

## Tasks to be addressed next
1. Run with lower constellation update frequency and verify GUI responsiveness / mainline stability.

## Known issues
- None added; no code-path modification required because option already exists.

## Current progress status
- Interpreted runtime warning `ISDBTSymbolSync: sink_backpressure sink=SignalFlow.FFTBlock.FFTBlockContext{ComplexF32}` and summarized operational implications.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed analysis/support task per AGENTS.md rules.

## Tasks to be addressed next
1. Check frequency of `sink_backpressure` events in latest run logs.
2. If frequent/continuous, tune pipeline buffering and logging load.

## Known issues
- Occasional backpressure can occur transiently (startup/load spikes). Continuous backpressure indicates FFT downstream cannot drain at input rate.

## Current progress status
- Investigated startup hang/interrupt stacktrace and fixed connection-time sink queue saturation.
- Root cause: `append_block!` blocked on `put!(src.new_sinks, sink)` because `new_sinks` capacity was too small during graph wiring before workers drained control queues.

## Files changed and reasons for the changes
- Updated: 20 source files under `src/` (all blocks creating `new_sinks`)
  - Changed `new_sinks = Channel{SignalFlowBlock}(4)` to `new_sinks = Channel{SignalFlowBlock}(64)`.
  - Affected modules: `ADFMCOMMS2Src`, `BandSNREstimator`, `BinPowerMonitor`, `FFTBlock`, `GainBlock`, `ISDBT1SegSymbolSync`, `ISDBTCPECorrector`, `ISDBTDataCarrierExtractor`, `ISDBTFrameSync`, `ISDBTPhaseSlopeCorrector`, `ISDBTPilotEqualizer`, `ISDBTPilotExtractor`, `ISDBTSymbolSync`, `LPF`, `PilotCorrelationMonitor`, `RateMonitor`, `SignalStatsMonitor`, `TMCCDBPSKDecoder`, `WBFM`, `WBFMStereoDemod`.
  - Reason: prevent blocking during startup graph wiring while preserving runtime asynchronous sink-registration behavior.
- Updated: `HANDOVER.md`
  - Reason: recorded completed fix task per AGENTS.md rules.

## Tasks to be addressed next
1. Re-run `examples/isdbt_demod.jl` command and confirm startup progresses past graph wiring without repeated Ctrl-C.
2. Verify graceful SIGINT shutdown still works under the same command.

## Known issues
- This fix increases queue buffering for sink registration events; monitor memory only if block count grows significantly (current scale is safe).

## Current progress status
- Assessed feasibility of TMCC decoding based on latest runtime log and current execution options.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded analysis-only task completion per AGENTS.md rules.

## Tasks to be addressed next
1. Run with `--tmcc-dbpsk` enabled to obtain direct TMCC decode evidence.
2. Evaluate `TMCCDBPSK[...]` logs for lock/fine-lock and bit-stream stability.

## Known issues
- Latest evaluated log did not include TMCCDBPSK decoder outputs because TMCC decode path was not enabled.

## Current progress status
- Evaluated `logs/eval_tmcc_dbpsk_515142857_20260213_013038.log` for TMCC DBPSK decode readiness.
- TMCCDBPSK path is active and producing sync/bit output for both `norm` and `flip` branches.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completed TMCC decode evaluation task per AGENTS.md rules.

## Tasks to be addressed next
1. Improve TMCC lock persistence (reduce `sync_locked=false` events), especially on `flip` branch.
2. If target branch is known, prioritize one branch and tighten thresholds for that branch.
3. Validate decoded TMCC bitstream consistency over longer runs.

## Known issues
- TMCC sync lock is not perfectly stable yet; intermittent unlocks remain.

## Evaluation summary
- Graceful stop: confirmed (`Interrupt received` -> `Shutdown complete`).
- `TMCCDBPSK[norm]`: total=9, `sync_locked=true`=8, `sync_fine_locked=true`=5, `sync_locked=false`=1.
- `TMCCDBPSK[flip]`: total=9, `sync_locked=true`=7, `sync_fine_locked=true`=5, `sync_locked=false`=2.
- Confidence/score snapshots:
  - norm: `conf` mean≈0.683 (min 0.553, max 0.868), `sync_score` max 0.688.
  - flip: `conf` mean≈0.597 (min 0.275, max 0.876), `sync_score` max 0.812.

## Current progress status
- Reviewed current implementation status of SP phase correction path (`PilotEQ -> PhaseSlope -> CPE`) and runtime wiring in `examples/isdbt_demod.jl`.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded status-audit task completion per AGENTS.md rules.

## Tasks to be addressed next
1. Continue milestone-2 stabilization with emphasis on temporal smoothness under low-SNR windows.
2. Re-run long capture with current thresholds to quantify residual unlock/update-chatter rates.

## Known issues
- TMCC decode path is active but lock persistence still fluctuates; SP correction quality under degraded SNR remains a key risk.

## Current progress status
- Added an EVM evaluation block implementation and wired it into `examples/isdbt_demod.jl` via runtime options.
- Block is non-intrusive by design (drops under pressure instead of backpressuring mainline).

## Files changed and reasons for the changes
- Added: `src/ISDBTEVMMonitor.jl`
  - Reason: implemented decision-directed EVM/MER monitor for `QPSK/16QAM/64QAM` with gain/phase fit and periodic logging.
- Updated: `src/SignalFlow.jl`
  - Reason: included `ISDBTEVMMonitor` module in the build.
- Updated: `examples/isdbt_demod.jl`
  - Reason: added runtime options `--evm`, `--evm-mod`, `--evm-log-interval`, and connected EVM monitor to `data_carriers` output.
- Updated: `HANDOVER.md`
  - Reason: recorded completion details per AGENTS.md rules.

## Tasks to be addressed next
1. Run field validation on 515.142857 MHz and verify EVM trend stability over 5+ minutes.
2. Correlate EVM changes with PhaseSlope/CPE `updated` and confidence logs.
3. If needed, add percentile metrics (e.g., EVM95) for better outlier sensitivity.

## Known issues
- EVM monitor requires correct modulation selection (`--evm-mod`); wrong modulation produces non-meaningful EVM values.

## Current progress status
- Enumerated and reported the current block connections in `examples/isdbt_demod.jl`, including conditional branches (`diag`, rate monitors, TMCC DBPSK, EVM, views).

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completion of block-connection reporting task per AGENTS.md rules.

## Tasks to be addressed next
1. If needed, add a runtime `--print-graph` option using `SignalFlow.flow_graph_snapshot()` for automatic connection dumps.

## Known issues
- Connection view is currently code-derived; there is no dedicated runtime graph printer in `isdbt_demod.jl` yet.

## Current progress status
- Clarified `data_carriers` scope in current wiring/configuration: output is segment-0 only (1seg), not 12seg nor 12seg+1seg.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completion of `data_carriers` scope clarification task per AGENTS.md rules.

## Tasks to be addressed next
1. If full-band decoding is needed, design multi-segment extractor path and segment-index mapping for all 13 segments.

## Known issues
- Current extractor configuration (`segment_carriers=432`, `segment_index=0`) limits output to seg0.

## Current progress status
- Analyzed likely causes of unstable/non-clustered QPSK constellation display in current ISDB-T pipeline and prioritized root-cause candidates.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded completion of troubleshooting analysis task per AGENTS.md rules.

## Tasks to be addressed next
1. Verify whether seg0 payload modulation is actually QPSK (TMCC decode result).
2. Correlate constellation instability with `FrameSync lock`, `PhaseSlope gate/updated`, and `CPE conf/gate/updated`.
3. If needed, relax/tune gate hysteresis and confidence thresholds for low-SNR windows.

## Known issues
- If modulation assumption is wrong (e.g., actual 16QAM/64QAM), QPSK-like stable clusters will not appear by design.

## Current progress status
- Assessed TMCC decode milestones using existing run results/log references and summarized current attainment level by stage.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded milestone attainment assessment task completion per AGENTS.md rules.

## Tasks to be addressed next
1. Advance milestone-4 by improving TMCC lock persistence (`sync_locked`/`fine_locked` hold time).
2. Establish bitstream-level validity criteria for milestone-6 and run long-duration validation.

## Known issues
- Milestones 6-7 remain blocked by intermittent TMCC unlock events and insufficient long-run consistency evidence.

## Current progress status
- Implemented a new `AWGNInjector` block for software low-SNR emulation.
- Verified package load and block create/update/stop path.

## Files changed and reasons for the changes
- Added: `src/AWGNInjector.jl`
  - Reason: added frame-based AWGN injection block with configurable `snr_db`, optional stats logging, and sequence trace continuity.
- Updated: `src/SignalFlow.jl`
  - Reason: included `AWGNInjector` module in package load path.
- Updated: `Project.toml`
  - Reason: added stdlib dependency `Random` required by `AWGNInjector`.
- Updated: `HANDOVER.md`
  - Reason: recorded task completion details per AGENTS.md rules.

## Tasks to be addressed next
1. Wire `AWGNInjector` into `examples/isdbt_demod.jl` via CLI options (e.g., `--awgn-snr-db`) when requested.
2. Run low-SNR sweeps and correlate TMCC lock persistence / EVM / gate chatter against injected SNR.

## Known issues
- Current `AWGNInjector` exists as a reusable block but is not yet wired into `isdbt_demod.jl` by default.

## Current progress status
- Added `examples/awgn_test.jl` to measure SNR before/after AWGN injection.
- Added labeled logging support to `BandSNREstimator` so before/after streams are distinguishable in logs.
- Verified runtime behavior with `timeout -s INT` and confirmed graceful shutdown path.

## Files changed and reasons for the changes
- Added: `examples/awgn_test.jl`
  - Reason: provide a standalone AWGN test program with topology:
    `SyntheticIQSource -> BandSNREstimator(before)` and `SyntheticIQSource -> AWGNInjector -> BandSNREstimator(after)`.
  - Includes CLI options for `--snr-db`, `--samplerate`, `--frame-size`, signal/noise band settings, and log interval.
- Updated: `src/BandSNREstimator.jl`
  - Reason: added optional `label` parameter and labeled log output:
    `BandSNREstimator[<label>]: snr=... dB`.
- Updated: `HANDOVER.md`
  - Reason: recorded task completion details per AGENTS.md rules.

## Tasks to be addressed next
1. If needed, wire `AWGNInjector` into `examples/isdbt_demod.jl` via runtime option (e.g., `--awgn-snr-db`) for end-to-end low-SNR tests.
2. Add optional summary output in `awgn_test.jl` (moving average before/after SNR and delta).

## Known issues
- `awgn_test.jl` uses multitone synthetic input; absolute post-AWGN SNR value depends on selected signal/noise bands and is not guaranteed to equal the configured `--snr-db` exactly.

## Current progress status
- Evaluated `logs/eval_awgn_test_20260213_202818.log` from `examples/awgn_test.jl`.
- Confirmed continuous before/after SNR logging and graceful shutdown behavior.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded AWGN test log evaluation results per AGENTS.md rules.

## Tasks to be addressed next
1. Add optional summary statistics output in `awgn_test.jl` (mean/std for before/after SNR and delta).
2. If target is matching configured `--snr-db` numerically, calibrate signal/noise bands or change source waveform for estimator consistency.

## Known issues
- In this run, injected setting was `snr_db=12.0`, while `after_awgn` estimator reported about `17.33 dB` mean; this is expected with current multitone/band-definition setup and should not be interpreted as injector malfunction.

## Current progress status
- Advanced milestone-2 (FrameSync stabilization) by adding unlock-suppression guard during reference-hold phases.
- Added visibility of suppression activity to periodic FrameSync logs.

## Files changed and reasons for the changes
- Updated: `src/ISDBTFrameSync.jl`
  - Reason: suppress unlock counter progression while `ref_update_hold` / warmup / outlier recovery is active, reducing low-SNR lock chatter.
  - Reason: added `unlock_suppressed_count` runtime counter and exposed it in periodic `corr/ema` log lines for traceability.
- Updated: `HANDOVER.md`
  - Reason: recorded milestone-2 stabilization change per AGENTS.md rules.

## Tasks to be addressed next
1. Run 5-min evaluation with current settings and compare `unlock` event count before/after this change.
2. If unlocks still persist, tune `lock_confirm/unlock_confirm` and `metric_guard_band` with fixed command sweep.

## Known issues
- Guard improves resilience to transient hold phases, but sustained deep fades can still trigger legitimate unlocks.

## Current progress status
- Evaluated `logs/eval_framesync_unlockguard_515142857_20260213_204212.log` after FrameSync unlock-guard change.
- Run indicates stable lock retention with no unlock/outlier/resync events in the measured interval.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded FrameSync unlock-guard evaluation results per AGENTS.md rules.

## Tasks to be addressed next
1. Run longer-duration validation (15-30 min) to confirm no regressions under varying channel conditions.
2. Stress test with software low-SNR injection (`AWGNInjector`) and verify `unlock_suppressed` behavior during degraded windows.

## Known issues
- In this particular run, `unlock_suppressed` remained zero because unlock-side guard conditions were not exercised; dedicated low-SNR stress is still required.

## Current progress status
- Enabled low-SNR stress testing on the full `isdbt_demod.jl` path by adding optional AWGN insertion before SymbolSync.
- Added a sweep script to run multiple AWGN SNR points and collect logs automatically.

## Files changed and reasons for the changes
- Updated: `examples/isdbt_demod.jl`
  - Reason: added CLI options `--awgn-snr-db` and `--awgn-log-interval`.
  - Reason: inserted optional path `rfsrc -> awgn -> (mon_sync) -> sync` while keeping original path unchanged when AWGN is disabled.
- Added: `scripts/run_framesync_lowsnr_sweep.sh`
  - Reason: automate low-SNR stress runs (multiple SNR points, redirected logs).
- Updated: `HANDOVER.md`
  - Reason: recorded task completion details per AGENTS.md rules.

## Tasks to be addressed next
1. Run sweep at `12/10/8/6 dB` and compare FrameSync `unlock`, `forced_resync`, and `unlock_suppressed`.
2. If instability starts at specific SNR, tune `lock_confirm/unlock_confirm` and `metric_guard_band` around that boundary.

## Known issues
- Quick local command check with `timeout` occasionally printed Julia SIGINT backtrace during precompile/interrupt timing race; this is unrelated to FrameSync logic and should be ignored for log-based evaluation.

## Current progress status
- Analyzed low-SNR sweep logs (`..._20260213_210625`) and identified invalid test condition: AWGN path dropped all frames due frame-size mismatch.
- Fixed AWGN insertion in `isdbt_demod.jl` to use source actual chunk frame size.

## Files changed and reasons for the changes
- Updated: `examples/isdbt_demod.jl`
  - Reason: changed AWGNInjector `frame_size` from fixed `SyncFrameSize` to runtime `src_frame_size = length(rfsrc.ringbuffer.bufs[1].buf)` to match upstream frame granularity.
- Updated: `HANDOVER.md`
  - Reason: recorded root cause and fix per AGENTS.md rules.

## Tasks to be addressed next
1. Re-run low-SNR sweep after this fix and re-evaluate FrameSync metrics.
2. Confirm FFT path is active (`FFTBlock input stats total_samples > 0`) in each new log.

## Known issues
- Previous sweep logs with timestamp `20260213_210625` are invalid for FrameSync evaluation because AWGN mismatch caused zero downstream samples.

## Current progress status
- Evaluated valid low-SNR AWGN sweep logs (`..._20260213_215430`) for SNR points `12/10/8/6 dB`.
- Confirmed FFT path remained active in all runs and FrameSync retained lock without unlock/outlier/resync events.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded low-SNR sweep evaluation results per AGENTS.md rules.

## Tasks to be addressed next
1. Extend stress sweep to lower points (`4/2/0 dB`) to intentionally trigger unlock-side guard behavior.
2. If still no unlock/outlier events, increase stress by adding CFO/phase impairment or stricter thresholds to expose boundary behavior.

## Known issues
- In `12/10/8/6 dB` runs, `unlock_suppressed` stayed `0`; unlock-guard path was not exercised because FrameSync remained stable.

## Current progress status
- Evaluated additional low-SNR sweep logs (`..._20260213_225952`) at `4/2/0 dB`.
- FrameSync remained locked across all three runs without unlock/outlier/forced-resync events.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded latest low-SNR sweep evaluation and boundary status per AGENTS.md rules.

## Tasks to be addressed next
1. Extend stress below `0 dB` (e.g., `-2/-4/-6 dB`) to force unlock-side behavior.
2. Add a stricter stress profile by tightening FrameSync thresholds (`unlock_threshold` up, `unlock_confirm` down) in a dedicated eval branch if unlock still does not appear.

## Known issues
- Even at `0 dB` injected AWGN in current setup, `unlock_suppressed` did not increase; unlock-guard path remains unexercised.

## Current progress status
- Clarified why FrameSync did not unlock under `0 dB` AWGN stress in current configuration.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded root-cause explanation task completion per AGENTS.md rules.

## Tasks to be addressed next
1. Run harsher stress (`-2/-4/-6 dB`) and check whether EMA metric crosses unlock gate.
2. Optionally add diagnostic logging of raw unlock-gate crossings (`metric <= unlock_threshold - guard`) count per interval.

## Known issues
- Current unlock gate is relatively permissive for stability (`unlock_threshold=0.2`, guard `0.03`, `unlock_confirm=20`), so AWGN-only degradation may not trigger unlock.

## Current progress status
- Verified whether data-valid frames are being produced under low-SNR runs (`4/2/0 dB`).
- Confirmed FFT/PilotEQ/CPE/DataCarrierExtractor all processed frames continuously with no mismatch/backpressure anomalies in logs.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded data-valid-frame verification results per AGENTS.md rules.

## Tasks to be addressed next
1. Add payload-level validity check (TMCC/TS CRC or BER proxy), because current confirmation is pipeline/frame-continuity level.
2. Keep collecting boundary logs where FrameSync unlock actually appears to validate unlock-suppression path.

## Known issues
- Current evidence proves frame flow continuity and processing activity, but does not yet prove transport payload correctness end-to-end.

## Current progress status
- Assessed GI-correlation state under low-SNR logs by comparing `ISDBTSymbolSync peak` distribution against configured lock/unlock thresholds.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded GI-correlation assessment task completion per AGENTS.md rules.

## Tasks to be addressed next
1. If needed, add percentile logging for `ISDBTSymbolSync peak` (e.g., p1/p50/p99) to track guard margin over long runs.

## Known issues
- `peak` occasionally dips near the lock threshold, but remains mostly above unlock threshold; this can preserve lock while reducing margin.

## Current progress status
- Computed GI-correlation (`ISDBTSymbolSync peak`) summary for `12/10/8/6 dB` AWGN sweep logs.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded requested per-SNR GI-correlation summary task completion per AGENTS.md rules.

## Tasks to be addressed next
1. Compare `12/10/8/6 dB` and `4/2/0 dB` peak distributions in one table to estimate practical unlock boundary.

## Known issues
- Peak minima can be close to lock threshold transiently; distribution-based monitoring is preferable to single-point checks.

## Current progress status
- Evaluated additional stress sweep logs (`..._20260213_231929`) for `-2/-4/-6 dB` AWGN.
- FrameSync still retained lock with no unlock/outlier/forced-resync events in all three runs.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded `-2/-4/-6 dB` evaluation and implications per AGENTS.md rules.

## Tasks to be addressed next
1. Introduce harsher impairment beyond AWGN (e.g., CFO/phase-jump injection) or tighten FrameSync unlock thresholds in test mode.
2. Add direct unlock-gate crossing counters to FrameSync logs to quantify near-miss behavior.

## Known issues
- AWGN-only stress (down to `-6 dB`) is still insufficient to trigger unlock in current FrameSync parameterization.

## Current progress status
- Extracted GI-correlation (`ISDBTSymbolSync peak`) summaries from low-SNR sweep logs and reported per-SNR mean/min/max.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded GI-correlation extraction task completion per AGENTS.md rules.

## Tasks to be addressed next
1. Track GI peak percentile trend (p1/p50/p99) across SNR sweep for clearer boundary visualization.

## Known issues
- Multiple runs exist for `4 dB`; compare by timestamp when interpreting GI-correlation values.

## Current progress status
- Clarified relationship between GI correlation metric (`ISDBTSymbolSync peak`) and FrameSync lock/unlock behavior.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded explanation task completion per AGENTS.md rules.

## Tasks to be addressed next
1. If desired, add cross-metric diagnostic logging that prints SymbolSync peak and FrameSync corr/ema in the same interval.

## Known issues
- Current architecture uses separate lock domains (SymbolSync metric vs FrameSync TMCC correlation), so one metric can degrade while the other still holds lock.

## Current progress status
- Extracted TMCC correlation metrics (`ISDBTFrameSync corr/ema`) from low-SNR sweep logs and summarized per-SNR statistics.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded TMCC correlation extraction task completion per AGENTS.md rules.

## Tasks to be addressed next
1. Plot `corr/ema` vs SNR and compare against unlock gate (`unlock_threshold - metric_guard_band`) to visualize margin.

## Known issues
- Multiple timestamps exist for `4 dB`; interpret TMCC correlation values with file timestamp context.

## Current progress status
- Explained why TMCC correlation metric can remain relatively high even when injected AWGN SNR is degraded.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded explanatory support task completion per AGENTS.md rules.

## Tasks to be addressed next
1. If requested, add a companion metric without strong EMA smoothing to observe raw correlation volatility under low SNR.

## Known issues
- Correlation metric interpretation is influenced by normalization and EMA; it does not map linearly to user-facing SNR numbers.

## Current progress status
- Assessed appropriateness of raising FrameSync `unlock_threshold` to `0.45` for stress testing.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded configuration-guidance task completion per AGENTS.md rules.

## Tasks to be addressed next
1. If requested, run threshold sweep (`0.30/0.35/0.40/0.45`) and compare unlock/forced_resync impact.

## Known issues
- `unlock_threshold=0.45` is likely too strict for production stability; suitable mainly for debug/stress boundary exploration.

## Current progress status
- Documented rationale for using a staged unlock-threshold sweep (`0.30 -> 0.35 -> 0.40 -> 0.45`) instead of a single-step jump.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded decision rationale task completion per AGENTS.md rules.

## Tasks to be addressed next
1. Execute threshold sweep and identify first threshold where unlock/resync behavior appears.

## Known issues
- Single-point threshold jumps can conflate boundary detection and over-tuning risk; staged sweep improves interpretability.

## Current progress status
- Implemented unlock-threshold sweep support for requested values `0.25 -> 0.30 -> 0.35`.
- Added CLI override for FrameSync unlock threshold and an automation script for the 3-point sweep.

## Files changed and reasons for the changes
- Updated: `examples/isdbt_demod.jl`
  - Reason: added `--framesync-unlock-threshold` argument and wired it into `ISDBTFrameSyncLockConfig(unlock_threshold=...)`.
  - Reason: print effective FrameSync unlock threshold at startup for traceability.
- Added: `scripts/run_framesync_unlockth_sweep.sh`
  - Reason: automate requested 3-point threshold sweep with AWGN stress in one command.
- Updated: `HANDOVER.md`
  - Reason: recorded task completion details per AGENTS.md rules.

## Tasks to be addressed next
1. Run the 3-point sweep and compare `unlock`, `forced_resync`, and `unlock_suppressed` across thresholds.
2. Choose the minimum threshold that exposes unlock behavior without excessive false unlock in nominal SNR.

## Known issues
- `--framesync-unlock-threshold` is a test/control knob; production default remains `0.2`.

## Current progress status
- Evaluated unlock-threshold sweep at requested points (`0.25/0.30/0.35`) under `AWGN=-6 dB`.
- Found first observable unlock-guard activation at `unlock_threshold=0.35` (`unlock_suppressed` increments), while `unlock` remained zero.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded threshold-sweep evaluation results and boundary finding per AGENTS.md rules.

## Tasks to be addressed next
1. Probe near-boundary values (`0.36/0.38/0.40`) to determine onset of actual unlock events.
2. Optionally reduce `unlock_confirm` in test mode to expose unlock transitions without over-tightening threshold.

## Known issues
- At current test settings, threshold changes mainly affect guard suppression counts; actual unlock was not triggered in this sweep.

## Current progress status
- Implemented next-step boundary test support after `0.25/0.30/0.35` sweep.
- Added control of `FrameSync unlock_confirm` from CLI and prepared boundary sweep script for `0.36/0.38/0.40`.

## Files changed and reasons for the changes
- Updated: `examples/isdbt_demod.jl`
  - Reason: added CLI option `--framesync-unlock-confirm` and connected it to FrameSync lock config.
  - Reason: startup print now includes both unlock threshold and unlock confirm for traceability.
- Added: `scripts/run_framesync_unlockth_boundary.sh`
  - Reason: automate next boundary sweep (`0.36/0.38/0.40`) with configurable AWGN SNR and unlock_confirm.
- Updated: `HANDOVER.md`
  - Reason: recorded completion details per AGENTS.md rules.

## Tasks to be addressed next
1. Run boundary sweep with current `unlock_confirm=20`.
2. If unlock still absent, rerun with lower `unlock_confirm` (e.g. 12 or 8) to expose transition.

## Known issues
- New CLI control is test-oriented; production behavior should keep conservative defaults unless re-validated.

## Current progress status
- Evaluated boundary sweep (`unlock_threshold=0.36/0.38/0.40`, `unlock_confirm=20`, `AWGN=-6 dB`).
- Observed first actual unlock event at `unlock_threshold=0.40` (with relock), while `0.36/0.38` remained no-unlock.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded boundary sweep evaluation outcome per AGENTS.md rules.

## Tasks to be addressed next
1. Run same boundary points with lower `unlock_confirm` (e.g., 12) to increase sensitivity and map unlock frequency.
2. Compare unlock/relock rate and `unlock_suppressed` growth across confirm values.

## Known issues
- At `unlock_threshold=0.40`, unlock appears but remains sparse with `unlock_confirm=20`; transition region is narrow and confirmation-length sensitive.

## Current progress status
- Evaluated repeat run (`..._20260214_003848`) for boundary points `0.36/0.38/0.40` at `AWGN=-6 dB`, `unlock_confirm=20`.
- Confirmed boundary behavior reproducibility: `0.40` again produced actual unlock/relock events; `0.36/0.38` did not.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded repeated-boundary-run evaluation results per AGENTS.md rules.

## Tasks to be addressed next
1. Run boundary sweep with `unlock_confirm=12` to quantify unlock frequency sensitivity.
2. Compare event rates (`unlock`, relock count, unlock_suppressed growth) between confirm 20 and 12.

## Known issues
- Boundary region shows run-to-run variance in suppression counts, but unlock onset at `0.40` is repeatable under current stress setting.

## Current progress status
- Updated milestone attainment estimate for constellation-display readiness based on latest low-SNR sweep and pipeline stability logs.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded milestone attainment reporting task completion per AGENTS.md rules.

## Tasks to be addressed next
1. Run unlock-threshold sweep (`0.25/0.30/0.35`) and integrate results into milestone-3/4 confidence.
2. Re-enable constellation view in a controlled run and score visual stability milestone (7/8).

## Known issues
- Visual-quality milestones (7/8) are still estimate-based until direct constellation-on runs are evaluated.

## Current progress status
- Re-evaluated boundary sweep logs from user run (`..._20260214_003848`) at `AWGN=-6 dB`, `unlock_confirm=20`.
- Quantified event counts from raw logs:
  - `unlock_th=0.36`: `lock=1`, `unlock=0`
  - `unlock_th=0.38`: `lock=1`, `unlock=0`
  - `unlock_th=0.40`: `lock=3`, `unlock=2`
- Data path integrity remained normal in all 3 runs (`FFTBlock sink_fail=0`, graceful shutdown).

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded concrete lock/unlock counts and integrity checks for the latest boundary-run evaluation per AGENTS.md rules.

## Tasks to be addressed next
1. Execute boundary sweep with `unlock_confirm=12` to map how often `unlock_th=0.38` starts to unlock.
2. Select operating point by balancing false-lock persistence vs unlock chattering under low-SNR stress.

## Known issues
- Event summary line format is not standardized in logs, so event counts currently require pattern-based extraction (`lock corr` / `unlock corr`).

## Current progress status
- Evaluated boundary sweep at `AWGN=-6 dB`, `unlock_confirm=12` using logs timestamped `20260214_023607`.
- Extracted lock/unlock counts:
  - `unlock_th=0.36`: `lock=3`, `unlock=2`
  - `unlock_th=0.38`: `lock=6`, `unlock=6`
  - `unlock_th=0.40`: `lock=8`, `unlock=8`
- Compared against prior `unlock_confirm=20` (`20260214_003848`):
  - `0.36`: `1/0` -> `3/2`
  - `0.38`: `1/0` -> `6/6`
  - `0.40`: `3/2` -> `8/8`
- Interpretation: decreasing confirm from 20 to 12 significantly increases unlock/relock chattering under `-6 dB` stress.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded latest boundary evaluation and confirm-sensitivity comparison per AGENTS.md rules.

## Tasks to be addressed next
1. Select operating region by objective:
   - stability-priority: keep `unlock_confirm=20` and choose threshold around `0.38`.
   - responsiveness-priority: use lower confirm but add hysteresis/hold to suppress chatter.
2. Validate chosen setting at non-stress SNR (`0/6/12 dB`) to avoid overfitting to `-6 dB`.

## Known issues
- Current logs do not emit a single consolidated FrameSync summary line (`lock/unlock/forced_resync`) at shutdown; event counting still relies on pattern extraction.

## Current progress status
- Reported current milestone status and attainment after completing FrameSync boundary comparison (`unlock_confirm=20` vs `12`).
- Updated attainment view to reflect that low-SNR unlock boundary is now identified, while TMCC persistence and visual constellation stability remain pending.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: logged milestone-status reporting task completion per AGENTS.md rules.

## Tasks to be addressed next
1. Fix FrameSync operating point (stability-first candidate: `unlock_th~0.38`, `unlock_confirm=20`) and validate at `AWGN 0/6/12 dB`.
2. Re-enable constellation and score visual stability with the fixed FrameSync setting.
3. Continue TMCC lock persistence improvement and bitstream consistency checks.

## Known issues
- Constellation stability milestone still depends on direct `--constellation` evaluation runs.

## Current progress status
- Implemented tooling to execute milestone-2 remaining work: fixing FrameSync operating point and re-validating under normal SNR levels.
- Added an automated run script for the fixed operating point and a checker script for pass/fail criteria.

## Files changed and reasons for the changes
- Added: `scripts/run_framesync_operatingpoint_validate.sh`
  - Reason: run fixed FrameSync settings (`unlock_th`, `unlock_confirm`) across normal-SNR AWGN points (default `12/6/0 dB`) with unified logging.
- Added: `scripts/check_framesync_operatingpoint.sh`
  - Reason: automatically evaluate each log with milestone-2 criteria (`lock>=1`, `unlock=0`, `forced/outlier=0`, `sink_fail=0`, graceful shutdown).
- Updated: `HANDOVER.md`
  - Reason: recorded completion of milestone-2 execution tooling task per AGENTS.md rules.

## Tasks to be addressed next
1. Run `scripts/run_framesync_operatingpoint_validate.sh` on target RF source and collect verdict.
2. If any run fails due to unlocks, adjust operating point slightly (e.g., threshold down to `0.36` while keeping `unlock_confirm=20`) and re-run.
3. After pass, proceed to constellation-on visual verification with the fixed setting.

## Known issues
- Checker relies on current log patterns (`lock corr`, `unlock corr`, `FFTBlock input stats`, `Shutdown complete`); if log format changes, parser updates will be needed.

## Current progress status
- Diagnosed latest operating-point validation failure (`20260214_025701`) as startup/connectivity failure, not FrameSync behavior.
- All three logs terminated at startup with `AssertionError: iio_context* null pointer` before `start rxTask`, so milestone-2 re-validation did not actually run.
- Improved checker to classify such cases as `INVALID` instead of misreporting as metric-based `FAIL`.

## Files changed and reasons for the changes
- Updated: `scripts/check_framesync_operatingpoint.sh`
  - Reason: added startup failure detection (`iio_context* null pointer`, missing `start rxTask`) and explicit `[INVALID]` reporting.
- Updated: `HANDOVER.md`
  - Reason: recorded diagnosis and checker improvement per AGENTS.md rules.

## Tasks to be addressed next
1. Re-run operating-point validation after restoring SDR/libiio connectivity (same command is fine).
2. Once logs contain runtime section (`start rxTask` .. `Shutdown complete`), evaluate M2 criteria.

## Known issues
- Current run timestamp `20260214_025701` cannot be used for M2 attainment judgement due to early startup failure.

## Current progress status
- Evaluated operating-point validation logs (`20260214_114539`) for `unlock_th=0.36`, `unlock_confirm=20` at normal-SNR points (`12/6/0 dB`).
- Result:
  - `12 dB`: `lock=17`, `unlock=16` (FAIL)
  - `6 dB`: `lock=5`, `unlock=4` (FAIL)
  - `0 dB`: `lock=1`, `unlock=0` (PASS)
- Interpretation: `unlock_th=0.36` is too strict for stability objective; false unlock/relock chattering occurs even at higher SNR points.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded operating-point evaluation result and conclusion per AGENTS.md rules.

## Tasks to be addressed next
1. Re-run operating-point validation with lower threshold (`unlock_th=0.30`, `unlock_confirm=20`) across `12/6/0 dB`.
2. If needed, confirm margin with `unlock_th=0.25` as fallback stability setting.
3. After selecting stable setting, proceed to constellation-on evaluation.

## Known issues
- Current lock/unlock behavior is highly sensitive to threshold in this run condition; operating-point must be chosen by normal-SNR stability first.

## Current progress status
- Evaluated operating-point validation at `unlock_th=0.30`, `unlock_confirm=20` (`20260214_151140`) for normal-SNR points (`12/6/0 dB`).
- Result:
  - `12 dB`: `lock=5`, `unlock=4` (FAIL)
  - `6 dB`: `lock=1`, `unlock=0` (PASS)
  - `0 dB`: `lock=1`, `unlock=0` (PASS)
- Conclusion: `0.30` still causes false unlock/relock in at least one normal-SNR point; not acceptable for stability-first operating point.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded latest operating-point validation and conclusion per AGENTS.md rules.

## Tasks to be addressed next
1. Validate `unlock_th=0.25`, `unlock_confirm=20` across `12/6/0 dB`.
2. If all-pass, freeze M2 operating point at `0.25/20` and proceed to constellation evaluation.
3. If 12 dB still fails intermittently, add 2-run criterion at 12 dB to measure reproducibility.

## Known issues
- Normal-SNR stability currently shows run variance around threshold setting; operating point must be selected with margin.

## Current progress status
- Completed milestone-2 remaining work (operating-point finalization + normal-SNR re-validation).
- At `unlock_th=0.25`, `unlock_confirm=20` and `AWGN=12/6/0 dB` (`20260214_152925`), all logs passed criteria:
  - lock>=1, unlock=0, forced_resync=0, outlier_resync=0, sink_fail=0, graceful shutdown.
- Fixed FrameSync operating point for stability-first objective as `unlock_th=0.25`, `unlock_confirm=20`.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded milestone-2 finalization evidence and fixed operating point per AGENTS.md rules.

## Tasks to be addressed next
1. Proceed to constellation-on evaluation using fixed FrameSync operating point.
2. Start TMCC persistence checks under the fixed operating point.

## Known issues
- Operating-point pass is confirmed for this run set; reproducibility over repeated long runs should still be monitored.

## Current progress status
- Reported latest milestone status after fixing FrameSync operating point (`unlock_th=0.25`, `unlock_confirm=20`) and confirming normal-SNR all-pass.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded milestone progress reporting task completion per AGENTS.md rules.

## Tasks to be addressed next
1. Run constellation-on stability evaluation with fixed FrameSync operating point.
2. Continue TMCC lock persistence and bitstream consistency validation.

## Known issues
- Visual constellation quality and TMCC long-run persistence are still pending final acceptance.

## Current progress status
- Updated default FrameSync operating point in `isdbt_demod.jl` so explicit CLI override is no longer required for the stabilized setting.

## Files changed and reasons for the changes
- Updated: `examples/isdbt_demod.jl`
  - Reason: changed default `framesync_unlock_threshold` from `0.2` to `0.25` (kept `framesync_unlock_confirm=20`) to match validated operating point.
- Updated: `HANDOVER.md`
  - Reason: recorded completion of default-setting update per AGENTS.md rules.

## Tasks to be addressed next
1. Run a short default-argument check (without `--framesync-unlock-*`) and confirm startup print shows `unlock_threshold: 0.25 unlock_confirm: 20`.
2. Proceed to constellation-on evaluation using defaults.

## Known issues
- None specific to this change; CLI override remains available for stress testing.

## Current progress status
- Compiled current milestone-3 (SP phase stabilization) issues after milestone-2 completion.
- Prioritized remaining M3 work items for next implementation step.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded M3 issue-status reporting per AGENTS.md rules.

## Tasks to be addressed next
1. PilotEQ temporal stabilization under disturbed/low-SNR windows (verify coefficient jitter reduction beyond stable-condition logs).
2. PhaseSlope gate retuning for responsiveness vs over-gating under disturbance (`min_used_pilots/min_used_ratio` and hysteresis balance).
3. CPE confidence-ramp validation in real low-confidence region (`conf < min_update_conf_on`) and adjustment of `conf_gain_floor` / ramp shape.
4. Stage-to-stage consistency check (PilotEQ -> PhaseSlope -> CPE) using correlated telemetry (`updated/gate/conf`) and EVM trend.

## Known issues
- Existing CPE validation logs were dominated by `conf=1.0`, so low-confidence suppression behavior is still not proven.
- PhaseSlope often remained near-zero update in prior stable runs, leaving disturbed-condition responsiveness as an open risk.

## Current progress status
- Prepared a concrete low-SNR validation plan for milestone-3 item 1 (PilotEQ temporal stabilization).
- Defined test matrix, metrics, pass criteria, and execution order to avoid mixing FrameSync effects with PilotEQ effects.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded M3 low-SNR validation planning task per AGENTS.md rules.

## Tasks to be addressed next
1. Baseline acquisition (fixed FrameSync operating point, no PilotEQ parameter change).
2. PilotEQ alpha sweep under low-SNR (`alpha=0.1/0.2/0.3`) with identical capture duration.
3. Metric extraction and comparison (`EVM`, `PhaseSlope/CPE update activity`, lock stability, seq continuity).
4. Candidate selection and confirmation run.

## Known issues
- Current automated scripts focus on FrameSync lock metrics; PilotEQ-specific jitter metrics still require log parser extension or manual extraction.

## Current progress status
- Saved M3 low-SNR PilotEQ validation plan as a standalone markdown document under `docs/`.

## Files changed and reasons for the changes
- Added: `docs/m3_piloteq_lowsnr_validation_plan.md`
  - Reason: documented the requested low-SNR test plan (objective, matrix, metrics, pass criteria, and execution order).
- Updated: `HANDOVER.md`
  - Reason: recorded plan-documentation task completion per AGENTS.md rules.

## Tasks to be addressed next
1. Execute baseline + alpha sweep according to the new plan.
2. Summarize results into a comparison table and select candidate alpha.

## Known issues
- Plan is defined; execution/analysis artifacts are pending.

## Current progress status
- Created a dedicated summary document for work completed up to M1/M2 attainment.

## Files changed and reasons for the changes
- Added: `docs/m1_m2_achievement_summary.md`
  - Reason: documented actions, fixes, evaluation flow, and final operating-point decision leading to M1/M2 completion.
- Updated: `HANDOVER.md`
  - Reason: recorded documentation task completion per AGENTS.md rules.

## Tasks to be addressed next
1. Continue M3 low-SNR PilotEQ validation using `docs/m3_piloteq_lowsnr_validation_plan.md`.
2. Link M3 result summary doc after sweep execution.

## Known issues
- This task is documentation-only; no runtime behavior changed.

## Current progress status
- Split combined M1/M2 summary document into separate files as requested.

## Files changed and reasons for the changes
- Added: `docs/m1_achievement_summary.md`
  - Reason: isolated M1 achievement history and outcomes.
- Added: `docs/m2_achievement_summary.md`
  - Reason: isolated M2 achievement history, boundary exploration, and operating-point finalization.
- Deleted: `docs/m1_m2_achievement_summary.md`
  - Reason: replaced by separate M1/M2 docs to avoid duplication/confusion.
- Updated: `HANDOVER.md`
  - Reason: recorded documentation split task completion per AGENTS.md rules.

## Tasks to be addressed next
1. Keep M3 docs separate in the same style (`plan` and `result`) after low-SNR sweep execution.

## Known issues
- None (documentation restructuring only).

## Current progress status
- Added executable sweep script for M3 low-SNR PilotEQ validation (`SNR x alpha` matrix).

## Files changed and reasons for the changes
- Added: `scripts/run_m3_piloteq_lowsnr_sweep.sh`
  - Reason: automate planned M3-1 test matrix with reproducible FrameSync operating point and quick lock/sink summary.
- Updated: `HANDOVER.md`
  - Reason: recorded script implementation task completion per AGENTS.md rules.

## Tasks to be addressed next
1. Execute the new script and collect logs for `6/0/-2 dB` x `alpha=0.1/0.2/0.3`.
2. Compare EVM and update-activity metrics to select candidate alpha.

## Known issues
- Script provides quick lock/sink summary only; detailed PilotEQ jitter/EVM analysis is still a follow-up step.

## Current progress status
- Evaluated first M3 low-SNR sweep run (`20260214_160733`) generated by `scripts/run_m3_piloteq_lowsnr_sweep.sh`.
- All 9 conditions passed pipeline stability checks (`lock=1`, `unlock=0`, `forced/outlier=0`, `sink_fail=0`, graceful shutdown).
- PilotEQ temporal metrics (`mean|H|`) differences across alpha (`0.1/0.2/0.3`) were small; no decisive winner from this run alone.
- `PhaseSlope updated` and `CPE updated` remained zero in all runs, and `CPE conf` stayed `1.0`, so disturbance-driven adaptation was not exercised.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded M3 sweep evaluation outcome per AGENTS.md rules.

## Tasks to be addressed next
1. Increase stress and observability for M3 discrimination:
   - extend SNR set to `0/-2/-4/-6 dB` and/or duration to `600 s`.
   - enable EVM path during sweep for quality tie-break.
2. Re-run selected alpha candidates with repeated runs to measure variance.

## Known issues
- Current sweep is stability-pass but not yet discriminative for alpha selection because adaptation paths were mostly idle.

## Current progress status
- Updated M3 low-SNR sweep script to support optional EVM logging so the extended 600s stress run can include a quality metric.

## Files changed and reasons for the changes
- Updated: `scripts/run_m3_piloteq_lowsnr_sweep.sh`
  - Reason: added env-controlled EVM options (`M3_ENABLE_EVM`, `M3_EVM_MOD`, `M3_EVM_LOG_INTERVAL`) and command-array execution.
- Updated: `HANDOVER.md`
  - Reason: recorded script enhancement task completion per AGENTS.md rules.

## Tasks to be addressed next
1. Run extended sweep with `SNR=0/-2/-4/-6`, `duration=600s`, and EVM enabled.
2. Evaluate logs for stability + EVM trend by alpha.

## Known issues
- EVM interpretation still depends on actual payload modulation (`qpsk/16qam/64qam`) selection.

## Current progress status
- Evaluated extended M3 low-SNR sweep with EVM enabled (`20260214_170229`, SNR `0/-2/-4/-6`, duration `600s`, alpha `0.1/0.2/0.3`).
- Pipeline stability remained clean across all 12 runs (`unlock=0`, `forced/outlier=0`, `sink_fail=0`).
- EVM-based comparison summary (run-mean EVM, lower is better):
  - alpha=0.1: overall mean `137.62`, mean std `46.36`
  - alpha=0.2: overall mean `145.05`, mean std `59.27`
  - alpha=0.3: overall mean `146.19`, mean std `66.22`
- By SNR, alpha=0.1 was best at `0/-4/-6 dB`; alpha=0.3 was slightly better at `-2 dB`.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded extended M3 sweep evaluation and alpha ranking per AGENTS.md rules.

## Tasks to be addressed next
1. Use `pilot_temporal_alpha=0.1` as provisional M3 baseline.
2. Run confirmation repeat at `-2 dB` (where alpha preference crossed) to confirm robustness.
3. Investigate why `PhaseSlope/CPE updated` remain mostly idle under stress despite EVM variation.

## Known issues
- `CPE conf` stayed `1.0` and `PhaseSlope/CPE updated` stayed `false` in these runs, limiting adaptation-path discrimination.

## Current progress status
- Added tooling to close remaining M3 issues by running reproducibility re-checks and producing deterministic analysis output.

## Files changed and reasons for the changes
- Added: `scripts/run_m3_piloteq_recheck.sh`
  - Reason: automate repeated A/B verification (default alpha `0.1` vs `0.3`) at a chosen SNR with EVM enabled.
- Added: `scripts/analyze_m3_piloteq_logs.py`
  - Reason: aggregate per-log metrics (`EVM`, `H`, lock/unlock, updated flags, sink_fail`) and alpha-level summary for selection decisions.
- Updated: `HANDOVER.md`
  - Reason: recorded remaining-issue closure tooling task per AGENTS.md rules.

## Tasks to be addressed next
1. Execute re-check at `-2 dB` (`repeats=3`) to resolve alpha crossover uncertainty.
2. If alpha `0.1` remains stable winner or tie, freeze as M3 baseline.
3. If crossover persists, add one more discriminator run at `-3 dB`.

## Known issues
- Adaptation-path activity (`PhaseSlope/CPE updated`) may still remain sparse; decision currently relies mostly on EVM + stability metrics.

## Current progress status
- Evaluated M3 reproducibility re-check at `AWGN=-2 dB` (`20260215_114612`, alpha `0.1` vs `0.3`, repeats=3).
- All 6 runs were stable (`lock=1`, `unlock=0`, `forced/outlier=0`, `sink_fail=0`, graceful shutdown).
- Re-check result:
  - alpha `0.1`: evm_mean(avg)=`148.74`, std_across_runs=`17.87`
  - alpha `0.3`: evm_mean(avg)=`142.25`, std_across_runs=`4.89`
- Conclusion: at the previously ambiguous `-2 dB` point, alpha `0.3` is better and more reproducible.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded re-check evaluation and updated M3-1 provisional decision per AGENTS.md rules.

## Tasks to be addressed next
1. Decide M3-1 baseline policy:
   - global low-SNR robustness priority: keep alpha `0.1`
   - include `-2 dB` reproducibility priority: choose alpha `0.3`
2. Optionally run one tie-break at `-3 dB` to define boundary behavior.

## Known issues
- `PhaseSlope/CPE updated` remained zero and `CPE conf=1.0`; adaptation-path discrimination is still limited to EVM/stability behavior.

## Current progress status
- Fixed M3-1 baseline policy to option 1 (overall low-SNR average priority): `pilot_temporal_alpha=0.1`.

## Files changed and reasons for the changes
- Updated: `examples/isdbt_demod.jl`
  - Reason: changed default `pilot_temporal_alpha` from `0.2` to `0.1` to reflect chosen M3-1 baseline policy.
- Updated: `HANDOVER.md`
  - Reason: recorded baseline-policy fixation per AGENTS.md rules.

## Tasks to be addressed next
1. Run a short no-override check to confirm startup print shows `PilotEQ temporal_alpha: 0.1`.
2. Proceed to M3 next issue (PhaseSlope gate responsiveness / adaptation path exercise).

## Known issues
- Adaptation-path activity remains sparse in previous logs (`PhaseSlope/CPE updated` mostly false).

## Current progress status
- Started M3 remaining-work implementation by exposing adaptation-gate controls via CLI and adding an adaptation-probe runner.

## Files changed and reasons for the changes
- Updated: `examples/isdbt_demod.jl`
  - Reason: added CLI knobs for adaptation-path probing:
    - `--slope-min-used-ratio`
    - `--cpe-min-update-conf`
    - `--cpe-min-update-conf-off`
  - Reason: wired these values into `ISDBTPhaseSlopeCorrector` / `ISDBTCPECorrector` construction.
- Added: `scripts/run_m3_adaptivity_probe.sh`
  - Reason: A/B run script (`strict` vs `relaxed` gate profiles) to force/observe `PhaseSlope/CPE updated` behavior under low-SNR stress.
- Updated: `HANDOVER.md`
  - Reason: recorded M3-remaining implementation kickoff per AGENTS.md rules.

## Tasks to be addressed next
1. Run `run_m3_adaptivity_probe.sh` at `-4 dB` and compare `phase_up/cpe_up` + EVM.
2. If relaxed profile improves EVM without unlock increase, fold part of settings into baseline.

## Known issues
- Local runtime parse check was limited by environment launcher-lockfile error; functional verification should be done on target runtime host.

## Current progress status
- Evaluated M3 adaptivity probe (`strict` vs `relaxed`) at `AWGN=-4 dB` (`20260215_145152`, alpha=0.1).
- Result:
  - strict: `evm_mean=145.08`, `evm_std=66.72`
  - relaxed: `evm_mean=151.20`, `evm_std=94.90`
  - both: `lock=1`, `unlock=0`, `forced/outlier=0`, `sink_fail=0`, `phase_up/cpe_up=0`, `conf_mean=1.0`
- Conclusion: relaxing current gate thresholds did not improve quality and did not activate adaptation path; remaining bottleneck is likely outside current gate thresholds.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded adaptivity-probe evaluation and conclusion per AGENTS.md rules.

## Tasks to be addressed next
1. Keep strict profile as baseline (do not adopt relaxed gate settings).
2. Add stronger disturbance dimensions beyond AWGN-only (e.g., phase jump / CFO perturbation) to force adaptation-path exercise.
3. Instrument `PhaseSlope/CPE` with explicit skip-reason counters to identify why updates remain zero.

## Known issues
- `CPE conf` remains saturated at `1.0`, and `PhaseSlope/CPE updated` remain zero under tested conditions, limiting M3 closure confidence.

## Current progress status
- Implemented AWGN-external impairment injection path to force adaptation-path exercise (CFO + phase jump).

## Files changed and reasons for the changes
- Added: `src/CFOPhaseInjector.jl`
  - Reason: new block to inject software impairments:
    - continuous CFO rotation (`cfo_hz`)
    - periodic phase jumps (`phase_jump_deg`, `phase_jump_interval_frames`)
  - Reason: keeps SeqTrace continuity and supports periodic runtime stats logging.
- Updated: `src/SignalFlow.jl`
  - Reason: included `CFOPhaseInjector` module in package load path.
- Updated: `examples/isdbt_demod.jl`
  - Reason: added CLI options:
    - `--impair-cfo-hz`
    - `--impair-phase-jump-deg`
    - `--impair-phase-jump-interval-frames`
    - `--impair-log-interval`
  - Reason: inserted optional impairment block into chain (`rfsrc/awgn -> impair -> sync`).
  - Reason: startup prints now indicate impairment settings when enabled.
- Updated: `HANDOVER.md`
  - Reason: recorded impairment-path implementation per AGENTS.md rules.

## Tasks to be addressed next
1. Run M3 probe with impairment enabled and compare `phase_up/cpe_up` activation against AWGN-only baseline.
2. Tune impairment strength to trigger updates without causing unlock.

## Known issues
- Runtime functional validation of new CLI path is pending on target environment (local launcher had lockfile restrictions in sandbox checks).

## Current progress status
- Extended `run_m3_adaptivity_probe.sh` to support CFO/phase-jump impairment injection during strict/relaxed A/B evaluation.

## Files changed and reasons for the changes
- Updated: `scripts/run_m3_adaptivity_probe.sh`
  - Reason: added impairment env knobs and command wiring:
    - `IMPAIR_CFO_HZ`
    - `IMPAIR_PHASE_JUMP_DEG`
    - `IMPAIR_PHASE_JUMP_INTERVAL_FRAMES`
    - `IMPAIR_LOG_INTERVAL`
  - Reason: keeps same script while enabling AWGN-external disturbance tests.
- Updated: `HANDOVER.md`
  - Reason: recorded impairment-probe script enhancement per AGENTS.md rules.

## Tasks to be addressed next
1. Run adaptivity probe with impairment enabled and compare `phase_up/cpe_up` activation and EVM shift.
2. Decide whether impairment profile should be standardized for M3 validation.

## Known issues
- Need target-runtime validation logs to confirm impairment path actually increases update activity.

## Current progress status
- Evaluated impairment-enabled adaptivity probe (`20260215_154646`) with CFO/phase-jump (`120 Hz`, `12 deg`, interval `8 frames`) at `AWGN=-4 dB`.
- Results:
  - strict: `evm_mean=139.14`, `evm_std=36.01`
  - relaxed: `evm_mean=153.83`, `evm_std=79.47`
  - both: `lock=1`, `unlock=0`, `forced/outlier=0`, `sink_fail=0`, `phase_up/cpe_up=0`, `conf_mean=1.0`
- Conclusion: even with added impairment, adaptation-path updates were not triggered; strict profile remains clearly better.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded impairment-probe evaluation outcome per AGENTS.md rules.

## Tasks to be addressed next
1. Add skip-reason counters in `PhaseSlope/CPE` logs (e.g., blocked by `min_step`, zero-fit residual, phase-selection freeze, confidence hysteresis state).
2. Consider stronger/structured impairment (frame-synchronous phase jump >20 deg or CFO ramp) after counters are available.

## Known issues
- `PhaseSlope/CPE updated` remains zero even under current impairment profile; M3 closure is blocked on root-cause observability.

## Current progress status
- Implemented skip-reason observability for M3 adaptation-path debugging.
- Added cumulative `skip_*` counters to PhaseSlope and CPE logs, and extended analysis script to surface these counters.

## Files changed and reasons for the changes
- Updated: `src/ISDBTPhaseSlopeCorrector.jl`
  - Reason: added counters for non-update reasons:
    - `skip_freeze`, `skip_gate`, `skip_fit_input`, `skip_fit_rms`, `skip_small_delta`, `skip_invalid_fit`.
  - Reason: appended counters to periodic `PhaseSlope:` log output.
- Updated: `src/ISDBTCPECorrector.jl`
  - Reason: added counters for non-update reasons:
    - `skip_freeze`, `skip_gate`, `skip_no_used`, `skip_small_err`, `skip_zero_delta`.
  - Reason: appended counters to periodic `CPE:` log output.
- Updated: `scripts/analyze_m3_piloteq_logs.py`
  - Reason: parse and report latest skip counters in per-log table.
- Updated: `HANDOVER.md`
  - Reason: recorded observability enhancement task completion per AGENTS.md rules.

## Tasks to be addressed next
1. Re-run adaptivity probe (with impairment) and inspect skip-counter columns.
2. Use dominant skip reasons to choose next implementation fix (e.g., phase estimator path vs gate thresholds).

## Known issues
- Historical logs produced before this change will show `skip_* = 0` because fields were not emitted.

## Current progress status
- Evaluated impairment probe with new skip counters (`20260215_194635`).
- Dominant skip reasons were identified:
  - PhaseSlope: `skip_small_delta` overwhelmingly dominant (`~4.4e5`), while `skip_gate` and `skip_fit_input` were near-zero.
  - CPE: `skip_small_err` and `skip_zero_delta` overwhelmingly dominant (`~4.4e5`), with `skip_gate` near-zero.
- Interpretation: update gate is not the bottleneck; effective phase/slope deltas are below minimum-step thresholds most of the time.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded skip-counter based root-cause identification for M3 per AGENTS.md rules.

## Tasks to be addressed next
1. Expose minimum-step knobs in CLI and reduce them for probe runs:
   - PhaseSlope: `min_slope_step`, `min_intercept_step_deg`
   - CPE: `min_phase_step_deg`
2. Re-run impairment probe and confirm `phase_up/cpe_up` become non-zero without inducing unlock.

## Known issues
- Current settings keep adaptation-path updates effectively quantized to zero (`small step` dominated), blocking M3 closure.

## Current progress status
- Implemented next M3 step by exposing minimum-step quantization knobs and wiring them into adaptivity probe automation.

## Files changed and reasons for the changes
- Updated: `examples/isdbt_demod.jl`
  - Reason: added new CLI knobs for adaptation quantization thresholds:
    - `--slope-min-slope-step`
    - `--slope-min-intercept-step-deg`
    - `--cpe-min-phase-step-deg`
  - Reason: connected these to `ISDBTPhaseSlopeCorrector` / `ISDBTCPECorrector` constructors and startup prints.
- Updated: `scripts/run_m3_adaptivity_probe.sh`
  - Reason: added strict/relaxed profile controls for min-step knobs via env vars and passed them to demod CLI.
- Updated: `HANDOVER.md`
  - Reason: recorded this M3 implementation step per AGENTS.md rules.

## Tasks to be addressed next
1. Run adaptivity probe with reduced min-step values in relaxed profile and verify `phase_up/cpe_up > 0`.
2. If updates appear without unlock increase, tune toward minimum EVM while keeping stability.

## Known issues
- Final validation depends on runtime logs from target SDR host.

## Current progress status
- Evaluated adaptivity probe run `20260216_212526` with impairment + min-step tuning profile.
- Outcome:
  - `relaxed` improved EVM vs `strict` in this run (`133.36` vs `158.66`).
  - Pipeline stability remained good (`lock=1`, `unlock=0`, `sink_fail=0`).
  - However adaptation updates still did not trigger (`phase_up/cpe_up=0`).
- Skip counters remain dominated by small-step/zero-delta reasons:
  - PhaseSlope: `skip_small_delta` dominant.
  - CPE: `skip_small_err` and `skip_zero_delta` dominant.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded latest adaptivity-probe evaluation and implications per AGENTS.md rules.

## Tasks to be addressed next
1. Re-run strict/relaxed with repeats (>=3) to confirm whether relaxed EVM gain is reproducible or single-run variance.
2. If reproducible, keep relaxed min-step settings for M3 provisional baseline.
3. To trigger non-zero updates, either lower min-step further (one order) or move impairment injection point downstream (post-FFT/pilot path).

## Known issues
- Despite impairment and threshold tuning, `updated` remains zero; M3 closure still blocked on adaptation-path activation evidence.

## Current progress status
- Extended adaptivity probe runner to support repeated strict/relaxed A/B runs in one command.

## Files changed and reasons for the changes
- Updated: `scripts/run_m3_adaptivity_probe.sh`
  - Reason: added optional `repeats` argument (`5th` positional), per-repeat log suffix (`rN`), and looped execution for reproducibility checks.
- Updated: `HANDOVER.md`
  - Reason: recorded repeat-run automation update per AGENTS.md rules.

## Tasks to be addressed next
1. Execute probe with `repeats=3` and compare strict/relaxed reproducibility.
2. Decide provisional M3 profile from repeated-run aggregate metrics.

## Known issues
- Increased run count means long total runtime (`duration * repeats * 2`).

## Current progress status
- Evaluated repeated adaptivity probe (`20260217_212234`, repeats=3 each profile) with impairment and min-step tuning.
- Aggregated profile comparison from run means:
  - relaxed EVM means: `145.53`, `145.55`, `153.00` -> avg `148.03`
  - strict EVM means: `149.68`, `151.94`, `141.50` -> avg `147.71`
- Difference between profiles is very small (`~0.32`), with no decisive winner.
- Crucially, adaptation path still inactive in all 6 runs:
  - `phase_up/cpe_up=0`
  - skip counters dominated by small-step paths (`PhaseSlope skip_small_delta`, `CPE skip_small_err/skip_zero_delta`).

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded repeated strict/relaxed evaluation and conclusion per AGENTS.md rules.

## Tasks to be addressed next
1. Keep strict profile as default (difference is negligible; strict is current baseline).
2. Add one more probe profile with 10x lower min-step thresholds to force non-zero updates:
   - `slope-min-slope-step=1e-6`
   - `slope-min-intercept-step-deg=0.02`
   - `cpe-min-phase-step-deg=0.02`
3. If updates still remain zero, shift impairment injection downstream (post-FFT/Pilot domain) for direct phase-path excitation.

## Known issues
- M3 remains blocked by lack of adaptation activation evidence; gate and confidence are not the limiting factors under current impairment model.

## Current progress status
- Evaluated further-reduced min-step probe (`20260217_223147`, repeats=2, strict vs relaxed with relaxed min-step at `1e-6/0.02/0.02`).
- Result: adaptation path still inactive in all runs (`phase_up/cpe_up=0`).
- Skip counters remain dominated by small-step paths (`PhaseSlope skip_small_delta`, `CPE skip_small_err/skip_zero_delta`).
- Stability remained good (`unlock=0`, `sink_fail=0`).

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded latest probe outcome and decision basis per AGENTS.md rules.

## Tasks to be addressed next
1. Move impairment injection point downstream (post-FFT / pilot-domain path) to directly excite SP correction stages.
2. Re-run strict/relaxed probe at same SNR and compare whether `phase_up/cpe_up` become non-zero.

## Known issues
- Even aggressive min-step lowering did not activate updates with current upstream impairment injection.

## Current progress status
- Implemented downstream impairment injection path (post-FFT) to directly excite SP correction stages for M3 debugging.

## Files changed and reasons for the changes
- Added: `src/OFDMSymbolImpairInjector.jl`
  - Reason: new FFT-domain symbol impairment block supporting:
    - symbol-domain CFO drift (`cfo_hz`),
    - periodic phase jump (`phase_jump_deg`, `phase_jump_interval_frames`),
    - phase slope across bins (`slope_rad_per_bin`).
- Updated: `src/SignalFlow.jl`
  - Reason: included `OFDMSymbolImpairInjector` module.
- Updated: `examples/isdbt_demod.jl`
  - Reason: added symbol-domain impairment CLI options:
    - `--sym-impair-cfo-hz`
    - `--sym-impair-phase-jump-deg`
    - `--sym-impair-phase-jump-interval-frames`
    - `--sym-impair-slope-rad-per-bin`
    - `--sym-impair-log-interval`
  - Reason: inserted optional `sym_impair` block after `fft_gain_block` and before pilot/TMCC paths.
- Updated: `scripts/run_m3_adaptivity_probe.sh`
  - Reason: added env-based symbol impairment wiring (`SYM_IMPAIR_*`) and CLI forwarding.
- Updated: `HANDOVER.md`
  - Reason: recorded downstream-injection implementation per AGENTS.md rules.

## Tasks to be addressed next
1. Run adaptivity probe with `SYM_IMPAIR_*` enabled and compare `phase_up/cpe_up` activation versus upstream-only impairment.
2. If updates become non-zero, tune symbol-impair strength to maximize activation while preserving lock.

## Known issues
- Behavioral validation of new symbol-domain impairment path requires runtime logs from target SDR host.

## Current progress status
- Evaluated downstream (post-FFT) symbol impairment probe (`20260218_000154`, repeats=2) with:
  - `SYM_IMPAIR_CFO_HZ=120`
  - `SYM_IMPAIR_PHASE_JUMP_DEG=12`
  - `SYM_IMPAIR_PHASE_JUMP_INTERVAL_FRAMES=8`
  - `SYM_IMPAIR_SLOPE_RAD_PER_BIN=0.0002`
- Outcome:
  - stability remained good (`lock=1`, `unlock=0`, `sink_fail=0`) in all runs.
  - `phase_up/cpe_up` still `0/0` in all runs.
  - skip counters still dominated by small-step paths (`PhaseSlope skip_small_delta`, `CPE skip_small_err/skip_zero_delta`).
- Conclusion: moving impairment to FFT-post path alone did not activate adaptation updates.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded downstream-impairment evaluation result and M3 blockage status per AGENTS.md rules.

## Tasks to be addressed next
1. Inject disturbance directly into pilot/reference phase domain (before PhaseSlope/CPE fit decisions), not only into generic symbol stream.
2. Add explicit test mode in PhaseSlope/CPE for forced minimal update to validate update plumbing independently from fit thresholds.

## Known issues
- M3 remains blocked: adaptation path not activating despite upstream/downstream impairment and min-step relaxation.

## Current progress status
- Implemented request "2": added force-update CLI controls in `examples/isdbt_demod.jl` and wired them into `PhaseSlope`/`CPE` block constructors.
- Startup log now prints configured force-update values for visibility.
- Runtime check confirmed new options are parsed and reflected in startup output.

## Files changed and reasons for the changes
- Updated: `examples/isdbt_demod.jl`
  - Added CLI args:
    - `--slope-force-update-eps`
    - `--cpe-force-update-eps-deg`
  - Added validation:
    - both must be `>= 0`
  - Extended `parse_args` return tuple and `main()` destructuring.
  - Passed values to constructors:
    - `ISDBTPhaseSlopeCorrector.CreateISDBTPhaseSlopeCorrector(... force_update_eps=...)`
    - `ISDBTCPECorrector.CreateISDBTCPECorrector(... force_update_eps_deg=...)`
  - Added startup print line:
    - `Phase/CPE force_update_eps: slope=... cpe_deg=...`

## Tasks to be addressed next
1. Complete CPE side forced-update execution path in `src/ISDBTCPECorrector.jl` task loop (currently parser/wiring exists, task-side force application/logging still pending).
2. Add env forwarding in `scripts/run_m3_adaptivity_probe.sh` for strict/relaxed force-update knobs.
3. Re-run adaptivity probe and confirm non-zero `phase_up/cpe_up` or explicit `force_update` counters.

## Known issues
- Current live run still shows repeated `ISDBTSymbolSync: sink_backpressure ... FFTBlock` under this environment.
- CPE forced-update behavior is not yet verified end-to-end until task-side force-update logic is completed.

## Current progress status
- Continued from prior step and completed script wiring for strict/relaxed force-update controls.
- `scripts/run_m3_adaptivity_probe.sh` now forwards force-update parameters to `examples/isdbt_demod.jl` in both profiles.
- Syntax check passed (`bash -n`).

## Files changed and reasons for the changes
- Updated: `scripts/run_m3_adaptivity_probe.sh`
  - Reason: added env knobs and CLI forwarding for force-update validation path:
    - `STRICT_SLOPE_FORCE_UPDATE_EPS`
    - `STRICT_CPE_FORCE_UPDATE_EPS_DEG`
    - `RELAXED_SLOPE_FORCE_UPDATE_EPS`
    - `RELAXED_CPE_FORCE_UPDATE_EPS_DEG`
  - Reason: extended `run_one()` argument list so strict/relaxed values are passed explicitly.
  - Reason: usage/help text updated to show new env variables.
- Updated: `HANDOVER.md`
  - Reason: recorded completed task per AGENTS.md requirement.

## Tasks to be addressed next
1. Run adaptivity probe with non-zero force-update settings and compare strict vs relaxed output logs.
2. Verify logs include expected non-zero update evidence (`phase_up/cpe_up` and/or force-update counters).
3. If still no activation, complete/verify CPE task-side force-update path in `src/ISDBTCPECorrector.jl`.

## Known issues
- This step only adds script/CLI plumbing; behavioral activation depends on runtime path in correction blocks.
- CPE force-update runtime behavior still needs explicit confirmation in logs.

## Current progress status
- Evaluated force-update enabled M3 adaptivity probe (`20260218_225358`, SNR=-4dB, repeats=2).
- Adaptation path is now active in both profiles:
  - `phase_up/cpe_up` changed from `0/0` to non-zero (`20/17..19`).
- Stream stability remained good (`lock=1`, `unlock=0`, `sink_fail=0`, `shutdown=1`).

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded latest runtime evaluation and conclusions per AGENTS.md requirement.

## Tasks to be addressed next
1. Use relaxed force-update settings as candidate operating point for debug-phase validation (better confidence and lower EVM spread than strict).
2. Run A/B confirmation against force-update disabled baseline at same conditions (`SNR=-4dB`, repeats>=3).
3. If relaxed remains better, proceed to reduce force-update magnitudes gradually toward minimal effective values.

## Known issues
- Strict profile shows large CPE gate skips (`cpe_skip gate` very high) and lower confidence (`conf_mean ~0.63-0.66`), with one high-EVM outlier run.
- Force-update mode is diagnostic; final operating point should be re-validated with reduced/disabled force values.

## Current progress status
- Reported current milestone status and completion percentages based on latest validated logs and scripts.
- Reflected latest force-update adaptivity result (`20260218_225358`) in milestone progress interpretation.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: logged completion of current status-reporting task per AGENTS.md rule.

## Tasks to be addressed next
1. Execute M3 A/B validation (force-update ON vs OFF) at `SNR=-4dB`, repeats>=3.
2. If ON path remains superior, reduce force-update magnitudes stepwise to find minimum effective settings.
3. Reconfirm performance on normal-SNR point after low-SNR tuning.

## Known issues
- M3 is functionally unblocked (updates now active with force-update), but final baseline is not fixed until ON/OFF and tapering validation completes.

## Current progress status
- Continued M3 remaining tasks by implementing missing CPE force-update runtime path and adding automation scripts for final validation steps.
- Added two runnable workflows:
  1. force-update `ON/OFF` A/B comparison at fixed conditions
  2. force-update magnitude taper sweep toward minimum effective settings
- Local compile/syntax checks passed (`using SignalFlow`, `bash -n` for new scripts).

## Files changed and reasons for the changes
- Updated: `src/ISDBTCPECorrector.jl`
  - Reason: force-update parameters were previously plumbed but not applied in runtime.
  - Added task-loop behavior:
    - apply tiny signed phase step when `!updated && !freeze_active && force_update_eps>0`
    - increment `force_update_count`
    - set `updated=true` for the symbol
  - Added log fields:
    - `force_update=...`
    - `force_count=...`
- Added: `scripts/run_m3_force_ab.sh`
  - Reason: automate M3 final A/B check (`force OFF` vs `force ON`) with repeated runs and analyzer output.
- Added: `scripts/run_m3_force_taper.sh`
  - Reason: automate force magnitude taper sweep to find minimal effective force settings.
- Updated: `HANDOVER.md`
  - Reason: recorded completed implementation and next actions per AGENTS.md requirements.

## Tasks to be addressed next
1. Run `run_m3_force_ab.sh` at `SNR=-4dB` with repeats>=3 and verify ON/OFF separation in `phase_up/cpe_up` and EVM consistency.
2. Run `run_m3_force_taper.sh` with candidate force levels and choose minimum levels that preserve update activity and stability.
3. Reconfirm selected setting at normal SNR (e.g., `6/12 dB`) to ensure no regression.

## Known issues
- Final M3 closure still depends on long runtime logs from target SDR host.
- Force-update mode is diagnostic and should be minimized/disabled for final baseline once adaptation behavior is validated.

## Current progress status
- Evaluated M3 force A/B run (`20260219_223052`, profile=relaxed, SNR=-4dB, repeats=3, impairment enabled).
- A/B separation is clear:
  - force OFF: `phase_up/cpe_up = 0/0` (no adaptation updates)
  - force ON: `phase_up/cpe_up = 20/20` (updates activated every logged interval)
- Stability stayed intact in all runs (`lock=1`, `unlock=0`, `sink_fail=0`, `shutdown=1`).
- Quality tradeoff observed: force ON significantly worsened EVM.
  - OFF EVM mean(avg across runs): ~138.52
  - ON  EVM mean(avg across runs): ~164.28

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded M3 A/B evaluation outcome and implications per AGENTS.md requirement.

## Tasks to be addressed next
1. Run force taper sweep with smaller force levels to find minimal effective update activation with lower EVM penalty.
2. Prioritize reducing CPE force first (likely larger EVM impact), keeping slope force small for activation probing.
3. Re-run selected candidate at repeats>=3 and compare against OFF baseline.

## Known issues
- Current ON setting (`slope=5e-6`, `cpe=0.05deg`) is too strong for quality and not suitable as final operating point.
- OFF keeps better EVM but does not activate adaptation updates under this test condition.

## Current progress status
- Evaluated force taper sweep (`20260220_003812`, profile=relaxed, SNR=-4dB, repeats=2).
- Stability remained good across all levels (`lock=1`, `unlock=0`, `sink_fail=0`, `shutdown=1`).
- OFF baseline (`0:0`) still gives best quality but no adaptation updates:
  - EVM mean avg: ~133.65, `phase_up/cpe_up=0/0`.
- All non-zero force levels triggered updates (`phase_up/cpe_up=20/20`) but increased EVM.
- Among tested non-zero levels, best EVM was:
  - `slope=2e-6, cpe=0.01deg` (avg ~144.81)
  - next: `slope=1e-6, cpe=0.005deg` (avg ~146.41)

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded taper sweep evaluation and narrowed candidates for M3 closure.

## Tasks to be addressed next
1. Re-run candidate levels with repeats>=3 for confidence:
   - `2e-6:0.01`
   - `1e-6:0.005`
   - `0:0` baseline
2. Add factor-isolation run to separate slope-force and cpe-force contributions:
   - slope-only (`slope>0, cpe=0`)
   - cpe-only (`slope=0, cpe>0`)
3. Choose provisional M3 debug operating point minimizing EVM penalty while keeping `phase_up/cpe_up>0`.

## Known issues
- EVM impact is non-monotonic across tiny force levels; current repeats=2 is insufficient for final fixation.

## Current progress status
- Proceeding with M3 next step #1: planned reproducibility re-run for three candidate force levels with repeats>=3.
- Prepared a single command using `run_m3_force_taper.sh` to execute:
  - `2e-6:0.01`
  - `1e-6:0.005`
  - `0:0` (OFF baseline)

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded the selected next execution step and run conditions per AGENTS.md requirements.

## Tasks to be addressed next
1. Run the prepared 3-level reproducibility command (repeats=3).
2. Compare EVM mean/variance and `phase_up/cpe_up` persistence across the three levels.
3. Decide provisional M3 debug operating point from reproducibility results.

## Known issues
- This run is long (`600s * 3 levels * 3 repeats = 5400s`), so completion depends on SDR host runtime availability.

## Current progress status
- Answered user question about EVM value meaning/interpretation based on actual implementation (`ISDBTEVMMonitor`).

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: logged completion of explanatory task per AGENTS.md rule.

## Tasks to be addressed next
1. Continue M3 validation runs (force A/B and taper finalization).

## Known issues
- None newly identified in this explanatory step.

## Current progress status
- Provided concrete action plan to reduce EVM under current M3 conditions.
- Plan prioritizes minimizing force-update side effects while preserving adaptation observability.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: logged completion of guidance task per AGENTS.md requirement.

## Tasks to be addressed next
1. Run slope-only / cpe-only factor isolation to identify dominant EVM degradation source.
2. Re-run narrowed candidate force levels with repeats>=3 and choose minimal effective point.
3. Validate selected point at normal SNR to ensure no regression.

## Known issues
- Current force settings that activate updates are still introducing notable EVM penalty.

## Current progress status
- Evaluated candidate reproducibility run (`20260220_212741`, profile=relaxed, SNR=-4dB, repeats=3) for:
  - `2e-6:0.01`
  - `1e-6:0.005`
  - `0:0`
- Stability stayed good for all candidates (`lock=1`, `unlock=0`, `sink_fail=0`, `shutdown=1`).
- Reproducibility summary:
  - `0:0` (OFF baseline):
    - `phase_up/cpe_up=0/0` (no adaptation updates)
    - EVM avg across runs: ~150.91
  - `1e-6:0.005`:
    - `phase_up/cpe_up=20/20` in all runs
    - EVM avg across runs: ~146.06 (best among update-enabled candidates)
  - `2e-6:0.01`:
    - `phase_up/cpe_up=20/20` in all runs
    - EVM avg across runs: ~152.74
- Conclusion: `1e-6:0.005` is currently the best M3 debug operating point (updates active with smallest EVM penalty among tested ON candidates).

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded reproducibility-run evaluation and provisional operating-point decision.

## Tasks to be addressed next
1. Run factor-isolation tests to reduce EVM further:
   - slope-only (`1e-6:0`)
   - cpe-only (`0:0.005`)
   - plus `0:0` baseline
2. If one-sided force keeps updates with lower EVM, adopt it as new provisional M3 point.
3. Recheck selected point at normal SNR (e.g., 6/12dB).

## Known issues
- OFF baseline (`0:0`) can still show high run-to-run EVM variance in this channel/impairment condition.
- Update-enabled points still incur EVM cost; further reduction requires slope/CPE contribution split.

## Current progress status
- Proceeded to next M3 task: prepared factor-isolation validation to split EVM impact between slope-force and CPE-force.
- Defined concrete run command set for three levels (`1e-6:0`, `0:0.005`, `0:0`) under same impairment/SNR conditions.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded progression to factor-isolation step and run plan per AGENTS.md requirements.

## Tasks to be addressed next
1. Execute factor-isolation run (repeats=3) with:
   - slope-only: `1e-6:0`
   - cpe-only: `0:0.005`
   - baseline: `0:0`
2. Compare EVM mean/variance and `phase_up/cpe_up` for each level.
3. Decide whether slope-only or cpe-only can replace mixed force point (`1e-6:0.005`).

## Known issues
- Final decision remains data-dependent until factor-isolation logs are collected.

## Current progress status
- Evaluated factor-isolation run (`20260220_232405`, profile=relaxed, SNR=-4dB, repeats=3) for:
  - slope-only: `1e-6:0`
  - cpe-only: `0:0.005`
  - baseline: `0:0`
- Stability was preserved for all levels (`lock=1`, `unlock=0`, `sink_fail=0`, `shutdown=1`).
- Group summary (EVM mean across runs):
  - baseline `0:0` -> ~143.67 (`phase_up/cpe_up=0/0`)
  - cpe-only `0:0.005` -> ~147.34 (`phase_up/cpe_up=0/20`)
  - slope-only `1e-6:0` -> ~156.91 (`phase_up/cpe_up=20/19`)
- Interpretation:
  - CPE-only triggers updates with smaller EVM penalty than slope-only.
  - Slope-only contributes larger EVM degradation (including outlier run).

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded factor-isolation outcome and updated M3 narrowing logic.

## Tasks to be addressed next
1. Run direct A/B for `cpe-only (0:0.005)` vs `baseline (0:0)` with repeats>=3 for decision confidence.
2. If cpe-only remains near baseline and update activation is required, adopt cpe-only as M3 debug operating point.
3. Keep slope-force disabled by default in debug operating point unless slope-update activation becomes mandatory.

## Known issues
- Run-to-run channel variance remains high; final operating-point decision should be based on same-batch A/B comparisons.

## Current progress status
- Evaluated direct A/B (`20260221_101726`, profile=relaxed, SNR=-4dB, repeats=3) between:
  - cpe-only `0:0.005`
  - baseline `0:0`
- Stability remained good in all runs (`lock=1`, `unlock=0`, `sink_fail=0`, `shutdown=1`).
- Decision metrics:
  - cpe-only (`0:0.005`): EVM mean avg ~145.70, `phase_up/cpe_up=0/20` (consistent CPE updates)
  - baseline (`0:0`): EVM mean avg ~151.30, `phase_up/cpe_up=0/0`
- Conclusion: cpe-only outperformed baseline in this batch and preserves update activation.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded final A/B evaluation and provisional M3 debug operating-point decision.

## Tasks to be addressed next
1. Set provisional M3 debug operating point to cpe-only:
   - `slope_force_update_eps=0`
   - `cpe_force_update_eps_deg=0.005`
2. Run confirmation at normal SNR points (e.g., 6dB and 12dB) to ensure no regression.
3. If normal-SNR checks pass, mark M3 debug-phase closure and proceed to constellation stabilization/tmcc path goals.

## Known issues
- Channel variance is still present; keep periodic rechecks when RF condition changes.

## Current progress status
- Proceeded to next phase after M3 low-SNR decision by adding normal-SNR confirmation automation.
- Added script to validate provisional M3 debug operating point across configurable normal SNR list.

## Files changed and reasons for the changes
- Added: `scripts/run_m3_operatingpoint_normalsnr.sh`
  - Reason: automate confirmation runs at normal SNR (default 12/6 dB) for selected force settings.
  - Supports env-configurable operating point:
    - `FORCE_SLOPE_EPS` (default `0`)
    - `FORCE_CPE_EPS_DEG` (default `0.005`)
  - Forwards impairment knobs (`IMPAIR_*`, `SYM_IMPAIR_*`) and runs analyzer on collected logs.
- Updated: `HANDOVER.md`
  - Reason: recorded completion of this implementation step per AGENTS.md.

## Tasks to be addressed next
1. Run normal-SNR confirmation with provisional point (`slope=0`, `cpe=0.005`).
2. Compare against baseline (`slope=0`, `cpe=0`) if needed for regression assessment.
3. If stable and non-regressive, mark M3 debug-phase closure.

## Known issues
- Final decision still depends on runtime logs from target SDR environment.

## Current progress status
- Evaluated normal-SNR confirmation run (`20260221_143720`) for provisional M3 debug operating point:
  - `profile=relaxed`
  - `slope_force_update_eps=0`
  - `cpe_force_update_eps_deg=0.005`
  - SNR points: `12dB`, `6dB`, repeats=2
- Results:
  - Stability good in all runs (`lock=1`, `unlock=0`, `sink_fail=0`, `shutdown=1`).
  - CPE updates stayed active (`phase_up/cpe_up=0/20`).
  - EVM at normal SNR improved compared to low-SNR campaigns (overall avg ~139.41 across 12/6dB set).
- Decision: provisional M3 debug operating point is validated on normal-SNR checkpoints.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: recorded normal-SNR validation outcome and milestone implication.

## Tasks to be addressed next
1. Optionally run baseline comparison at normal SNR (`cpe_force=0`) for formal regression table.
2. If no regression concern remains, mark M3 debug-phase as complete and shift focus to constellation/TMCC end goals.

## Known issues
- Force-update remains a debug aid; production default decision should be revisited after final constellation/TMCC acceptance criteria are fixed.

## Current progress status
- Proceeded with next step #1 request: prepared baseline comparison run at normal SNR with force disabled.
- Target comparison against current provisional debug point (`cpe_force=0.005`) is now explicitly defined.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: logged execution plan for normal-SNR baseline comparison per AGENTS.md.

## Tasks to be addressed next
1. Run normal-SNR baseline with `FORCE_CPE_EPS_DEG=0` and collect logs.
2. Compare baseline vs provisional (`0.005`) on EVM mean/variance and lock stability.
3. Finalize whether to keep or remove cpe-force in M3 debug operating point.

## Known issues
- Final decision remains pending until baseline logs are collected and compared.

## Current progress status
- Completed requested normal-SNR baseline comparison (`20260221_152743`) with force disabled:
  - `FORCE_SLOPE_EPS=0`
  - `FORCE_CPE_EPS_DEG=0`
  - SNR `12/6 dB`, repeats=2
- Compared against prior run (`20260221_143720`, `FORCE_CPE_EPS_DEG=0.005`).
- Comparison summary:
  - `cf=0.005`: EVM avg ~139.41, `phase_up/cpe_up=0/20`
  - `cf=0`: EVM avg ~139.10, `phase_up/cpe_up=0/0`
  - Stability equal in both (`lock=1`, `unlock=0`, `sink_fail=0`, `shutdown=1`).
- Interpretation: quality at normal SNR is essentially equivalent, with tiny edge to `cf=0`; update activation is the only clear benefit of `cf=0.005`.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: logged baseline comparison result and decision basis per AGENTS.md.

## Tasks to be addressed next
1. Split operating policy:
   - default (quality-first): `FORCE_CPE_EPS_DEG=0`
   - debug/probing mode: `FORCE_CPE_EPS_DEG=0.005`
2. If desired, document this policy in docs/HANDOVER milestone section.
3. Continue toward constellation/TMCC acceptance checks.

## Known issues
- Run-to-run EVM spread remains high due to channel/impairment variability; use repeated runs for any final acceptance gate.

## Current progress status
- Reported current milestone status and completion percentages after normal-SNR baseline comparison.

## Files changed and reasons for the changes
- Updated: `HANDOVER.md`
  - Reason: logged completion of milestone/progress reporting task per AGENTS.md.

## Tasks to be addressed next
1. If approved, formalize split policy (default quality mode vs debug mode) in docs.
2. Move to constellation/TMCC acceptance-phase tasks.

## Known issues
- Final acceptance still needs explicit constellation/TMCC success criteria closure.

## Current progress status
- Continued M3 remaining tasks by formalizing operating policy and documentation.
- Switched normal-SNR validation script default to quality-first mode (`FORCE_CPE_EPS_DEG=0`).
- Added M3 achievement summary document under `docs/` with final findings and policy split.

## Files changed and reasons for the changes
- Updated: `scripts/run_m3_operatingpoint_normalsnr.sh`
  - Reason: align default behavior with validated quality-first policy:
    - default `FORCE_CPE_EPS_DEG` changed from `0.005` to `0`.
- Added: `docs/m3_achievement_summary.md`
  - Reason: capture M3 implementation/results and provisional operating policy (default vs debug mode).
- Updated: `HANDOVER.md`
  - Reason: recorded this completion step per AGENTS.md.

## Tasks to be addressed next
1. If approved, treat M3 as complete and move to constellation/TMCC acceptance phase.
2. Optionally add a short pointer in higher-level docs to `docs/m3_achievement_summary.md`.

## Known issues
- Force-based debug mode remains intentionally non-default and should be enabled only for adaptivity probing.
