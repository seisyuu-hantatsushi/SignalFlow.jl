module ISDBTFrameSync

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer
import ..AsyncLogger

struct ISDBTFrameSyncCoreConfig
    nfft::Int
    frame_symbols::Int
    tmcc_bins::Vector{Int}
    poolsize::Int
end

Base.@kwdef struct ISDBTFrameSyncLockConfig
    lock_threshold::Float64 = 0.7
    unlock_threshold::Float64 = 0.4
    lock_confirm::Int = 5
    unlock_confirm::Int = 5
    min_lock_symbols::Int = 64
    metric_guard_band::Float64 = 0.03
    corr_alpha::Float64 = 0.2
end

Base.@kwdef struct ISDBTFrameSyncCycleConfig
    expected_frame_ms::Float64 = 0.0
    cycle_outlier_ratio::Float64 = 0.15
    max_cycle_fold::Int = 4
    outlier_relock_count::Int = 3
    warmup_cycle_count::Int = 2
    ref_release_good_cycles::Int = 2
    cycle_ema_outlier_ratio::Float64 = 0.15
end

Base.@kwdef struct ISDBTFrameSyncGapConfig
    input_gap_threshold_ratio::Float64 = 2.5
    gap_freeze_min_ms::Float64 = 45.0
    gap_freeze_symbols::Int = 8
end

Base.@kwdef struct ISDBTFrameSyncLogConfig
    log_interval::Int = 200
    cycle_log_interval::Int = 20
    input_gap_log_interval_sec::Float64 = 1.0
end

struct ISDBTFrameSyncParams
    lock_threshold::Float64
    unlock_threshold::Float64
    lock_confirm::Int
    unlock_confirm::Int
    min_lock_symbols::Int
    expected_frame_ms::Float64
    cycle_outlier_ratio::Float64
    max_cycle_fold::Int
    outlier_relock_count::Int
    warmup_cycle_count::Int
    ref_release_good_cycles::Int
    input_gap_threshold_ratio::Float64
    gap_freeze_min_ms::Float64
    gap_freeze_symbols::Int
    metric_guard_band::Float64
    cycle_ema_outlier_ratio::Float64
    corr_alpha::Float64
    expected_symbol_ms::Float64
end

mutable struct ISDBTFrameSyncLogState
    log_interval::Int
    log_count::Int
    cycle_log_interval::Int
    cycle_log_count::Int
    input_gap_log_interval_sec::Float64
    last_input_gap_log_time::Float64
    input_gap_suppressed::Int
end

mutable struct ISDBTFrameSyncRuntime
    gap_freeze_countdown::Int
    corr_ema::Float64
    corr_ema_ready::Bool
    locked::Bool
    lock_count::Int
    unlock_count::Int
    lock_age::Int
    last_cycle_ns::Int64
    cycle_ema_ms::Float64
    cycle_count::Int
    outlier_streak::Int
    forced_resync_count::Int
    warmup_cycles_left::Int
    good_cycle_streak::Int
    ref_update_hold::Bool
    symbol_index::Int
    total_symbols::Int64
    last_wrap_symbols::Int64
    input_overrun_count::Int
    last_input_ns::Int64
    input_gap_count::Int
    input_gap_event_streak::Int
    input_gap_last_event_ns::Int64
end

mutable struct ISDBTFrameSyncContext <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    nfft::Int
    frame_symbols::Int
    tmcc_bins::Vector{Int}
    params::ISDBTFrameSyncParams
    logs::ISDBTFrameSyncLogState
    state::ISDBTFrameSyncRuntime
    symbol_index_ref::Base.Threads.Atomic{Int}
    gap_freeze_ref::Base.Threads.Atomic{Int}
    tmcc_ring::Vector{Vector{ComplexF32}}
    filled::Int
    outbuf::Vector{ComplexF32}
    ringbuffer::RingFrameBuffer{ComplexF32}
    holdbuf::Union{Nothing, Int}
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
end

function CreateISDBTFrameSync(core::ISDBTFrameSyncCoreConfig;
                              lock::ISDBTFrameSyncLockConfig = ISDBTFrameSyncLockConfig(),
                              cycle::ISDBTFrameSyncCycleConfig = ISDBTFrameSyncCycleConfig(),
                              gap::ISDBTFrameSyncGapConfig = ISDBTFrameSyncGapConfig(),
                              log::ISDBTFrameSyncLogConfig = ISDBTFrameSyncLogConfig())
    return CreateISDBTFrameSync(; nfft = core.nfft,
                                frame_symbols = core.frame_symbols,
                                tmcc_bins = core.tmcc_bins,
                                poolsize = core.poolsize,
                                lock_threshold = lock.lock_threshold,
                                unlock_threshold = lock.unlock_threshold,
                                lock_confirm = lock.lock_confirm,
                                unlock_confirm = lock.unlock_confirm,
                                min_lock_symbols = lock.min_lock_symbols,
                                metric_guard_band = lock.metric_guard_band,
                                corr_alpha = lock.corr_alpha,
                                expected_frame_ms = cycle.expected_frame_ms,
                                cycle_outlier_ratio = cycle.cycle_outlier_ratio,
                                max_cycle_fold = cycle.max_cycle_fold,
                                outlier_relock_count = cycle.outlier_relock_count,
                                warmup_cycle_count = cycle.warmup_cycle_count,
                                ref_release_good_cycles = cycle.ref_release_good_cycles,
                                cycle_ema_outlier_ratio = cycle.cycle_ema_outlier_ratio,
                                input_gap_threshold_ratio = gap.input_gap_threshold_ratio,
                                gap_freeze_min_ms = gap.gap_freeze_min_ms,
                                gap_freeze_symbols = gap.gap_freeze_symbols,
                                log_interval = log.log_interval,
                                cycle_log_interval = log.cycle_log_interval,
                                input_gap_log_interval_sec = log.input_gap_log_interval_sec)
end

function CreateISDBTFrameSync(; nfft::Int = 8192,
                              frame_symbols::Int = 204,
                              tmcc_bins::Vector{Int},
                              lock_threshold::Real = 0.7,
                              unlock_threshold::Real = 0.4,
                              lock_confirm::Int = 5,
                              unlock_confirm::Int = 5,
                              min_lock_symbols::Int = 64,
                              expected_frame_ms::Real = 0.0,
                              cycle_outlier_ratio::Real = 0.15,
                              max_cycle_fold::Int = 4,
                              outlier_relock_count::Int = 3,
                              warmup_cycle_count::Int = 2,
                              ref_release_good_cycles::Int = 2,
                              input_gap_threshold_ratio::Real = 2.5,
                              gap_freeze_min_ms::Real = 45.0,
                              gap_freeze_symbols::Int = 8,
                              metric_guard_band::Real = 0.03,
                              cycle_ema_outlier_ratio::Real = 0.15,
                              corr_alpha::Real = 0.2,
                              log_interval::Int = 200,
                              cycle_log_interval::Int = 20,
                              input_gap_log_interval_sec::Real = 1.0,
                              poolsize::Int = 8)
    nfft < 32 && error("ISDBTFrameSync: nfft must be >= 32.")
    frame_symbols < 1 && error("ISDBTFrameSync: frame_symbols must be >= 1.")
    isempty(tmcc_bins) && error("ISDBTFrameSync: tmcc_bins must not be empty.")
    lock_threshold <= 0 && error("ISDBTFrameSync: lock_threshold must be positive.")
    unlock_threshold <= 0 && error("ISDBTFrameSync: unlock_threshold must be positive.")
    lock_confirm < 1 && error("ISDBTFrameSync: lock_confirm must be >= 1.")
    unlock_confirm < 1 && error("ISDBTFrameSync: unlock_confirm must be >= 1.")
    min_lock_symbols < 0 && error("ISDBTFrameSync: min_lock_symbols must be >= 0.")
    expected_frame_ms < 0 && error("ISDBTFrameSync: expected_frame_ms must be >= 0.")
    (cycle_outlier_ratio < 0 || cycle_outlier_ratio >= 1) &&
        error("ISDBTFrameSync: cycle_outlier_ratio must be in [0, 1).")
    max_cycle_fold < 1 && error("ISDBTFrameSync: max_cycle_fold must be >= 1.")
    outlier_relock_count < 1 && error("ISDBTFrameSync: outlier_relock_count must be >= 1.")
    warmup_cycle_count < 0 && error("ISDBTFrameSync: warmup_cycle_count must be >= 0.")
    ref_release_good_cycles < 1 && error("ISDBTFrameSync: ref_release_good_cycles must be >= 1.")
    input_gap_threshold_ratio <= 0 && error("ISDBTFrameSync: input_gap_threshold_ratio must be > 0.")
    gap_freeze_min_ms <= 0 && error("ISDBTFrameSync: gap_freeze_min_ms must be > 0.")
    gap_freeze_symbols < 0 && error("ISDBTFrameSync: gap_freeze_symbols must be >= 0.")
    metric_guard_band < 0 && error("ISDBTFrameSync: metric_guard_band must be >= 0.")
    cycle_ema_outlier_ratio < 0 && error("ISDBTFrameSync: cycle_ema_outlier_ratio must be >= 0.")
    (corr_alpha <= 0 || corr_alpha > 1) && error("ISDBTFrameSync: corr_alpha must be in (0, 1].")
    log_interval < 1 && error("ISDBTFrameSync: log_interval must be >= 1.")
    cycle_log_interval < 1 && error("ISDBTFrameSync: cycle_log_interval must be >= 1.")
    input_gap_log_interval_sec <= 0 && error("ISDBTFrameSync: input_gap_log_interval_sec must be > 0.")
    poolsize < 1 && error("ISDBTFrameSync: poolsize must be at least 1.")

    tmcc_ring = [Vector{ComplexF32}(undef, length(tmcc_bins)) for _ in 1:frame_symbols]
    new_sinks = Channel{SignalFlowBlock}(4)
    sinks = Vector{SignalFlowBlock}()
    params = ISDBTFrameSyncParams(Float64(lock_threshold),
                                  Float64(unlock_threshold),
                                  Int(lock_confirm),
                                  Int(unlock_confirm),
                                  Int(min_lock_symbols),
                                  Float64(expected_frame_ms),
                                  Float64(cycle_outlier_ratio),
                                  Int(max_cycle_fold),
                                  Int(outlier_relock_count),
                                  Int(warmup_cycle_count),
                                  Int(ref_release_good_cycles),
                                  Float64(input_gap_threshold_ratio),
                                  Float64(gap_freeze_min_ms),
                                  Int(gap_freeze_symbols),
                                  Float64(metric_guard_band),
                                  Float64(cycle_ema_outlier_ratio),
                                  Float64(corr_alpha),
                                  frame_symbols > 0 && expected_frame_ms > 0 ? Float64(expected_frame_ms) / frame_symbols : 0.0)
    logs = ISDBTFrameSyncLogState(Int(log_interval),
                                  0,
                                  Int(cycle_log_interval),
                                  0,
                                  Float64(input_gap_log_interval_sec),
                                  0.0,
                                  0)
    state = ISDBTFrameSyncRuntime(0,
                                  0.0,
                                  false,
                                  false,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0.0,
                                  0,
                                  0,
                                  0,
                                  Int(warmup_cycle_count),
                                  0,
                                  true,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0)
    ctx = ISDBTFrameSyncContext(Base.Threads.Atomic{Bool}(true),
                                nfft,
                                frame_symbols,
                                tmcc_bins,
                                params,
                                logs,
                                state,
                                Base.Threads.Atomic{Int}(0),
                                Base.Threads.Atomic{Int}(0),
                                tmcc_ring,
                                0,
                                Vector{ComplexF32}(undef, nfft),
                                RingFrameBuffer(ComplexF32, nfft, poolsize),
                                nothing,
                                nothing,
                                new_sinks,
                                sinks)
    ctx.worker = Threads.@spawn task!(ctx)
    return ctx
end

function task!(context::ISDBTFrameSyncContext)
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                if rd_buffer.store_size == context.nfft
                    context.state.total_symbols += 1
                    skip_ref_update = false
                    idx = (context.state.symbol_index % context.frame_symbols) + 1
                    prev = context.tmcc_ring[idx]
                    corr = 0.0
                    if context.filled >= context.frame_symbols
                        s_re = 0.0
                        s_im = 0.0
                        p_cur = 0.0
                        p_prev = 0.0
                        @inbounds for i in 1:length(context.tmcc_bins)
                            v = rd_buffer.buf[context.tmcc_bins[i]]
                            p = prev[i]
                            s_re += real(v) * real(p) + imag(v) * imag(p)
                            s_im += real(v) * imag(p) - imag(v) * real(p)
                            p_cur += real(v) * real(v) + imag(v) * imag(v)
                            p_prev += real(p) * real(p) + imag(p) * imag(p)
                        end
                        den = sqrt(p_cur * p_prev)
                        corr = den > 0 ? sqrt(s_re * s_re + s_im * s_im) / den : 0.0
                        freeze_active = context.state.gap_freeze_countdown > 0
                        if freeze_active
                            context.state.gap_freeze_countdown -= 1
                            context.gap_freeze_ref[] = context.state.gap_freeze_countdown
                        end
                        if !freeze_active && context.state.corr_ema_ready
                            context.state.corr_ema = context.params.corr_alpha * corr +
                                               (1.0 - context.params.corr_alpha) * context.state.corr_ema
                        elseif !freeze_active
                            context.state.corr_ema = corr
                            context.state.corr_ema_ready = true
                        else
                            skip_ref_update = true
                        end
                        metric = context.state.corr_ema_ready ? context.state.corr_ema : corr

                        context.logs.log_count += 1
                        if context.logs.log_count >= context.logs.log_interval
                            context.logs.log_count = 0
                            AsyncLogger.log_async("ISDBTFrameSync: corr=", corr,
                                    " ema=", round(metric, digits = 4),
                                    " locked=", context.state.locked)
                        end
                        if !freeze_active
                            if metric >= context.params.lock_threshold + context.params.metric_guard_band
                                context.state.lock_count += 1
                                context.state.unlock_count = 0
                            elseif metric <= context.params.unlock_threshold - context.params.metric_guard_band
                                context.state.lock_count = 0
                                if context.state.locked && context.state.lock_age >= context.params.min_lock_symbols
                                    context.state.unlock_count += 1
                                else
                                    context.state.unlock_count = 0
                                end
                            else
                                # Guard band around thresholds: keep state to avoid noisy flips.
                                context.state.unlock_count = 0
                                context.state.lock_count = 0
                            end
                        end
                        if !context.state.locked && context.state.lock_count >= context.params.lock_confirm
                            context.state.locked = true
                            context.state.lock_age = 0
                            context.state.symbol_index = 0
                            context.symbol_index_ref[] = 0
                            context.state.last_cycle_ns = 0
                            context.state.cycle_count = 0
                            context.state.warmup_cycles_left = context.params.warmup_cycle_count
                            context.state.good_cycle_streak = 0
                            context.state.ref_update_hold = true
                            AsyncLogger.log_async("ISDBTFrameSync: lock corr=", corr, " ema=", round(metric, digits = 4))
                        elseif context.state.locked && context.state.unlock_count >= context.params.unlock_confirm
                            context.state.locked = false
                            context.state.lock_age = 0
                            context.state.ref_update_hold = true
                            AsyncLogger.log_async("ISDBTFrameSync: unlock corr=", corr, " ema=", round(metric, digits = 4))
                        end
                    end

                    @inbounds for i in 1:length(context.tmcc_bins)
                        context.tmcc_ring[idx][i] = rd_buffer.buf[context.tmcc_bins[i]]
                    end
                    context.filled = min(context.filled + 1, context.frame_symbols)
                    wrapped = false
                    context.state.symbol_index += 1
                    if context.state.symbol_index >= context.frame_symbols
                        context.state.symbol_index = 0
                        wrapped = true
                    end
                    if wrapped
                        now_ns = time_ns()
                        symbol_delta = context.state.total_symbols - context.state.last_wrap_symbols
                        context.state.last_wrap_symbols = context.state.total_symbols
                        if context.state.last_cycle_ns > 0
                            dt_ms_wall = (now_ns - context.state.last_cycle_ns) / 1_000_000.0
                            dt_ms_stream = context.params.expected_symbol_ms > 0 ?
                                           symbol_delta * context.params.expected_symbol_ms :
                                           dt_ms_wall
                            # Use stream-time (symbol count) for cycle validation to avoid
                            # false outliers caused by upstream receive stalls.
                            dt_ms_raw = dt_ms_stream
                            dt_ms = dt_ms_raw
                            fold = 1
                            outlier = false
                            warmup = context.state.warmup_cycles_left > 0
                            if warmup
                                context.state.warmup_cycles_left -= 1
                                context.state.good_cycle_streak = 0
                                context.state.ref_update_hold = true
                                skip_ref_update = true
                                AsyncLogger.log_async("ISDBTFrameSync: frame_cycle_warmup_ms=",
                                        round(dt_ms_raw, digits = 3),
                                        " wall_ms=",
                                        round(dt_ms_wall, digits = 3),
                                        " symbols=",
                                        symbol_delta,
                                        " left=",
                                        context.state.warmup_cycles_left,
                                        " locked=",
                                        context.state.locked)
                            elseif context.params.expected_frame_ms > 0
                                err = abs(dt_ms - context.params.expected_frame_ms) / context.params.expected_frame_ms
                                if err > context.params.cycle_outlier_ratio && context.params.max_cycle_fold >= 2
                                    fold_cand = round(Int, dt_ms_raw / context.params.expected_frame_ms)
                                    if fold_cand >= 2 && fold_cand <= context.params.max_cycle_fold
                                        dt_fold = dt_ms_raw / fold_cand
                                        err_fold = abs(dt_fold - context.params.expected_frame_ms) / context.params.expected_frame_ms
                                        if err_fold < err
                                            dt_ms = dt_fold
                                            err = err_fold
                                            fold = fold_cand
                                        end
                                    end
                                end
                                outlier = err > context.params.cycle_outlier_ratio
                                if !outlier && context.state.cycle_count > 4 && context.state.cycle_ema_ms > 0
                                    ema_err = abs(dt_ms - context.state.cycle_ema_ms) / context.state.cycle_ema_ms
                                    outlier = ema_err > context.params.cycle_ema_outlier_ratio
                                end
                            end
                            if !warmup && !outlier
                                context.state.outlier_streak = 0
                                if context.state.cycle_count == 0
                                    context.state.cycle_ema_ms = dt_ms
                                else
                                    context.state.cycle_ema_ms = 0.2 * dt_ms + 0.8 * context.state.cycle_ema_ms
                                end
                                context.state.cycle_count += 1
                                context.state.good_cycle_streak += 1
                                if context.state.good_cycle_streak >= context.params.ref_release_good_cycles
                                    context.state.ref_update_hold = false
                                else
                                    context.state.ref_update_hold = true
                                    skip_ref_update = true
                                end
                                context.logs.cycle_log_count += 1
                                if context.logs.cycle_log_count >= context.logs.cycle_log_interval
                                    context.logs.cycle_log_count = 0
                                    AsyncLogger.log_async("ISDBTFrameSync: frame_cycle_ms=",
                                            round(dt_ms, digits = 3),
                                            " raw_stream_ms=",
                                            round(dt_ms_raw, digits = 3),
                                            " wall_ms=",
                                            round(dt_ms_wall, digits = 3),
                                            " symbols=",
                                            symbol_delta,
                                            " fold=",
                                            fold,
                                            " good_streak=",
                                            context.state.good_cycle_streak,
                                            " ref_hold=",
                                            context.state.ref_update_hold,
                                            " ema_ms=",
                                            round(context.state.cycle_ema_ms, digits = 3),
                                            " locked=",
                                            context.state.locked)
                                end
                            elseif !warmup
                                context.state.outlier_streak += 1
                                context.state.good_cycle_streak = 0
                                context.state.ref_update_hold = true
                                skip_ref_update = true
                                AsyncLogger.log_async("ISDBTFrameSync: frame_cycle_outlier_ms=",
                                        round(dt_ms_stream, digits = 3),
                                        " wall_ms=",
                                        round(dt_ms_wall, digits = 3),
                                        " eval_ms=",
                                        round(dt_ms, digits = 3),
                                        " symbols=",
                                        symbol_delta,
                                        " fold=",
                                        fold,
                                        " expected_ms=",
                                        round(context.params.expected_frame_ms, digits = 3),
                                        " streak=",
                                        context.state.outlier_streak,
                                        " locked=",
                                        context.state.locked)
                                if context.state.outlier_streak >= context.params.outlier_relock_count
                                    context.state.forced_resync_count += 1
                                    context.state.symbol_index = 0
                                    context.symbol_index_ref[] = 0
                                    context.state.last_cycle_ns = 0
                                    context.state.cycle_count = 0
                                    context.state.cycle_ema_ms = 0.0
                                    context.state.lock_age = 0
                                    context.state.outlier_streak = 0
                                    context.state.warmup_cycles_left = context.params.warmup_cycle_count
                                    context.state.good_cycle_streak = 0
                                    context.state.ref_update_hold = true
                                    AsyncLogger.log_async("ISDBTFrameSync: forced_resync count=",
                                            context.state.forced_resync_count,
                                            " reason=cycle_outlier")
                                end
                            end
                        end
                        context.state.last_cycle_ns = now_ns
                    end
                    if context.state.locked
                        context.state.lock_age += 1
                    end
                    # Keep phase running even when unlocked; skip ref update on outlier cycle.
                    if !skip_ref_update && !context.state.ref_update_hold
                        context.symbol_index_ref[] = context.state.symbol_index
                    end

                    copyto!(context.outbuf, 1, rd_buffer.buf, 1, context.nfft)
                    while isready(context.new_sinks)
                        push!(context.sinks, take!(context.new_sinks))
                    end
                    for sink in context.sinks
                        input!(sink, context.outbuf, context.nfft)
                    end
                end
                rd_buffer.store_size = 0
                put!(context.ringbuffer.freeQ, rd_index)
            else
                yield()
            end
        end
    catch e
        if !(e isa InterruptException)
            AsyncLogger.log_async("ISDBTFrameSync error: ", e)
        end
    end
    return nothing
end

function input!(context::ISDBTFrameSyncContext, samples::AbstractVector{ComplexF32}, samples_size::Int)
    if !context.running[] || samples_size <= 0
        return false
    end

    actual_size = min(samples_size, length(samples))
    if actual_size <= 0
        return 0
    end

    if actual_size != context.nfft
        return -1
    end

    wait_loops = 0
    while !isready(context.ringbuffer.freeQ) && context.running[]
        wait_loops += 1
        if wait_loops == 2000
            context.state.input_overrun_count += 1
            AsyncLogger.log_async("ISDBTFrameSync: input_backpressure count=", context.state.input_overrun_count)
            wait_loops = 0
        end
        yield()
    end
    if !context.running[]
        return false
    end
    idx = take!(context.ringbuffer.freeQ)
    buf = context.ringbuffer.bufs[idx]
    copyto!(buf.buf, 1, samples, 1, actual_size)
    buf.store_size = actual_size
    put!(context.ringbuffer.fullQ, idx)
    if context.params.expected_symbol_ms > 0
        now_ns = time_ns()
        if context.state.last_input_ns > 0
            gap_ms = (now_ns - context.state.last_input_ns) / 1_000_000.0
            gap_thresh_ms = max(context.params.expected_symbol_ms * context.params.input_gap_threshold_ratio,
                                context.params.gap_freeze_min_ms)
            if gap_ms > gap_thresh_ms
                context.state.input_gap_count += 1
                event_window_ns = Int64(250_000_000) # 250ms: treat nearby bursts as one event cluster
                if context.state.input_gap_last_event_ns > 0 &&
                   (now_ns - context.state.input_gap_last_event_ns) <= event_window_ns
                    context.state.input_gap_event_streak += 1
                else
                    context.state.input_gap_event_streak = 1
                end
                context.state.input_gap_last_event_ns = now_ns

                # Guard isolated scheduler hiccups: freeze only on repeated bursts or clearly huge gaps.
                huge_gap = gap_ms >= (2.0 * gap_thresh_ms)
                apply_freeze = huge_gap || (context.state.input_gap_event_streak >= 2)
                freeze_set = 0
                    if apply_freeze
                    # Gap-aware freeze extension: longer upstream stalls get longer hold.
                    # This keeps reference updates from being contaminated right after bursts.
                    burst_freeze = 0
                    if context.params.expected_symbol_ms > 0
                        burst_freeze = ceil(Int, gap_ms / context.params.expected_symbol_ms)
                    end
                    freeze_set = max(context.params.gap_freeze_symbols, burst_freeze)
                    freeze_set = min(freeze_set, context.frame_symbols)
                        if freeze_set > 0
                            context.state.gap_freeze_countdown = max(context.state.gap_freeze_countdown, freeze_set)
                            context.gap_freeze_ref[] = context.state.gap_freeze_countdown
                        end
                    end
                now_s = time()
                if context.logs.last_input_gap_log_time == 0.0 ||
                   (now_s - context.logs.last_input_gap_log_time) >= context.logs.input_gap_log_interval_sec
                    burst = freeze_set > context.params.gap_freeze_symbols
                    AsyncLogger.log_async("ISDBTFrameSync: input_gap_ms=",
                            round(gap_ms, digits = 3),
                            " expected_symbol_ms=",
                            round(context.params.expected_symbol_ms, digits = 3),
                            " freeze_thresh_ms=",
                            round(gap_thresh_ms, digits = 3),
                            " huge=",
                            huge_gap,
                            " streak=",
                            context.state.input_gap_event_streak,
                            " apply_freeze=",
                            apply_freeze,
                            " burst=",
                            burst,
                            " freeze_set=",
                            freeze_set,
                            " count=",
                            context.state.input_gap_count,
                            " freeze_left=",
                            context.state.gap_freeze_countdown,
                            " suppressed=",
                            context.logs.input_gap_suppressed)
                    context.logs.last_input_gap_log_time = now_s
                    context.logs.input_gap_suppressed = 0
                else
                    context.logs.input_gap_suppressed += 1
                end
            end
        end
        context.state.last_input_ns = now_ns
    end

    return samples_size
end

function stop!(context::ISDBTFrameSyncContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

end
