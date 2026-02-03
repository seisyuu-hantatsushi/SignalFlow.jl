module ISDBTFrameSync

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer

mutable struct ISDBTFrameSyncContext <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    nfft::Int
    frame_symbols::Int
    tmcc_bins::Vector{Int}
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
    corr_alpha::Float64
    corr_ema::Float64
    corr_ema_ready::Bool
    log_interval::Int
    log_count::Int
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
    symbol_index_ref::Base.Threads.Atomic{Int}
    tmcc_ring::Vector{Vector{ComplexF32}}
    filled::Int
    outbuf::Vector{ComplexF32}
    ringbuffer::RingFrameBuffer{ComplexF32}
    holdbuf::Union{Nothing, Int}
    input_overrun_count::Int
    expected_symbol_ms::Float64
    last_input_ns::Int64
    input_gap_count::Int
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
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
                              corr_alpha::Real = 0.2,
                              log_interval::Int = 200,
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
    (corr_alpha <= 0 || corr_alpha > 1) && error("ISDBTFrameSync: corr_alpha must be in (0, 1].")
    log_interval < 1 && error("ISDBTFrameSync: log_interval must be >= 1.")
    poolsize < 1 && error("ISDBTFrameSync: poolsize must be at least 1.")

    tmcc_ring = [Vector{ComplexF32}(undef, length(tmcc_bins)) for _ in 1:frame_symbols]
    new_sinks = Channel{SignalFlowBlock}(4)
    sinks = Vector{SignalFlowBlock}()
    ctx = ISDBTFrameSyncContext(Base.Threads.Atomic{Bool}(true),
                                nfft,
                                frame_symbols,
                                tmcc_bins,
                                Float64(lock_threshold),
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
                                Float64(corr_alpha),
                                0.0,
                                false,
                                Int(log_interval),
                                0,
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
                                Base.Threads.Atomic{Int}(0),
                                tmcc_ring,
                                0,
                                Vector{ComplexF32}(undef, nfft),
                                RingFrameBuffer(ComplexF32, nfft, poolsize),
                                nothing,
                                0,
                                frame_symbols > 0 && expected_frame_ms > 0 ? Float64(expected_frame_ms) / frame_symbols : 0.0,
                                0,
                                0,
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
                    context.total_symbols += 1
                    skip_ref_update = false
                    idx = (context.symbol_index % context.frame_symbols) + 1
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
                        if context.corr_ema_ready
                            context.corr_ema = context.corr_alpha * corr + (1.0 - context.corr_alpha) * context.corr_ema
                        else
                            context.corr_ema = corr
                            context.corr_ema_ready = true
                        end
                        metric = context.corr_ema

                        context.log_count += 1
                        if context.log_count >= context.log_interval
                            context.log_count = 0
                            println("ISDBTFrameSync: corr=", corr,
                                    " ema=", round(metric, digits = 4),
                                    " locked=", context.locked)
                        end
                        if metric >= context.lock_threshold
                            context.lock_count += 1
                            context.unlock_count = 0
                        elseif metric <= context.unlock_threshold
                            context.lock_count = 0
                            if context.locked && context.lock_age >= context.min_lock_symbols
                                context.unlock_count += 1
                            else
                                context.unlock_count = 0
                            end
                        else
                            context.lock_count = 0
                            context.unlock_count = 0
                        end
                        if !context.locked && context.lock_count >= context.lock_confirm
                            context.locked = true
                            context.lock_age = 0
                            context.symbol_index = 0
                            context.symbol_index_ref[] = 0
                            context.last_cycle_ns = 0
                            context.cycle_count = 0
                            context.warmup_cycles_left = context.warmup_cycle_count
                            context.good_cycle_streak = 0
                            context.ref_update_hold = true
                            println("ISDBTFrameSync: lock corr=", corr, " ema=", round(metric, digits = 4))
                        elseif context.locked && context.unlock_count >= context.unlock_confirm
                            context.locked = false
                            context.lock_age = 0
                            context.ref_update_hold = true
                            println("ISDBTFrameSync: unlock corr=", corr, " ema=", round(metric, digits = 4))
                        end
                    end

                    @inbounds for i in 1:length(context.tmcc_bins)
                        context.tmcc_ring[idx][i] = rd_buffer.buf[context.tmcc_bins[i]]
                    end
                    context.filled = min(context.filled + 1, context.frame_symbols)
                    wrapped = false
                    context.symbol_index += 1
                    if context.symbol_index >= context.frame_symbols
                        context.symbol_index = 0
                        wrapped = true
                    end
                    if wrapped
                        now_ns = time_ns()
                        symbol_delta = context.total_symbols - context.last_wrap_symbols
                        context.last_wrap_symbols = context.total_symbols
                        if context.last_cycle_ns > 0
                            dt_ms_raw = (now_ns - context.last_cycle_ns) / 1_000_000.0
                            dt_ms = dt_ms_raw
                            fold = 1
                            outlier = false
                            warmup = context.warmup_cycles_left > 0
                            if warmup
                                context.warmup_cycles_left -= 1
                                context.good_cycle_streak = 0
                                context.ref_update_hold = true
                                skip_ref_update = true
                                println("ISDBTFrameSync: frame_cycle_warmup_ms=",
                                        round(dt_ms_raw, digits = 3),
                                        " symbols=",
                                        symbol_delta,
                                        " left=",
                                        context.warmup_cycles_left,
                                        " locked=",
                                        context.locked)
                            elseif context.expected_frame_ms > 0
                                err = abs(dt_ms - context.expected_frame_ms) / context.expected_frame_ms
                                if err > context.cycle_outlier_ratio && context.max_cycle_fold >= 2
                                    fold_cand = round(Int, dt_ms_raw / context.expected_frame_ms)
                                    if fold_cand >= 2 && fold_cand <= context.max_cycle_fold
                                        dt_fold = dt_ms_raw / fold_cand
                                        err_fold = abs(dt_fold - context.expected_frame_ms) / context.expected_frame_ms
                                        if err_fold < err
                                            dt_ms = dt_fold
                                            err = err_fold
                                            fold = fold_cand
                                        end
                                    end
                                end
                                outlier = err > context.cycle_outlier_ratio
                            end
                            if !warmup && !outlier
                                context.outlier_streak = 0
                                if context.cycle_count == 0
                                    context.cycle_ema_ms = dt_ms
                                else
                                    context.cycle_ema_ms = 0.2 * dt_ms + 0.8 * context.cycle_ema_ms
                                end
                                context.cycle_count += 1
                                context.good_cycle_streak += 1
                                if context.good_cycle_streak >= context.ref_release_good_cycles
                                    context.ref_update_hold = false
                                else
                                    context.ref_update_hold = true
                                    skip_ref_update = true
                                end
                                println("ISDBTFrameSync: frame_cycle_ms=",
                                        round(dt_ms, digits = 3),
                                        " raw_ms=",
                                        round(dt_ms_raw, digits = 3),
                                        " symbols=",
                                        symbol_delta,
                                        " fold=",
                                        fold,
                                        " good_streak=",
                                        context.good_cycle_streak,
                                        " ref_hold=",
                                        context.ref_update_hold,
                                        " ema_ms=",
                                        round(context.cycle_ema_ms, digits = 3),
                                        " locked=",
                                        context.locked)
                            elseif !warmup
                                context.outlier_streak += 1
                                context.good_cycle_streak = 0
                                context.ref_update_hold = true
                                skip_ref_update = true
                                println("ISDBTFrameSync: frame_cycle_outlier_ms=",
                                        round(dt_ms_raw, digits = 3),
                                        " eval_ms=",
                                        round(dt_ms, digits = 3),
                                        " symbols=",
                                        symbol_delta,
                                        " fold=",
                                        fold,
                                        " expected_ms=",
                                        round(context.expected_frame_ms, digits = 3),
                                        " streak=",
                                        context.outlier_streak,
                                        " locked=",
                                        context.locked)
                                if context.outlier_streak >= context.outlier_relock_count
                                    context.forced_resync_count += 1
                                    context.symbol_index = 0
                                    context.symbol_index_ref[] = 0
                                    context.last_cycle_ns = 0
                                    context.cycle_count = 0
                                    context.cycle_ema_ms = 0.0
                                    context.lock_age = 0
                                    context.outlier_streak = 0
                                    context.warmup_cycles_left = context.warmup_cycle_count
                                    context.good_cycle_streak = 0
                                    context.ref_update_hold = true
                                    println("ISDBTFrameSync: forced_resync count=",
                                            context.forced_resync_count,
                                            " reason=cycle_outlier")
                                end
                            end
                        end
                        context.last_cycle_ns = now_ns
                    end
                    if context.locked
                        context.lock_age += 1
                    end
                    # Keep phase running even when unlocked; skip ref update on outlier cycle.
                    if !skip_ref_update && !context.ref_update_hold
                        context.symbol_index_ref[] = context.symbol_index
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
            println("ISDBTFrameSync error: ", e)
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
            context.input_overrun_count += 1
            println("ISDBTFrameSync: input_backpressure count=", context.input_overrun_count)
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
    if context.expected_symbol_ms > 0
        now_ns = time_ns()
        if context.last_input_ns > 0
            gap_ms = (now_ns - context.last_input_ns) / 1_000_000.0
            if gap_ms > context.expected_symbol_ms * context.input_gap_threshold_ratio
                context.input_gap_count += 1
                println("ISDBTFrameSync: input_gap_ms=",
                        round(gap_ms, digits = 3),
                        " expected_symbol_ms=",
                        round(context.expected_symbol_ms, digits = 3),
                        " count=",
                        context.input_gap_count)
            end
        end
        context.last_input_ns = now_ns
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
