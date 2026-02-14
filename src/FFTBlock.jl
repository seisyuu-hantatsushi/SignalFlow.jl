module FFTBlock

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer
import ..SeqTrace
import ..AsyncLogger

using DSP
using FFTW
using LinearAlgebra

@enum WindowType begin
    Rectangular
    Hann
    Hamming
    Blackman
    BartlettHann
end

@enum FFTScale begin
    FFTScaleNone
    FFTScaleUnity
    FFTScaleSqrt
end

mutable struct FFTBlockContext{T} <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    fft_size::Int
    window::Vector{Float32}
    tmp::Vector{ComplexF32}
    outbuf::Vector{ComplexF32}
    plan::Any
    ringbuffer::RingFrameBuffer{T}
    holdbuf::Union{Nothing, Int}
    frame_composed::Vector{Bool}
    out_seq_local::UInt64
    last_in_seq::UInt64
    in_missing_count::UInt64
    in_missing_log_interval::UInt64
    perf::Any
    in_write_count::UInt64
    in_full_enqueued::UInt64
    out_dispatched::UInt64
    sink_push_fail::UInt64
    in_seq_anomaly_count::UInt64
    fs_handoff_last_raw_seq::UInt64
    fs_handoff_last_guard_seq::UInt64
    fs_handoff_anomaly_count::UInt64
    fs_handoff_guard_count::UInt64
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
end

mutable struct FFTPerfStats
    count::UInt64
    metric_count::UInt64
    total_ns::UInt64
    max_ns::UInt64
    raw_max_ns::UInt64
    log_interval::UInt64
    warmup_frames::UInt64
    window::Vector{UInt64}
    hist::Vector{Int}
    hist_bin_ns::UInt64
    window_pos::Int
    window_count::Int
    window_sum_ns::UInt64
    first_logged::Bool
    total_logged::Bool
    drop_count::UInt64
    samples::Channel{UInt64}
    task::Union{Nothing,Task}
end

function CreateFFTPerfStats(; log_interval::Int, warmup_frames::Int, window_size::Int)
    hist_bin_ns = UInt64(2_000)     # 2us per bin
    hist_bins = 8192                # covers ~16.384ms, tail clips into last bin
    return FFTPerfStats(UInt64(0),
                        UInt64(0),
                        UInt64(0),
                        UInt64(0),
                        UInt64(0),
                        UInt64(log_interval),
                        UInt64(warmup_frames),
                        Vector{UInt64}(undef, window_size),
                        zeros(Int, hist_bins),
                        hist_bin_ns,
                        1,
                        0,
                        UInt64(0),
                        false,
                        false,
                        UInt64(0),
                        Channel{UInt64}(max(window_size, 4096)),
                        nothing)
end

@inline function perf_hist_bin(perf::FFTPerfStats, dt::UInt64)
    idx = Int(dt ÷ perf.hist_bin_ns) + 1
    return idx > length(perf.hist) ? length(perf.hist) : idx
end

function window_coeffs(n::Int, window::WindowType)
    if window == Rectangular
        return ones(Float32, n)
    elseif window == Hann
        return Float32.(DSP.hann(n))
    elseif window == Hamming
        return Float32.(DSP.hamming(n))
    elseif window == Blackman
        return Float32.(DSP.blackman(n))
    else
        return Float32.(DSP.bartlett_hann(n))
    end
end

function CreateFFTBlock(::Type{T}, fft_size::Int;
                        window::WindowType = Hann,
                        scale::FFTScale = FFTScaleNone,
                        perf_log_interval::Int = 0,
                        perf_warmup_frames::Int = 2000,
                        perf_window_size::Int = 2048,
                        poolsize::Int = 64) where {T}
    fft_size < 32 && error("FFTBlock: fft_size must be >= 32.")
    fft_size > 128 * 1024 && error("FFTBlock: fft_size must be <= 131072.")
    poolsize < 1 && error("FFTBlock: poolsize must be at least 1.")
    perf_log_interval < 0 && error("FFTBlock: perf_log_interval must be >= 0.")
    perf_warmup_frames < 0 && error("FFTBlock: perf_warmup_frames must be >= 0.")
    perf_window_size < 1 && error("FFTBlock: perf_window_size must be >= 1.")

    new_sinks = Channel{SignalFlowBlock}(64)
    sinks = Vector{SignalFlowBlock}()
    gain = if scale == FFTScaleUnity
        1.0f0 / Float32(fft_size)
    elseif scale == FFTScaleSqrt
        inv(sqrt(Float32(fft_size)))
    else
        1.0f0
    end
    window = window_coeffs(fft_size, window)
    if gain != 1.0f0
        @inbounds for k in eachindex(window)
            window[k] *= gain
        end
    end
    outbuf = Vector{ComplexF32}(undef, fft_size)
    plan = FFTW.plan_fft!(outbuf; flags = FFTW.ESTIMATE)
    perf = CreateFFTPerfStats(log_interval = perf_log_interval,
                              warmup_frames = perf_warmup_frames,
                              window_size = perf_window_size)
    ctx = FFTBlockContext(Base.Threads.Atomic{Bool}(true),
                          fft_size,
                          window,
                          Vector{ComplexF32}(undef, fft_size),
                          outbuf,
                          plan,
                          RingFrameBuffer(T, fft_size, poolsize),
                          nothing,
                          fill(false, poolsize),
                          UInt64(0),
                          UInt64(0),
                          UInt64(0),
                          UInt64(200),
                          perf,
                          UInt64(0),
                          UInt64(0),
                          UInt64(0),
                          UInt64(0),
                          UInt64(0),
                          UInt64(0),
                          UInt64(0),
                          UInt64(0),
                          UInt64(0),
                          nothing,
                          new_sinks,
                          sinks)
    if perf_log_interval > 0
        ctx.perf.task = Threads.@spawn perf_task!(ctx.perf, ctx.running)
    end
    ctx.worker = Threads.@spawn task!(ctx)
    return ctx
end

@inline function update_perf_window!(perf::FFTPerfStats, dt::UInt64)
    n = length(perf.window)
    new_bin = perf_hist_bin(perf, dt)
    if perf.window_count < n
        pos = perf.window_count + 1
        perf.window[pos] = dt
        perf.window_count += 1
        perf.window_sum_ns += dt
        perf.hist[new_bin] += 1
        perf.window_pos = (pos == n) ? 1 : (pos + 1)
    else
        pos = perf.window_pos
        old = perf.window[pos]
        old_bin = perf_hist_bin(perf, old)
        perf.window[pos] = dt
        perf.window_sum_ns = perf.window_sum_ns - old + dt
        if perf.hist[old_bin] > 0
            perf.hist[old_bin] -= 1
        end
        perf.hist[new_bin] += 1
        perf.window_pos = (pos == n) ? 1 : (pos + 1)
    end
    return nothing
end

@inline function window_percentile_ns(perf::FFTPerfStats, q::Float64)
    n = perf.window_count
    n == 0 && return UInt64(0)
    target = clamp(ceil(Int, q * n), 1, n)
    acc = 0
    @inbounds for i in eachindex(perf.hist)
        acc += perf.hist[i]
        if acc >= target
            lo = (UInt64(i - 1) * perf.hist_bin_ns)
            return lo + (perf.hist_bin_ns ÷ 2)
        end
    end
    return UInt64(length(perf.hist) - 1) * perf.hist_bin_ns
end

function maybe_update_perf!(perf::FFTPerfStats, dt::UInt64)
    perf.count += 1
    if dt > perf.raw_max_ns
        perf.raw_max_ns = dt
    end

    if perf.count <= perf.warmup_frames
        return nothing
    end

    perf.metric_count += 1
    perf.total_ns += dt
    if dt > perf.max_ns
        perf.max_ns = dt
    end
    update_perf_window!(perf, dt)

    if (perf.metric_count % perf.log_interval) == 0
        avg_ns = perf.total_ns ÷ perf.metric_count
        window_n = max(perf.window_count, 1)
        win_avg_ns = perf.window_sum_ns ÷ UInt64(window_n)
        p95_ns = window_percentile_ns(perf, 0.95)
        p99_ns = window_percentile_ns(perf, 0.99)
        println("FFTBlock perf: avg_us=", Float64(avg_ns) / 1000.0,
                " win_avg_us=", Float64(win_avg_ns) / 1000.0,
                " p95_us=", Float64(p95_ns) / 1000.0,
                " p99_us=", Float64(p99_ns) / 1000.0,
                " max_us=", Float64(perf.max_ns) / 1000.0,
                " raw_max_us=", Float64(perf.raw_max_ns) / 1000.0,
                " warmup=", Int64(perf.warmup_frames),
                " count=", Int64(perf.metric_count),
                " total_count=", Int64(perf.count))
    end
    return nothing
end

function enqueue_perf_sample!(context::FFTBlockContext, dt::UInt64)
    perf = context.perf
    if !perf.first_logged
        println("FFTBlock perf: first_frame")
        perf.first_logged = true
    end
    if isfull(perf.samples)
        # Metrics can be degraded under load; never block signal processing.
        perf.drop_count += 1
        return nothing
    end
    put!(perf.samples, dt)
    return nothing
end

function perf_task!(perf::FFTPerfStats, running::Base.Threads.Atomic{Bool})
    try
        while running[] || isready(perf.samples)
            if isready(perf.samples)
                dt = take!(perf.samples)
                maybe_update_perf!(perf, dt)
            else
                yield()
            end
        end
    catch e
        if !(e isa InterruptException)
            println("FFTBlock perf task error: ", e)
        end
    end
    return nothing
end

const FRAME_SYNC_HANDOFF_JUMP_WARN = Int64(64)
const FFTBLOCK_INPUT_JUMP_WARN = Int64(64)

@inline function is_framesync_sink(sink)::Bool
    # Keep FFTBlock decoupled from concrete FrameSync type declaration order.
    return occursin("ISDBTFrameSyncContext", string(typeof(sink)))
end

@inline function find_framesync_sink(sinks)::Union{Nothing, SignalFlowBlock}
    for sink in sinks
        if is_framesync_sink(sink)
            return sink
        end
    end
    return nothing
end

@inline function guard_framesync_handoff_seq!(context::FFTBlockContext, raw_seq::UInt64, sink)
    if raw_seq == 0 || !SeqTrace.is_enabled() || !SeqTrace.stage_allowed("ISDBTFrameSync")
        return raw_seq
    end
    prev_raw = context.fs_handoff_last_raw_seq
    if prev_raw != 0
        delta = Int64(raw_seq) - Int64(prev_raw)
        if abs(delta) > FRAME_SYNC_HANDOFF_JUMP_WARN
            context.fs_handoff_anomaly_count += UInt64(1)
            AsyncLogger.log_async("FFTBlock: fs_handoff_seq_probe sink=",
                                  string(typeof(sink)),
                                  " prev=",
                                  Int64(prev_raw),
                                  " cur=",
                                  Int64(raw_seq),
                                  " delta=",
                                  delta,
                                  " anomalies=",
                                  Int64(context.fs_handoff_anomaly_count))
        end
    end

    guarded = raw_seq
    prev_guard = context.fs_handoff_last_guard_seq
    if prev_guard != 0 && raw_seq <= prev_guard
        if prev_guard < typemax(UInt64)
            guarded = prev_guard + UInt64(1)
        end
        context.fs_handoff_guard_count += UInt64(1)
        AsyncLogger.log_async("FFTBlock: fs_handoff_guard sink=",
                              string(typeof(sink)),
                              " raw=",
                              Int64(raw_seq),
                              " guarded=",
                              Int64(guarded),
                              " prev_guard=",
                              Int64(prev_guard),
                              " guards=",
                              Int64(context.fs_handoff_guard_count))
    end
    context.fs_handoff_last_raw_seq = raw_seq
    context.fs_handoff_last_guard_seq = guarded
    return guarded
end

@inline function maybe_log_fftblock_input_probe!(context::FFTBlockContext, in_seq::UInt64)
    if in_seq == 0 || !SeqTrace.is_enabled() || !SeqTrace.stage_allowed("FFTBlock")
        return nothing
    end
    prev = context.last_in_seq
    if prev != 0
        delta = Int64(in_seq) - Int64(prev)
        if abs(delta) > FFTBLOCK_INPUT_JUMP_WARN
            context.in_seq_anomaly_count += UInt64(1)
            AsyncLogger.log_async("FFTBlock: in_seq_probe prev=",
                                  Int64(prev),
                                  " cur=",
                                  Int64(in_seq),
                                  " delta=",
                                  delta,
                                  " anomalies=",
                                  Int64(context.in_seq_anomaly_count))
        end
    end
    return nothing
end

@inline function fill_fft_input!(context::FFTBlockContext{ComplexF32},
                                 src::AbstractVector{ComplexF32},
                                 n::Int)
    @inbounds @simd for k in 1:n
        # SIMD-friendly scalar multiply for both real/imag parts.
        @fastmath context.outbuf[k] = src[k] * context.window[k]
    end
    return nothing
end

@inline function fill_fft_input!(context::FFTBlockContext{T},
                                 src::AbstractVector{T},
                                 n::Int) where {T<:Complex}
    @inbounds @simd for k in 1:n
        v = src[k]
        w = context.window[k]
        @fastmath context.outbuf[k] = ComplexF32(Float32(real(v)) * w, Float32(imag(v)) * w)
    end
    return nothing
end

@inline function fill_fft_input!(context::FFTBlockContext{T},
                                 src::AbstractVector{T},
                                 n::Int) where {T<:Real}
    @inbounds @simd for k in 1:n
        @fastmath context.outbuf[k] = ComplexF32(Float32(src[k]) * context.window[k], 0f0)
    end
    return nothing
end

function task!(context::FFTBlockContext{T}) where {T}
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                if rd_buffer.store_size == context.fft_size
                    in_seq = UInt64(0)
                    out_seq = UInt64(0)
                    composed = context.frame_composed[rd_index]
                    if SeqTrace.is_enabled()
                        in_seq = SeqTrace.get_seq(rd_buffer.buf)
                        SeqTrace.log_in!("FFTBlock", context, in_seq; strict = false)
                    end
                    t0 = context.perf.log_interval > 0 ? time_ns() : UInt64(0)
                    fill_fft_input!(context, rd_buffer.buf, context.fft_size)
                    mul!(context.outbuf, context.plan, context.outbuf)
                    if context.perf.log_interval > 0
                        enqueue_perf_sample!(context, time_ns() - t0)
                    end
                    while isready(context.new_sinks)
                        push!(context.sinks, take!(context.new_sinks))
                    end
                    if SeqTrace.is_enabled()
                        out_seq = in_seq
                        if composed || in_seq == 0
                            context.out_seq_local += UInt64(1)
                            out_seq = context.out_seq_local
                        else
                            context.out_seq_local = in_seq
                        end
                        fs_sink = find_framesync_sink(context.sinks)
                        if fs_sink !== nothing
                            out_seq = guard_framesync_handoff_seq!(context, out_seq, fs_sink)
                        end
                        SeqTrace.set_seq!(context.outbuf, out_seq)
                        SeqTrace.log_out!("FFTBlock", context, out_seq; strict = false)
                    end
                    for sink in context.sinks
                        ok = input!(sink, context.outbuf, context.fft_size)
                        if ok == -1
                            context.sink_push_fail += UInt64(1)
                        end
                    end
                    context.out_dispatched += UInt64(1)
                end
                context.frame_composed[rd_index] = false
                rd_buffer.store_size = 0
                put!(context.ringbuffer.freeQ, rd_index)
            else
                yield()
            end
        end
    catch e
        if !(e isa InterruptException)
            println("FFTBlock error: ", e)
        end
    end
    return nothing
end

function input!(context::FFTBlockContext{T}, samples::AbstractVector{T}, samples_size::Int) where {T}
    if !context.running[] || samples_size <= 0
        return false
    end

    actual_size = min(samples_size, length(samples))
    if actual_size <= 0
        return 0
    end

    if SeqTrace.is_enabled()
        in_seq = SeqTrace.get_seq(samples)
        if in_seq == 0
            context.in_missing_count += 1
            if SeqTrace.stage_allowed("FFTBlock") &&
               (context.in_missing_count % context.in_missing_log_interval) == 0
                AsyncLogger.log_async("SeqTrace[FFTBlock] in_missing count=",
                                      Int64(context.in_missing_count))
            end
        else
            maybe_log_fftblock_input_probe!(context, in_seq)
            if SeqTrace.stage_allowed("FFTBlock") &&
               context.last_in_seq != 0 &&
               in_seq != context.last_in_seq + 1
                AsyncLogger.log_async("SeqTrace[FFTBlock] in_jump expected=",
                                      Int64(context.last_in_seq + 1),
                                      " actual=", Int64(in_seq),
                                      " delta=", Int64(in_seq) - Int64(context.last_in_seq))
            end
            context.last_in_seq = in_seq
        end
    end

    if actual_size == context.ringbuffer.frame_size && context.holdbuf === nothing
        if !isready(context.ringbuffer.freeQ)
            return -1
        end
        wr_index = take!(context.ringbuffer.freeQ)
        wr_buffer = context.ringbuffer.bufs[wr_index]
        copyto!(wr_buffer.buf, 1, samples, 1, actual_size)
        if SeqTrace.is_enabled()
            SeqTrace.inherit_seq!(samples, wr_buffer.buf)
        end
        context.frame_composed[wr_index] = false
        wr_buffer.store_size = actual_size
        put!(context.ringbuffer.fullQ, wr_index)
        context.in_write_count += UInt64(actual_size)
        context.in_full_enqueued += UInt64(1)
        return samples_size
    end

    remain_size = actual_size
    while remain_size > 0
        if context.holdbuf == nothing && isready(context.ringbuffer.freeQ)
            context.holdbuf = take!(context.ringbuffer.freeQ)
            context.frame_composed[context.holdbuf] = true
        end

        if context.holdbuf == nothing
            return -1
        else
            write_frame = context.ringbuffer.bufs[context.holdbuf]
            if SeqTrace.is_enabled() && write_frame.store_size == 0
                SeqTrace.inherit_seq!(samples, write_frame.buf)
            end
            copy_size = min(remain_size, context.ringbuffer.frame_size - write_frame.store_size)
            copyto!(write_frame.buf, write_frame.store_size + 1, samples, actual_size - remain_size + 1, copy_size)
            write_frame.store_size += copy_size
            remain_size -= copy_size
            context.in_write_count += UInt64(copy_size)
            if write_frame.store_size >= context.ringbuffer.frame_size
                put!(context.ringbuffer.fullQ, context.holdbuf)
                context.in_full_enqueued += UInt64(1)
                context.holdbuf = nothing
            end
        end
    end

    return samples_size
end

function stop!(context::FFTBlockContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    if context.perf.task !== nothing
        wait(context.perf.task)
    end
    if context.perf.log_interval > 0 && !context.perf.total_logged
        println("FFTBlock perf: total_frames=", Int64(context.perf.count))
        if context.perf.metric_count > 0
            avg_ns = context.perf.total_ns ÷ context.perf.metric_count
            println("FFTBlock perf: final_avg_us=", Float64(avg_ns) / 1000.0,
                    " final_max_us=", Float64(context.perf.max_ns) / 1000.0,
                    " raw_max_us=", Float64(context.perf.raw_max_ns) / 1000.0,
                    " warmup=", Int64(context.perf.warmup_frames),
                    " measured_frames=", Int64(context.perf.metric_count),
                    " dropped_metrics=", Int64(context.perf.drop_count))
        else
            println("FFTBlock perf: final_avg_us=NA final_max_us=NA raw_max_us=",
                    Float64(context.perf.raw_max_ns) / 1000.0,
                    " warmup=", Int64(context.perf.warmup_frames),
                    " measured_frames=0",
                    " dropped_metrics=", Int64(context.perf.drop_count))
        end
        context.perf.total_logged = true
    end
    println("FFTBlock input stats: total_samples=", Int64(context.in_write_count),
            " full_frames=", Int64(context.in_full_enqueued),
            " out_frames=", Int64(context.out_dispatched),
            " sink_fail=", Int64(context.sink_push_fail))
    return nothing
end

end
