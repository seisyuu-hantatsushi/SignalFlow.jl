module ADFMCOMMS2Src

using ADFMCOMMS2

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer
import ..AsyncLogger

mutable struct ADFMCOMMS2Src{T} <: SignalFlowBlock 
    running::Base.Threads.Atomic{Bool}
    adapter::ADFMCOMMS2.SDR_RxAdapter{T}
    ringbuffer::RingFrameBuffer{T}
    drop_on_backpressure::Bool
    backpressure_log_interval::Int
    recv_task::Union{Nothing, Task}
    dispatch_task::Union{Nothing, Task}
    recv_overrun_count::Int
    dropped_frame_count::Int
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
end

mutable struct RecvGapStats
    count::Int
    sum_gap_ns::Int64
    max_gap_ns::Int64
    over_10ms::Int
    over_20ms::Int
    over_40ms::Int
    prev_recv_ns::Int64
    recent_gaps_ns::Vector{Int64}
    recent_count::Int
    recent_head::Int
    last_burst_log_ns::Int64
end

mutable struct DispatchSinkStats
    calls::Int
    total_ns::Int64
    max_ns::Int64
    over_1ms::Int
    over_5ms::Int
end

DispatchSinkStats() = DispatchSinkStats(0, 0, 0, 0, 0)

@inline function update_dispatch_sink_stats!(stats::DispatchSinkStats, elapsed_ns::Int64)
    stats.calls += 1
    stats.total_ns += elapsed_ns
    if elapsed_ns > stats.max_ns
        stats.max_ns = elapsed_ns
    end
    if elapsed_ns >= 1_000_000
        stats.over_1ms += 1
    end
    if elapsed_ns >= 5_000_000
        stats.over_5ms += 1
    end
    return nothing
end

@inline function reset_dispatch_sink_stats!(stats::DispatchSinkStats)
    stats.calls = 0
    stats.total_ns = 0
    stats.max_ns = 0
    stats.over_1ms = 0
    stats.over_5ms = 0
    return nothing
end

function RecvGapStats(window_size::Int = 64)
    window_size < 1 && error("ADFMCOMMS2Src: gap window_size must be at least 1.")
    return RecvGapStats(0, 0, 0, 0, 0, 0, 0, Vector{Int64}(undef, window_size), 0, 1, 0)
end

@inline function push_recent_gap!(stats::RecvGapStats, gap_ns::Int64)
    stats.recent_gaps_ns[stats.recent_head] = gap_ns
    stats.recent_head += 1
    if stats.recent_head > length(stats.recent_gaps_ns)
        stats.recent_head = 1
    end
    if stats.recent_count < length(stats.recent_gaps_ns)
        stats.recent_count += 1
    end
    return nothing
end

@inline function recent_gap_summary(stats::RecvGapStats)
    n = stats.recent_count
    if n == 0
        return 0.0, 0.0, 0.0, 0
    end
    min_gap = typemax(Int64)
    max_gap = typemin(Int64)
    sum_gap = 0.0
    @inbounds for i in 1:n
        g = stats.recent_gaps_ns[i]
        g < min_gap && (min_gap = g)
        g > max_gap && (max_gap = g)
        sum_gap += g
    end
    return Float64(min_gap) / 1_000_000,
           (sum_gap / n) / 1_000_000,
           Float64(max_gap) / 1_000_000,
           n
end

@inline function update_recv_gap!(stats::RecvGapStats, recv_ns::Integer)
    recv_ns_i = Int64(recv_ns)
    if stats.prev_recv_ns != 0
        gap_ns = recv_ns_i - stats.prev_recv_ns
        stats.count += 1
        stats.sum_gap_ns += gap_ns
        push_recent_gap!(stats, gap_ns)
        if gap_ns > stats.max_gap_ns
            stats.max_gap_ns = gap_ns
        end
        if gap_ns >= 10_000_000
            stats.over_10ms += 1
        end
        if gap_ns >= 20_000_000
            stats.over_20ms += 1
        end
        if gap_ns >= 40_000_000
            stats.over_40ms += 1
        end
        stats.prev_recv_ns = recv_ns_i
        return gap_ns
    end
    stats.prev_recv_ns = recv_ns_i
    return Int64(0)
end

@inline function maybe_log_gap_burst!(stats::RecvGapStats,
                                      gap_ns::Int64,
                                      now_ns::Int64;
                                      threshold_ns::Int64 = 40_000_000,
                                      min_interval_ns::Int64 = 500_000_000)
    if gap_ns < threshold_ns
        return nothing
    end
    if stats.last_burst_log_ns != 0 && (now_ns - stats.last_burst_log_ns) < min_interval_ns
        return nothing
    end
    recent_min_ms, recent_mean_ms, recent_max_ms, recent_n = recent_gap_summary(stats)
    AsyncLogger.log_async("recv gap burst: gap_ms=", round(Float64(gap_ns) / 1_000_000, digits = 3),
                          " recent_mean_ms=", round(recent_mean_ms, digits = 3),
                          " recent_min_ms=", round(recent_min_ms, digits = 3),
                          " recent_max_ms=", round(recent_max_ms, digits = 3),
                          " recent_n=", recent_n,
                          " over10=", stats.over_10ms,
                          " over20=", stats.over_20ms,
                          " over40=", stats.over_40ms)
    stats.last_burst_log_ns = now_ns
    return nothing
end

@inline function reset_recv_gap_interval!(stats::RecvGapStats)
    stats.count = 0
    stats.sum_gap_ns = 0
    stats.max_gap_ns = 0
    stats.over_10ms = 0
    stats.over_20ms = 0
    stats.over_40ms = 0
    return nothing
end

@inline function sample_fullscale(::Type{Complex{R}}) where {R<:AbstractFloat}
    return Float64(one(R))
end

@inline function sample_fullscale(::Type{Complex{R}}) where {R<:Integer}
    return Float64(typemax(R))
end

@inline function sample_fullscale(::Type{R}) where {R<:AbstractFloat}
    return Float64(one(R))
end

@inline function sample_fullscale(::Type{R}) where {R<:Integer}
    return Float64(typemax(R))
end

@inline function sample_clip_peak(v::Complex)
    return max(abs(real(v)), abs(imag(v)))
end

@inline function sample_clip_peak(v::Real)
    return abs(v)
end

function open(::Type{T}, uri::String, frequency::UInt64, samplerate::UInt32, bandwidth::UInt32;
              poolsize::Int = 256,
              chunk_size::Int = 32768,
              dispatch_burst::Int = 8,
              drop_on_backpressure::Bool = true,
              backpressure_log_interval::Int = 200) where {T}
    poolsize < 1 && error("ADFMCOMMS2Src: poolsize must be at least 1.")
    chunk_size < 1 && error("ADFMCOMMS2Src: chunk_size must be at least 1.")
    dispatch_burst < 1 && error("ADFMCOMMS2Src: dispatch_burst must be at least 1.")
    backpressure_log_interval < 1 && error("ADFMCOMMS2Src: backpressure_log_interval must be at least 1.")

    adapter = ADFMCOMMS2.SDR_RxAdapter(uri,
                                       frequency,
                                       samplerate,
                                       bandwidth,
                                       T)
    frame_size = Int(ADFMCOMMS2.SamplingFrameSize(adapter))
    rb_frame_size = min(frame_size, chunk_size)
    ringbuffer = RingFrameBuffer(T, rb_frame_size, poolsize)
    
    new_sinks = Channel{SignalFlowBlock}(4)
    sinks = Vector{SignalFlowBlock}()
    src = ADFMCOMMS2Src(Base.Threads.Atomic{Bool}(true),
                        adapter,
                        ringbuffer,
                        drop_on_backpressure,
                        backpressure_log_interval,
                        nothing,
                        nothing,
                        0,
                        0,
                        new_sinks,
                        sinks)
    src.recv_task = Threads.@spawn recv_task!(src)
    src.dispatch_task = Threads.@spawn dispatch_task!(src, dispatch_burst)
    return src
end

function close!(context::ADFMCOMMS2Src)

    if !context.running[]
        return nothing
    end

    context.running[] = false
    # Stop device first to unblock recv! / iio_buffer_refill wait promptly.
    try
        ADFMCOMMS2.stop!(context.adapter)
    catch
    end
    if context.recv_task !== nothing
        Base.disable_sigint() do
            try
                wait(context.recv_task)
            catch e
                if !(e isa InterruptException)
                    rethrow()
                end
            end
        end
    end
    if context.dispatch_task !== nothing
        Base.disable_sigint() do
            try
                wait(context.dispatch_task)
            catch e
                if !(e isa InterruptException)
                    rethrow()
                end
            end
        end
    end

end

function recv_task!(context::ADFMCOMMS2Src{T}) where {T}
    total_recv_samples::UInt64 = 0
    prev_total_recv_samples::UInt64 = 0
    prev_dropped_frames::Int = 0
    clip_count::UInt64 = 0
    clip_peak = 0.0
    clip_threshold = sample_fullscale(T) * 0.98
    tiny_eps = 1e-12
    recv_gap_stats = RecvGapStats()
    ADFMCOMMS2.start!(context.adapter)
    try
        recv_buffer = Vector{T}(undef, Int(ADFMCOMMS2.SamplingFrameSize(context.adapter)))
        rb_frame_size = length(context.ringbuffer.bufs[1].buf)
        prev_time = now_time = time_ns()
        while context.running[]
            now_time = time_ns()

            recv_size = ADFMCOMMS2.recv!(context.adapter, recv_buffer)
            if recv_size < 0
                error("ADFMCOMMS2Src: RF Receive Error")
            end
            recv_done_ns = time_ns()
            gap_ns = update_recv_gap!(recv_gap_stats, recv_done_ns)
            maybe_log_gap_burst!(recv_gap_stats, gap_ns, Int64(recv_done_ns))
            total_recv_samples += recv_size
            @inbounds for i in 1:recv_size
                p = Float64(sample_clip_peak(recv_buffer[i]))
                if p > clip_peak
                    clip_peak = p
                end
                if p >= clip_threshold
                    clip_count += 1
                end
            end

            offset = 1
            while offset <= recv_size && context.running[]
                idx = 0
                if isready(context.ringbuffer.freeQ)
                    idx = take!(context.ringbuffer.freeQ)
                elseif context.drop_on_backpressure && isready(context.ringbuffer.fullQ)
                    # Keep receiver real-time by dropping the oldest queued frame.
                    idx = take!(context.ringbuffer.fullQ)
                    context.dropped_frame_count += 1
                    if (context.dropped_frame_count % context.backpressure_log_interval) == 0
                        AsyncLogger.log_async("ADFMCOMMS2Src: dropped_backpressure_frames=", context.dropped_frame_count)
                    end
                else
                    context.recv_overrun_count += 1
                    if (context.recv_overrun_count % context.backpressure_log_interval) == 0
                        AsyncLogger.log_async("ADFMCOMMS2Src: recv_backpressure count=", context.recv_overrun_count)
                    end
                    yield()
                    continue
                end
                chunk_n = min(rb_frame_size, recv_size - offset + 1)
                buf = context.ringbuffer.bufs[idx]
                copyto!(buf.buf, 1, recv_buffer, offset, chunk_n)
                buf.store_size = chunk_n
                put!(context.ringbuffer.fullQ, idx)
                offset += chunk_n
            end
            
            if now_time - prev_time >= 1_000_000_000
                diff_time = Float32(now_time - prev_time)/1000_000_000
                diff_samples = total_recv_samples - prev_total_recv_samples
                AsyncLogger.log_async("recv rate: ",Float32(diff_samples)/diff_time, "S/sec")
                diff_samples_u = max(diff_samples, UInt64(1))
                clip_ratio = Float64(clip_count) / Float64(diff_samples_u)
                peak_norm = clip_peak / max(sample_fullscale(T), tiny_eps)
                AsyncLogger.log_async("recv clip: ratio=", round(clip_ratio, digits = 6),
                                      " peak_norm=", round(peak_norm, digits = 4),
                                      " threshold=", round(clip_threshold / max(sample_fullscale(T), tiny_eps), digits = 3))
                if recv_gap_stats.count > 0
                    gap_mean_ms = (Float64(recv_gap_stats.sum_gap_ns) / recv_gap_stats.count) / 1_000_000
                    gap_max_ms = Float64(recv_gap_stats.max_gap_ns) / 1_000_000
                    AsyncLogger.log_async("recv gap: mean_ms=", round(gap_mean_ms, digits = 3),
                                          " max_ms=", round(gap_max_ms, digits = 3),
                                          " over10=", recv_gap_stats.over_10ms,
                                          " over20=", recv_gap_stats.over_20ms,
                                          " over40=", recv_gap_stats.over_40ms,
                                          " n=", recv_gap_stats.count)
                end
                dropped_delta = context.dropped_frame_count - prev_dropped_frames
                if dropped_delta > 0
                    AsyncLogger.log_async("recv drop: frames=", dropped_delta, " total=", context.dropped_frame_count)
                end
                prev_time = now_time
                prev_total_recv_samples = total_recv_samples
                prev_dropped_frames = context.dropped_frame_count
                clip_count = 0
                clip_peak = 0.0
                reset_recv_gap_interval!(recv_gap_stats)
            end
        end
        
    catch e
        AsyncLogger.log_async("ADFMCOMMS2Src error: ", e)
    end
    try
        ADFMCOMMS2.stop!(context.adapter)
    catch
    end
end

function dispatch_task!(context::ADFMCOMMS2Src{T}, dispatch_burst::Int) where {T}
    sink_stats = Dict{String,DispatchSinkStats}()
    sink_stats_last_log_ns = time_ns()
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                while isready(context.new_sinks)
                    sink = take!(context.new_sinks)
                    push!(context.sinks, sink)
                    sink_key = string(typeof(sink))
                    if !haskey(sink_stats, sink_key)
                        sink_stats[sink_key] = DispatchSinkStats()
                    end
                end
                # Drain available frames in a burst to reduce queue latency spikes.
                processed = 0
                while isready(context.ringbuffer.fullQ)
                    idx = take!(context.ringbuffer.fullQ)
                    buf = context.ringbuffer.bufs[idx]
                    for sink in context.sinks
                        t0_ns = time_ns()
                        input!(sink, buf.buf, buf.store_size)
                        elapsed_ns = Int64(time_ns() - t0_ns)
                        sink_key = string(typeof(sink))
                        stats = get!(sink_stats, sink_key, DispatchSinkStats())
                        update_dispatch_sink_stats!(stats, elapsed_ns)
                    end
                    buf.store_size = 0
                    put!(context.ringbuffer.freeQ, idx)
                    processed += 1
                    if processed >= dispatch_burst
                        # Avoid starving other tasks during heavy bursts.
                        yield()
                        processed = 0
                    end
                end
                now_ns = time_ns()
                if now_ns - sink_stats_last_log_ns >= 1_000_000_000
                    worst_key = ""
                    worst_max_ns = Int64(0)
                    for (k, s) in sink_stats
                        if s.max_ns > worst_max_ns
                            worst_max_ns = s.max_ns
                            worst_key = k
                        end
                    end
                    if !isempty(worst_key)
                        ws = sink_stats[worst_key]
                        mean_ms = ws.calls > 0 ? (Float64(ws.total_ns) / ws.calls) / 1_000_000 : 0.0
                        max_ms = Float64(ws.max_ns) / 1_000_000
                        if ws.max_ns >= 1_000_000
                            AsyncLogger.log_async("dispatch sink lag: sink=", worst_key,
                                                  " mean_ms=", round(mean_ms, digits = 3),
                                                  " max_ms=", round(max_ms, digits = 3),
                                                  " over1=", ws.over_1ms,
                                                  " over5=", ws.over_5ms,
                                                  " calls=", ws.calls)
                        end
                    end
                    for s in values(sink_stats)
                        reset_dispatch_sink_stats!(s)
                    end
                    sink_stats_last_log_ns = now_ns
                end
            else
                yield()
            end
        end
    catch e
        if !(e isa InterruptException)
            AsyncLogger.log_async("ADFMCOMMS2Src dispatch error: ", e)
        end
    end
    return nothing
end

end
