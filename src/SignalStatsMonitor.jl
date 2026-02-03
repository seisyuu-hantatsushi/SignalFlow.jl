module SignalStatsMonitor

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer

mutable struct SignalStatsMonitorContext <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    frame_size::Int
    label::String
    log_interval::Float64
    last_log_time::Float64
    ringbuffer::RingFrameBuffer{ComplexF32}
    holdbuf::Union{Nothing, Int}
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
end

const LOG_LOCK = ReentrantLock()

function CreateSignalStatsMonitor(; frame_size::Int,
                                  label::AbstractString,
                                  log_interval::Real = 1.0,
                                  poolsize::Int = 8)
    frame_size < 1 && error("SignalStatsMonitor: frame_size must be >= 1.")
    poolsize < 1 && error("SignalStatsMonitor: poolsize must be at least 1.")
    log_interval <= 0 && error("SignalStatsMonitor: log_interval must be > 0.")

    new_sinks = Channel{SignalFlowBlock}(4)
    sinks = Vector{SignalFlowBlock}()
    ctx = SignalStatsMonitorContext(Base.Threads.Atomic{Bool}(true),
                                    frame_size,
                                    String(label),
                                    Float64(log_interval),
                                    0.0,
                                    RingFrameBuffer(ComplexF32, frame_size, poolsize),
                                    nothing,
                                    nothing,
                                    new_sinks,
                                    sinks)
    ctx.worker = Threads.@spawn task!(ctx)
    return ctx
end

function task!(context::SignalStatsMonitorContext)
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                if rd_buffer.store_size == context.frame_size
                    now = time()
                    if now - context.last_log_time >= context.log_interval
                        n = context.frame_size
                        sum_re = 0.0
                        sum_im = 0.0
                        sum_p = 0.0
                        @inbounds for i in 1:n
                            v = rd_buffer.buf[i]
                            re = real(v)
                            im = imag(v)
                            sum_re += re
                            sum_im += im
                            sum_p += re * re + im * im
                        end
                        mean_re = sum_re / n
                        mean_im = sum_im / n
                        mean_p = sum_p / n
                        p_db = 10 * log10(max(mean_p, 1e-12))
                        lock(LOG_LOCK) do
                            println("SignalStats[", context.label, "]: p=",
                                    round(p_db, digits = 2), " dB, mean=(",
                                    round(mean_re, digits = 4), ", ",
                                    round(mean_im, digits = 4), ")")
                        end
                        context.last_log_time = now
                    end

                    while isready(context.new_sinks)
                        push!(context.sinks, take!(context.new_sinks))
                    end
                    for sink in context.sinks
                        input!(sink, rd_buffer.buf, context.frame_size)
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
            println("SignalStatsMonitor error: ", e)
        end
    end
    return nothing
end

function input!(context::SignalStatsMonitorContext, samples::AbstractVector{ComplexF32}, samples_size::Int)
    if !context.running[] || samples_size <= 0
        return false
    end

    actual_size = min(samples_size, length(samples))
    if actual_size != context.frame_size
        return -1
    end

    if isready(context.ringbuffer.freeQ)
        idx = take!(context.ringbuffer.freeQ)
        buf = context.ringbuffer.bufs[idx]
        copyto!(buf.buf, 1, samples, 1, actual_size)
        buf.store_size = actual_size
        put!(context.ringbuffer.fullQ, idx)
    else
        return -1
    end

    return samples_size
end

function stop!(context::SignalStatsMonitorContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

end
