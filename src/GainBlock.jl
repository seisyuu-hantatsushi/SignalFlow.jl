module GainBlock

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer

mutable struct GainBlockContext <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    gain::Float64
    frame_size::Int
    outbuf::Vector{ComplexF32}
    ringbuffer::RingFrameBuffer{ComplexF32}
    holdbuf::Union{Nothing, Int}
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
end

function CreateGainBlock(::Type{T}; gain::Real = 1.0, frame_size::Int, poolsize::Int = 8) where {T<:ComplexF32}
    frame_size < 1 && error("GainBlock: frame_size must be >= 1.")
    poolsize < 1 && error("GainBlock: poolsize must be at least 1.")

    new_sinks = Channel{SignalFlowBlock}(4)
    sinks = Vector{SignalFlowBlock}()
    ctx = GainBlockContext(Base.Threads.Atomic{Bool}(true),
                           Float64(gain),
                           frame_size,
                           Vector{ComplexF32}(undef, frame_size),
                           RingFrameBuffer(ComplexF32, frame_size, poolsize),
                           nothing,
                           nothing,
                           new_sinks,
                           sinks)
    ctx.worker = Threads.@spawn task!(ctx)
    return ctx
end

function task!(context::GainBlockContext)
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                if rd_buffer.store_size == context.frame_size
                    g = Float32(context.gain)
                    @inbounds for k in 1:context.frame_size
                        context.outbuf[k] = rd_buffer.buf[k] * g
                    end
                    while isready(context.new_sinks)
                        push!(context.sinks, take!(context.new_sinks))
                    end
                    for sink in context.sinks
                        input!(sink, context.outbuf, context.frame_size)
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
            println("GainBlock error: ", e)
        end
    end
    return nothing
end

function input!(context::GainBlockContext, samples::AbstractVector{ComplexF32}, samples_size::Int)
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

function stop!(context::GainBlockContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

end
