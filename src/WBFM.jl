module WBFM

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer

mutable struct WBFM{T} <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    samplerate::Float64
    deviation::Float64
    gain::Float32
    decimation::Int
    last_sample::T
    outbuf::Vector{Float32}
    ringbuffer::RingFrameBuffer{T}
    holdbuf::Union{Nothing, Int}
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
end

function CreateWBFM(::Type{T}, samplerate::Real;
                    deviation::Real = 75e3,
                    poolsize::Int = 8,
                    frame_size::Int = 4096,
                    decimation::Int = 1,
                    gain::Real = samplerate / (2 * pi * deviation)) where {T}
    samplerate <= 0 && error("WBFM: samplerate must be positive.")
    deviation <= 0 && error("WBFM: deviation must be positive.")
    poolsize < 1 && error("WBFM: poolsize must be at least 1.")
    frame_size < 1 && error("WBFM: frame_size must be at least 1.")
    decimation < 1 && error("WBFM: decimation must be >= 1.")

    new_sinks = Channel{SignalFlowBlock}(4)
    sinks = Vector{SignalFlowBlock}()
    outbuf = Vector{Float32}(undef, 0)
    ringbuffer = RingFrameBuffer(T, frame_size, poolsize)

    ctx = WBFM(Base.Threads.Atomic{Bool}(true),
               Float64(samplerate),
               Float64(deviation),
               Float32(gain),
               decimation,
               one(T),
               outbuf,
               ringbuffer,
               nothing,
               nothing,
               new_sinks,
               sinks)
    ctx.worker = Threads.@spawn task!(ctx)
    return ctx
end

function task!(context::WBFM{T}) where {T}
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                actual_size = rd_buffer.store_size
                if actual_size > 0
                    if length(context.outbuf) < actual_size
                        resize!(context.outbuf, actual_size)
                    end
                    prev = context.last_sample
                    @inbounds for i in 1:actual_size
                        x = rd_buffer.buf[i]
                        prod = x * conj(prev)
                        phase = atan(imag(prod), real(prod))
                        context.outbuf[i] = Float32(phase) * context.gain
                        prev = x
                    end
                    context.last_sample = prev

                    if context.decimation > 1
                        out_size = (actual_size - 1) ÷ context.decimation + 1
                        write_pos = 1
                        read_pos = 1
                        @inbounds while read_pos <= actual_size
                            context.outbuf[write_pos] = context.outbuf[read_pos]
                            write_pos += 1
                            read_pos += context.decimation
                        end
                        actual_size = out_size
                    end

                    while isready(context.new_sinks)
                        push!(context.sinks, take!(context.new_sinks))
                    end
                    for sink in context.sinks
                        input!(sink, context.outbuf, actual_size)
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
            println("WBFM error: ", e)
        end
    end
    return nothing
end

function input!(context::WBFM{T}, samples::AbstractVector{T}, samples_size::Int) where {T}
    if !context.running[] || samples_size <= 0
        return false
    end

    actual_size = min(samples_size, length(samples))
    if actual_size <= 0
        return 0
    end

    remain_size = actual_size
    while remain_size > 0
        if context.holdbuf == nothing && isready(context.ringbuffer.freeQ)
            context.holdbuf = take!(context.ringbuffer.freeQ)
        end

        if context.holdbuf == nothing
            return -1
        else
            write_frame = context.ringbuffer.bufs[context.holdbuf]
            copy_size = min(remain_size, context.ringbuffer.frame_size - write_frame.store_size)
            copyto!(write_frame.buf, write_frame.store_size + 1, samples, actual_size - remain_size + 1, copy_size)
            write_frame.store_size += copy_size
            remain_size -= copy_size
            if write_frame.store_size >= context.ringbuffer.frame_size
                put!(context.ringbuffer.fullQ, context.holdbuf)
                context.holdbuf = nothing
            end
        end
    end

    return samples_size
end

function stop!(context::WBFM)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

end
