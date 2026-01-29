module IQFileSink

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer

mutable struct IQFileSink{T} <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    ringbuffer::RingFrameBuffer{T}
    holdbuf::Union{Nothing, Int}
    io::IOStream
    flush_interval_frames::Int
    frames_since_flush::Int
    worker::Union{Nothing,Task}
end

function CreateIQFileSink(::Type{T}, filepath::AbstractString;
                          append::Bool = false,
                          poolsize::Int = 8,
                          frame_size::Int = 4096,
                          flush_interval_frames::Int = 0) where {T}
    poolsize < 1 && error("IQFileSink: poolsize must be at least 1.")
    frame_size < 1 && error("IQFileSink: frame_size must be at least 1.")
    flush_interval_frames < 0 && error("IQFileSink: flush_interval_frames must be >= 0.")
    !isbitstype(T) && error("IQFileSink: element type must be a bits type.")

    io = open(filepath, append ? "a" : "w")
    ringbuffer = RingFrameBuffer(T, frame_size, poolsize)
    sink = IQFileSink(Base.Threads.Atomic{Bool}(true),
                      ringbuffer,
                      nothing,
                      io,
                      flush_interval_frames,
                      0,
                      nothing)
    sink.worker = Threads.@spawn task!(sink)
    return sink
end

function task!(context::IQFileSink{T}) where {T}
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                actual_size = rd_buffer.store_size
                if actual_size > 0
                    write(context.io, view(rd_buffer.buf, 1:actual_size))
                    if context.flush_interval_frames > 0
                        context.frames_since_flush += 1
                        if context.frames_since_flush >= context.flush_interval_frames
                            flush(context.io)
                            context.frames_since_flush = 0
                        end
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
            println("IQFileSink error: ", e)
        end
    end
    return nothing
end

function input!(context::IQFileSink{T}, samples::AbstractVector{T}, samples_size::Int) where {T}
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

function stop!(context::IQFileSink)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    flush(context.io)
    close(context.io)
    return nothing
end

end
