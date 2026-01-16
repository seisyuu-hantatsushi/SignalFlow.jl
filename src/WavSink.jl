module WavSink

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer

@enum WavSampleFormat begin
    Int16PCM
    Float32PCM
end

mutable struct WavSink{T} <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    samplerate::UInt32
    channels::UInt16
    format::WavSampleFormat
    bytes_per_sample::UInt16
    block_align::UInt16
    ringbuffer::RingFrameBuffer{T}
    holdbuf::Union{Nothing, Int}
    io::IOStream
    data_bytes::UInt32
    worker::Union{Nothing,Task}
end

function write_u16(io::IO, value::UInt16)
    write(io, reinterpret(UInt8, [value]))
    return nothing
end

function write_u32(io::IO, value::UInt32)
    write(io, reinterpret(UInt8, [value]))
    return nothing
end

function write_header(io::IO, samplerate::UInt32, channels::UInt16, format::WavSampleFormat)
    audio_format = format == Float32PCM ? UInt16(3) : UInt16(1)
    bits_per_sample = format == Float32PCM ? UInt16(32) : UInt16(16)
    bytes_per_sample = UInt16(bits_per_sample ÷ 8)
    block_align = UInt16(channels * bytes_per_sample)
    byte_rate = samplerate * UInt32(block_align)

    write(io, "RIFF")
    write_u32(io, 0x00000000)
    write(io, "WAVE")
    write(io, "fmt ")
    write_u32(io, UInt32(16))
    write_u16(io, audio_format)
    write_u16(io, channels)
    write_u32(io, samplerate)
    write_u32(io, byte_rate)
    write_u16(io, block_align)
    write_u16(io, bits_per_sample)
    write(io, "data")
    write_u32(io, 0x00000000)
    return bytes_per_sample, block_align
end

function finalize_header(io::IO, data_bytes::UInt32)
    riff_size = UInt32(36) + data_bytes
    seek(io, 4)
    write_u32(io, riff_size)
    seek(io, 40)
    write_u32(io, data_bytes)
    seekend(io)
    return nothing
end

function CreateWavSink(::Type{T}, filepath::AbstractString;
                       samplerate::Real,
                       channels::Int = 2,
                       format::WavSampleFormat = UInt16PCM,
                       poolsize::Int = 8,
                       frame_size::Int = 4096) where {T}
    samplerate <= 0 && error("WavSink: samplerate must be positive.")
    channels < 1 && error("WavSink: channels must be at least 1.")
    poolsize < 1 && error("WavSink: poolsize must be at least 1.")
    frame_size < 1 && error("WavSink: frame_size must be at least 1.")

    if format == Int16PCM && T != Int16
        error("WavSink: format Int16PCM requires input element type Int16.")
    end
    if format == Float32PCM && T != Float32
        error("WavSink: format Float32PCM requires input element type Float32.")
    end

    io = open(filepath, "w")
    bytes_per_sample, block_align = write_header(io, UInt32(round(samplerate)), UInt16(channels), format)

    ringbuffer = RingFrameBuffer(T, frame_size, poolsize)
    sink = WavSink(Base.Threads.Atomic{Bool}(true),
                   UInt32(round(samplerate)),
                   UInt16(channels),
                   format,
                   bytes_per_sample,
                   block_align,
                   ringbuffer,
                   nothing,
                   io,
                   0x00000000,
                   nothing)
    sink.worker = Threads.@spawn task!(sink)
    return sink
end

function task!(context::WavSink{T}) where {T}
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                actual_size = rd_buffer.store_size
                if actual_size > 0
                    write(context.io, view(rd_buffer.buf, 1:actual_size))
                    context.data_bytes += UInt32(actual_size * sizeof(T))
                end
                rd_buffer.store_size = 0
                put!(context.ringbuffer.freeQ, rd_index)
            else
                yield()
            end
        end
    catch e
        if !(e isa InterruptException)
            println("WavSink error: ", e)
        end
    end
    return nothing
end

function input!(context::WavSink{T}, samples::AbstractVector{T}, samples_size::Int) where {T}
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

function stop!(context::WavSink)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    finalize_header(context.io, context.data_bytes)
    close(context.io)
    return nothing
end

end
