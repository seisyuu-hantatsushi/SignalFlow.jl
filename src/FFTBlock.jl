module FFTBlock

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer

using DSP
using FFTW

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
    scale::FFTScale
    scale_gain::Float32
    ringbuffer::RingFrameBuffer{T}
    holdbuf::Union{Nothing, Int}
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
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
                        poolsize::Int = 8) where {T}
    fft_size < 32 && error("FFTBlock: fft_size must be >= 32.")
    fft_size > 128 * 1024 && error("FFTBlock: fft_size must be <= 131072.")
    poolsize < 1 && error("FFTBlock: poolsize must be at least 1.")

    new_sinks = Channel{SignalFlowBlock}(4)
    sinks = Vector{SignalFlowBlock}()
    gain = if scale == FFTScaleUnity
        1.0f0 / Float32(fft_size)
    elseif scale == FFTScaleSqrt
        inv(sqrt(Float32(fft_size)))
    else
        1.0f0
    end
    ctx = FFTBlockContext(Base.Threads.Atomic{Bool}(true),
                          fft_size,
                          window_coeffs(fft_size, window),
                          Vector{ComplexF32}(undef, fft_size),
                          Vector{ComplexF32}(undef, fft_size),
                          scale,
                          gain,
                          RingFrameBuffer(T, fft_size, poolsize),
                          nothing,
                          nothing,
                          new_sinks,
                          sinks)
    ctx.worker = Threads.@spawn task!(ctx)
    return ctx
end

function task!(context::FFTBlockContext{ComplexF32})
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                if rd_buffer.store_size == context.fft_size
                    @inbounds for k in 1:context.fft_size
                        context.tmp[k] = rd_buffer.buf[k] * context.window[k]
                    end
                    context.outbuf .= fft(context.tmp)
                    if context.scale != FFTScaleNone
                        @inbounds for k in 1:context.fft_size
                            context.outbuf[k] *= context.scale_gain
                        end
                    end
                    while isready(context.new_sinks)
                        push!(context.sinks, take!(context.new_sinks))
                    end
                    for sink in context.sinks
                        input!(sink, context.outbuf, context.fft_size)
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
            println("FFTBlock error: ", e)
        end
    end
    return nothing
end

function task!(context::FFTBlockContext{T}) where {T<:Real}
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                if rd_buffer.store_size == context.fft_size
                    @inbounds for k in 1:context.fft_size
                        context.tmp[k] = ComplexF32(Float32(rd_buffer.buf[k]) * context.window[k], 0f0)
                    end
                    context.outbuf .= fft(context.tmp)
                    if context.scale != FFTScaleNone
                        @inbounds for k in 1:context.fft_size
                            context.outbuf[k] *= context.scale_gain
                        end
                    end
                    while isready(context.new_sinks)
                        push!(context.sinks, take!(context.new_sinks))
                    end
                    for sink in context.sinks
                        input!(sink, context.outbuf, context.fft_size)
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

function stop!(context::FFTBlockContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

end
