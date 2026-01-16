
module LPF

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer

using DSP

@enum FilterType begin
    IIR
    FIR
end

mutable struct LPF{T,F} <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    filter_type::FilterType
    filter::F
    decimation::Int
    outbuf::Vector{T}
    ringbuffer::RingFrameBuffer{T}
    holdbuf::Union{Nothing, Int}
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
end

#=
 - transitionwidth: 低域通過帯域から遮断帯域までの「遷移帯域の幅」です。小さい
    ほど急峻なフィルタになり、その分タップ数が増えます。単位は Hz で指定していま
    す（内部で正規化）。
 - attenuation: 遮断帯域でどれだけ減衰させるか（dB）です。大きいほど遮断帯域の
    漏れが減り、タップ数が増えます。

Usage example:

    using SignalFlow
    using SignalFlow.LPF

    samplerate = 2_000_000.0
    cutoff = 200_000.0
    lpf = CreateLPF(ComplexF32, samplerate, cutoff; filter_type = FIR)

    # If you have a source block that provides ComplexF32 samples:
    # append_block!(src, lpf)
=#

function CreateLPF(::Type{T}, samplerate::Real, cutoff::Real;
                   filter_type::FilterType = FIR,
                   transitionwidth::Real = cutoff * 0.1,
                   attenuation::Real = 60.0,
                   iir_order::Int = 4,
                   poolsize::Int = 8,
                   frame_size::Int = 4096,
                   decimation::Int = 1) where {T}

    samplerate <= 0 && error("LPF: samplerate must be positive.")
    cutoff <= 0 && error("LPF: cutoff must be positive.")
    cutoff >= samplerate / 2 && error("LPF: cutoff must be less than Nyquist.")
    poolsize < 1 && error("LPF: poolsize must be at least 1.")
    frame_size < 1 && error("LPF: frame_size must be at least 1.")
    decimation < 1 && error("LPF: decimation must be >= 1.")
    cutoff >= samplerate / (2 * decimation) && error("LPF: cutoff must be less than Nyquist after decimation.")

    filter = if filter_type == FIR
        transitionwidth <= 0 && error("LPF: transitionwidth must be positive.")
        tw = 2 * transitionwidth / samplerate
        fir_window = DSP.FIRWindow(; transitionwidth = tw, attenuation = attenuation)
        coeffs = DSP.digitalfilter(DSP.Lowpass(cutoff), fir_window; fs = samplerate)
        DSP.FIRFilter(coeffs)
    else
        iir_order < 1 && error("LPF: iir_order must be >= 1.")
        coeffs = DSP.digitalfilter(DSP.Lowpass(cutoff), DSP.Butterworth(iir_order); fs = samplerate)
        sos = DSP.SecondOrderSections(coeffs)
        DSP.DF2TFilter(sos, zeros(T, 2, length(sos.biquads)))
    end

    new_sinks = Channel{SignalFlowBlock}(4)
    sinks = Vector{SignalFlowBlock}()
    outbuf = Vector{T}(undef, 0)
    ringbuffer = RingFrameBuffer(T, frame_size, poolsize)

    lpf = LPF(Base.Threads.Atomic{Bool}(true),
              filter_type,
              filter,
              decimation,
              outbuf,
              ringbuffer,
              nothing,
              nothing,
              new_sinks,
              sinks)
    lpf.worker = Threads.@spawn task!(lpf)
    return lpf
end

function task!(context::LPF{T}) where {T}
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
                    out = view(context.outbuf, 1:actual_size)
                    DSP.filt!(out, context.filter, view(rd_buffer.buf, 1:actual_size))

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
            println("LPF error: ", e)
        end
    end
    return nothing
end

function input!(context::LPF{T}, samples::AbstractVector{T}, samples_size::Int) where {T}
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

function stop!(context::LPF)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

end
