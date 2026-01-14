
module LPF

import ..SignalFlowBlock
import ..input!

using DSP

@enum FilterType begin
    IIR
    FIR
end

mutable struct LPF{T,F} <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    filter_type::FilterType
    filter::F
    outbuf::Vector{T}
    freeQ::Channel{Vector{T}}
    fullQ::Channel{Vector{T}}
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
                   poolsize::Int = 8) where {T}

    samplerate <= 0 && error("samplerate must be positive.")
    cutoff <= 0 && error("cutoff must be positive.")
    cutoff >= samplerate / 2 && error("cutoff must be less than Nyquist.")
    poolsize < 1 && error("poolsize must be at least 1.")

    filter = if filter_type == FIR
        transitionwidth <= 0 && error("transitionwidth must be positive.")
        tw = 2 * transitionwidth / samplerate
        fir_window = DSP.FIRWindow(; transitionwidth = tw, attenuation = attenuation)
        coeffs = DSP.digitalfilter(DSP.Lowpass(cutoff), fir_window; fs = samplerate)
        DSP.FIRFilter(coeffs)
    else
        iir_order < 1 && error("iir_order must be >= 1.")
        coeffs = DSP.digitalfilter(DSP.Lowpass(cutoff), DSP.Butterworth(iir_order); fs = samplerate)
        sos = DSP.SecondOrderSections(coeffs)
        DSP.DF2TFilter(sos, zeros(T, 2, length(sos.biquads)))
    end

    new_sinks = Channel{SignalFlowBlock}(4)
    sinks = Vector{SignalFlowBlock}()
    outbuf = Vector{T}(undef, 0)
    freeQ = Channel{Vector{T}}(poolsize)
    fullQ = Channel{Vector{T}}(poolsize)
    for _ in 1:poolsize
        put!(freeQ, Vector{T}(undef, 0))
    end

    lpf = LPF(Base.Threads.Atomic{Bool}(true),
              filter_type,
              filter,
              outbuf,
              freeQ,
              fullQ,
              nothing,
              new_sinks,
              sinks)
    lpf.worker = Threads.@spawn task!(lpf)
    return lpf
end

function task!(context::LPF{T}) where {T}
    try
        while context.running[]
            if isready(context.fullQ)
                buf = take!(context.fullQ)
                actual_size = length(buf)
                if actual_size > 0
                    if length(context.outbuf) < actual_size
                        resize!(context.outbuf, actual_size)
                    end
                    out = view(context.outbuf, 1:actual_size)
                    DSP.filt!(out, context.filter, buf)

                    while isready(context.new_sinks)
                        push!(context.sinks, take!(context.new_sinks))
                    end
                    for sink in context.sinks
                        input!(sink, context.outbuf, actual_size)
                    end
                end
                put!(context.freeQ, buf)
            else
                yield()
            end
        end
    catch e
        if !(e isa InterruptException)
            println(e)
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

    if isready(context.freeQ)
        buf = take!(context.freeQ)
        if length(buf) < actual_size
            resize!(buf, actual_size)
        end
        copyto!(buf, 1, samples, 1, actual_size)
        put!(context.fullQ, buf)
    else
        return -1
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
