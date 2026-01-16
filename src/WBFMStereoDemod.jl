module WBFMStereoDemod

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer

using DSP

@enum OutputFormat begin
    Int16PCM
    Float32PCM
end

mutable struct PLLState
    phase::Float64
    freq::Float64
    alpha::Float64
    beta::Float64
end

mutable struct WBFMStereoDemod{T, PilotFilterT, LPRFilterT, LRFilterT, LMRFilterT, DeEmpFilterT, O} <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    samplerate::Float64
    output_rate::Float64
    output_format::OutputFormat
    deemphasis_us::Float64
    pll::PLLState
    pilot_filter::PilotFilterT
    lpr_filter::LPRFilterT
    lr_filter::LRFilterT
    lmr_filter::LMRFilterT
    deemph_filter::DeEmpFilterT
    ringbuffer::RingFrameBuffer{T}
    holdbuf::Union{Nothing, Int}
    outbuf::Vector{O}
    work_pilot::Vector{Float32}
    work_lpr::Vector{Float32}
    work_lr::Vector{Float32}
    work_lmr::Vector{Float32}
    work_l::Vector{Float32}
    work_r::Vector{Float32}
    work_l_de::Vector{Float32}
    work_r_de::Vector{Float32}
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
end

function pll_step!(pll::PLLState, sample::Float32)
    vco_sin = sin(pll.phase)
    vco_cos = cos(pll.phase)
    error = sample * vco_cos
    pll.freq += pll.beta * error
    pll.phase += pll.freq + pll.alpha * error
    if pll.phase > 2pi
        pll.phase -= 2pi
    elseif pll.phase < 0
        pll.phase += 2pi
    end
    return vco_sin
end

function make_iir(filter, ::Type{T}) where {T}
    sos = DSP.SecondOrderSections(filter)
    return DSP.DF2TFilter(sos, zeros(T, 2, length(sos.biquads)))
end

function CreateWBFMStereoDemod(::Type{T}, samplerate::Real;
                           output_rate::Real = 48_000,
                           output_format::OutputFormat = Float32PCM,
                           deemphasis_us::Real = 75.0,
                           poolsize::Int = 8,
                           frame_size::Int = 4096,
                           pilot_band::Tuple{Real,Real} = (18.5e3, 19.5e3),
                           lpr_cutoff::Real = 15e3,
                           lr_band::Tuple{Real,Real} = (23e3, 53e3),
                           pll_bandwidth::Real = 50.0) where {T}
    samplerate <= 0 && error("WBFMStereoDemod: samplerate must be positive.")
    output_rate <= 0 && error("WBFMStereoDemod: output_rate must be positive.")
    deemphasis_us <= 0 && error("WBFMStereoDemod: deemphasis_us must be positive.")
    poolsize < 1 && error("WBFMStereoDemod: poolsize must be at least 1.")
    frame_size < 1 && error("WBFMStereoDemod: frame_size must be at least 1.")
    lpr_cutoff <= 0 && error("WBFMStereoDemod: lpr_cutoff must be positive.")

    pilot_filter = DSP.digitalfilter(DSP.Bandpass(pilot_band[1], pilot_band[2]),
                                     DSP.Butterworth(4); fs = samplerate)
    lpr_filter = DSP.digitalfilter(DSP.Lowpass(lpr_cutoff),
                                   DSP.Butterworth(6); fs = samplerate)
    lr_filter = DSP.digitalfilter(DSP.Bandpass(lr_band[1], lr_band[2]),
                                  DSP.Butterworth(4); fs = samplerate)
    lmr_filter = DSP.digitalfilter(DSP.Lowpass(lpr_cutoff),
                                   DSP.Butterworth(6); fs = samplerate)
    de_tau = Float64(deemphasis_us) * 1e-6
    de_fc = 1.0 / (2 * pi * de_tau)
    deemph_filter = DSP.digitalfilter(DSP.Lowpass(de_fc),
                                      DSP.Butterworth(1); fs = output_rate)

    f0 = 2pi * 19_000.0 / Float64(samplerate)
    bw = pll_bandwidth
    zeta = 0.707
    theta = bw / Float64(samplerate)
    alpha = 2 * zeta * theta
    beta = theta * theta
    pll = PLLState(0.0, f0, alpha, beta)

    ringbuffer = RingFrameBuffer(T, frame_size, poolsize)
    new_sinks = Channel{SignalFlowBlock}(4)
    sinks = Vector{SignalFlowBlock}()

    outbuf = output_format == Int16PCM ? Vector{Int16}(undef, 0) : Vector{Float32}(undef, 0)
    ctx = WBFMStereoDemod(Base.Threads.Atomic{Bool}(true),
                      Float64(samplerate),
                      Float64(output_rate),
                      output_format,
                      Float64(deemphasis_us),
                      pll,
                      make_iir(pilot_filter, Float32),
                      make_iir(lpr_filter, Float32),
                      make_iir(lr_filter, Float32),
                      make_iir(lmr_filter, Float32),
                      make_iir(deemph_filter, Float32),
                      ringbuffer,
                      nothing,
                      outbuf,
                      Vector{Float32}(undef, 0),
                      Vector{Float32}(undef, 0),
                      Vector{Float32}(undef, 0),
                      Vector{Float32}(undef, 0),
                      Vector{Float32}(undef, 0),
                      Vector{Float32}(undef, 0),
                      Vector{Float32}(undef, 0),
                      Vector{Float32}(undef, 0),
                      nothing,
                      new_sinks,
                      sinks)
    ctx.worker = Threads.@spawn task!(ctx)
    return ctx
end

function resample_if_needed(x::Vector{Float32}, ratio::Float64)
    if ratio == 1.0
        return x
    end
    return Float32.(DSP.resample(x, ratio))
end

function task!(context::WBFMStereoDemod{T, PilotFilterT, LPRFilterT, LRFilterT, LMRFilterT, DeEmpFilterT, O}) where {T, PilotFilterT, LPRFilterT, LRFilterT, LMRFilterT, DeEmpFilterT, O}
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                actual_size = rd_buffer.store_size
                if actual_size > 0
                    if length(context.work_pilot) < actual_size
                        resize!(context.work_pilot, actual_size)
                        resize!(context.work_lpr, actual_size)
                        resize!(context.work_lr, actual_size)
                        resize!(context.work_lmr, actual_size)
                        resize!(context.work_l, actual_size)
                        resize!(context.work_r, actual_size)
                        resize!(context.work_l_de, actual_size)
                        resize!(context.work_r_de, actual_size)
                    end

                    inbuf = view(rd_buffer.buf, 1:actual_size)
                    DSP.filt!(context.work_pilot, context.pilot_filter, inbuf)
                    DSP.filt!(context.work_lpr, context.lpr_filter, inbuf)
                    DSP.filt!(context.work_lr, context.lr_filter, inbuf)

                    @inbounds for i in 1:actual_size
                        carrier = pll_step!(context.pll, context.work_pilot[i])
                        context.work_lmr[i] = 2f0 * context.work_lr[i] * Float32(sin(2.0 * context.pll.phase))
                    end
                    DSP.filt!(context.work_lmr, context.lmr_filter, context.work_lmr)

                    @inbounds for i in 1:actual_size
                        l = 0.5f0 * (context.work_lpr[i] + context.work_lmr[i])
                        r = 0.5f0 * (context.work_lpr[i] - context.work_lmr[i])
                        context.work_l[i] = l
                        context.work_r[i] = r
                    end

                    ratio = context.output_rate / context.samplerate
                    l_out = resample_if_needed(context.work_l, ratio)
                    r_out = resample_if_needed(context.work_r, ratio)
                    l_len = length(l_out)
                    r_len = length(r_out)
                    if length(context.work_l_de) < l_len
                        resize!(context.work_l_de, l_len)
                    end
                    if length(context.work_r_de) < r_len
                        resize!(context.work_r_de, r_len)
                    end
                    l_view = view(context.work_l_de, 1:l_len)
                    r_view = view(context.work_r_de, 1:r_len)
                    DSP.filt!(l_view, context.deemph_filter, l_out)
                    DSP.filt!(r_view, context.deemph_filter, r_out)
                    l_out = l_view
                    r_out = r_view
                    out_frames = min(length(l_out), length(r_out))
                    out_samples = out_frames * 2
                    if length(context.outbuf) < out_samples
                        resize!(context.outbuf, out_samples)
                    end

                    if context.output_format == Float32PCM
                        @inbounds for i in 1:out_frames
                            context.outbuf[2i - 1] = l_out[i]
                            context.outbuf[2i] = r_out[i]
                        end
                    else
                        @inbounds for i in 1:out_frames
                            l = clamp(l_out[i], -1f0, 1f0)
                            r = clamp(r_out[i], -1f0, 1f0)
                            context.outbuf[2i - 1] = Int16(round(Int, l * 32767f0))
                            context.outbuf[2i] = Int16(round(Int, r * 32767f0))
                        end
                    end

                    while isready(context.new_sinks)
                        push!(context.sinks, take!(context.new_sinks))
                    end
                    for sink in context.sinks
                        input!(sink, context.outbuf, out_samples)
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
            println("WBFMStereoDemod error: ", e)
        end
    end
    return nothing
end

function input!(context::WBFMStereoDemod{T}, samples::AbstractVector{T}, samples_size::Int) where {T}
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

function stop!(context::WBFMStereoDemod)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

end
