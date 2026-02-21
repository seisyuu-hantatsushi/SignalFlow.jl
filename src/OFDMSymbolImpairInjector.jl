module OFDMSymbolImpairInjector

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer
import ..SeqTrace

const TWO_PI = 2.0 * Float64(pi)

mutable struct OFDMSymbolImpairInjectorContext <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    nfft::Int
    sample_rate::Float64
    ncp::Int
    cfo_hz::Float64
    phase_jump_deg::Float64
    phase_jump_interval_frames::Int
    slope_rad_per_bin::Float64
    log_stats::Bool
    log_interval::Float64
    last_log_time::Float64
    outbuf::Vector{ComplexF32}
    ringbuffer::RingFrameBuffer{ComplexF32}
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
    phase_acc::Float64
    frame_count::UInt64
    jump_count::UInt64
end

@inline function wrap_phase(phase::Float64)
    if phase > Float64(pi)
        return phase - TWO_PI
    elseif phase < -Float64(pi)
        return phase + TWO_PI
    end
    return phase
end

function CreateOFDMSymbolImpairInjector(::Type{T};
                                        nfft::Int,
                                        sample_rate::Real,
                                        ncp::Int = 0,
                                        cfo_hz::Real = 0.0,
                                        phase_jump_deg::Real = 0.0,
                                        phase_jump_interval_frames::Int = 0,
                                        slope_rad_per_bin::Real = 0.0,
                                        log_stats::Bool = false,
                                        log_interval::Real = 1.0,
                                        poolsize::Int = 8) where {T<:ComplexF32}
    nfft < 1 && error("OFDMSymbolImpairInjector: nfft must be >= 1.")
    sample_rate <= 0 && error("OFDMSymbolImpairInjector: sample_rate must be > 0.")
    ncp < 0 && error("OFDMSymbolImpairInjector: ncp must be >= 0.")
    phase_jump_interval_frames < 0 && error("OFDMSymbolImpairInjector: phase_jump_interval_frames must be >= 0.")
    isfinite(cfo_hz) || error("OFDMSymbolImpairInjector: cfo_hz must be finite.")
    isfinite(phase_jump_deg) || error("OFDMSymbolImpairInjector: phase_jump_deg must be finite.")
    isfinite(slope_rad_per_bin) || error("OFDMSymbolImpairInjector: slope_rad_per_bin must be finite.")
    log_interval <= 0 && error("OFDMSymbolImpairInjector: log_interval must be > 0.")
    poolsize < 1 && error("OFDMSymbolImpairInjector: poolsize must be at least 1.")

    new_sinks = Channel{SignalFlowBlock}(64)
    sinks = Vector{SignalFlowBlock}()
    ctx = OFDMSymbolImpairInjectorContext(Base.Threads.Atomic{Bool}(true),
                                          nfft,
                                          Float64(sample_rate),
                                          ncp,
                                          Float64(cfo_hz),
                                          Float64(phase_jump_deg),
                                          phase_jump_interval_frames,
                                          Float64(slope_rad_per_bin),
                                          log_stats,
                                          Float64(log_interval),
                                          0.0,
                                          Vector{ComplexF32}(undef, nfft),
                                          RingFrameBuffer(ComplexF32, nfft, poolsize),
                                          nothing,
                                          new_sinks,
                                          sinks,
                                          0.0,
                                          0,
                                          0)
    ctx.worker = Threads.@spawn task!(ctx)
    return ctx
end

function task!(context::OFDMSymbolImpairInjectorContext)
    try
        center = 0.5 * (context.nfft + 1)
        symbol_dt = (context.nfft + context.ncp) / context.sample_rate
        dphi_sym = TWO_PI * context.cfo_hz * symbol_dt

        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                if rd_buffer.store_size == context.nfft
                    in_seq = UInt64(0)
                    if SeqTrace.is_enabled()
                        in_seq = SeqTrace.get_seq(rd_buffer.buf)
                        SeqTrace.log_in!("OFDMSymbolImpairInjector", context, in_seq; strict = false)
                    end

                    context.frame_count += 1
                    if context.phase_jump_interval_frames > 0 &&
                       context.phase_jump_deg != 0.0 &&
                       (context.frame_count % UInt64(context.phase_jump_interval_frames) == 0)
                        context.phase_acc = wrap_phase(context.phase_acc +
                                                       context.phase_jump_deg * Float64(pi) / 180.0)
                        context.jump_count += 1
                    end

                    @inbounds for k in 1:context.nfft
                        phi = context.phase_acc + context.slope_rad_per_bin * (k - center)
                        rot = ComplexF32(cos(phi), sin(phi))
                        context.outbuf[k] = rd_buffer.buf[k] * rot
                    end
                    context.phase_acc = wrap_phase(context.phase_acc + dphi_sym)

                    if context.log_stats
                        now = time()
                        if now - context.last_log_time >= context.log_interval
                            println("OFDMSymbolImpairInjector: cfo_hz=", round(context.cfo_hz, digits = 2),
                                    " slope_rad_per_bin=", round(context.slope_rad_per_bin, digits = 6),
                                    " phase_jump_deg=", round(context.phase_jump_deg, digits = 2),
                                    " jump_interval_frames=", context.phase_jump_interval_frames,
                                    " jumps=", context.jump_count)
                            context.last_log_time = now
                        end
                    end

                    if SeqTrace.is_enabled()
                        SeqTrace.set_seq!(context.outbuf, in_seq)
                        SeqTrace.log_out!("OFDMSymbolImpairInjector", context, in_seq; strict = false)
                    end
                    while isready(context.new_sinks)
                        push!(context.sinks, take!(context.new_sinks))
                    end
                    for sink in context.sinks
                        input!(sink, context.outbuf, context.nfft)
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
            println("OFDMSymbolImpairInjector error: ", e)
        end
    end
    return nothing
end

function input!(context::OFDMSymbolImpairInjectorContext, samples::AbstractVector{ComplexF32}, samples_size::Int)
    if !context.running[] || samples_size <= 0
        return false
    end
    actual_size = min(samples_size, length(samples))
    if actual_size != context.nfft
        return -1
    end

    if isready(context.ringbuffer.freeQ)
        idx = take!(context.ringbuffer.freeQ)
        buf = context.ringbuffer.bufs[idx]
        copyto!(buf.buf, 1, samples, 1, actual_size)
        if SeqTrace.is_enabled()
            SeqTrace.inherit_seq!(samples, buf.buf)
        end
        buf.store_size = actual_size
        put!(context.ringbuffer.fullQ, idx)
    else
        return -1
    end
    return samples_size
end

function stop!(context::OFDMSymbolImpairInjectorContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

end
