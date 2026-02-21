module CFOPhaseInjector

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer
import ..SeqTrace

const TWO_PI = 2.0 * Float64(pi)

mutable struct CFOPhaseInjectorContext <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    frame_size::Int
    sample_rate::Float64
    cfo_hz::Float64
    phase_jump_deg::Float64
    phase_jump_interval_frames::Int
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

function CreateCFOPhaseInjector(::Type{T};
                                frame_size::Int,
                                sample_rate::Real,
                                cfo_hz::Real = 0.0,
                                phase_jump_deg::Real = 0.0,
                                phase_jump_interval_frames::Int = 0,
                                log_stats::Bool = false,
                                log_interval::Real = 1.0,
                                poolsize::Int = 8) where {T<:ComplexF32}
    frame_size < 1 && error("CFOPhaseInjector: frame_size must be >= 1.")
    sample_rate <= 0 && error("CFOPhaseInjector: sample_rate must be > 0.")
    isfinite(cfo_hz) || error("CFOPhaseInjector: cfo_hz must be finite.")
    isfinite(phase_jump_deg) || error("CFOPhaseInjector: phase_jump_deg must be finite.")
    phase_jump_interval_frames < 0 && error("CFOPhaseInjector: phase_jump_interval_frames must be >= 0.")
    log_interval <= 0 && error("CFOPhaseInjector: log_interval must be > 0.")
    poolsize < 1 && error("CFOPhaseInjector: poolsize must be at least 1.")

    new_sinks = Channel{SignalFlowBlock}(64)
    sinks = Vector{SignalFlowBlock}()
    ctx = CFOPhaseInjectorContext(Base.Threads.Atomic{Bool}(true),
                                  frame_size,
                                  Float64(sample_rate),
                                  Float64(cfo_hz),
                                  Float64(phase_jump_deg),
                                  phase_jump_interval_frames,
                                  log_stats,
                                  Float64(log_interval),
                                  0.0,
                                  Vector{ComplexF32}(undef, frame_size),
                                  RingFrameBuffer(ComplexF32, frame_size, poolsize),
                                  nothing,
                                  new_sinks,
                                  sinks,
                                  0.0,
                                  0,
                                  0)
    ctx.worker = Threads.@spawn task!(ctx)
    return ctx
end

function task!(context::CFOPhaseInjectorContext)
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                if rd_buffer.store_size == context.frame_size
                    in_seq = UInt64(0)
                    if SeqTrace.is_enabled()
                        in_seq = SeqTrace.get_seq(rd_buffer.buf)
                        SeqTrace.log_in!("CFOPhaseInjector", context, in_seq; strict = false)
                    end

                    context.frame_count += 1
                    if context.phase_jump_interval_frames > 0 &&
                       context.phase_jump_deg != 0.0 &&
                       (context.frame_count % UInt64(context.phase_jump_interval_frames) == 0)
                        context.phase_acc = wrap_phase(context.phase_acc +
                                                       context.phase_jump_deg * Float64(pi) / 180.0)
                        context.jump_count += 1
                    end

                    omega = TWO_PI * context.cfo_hz / context.sample_rate
                    phase = context.phase_acc
                    @inbounds for i in 1:context.frame_size
                        rot = ComplexF32(cos(phase), sin(phase))
                        context.outbuf[i] = rd_buffer.buf[i] * rot
                        phase = wrap_phase(phase + omega)
                    end
                    context.phase_acc = phase

                    if context.log_stats
                        now = time()
                        if now - context.last_log_time >= context.log_interval
                            println("CFOPhaseInjector: cfo_hz=", round(context.cfo_hz, digits = 2),
                                    " phase_jump_deg=", round(context.phase_jump_deg, digits = 2),
                                    " jump_interval_frames=", context.phase_jump_interval_frames,
                                    " jumps=", context.jump_count)
                            context.last_log_time = now
                        end
                    end

                    if SeqTrace.is_enabled()
                        SeqTrace.set_seq!(context.outbuf, in_seq)
                        SeqTrace.log_out!("CFOPhaseInjector", context, in_seq; strict = false)
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
            println("CFOPhaseInjector error: ", e)
        end
    end
    return nothing
end

function input!(context::CFOPhaseInjectorContext, samples::AbstractVector{ComplexF32}, samples_size::Int)
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

function stop!(context::CFOPhaseInjectorContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

end
