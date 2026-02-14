module AWGNInjector

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer
import ..SeqTrace

using Random

mutable struct AWGNInjectorContext <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    snr_db::Float64
    frame_size::Int
    log_stats::Bool
    log_interval::Float64
    last_log_time::Float64
    outbuf::Vector{ComplexF32}
    ringbuffer::RingFrameBuffer{ComplexF32}
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
    rng::Xoshiro
end

function CreateAWGNInjector(::Type{T};
                            snr_db::Real,
                            frame_size::Int,
                            log_stats::Bool = false,
                            log_interval::Real = 1.0,
                            poolsize::Int = 8,
                            seed::Integer = 0) where {T<:ComplexF32}
    frame_size < 1 && error("AWGNInjector: frame_size must be >= 1.")
    poolsize < 1 && error("AWGNInjector: poolsize must be at least 1.")
    log_interval <= 0 && error("AWGNInjector: log_interval must be > 0.")
    isfinite(snr_db) || error("AWGNInjector: snr_db must be finite.")
    rng = seed == 0 ? Xoshiro() : Xoshiro(UInt64(seed))

    new_sinks = Channel{SignalFlowBlock}(64)
    sinks = Vector{SignalFlowBlock}()
    ctx = AWGNInjectorContext(Base.Threads.Atomic{Bool}(true),
                              Float64(snr_db),
                              frame_size,
                              log_stats,
                              Float64(log_interval),
                              0.0,
                              Vector{ComplexF32}(undef, frame_size),
                              RingFrameBuffer(ComplexF32, frame_size, poolsize),
                              nothing,
                              new_sinks,
                              sinks,
                              rng)
    ctx.worker = Threads.@spawn task!(ctx)
    return ctx
end

function set_snr_db!(context::AWGNInjectorContext, snr_db::Real)
    isfinite(snr_db) || error("AWGNInjector: snr_db must be finite.")
    context.snr_db = Float64(snr_db)
    return nothing
end

function task!(context::AWGNInjectorContext)
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                if rd_buffer.store_size == context.frame_size
                    in_seq = UInt64(0)
                    if SeqTrace.is_enabled()
                        in_seq = SeqTrace.get_seq(rd_buffer.buf)
                        SeqTrace.log_in!("AWGNInjector", context, in_seq; strict = false)
                    end

                    sum_p = 0.0f0
                    @inbounds for i in 1:context.frame_size
                        v = rd_buffer.buf[i]
                        sum_p += abs2(v)
                    end
                    p_sig = Float64(sum_p) / context.frame_size
                    p_noise = p_sig / (10.0^(context.snr_db / 10.0))
                    sigma = sqrt(max(p_noise, 0.0) / 2.0)
                    sigma32 = Float32(sigma)

                    @inbounds for i in 1:context.frame_size
                        nre = sigma32 * randn(context.rng, Float32)
                        nim = sigma32 * randn(context.rng, Float32)
                        context.outbuf[i] = rd_buffer.buf[i] + ComplexF32(nre, nim)
                    end

                    if context.log_stats
                        now = time()
                        if now - context.last_log_time >= context.log_interval
                            println("AWGNInjector: snr_db=", round(context.snr_db, digits = 2),
                                    " p_sig_db=", round(10 * log10(max(p_sig, 1e-12)), digits = 2),
                                    " p_noise_db=", round(10 * log10(max(p_noise, 1e-12)), digits = 2))
                            context.last_log_time = now
                        end
                    end

                    if SeqTrace.is_enabled()
                        SeqTrace.set_seq!(context.outbuf, in_seq)
                        SeqTrace.log_out!("AWGNInjector", context, in_seq; strict = false)
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
            println("AWGNInjector error: ", e)
        end
    end
    return nothing
end

function input!(context::AWGNInjectorContext, samples::AbstractVector{ComplexF32}, samples_size::Int)
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

function stop!(context::AWGNInjectorContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

end
