module ISDBTEVMMonitor

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer
import ..AsyncLogger

mutable struct ISDBTEVMMonitorContext <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    frame_size::Int
    label::String
    log_interval::Float64
    last_log_time::Float64
    drop_log_interval::Int
    dropped_frames::Int
    input_mismatch_count::Int
    modulation::Symbol
    points::Vector{ComplexF32}
    decisions::Vector{ComplexF32}
    ringbuffer::RingFrameBuffer{ComplexF32}
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
end

const LOG_LOCK = ReentrantLock()

function parse_modulation(modulation::Union{Symbol,AbstractString})
    mod = Symbol(lowercase(String(modulation)))
    if !(mod in (:qpsk, Symbol("16qam"), Symbol("64qam")))
        error("ISDBTEVMMonitor: modulation must be one of qpsk/16qam/64qam.")
    end
    return mod
end

function constellation_points(modulation::Symbol)
    if modulation == :qpsk
        s = Float32(inv(sqrt(2.0)))
        return ComplexF32[
            ComplexF32(s, s),
            ComplexF32(-s, s),
            ComplexF32(-s, -s),
            ComplexF32(s, -s),
        ]
    elseif modulation == Symbol("16qam")
        a = Float32(inv(sqrt(10.0)))
        lv = (-3, -1, 1, 3)
        pts = ComplexF32[]
        for re in lv, im in lv
            push!(pts, ComplexF32(a * re, a * im))
        end
        return pts
    else
        a = Float32(inv(sqrt(42.0)))
        lv = (-7, -5, -3, -1, 1, 3, 5, 7)
        pts = ComplexF32[]
        for re in lv, im in lv
            push!(pts, ComplexF32(a * re, a * im))
        end
        return pts
    end
end

function nearest_point(v::ComplexF32, points::Vector{ComplexF32})
    best = points[1]
    best_d2 = typemax(Float32)
    @inbounds for i in eachindex(points)
        d = v - points[i]
        d2 = abs2(d)
        if d2 < best_d2
            best_d2 = d2
            best = points[i]
        end
    end
    return best
end

function CreateISDBTEVMMonitor(; frame_size::Int,
                               modulation::Union{Symbol,AbstractString} = :qpsk,
                               label::AbstractString = "DataCarriers",
                               log_interval::Real = 1.0,
                               drop_log_interval::Int = 500,
                               poolsize::Int = 8)
    frame_size < 1 && error("ISDBTEVMMonitor: frame_size must be >= 1.")
    poolsize < 1 && error("ISDBTEVMMonitor: poolsize must be at least 1.")
    log_interval <= 0 && error("ISDBTEVMMonitor: log_interval must be > 0.")
    drop_log_interval < 1 && error("ISDBTEVMMonitor: drop_log_interval must be >= 1.")

    mod = parse_modulation(modulation)
    pts = constellation_points(mod)
    new_sinks = Channel{SignalFlowBlock}(64)
    sinks = Vector{SignalFlowBlock}()

    ctx = ISDBTEVMMonitorContext(Base.Threads.Atomic{Bool}(true),
                                 frame_size,
                                 String(label),
                                 Float64(log_interval),
                                 0.0,
                                 drop_log_interval,
                                 0,
                                 0,
                                 mod,
                                 pts,
                                 Vector{ComplexF32}(undef, frame_size),
                                 RingFrameBuffer(ComplexF32, frame_size, poolsize),
                                 nothing,
                                 new_sinks,
                                 sinks)
    ctx.worker = Threads.@spawn task!(ctx)
    return ctx
end

function task!(context::ISDBTEVMMonitorContext)
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                if rd_buffer.store_size == context.frame_size
                    now = time()
                    if now - context.last_log_time >= context.log_interval
                        cross = ComplexF64(0)
                        sum_dec = 0.0
                        sum_rx = 0.0
                        @inbounds for i in 1:context.frame_size
                            v = rd_buffer.buf[i]
                            d = nearest_point(v, context.points)
                            context.decisions[i] = d
                            cross += ComplexF64(v) * conj(ComplexF64(d))
                            sum_dec += abs2(d)
                            sum_rx += abs2(v)
                        end

                        if sum_dec > 0.0
                            g = cross / sum_dec
                            sum_err = 0.0
                            @inbounds for i in 1:context.frame_size
                                e = ComplexF64(rd_buffer.buf[i]) - g * ComplexF64(context.decisions[i])
                                sum_err += abs2(e)
                            end
                            ref_pow = abs2(g) * sum_dec
                            evm_rms = sqrt(sum_err / max(ref_pow, 1e-12))
                            evm_pct = 100.0 * evm_rms
                            evm_db = 20.0 * log10(max(evm_rms, 1e-12))
                            mer_db = 10.0 * log10(max(ref_pow / max(sum_err, 1e-12), 1e-12))
                            gain_abs = abs(g)
                            gain_phase_deg = angle(g) * 180.0 / pi
                            rx_db = 10.0 * log10(max(sum_rx / context.frame_size, 1e-12))
                            lock(LOG_LOCK) do
                                println("EVM[", context.label, ":", String(context.modulation), "]: evm=",
                                        round(evm_pct, digits = 2), "% (",
                                        round(evm_db, digits = 2), " dB), MER=",
                                        round(mer_db, digits = 2), " dB, gain=",
                                        round(gain_abs, digits = 3), "@",
                                        round(gain_phase_deg, digits = 2), " deg, rx=",
                                        round(rx_db, digits = 2), " dB")
                            end
                        end
                        context.last_log_time = now
                    end

                    while isready(context.new_sinks)
                        push!(context.sinks, take!(context.new_sinks))
                    end
                    for sink in context.sinks
                        input!(sink, rd_buffer.buf, context.frame_size)
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
            println("ISDBTEVMMonitor error: ", e)
        end
    end
    return nothing
end

function input!(context::ISDBTEVMMonitorContext, samples::AbstractVector{ComplexF32}, samples_size::Int)
    if !context.running[] || samples_size <= 0
        return false
    end

    actual_size = min(samples_size, length(samples))
    if actual_size != context.frame_size
        context.input_mismatch_count += 1
        if (context.input_mismatch_count % context.drop_log_interval) == 0
            AsyncLogger.log_async("EVM[", context.label, "]: dropped_mismatch_frames=",
                                  context.input_mismatch_count,
                                  " expected=", context.frame_size,
                                  " actual=", actual_size)
        end
        return samples_size
    end

    if isready(context.ringbuffer.freeQ)
        idx = take!(context.ringbuffer.freeQ)
        buf = context.ringbuffer.bufs[idx]
        copyto!(buf.buf, 1, samples, 1, actual_size)
        buf.store_size = actual_size
        put!(context.ringbuffer.fullQ, idx)
    else
        context.dropped_frames += 1
        if (context.dropped_frames % context.drop_log_interval) == 0
            AsyncLogger.log_async("EVM[", context.label, "]: dropped_backpressure_frames=",
                                  context.dropped_frames)
        end
        return samples_size
    end

    return samples_size
end

function stop!(context::ISDBTEVMMonitorContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

end
