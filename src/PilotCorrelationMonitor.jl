module PilotCorrelationMonitor

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer
import ..ISDBTPRBS

mutable struct PilotCorrelationMonitorContext <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    nfft::Int
    normal_bins::Vector{Vector{Int}}
    normal_refs::Vector{Vector{ComplexF32}}
    flip_bins::Vector{Vector{Int}}
    flip_refs::Vector{Vector{ComplexF32}}
    label::String
    log_interval::Float64
    last_log_time::Float64
    ringbuffer::RingFrameBuffer{ComplexF32}
    holdbuf::Union{Nothing, Int}
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
end

const LOG_LOCK = ReentrantLock()

function build_pilot_sets(nfft::Int,
                          pilot_spacing::Int,
                          pilot_offset0::Int,
                          pilot_offset_step::Int,
                          segment_carriers::Int,
                          segment_index::Int)
    prbs_bits = ISDBTPRBS.mode3_segment_prbs(segment_index; carriers = segment_carriers)
    normal_bins = Vector{Vector{Int}}(undef, 4)
    normal_refs = Vector{Vector{ComplexF32}}(undef, 4)
    flip_bins = Vector{Vector{Int}}(undef, 4)
    flip_refs = Vector{Vector{ComplexF32}}(undef, 4)

    for phase in 0:3
        offset = pilot_offset0 + pilot_offset_step * phase
        n_bins = Int[]
        n_refs = ComplexF32[]
        f_bins = Int[]
        f_refs = ComplexF32[]
        @inbounds for carrier in 0:segment_carriers - 1
            if mod(carrier - offset, pilot_spacing) == 0
                bit = prbs_bits[carrier + 1]
                ref = ISDBTPRBS.pilot_value_unit_from_bit(bit)
                b = ISDBTPRBS.seg0_carrier_to_bin(nfft, carrier, segment_carriers)
                bf = ISDBTPRBS.seg0_carrier_to_bin(nfft, segment_carriers - 1 - carrier, segment_carriers)
                if b > 0
                    push!(n_bins, b)
                    push!(n_refs, ref)
                end
                if bf > 0
                    push!(f_bins, bf)
                    push!(f_refs, ref)
                end
            end
        end
        normal_bins[phase + 1] = n_bins
        normal_refs[phase + 1] = n_refs
        flip_bins[phase + 1] = f_bins
        flip_refs[phase + 1] = f_refs
    end
    return normal_bins, normal_refs, flip_bins, flip_refs
end

function CreatePilotCorrelationMonitor(; nfft::Int = 8192,
                                       pilot_spacing::Int = 12,
                                       pilot_offset0::Int = 3,
                                       pilot_offset_step::Int = 3,
                                       segment_carriers::Int = 432,
                                       segment_index::Int = 0,
                                       label::AbstractString = "PilotCorr",
                                       log_interval::Real = 1.0,
                                       poolsize::Int = 8)
    nfft < 32 && error("PilotCorrelationMonitor: nfft must be >= 32.")
    pilot_spacing < 1 && error("PilotCorrelationMonitor: pilot_spacing must be >= 1.")
    poolsize < 1 && error("PilotCorrelationMonitor: poolsize must be at least 1.")
    log_interval <= 0 && error("PilotCorrelationMonitor: log_interval must be > 0.")
    segment_carriers < 1 && error("PilotCorrelationMonitor: segment_carriers must be >= 1.")

    normal_bins, normal_refs, flip_bins, flip_refs =
        build_pilot_sets(nfft, pilot_spacing, pilot_offset0, pilot_offset_step,
                         segment_carriers, segment_index)

    new_sinks = Channel{SignalFlowBlock}(64)
    sinks = Vector{SignalFlowBlock}()
    ctx = PilotCorrelationMonitorContext(Base.Threads.Atomic{Bool}(true),
                                         nfft,
                                         normal_bins,
                                         normal_refs,
                                         flip_bins,
                                         flip_refs,
                                         String(label),
                                         Float64(log_interval),
                                         0.0,
                                         RingFrameBuffer(ComplexF32, nfft, poolsize),
                                         nothing,
                                         nothing,
                                         new_sinks,
                                         sinks)
    ctx.worker = Threads.@spawn task!(ctx)
    return ctx
end

function corr_metric(buf::Vector{ComplexF32},
                     bins::Vector{Int},
                     refs::Vector{ComplexF32})
    s_re = 0.0
    s_im = 0.0
    @inbounds for i in 1:length(bins)
        v = buf[bins[i]]
        r = refs[i]
        s_re += real(v) * real(r) + imag(v) * imag(r)
        s_im += real(v) * imag(r) - imag(v) * real(r)
    end
    return sqrt(s_re * s_re + s_im * s_im) / max(length(bins), 1)
end

function task!(context::PilotCorrelationMonitorContext)
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                if rd_buffer.store_size == context.nfft
                    now = time()
                    if now - context.last_log_time >= context.log_interval
                        best_norm = -1.0
                        best_flip = -1.0
                        best_norm_phase = 0
                        best_flip_phase = 0
                        for phase in 1:4
                            m = corr_metric(rd_buffer.buf,
                                            context.normal_bins[phase],
                                            context.normal_refs[phase])
                            if m > best_norm
                                best_norm = m
                                best_norm_phase = phase - 1
                            end
                            mf = corr_metric(rd_buffer.buf,
                                             context.flip_bins[phase],
                                             context.flip_refs[phase])
                            if mf > best_flip
                                best_flip = mf
                                best_flip_phase = phase - 1
                            end
                        end
                        lock(LOG_LOCK) do
                            println("PilotCorr[", context.label, "]: normal=",
                                    round(best_norm, digits = 3), " (phase ",
                                    best_norm_phase, "), flip=",
                                    round(best_flip, digits = 3), " (phase ",
                                    best_flip_phase, ")")
                        end
                        context.last_log_time = now
                    end

                    while isready(context.new_sinks)
                        push!(context.sinks, take!(context.new_sinks))
                    end
                    for sink in context.sinks
                        input!(sink, rd_buffer.buf, context.nfft)
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
            println("PilotCorrelationMonitor error: ", e)
        end
    end
    return nothing
end

function input!(context::PilotCorrelationMonitorContext, samples::AbstractVector{ComplexF32}, samples_size::Int)
    if !context.running[] || samples_size <= 0
        return false
    end

    actual_size = min(samples_size, length(samples))
    if actual_size <= 0
        return 0
    end

    if actual_size != context.nfft
        return -1
    end

    if isready(context.ringbuffer.freeQ)
        idx = take!(context.ringbuffer.freeQ)
        buf = context.ringbuffer.bufs[idx]
        copyto!(buf.buf, 1, samples, 1, actual_size)
        buf.store_size = actual_size
        put!(context.ringbuffer.fullQ, idx)
    else
        return -1
    end

    return samples_size
end

function stop!(context::PilotCorrelationMonitorContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

end
