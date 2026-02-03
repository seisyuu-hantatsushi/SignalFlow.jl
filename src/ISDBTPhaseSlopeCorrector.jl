module ISDBTPhaseSlopeCorrector

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer
import ..ISDBTPRBS

const LOG_LOCK = ReentrantLock()

mutable struct ISDBTPhaseSlopeCorrectorContext <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    nfft::Int
    pilot_spacing::Int
    pilot_offset0::Int
    pilot_offset_step::Int
    pilot_values::Vector{ComplexF32}
    samplerate::Float64
    band_limit_hz::Float64
    segment_carriers::Int
    prbs_bits::Vector{Int}
    alpha::Float64
    auto_sp_phase::Bool
    symbol_index_ref::Union{Nothing,Base.Threads.Atomic{Int}}
    slope::Float64
    intercept::Float64
    log_stats::Bool
    log_interval::Float64
    last_log_time::Float64
    symbol_index::Int
    outbuf::Vector{ComplexF32}
    ringbuffer::RingFrameBuffer{ComplexF32}
    holdbuf::Union{Nothing, Int}
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
end

function pilot_positions(nfft::Int, spacing::Int, offset::Int)
    start = 1 + mod(offset, spacing)
    return collect(start:spacing:nfft)
end

function seg0_pilot_bins(nfft::Int, spacing::Int, offset::Int, segment_carriers::Int)
    bins = Int[]
    @inbounds for carrier in 0:segment_carriers - 1
        if mod(carrier - offset, spacing) == 0
            bin = ISDBTPRBS.seg0_carrier_to_bin(nfft, carrier, segment_carriers)
            bin > 0 && push!(bins, bin)
        end
    end
    sort!(bins)
    return bins
end

function unwrap_phases!(phases::Vector{Float64})
    n = length(phases)
    n < 2 && return
    @inbounds for i in 2:n
        dp = phases[i] - phases[i - 1]
        if dp > Float64(π)
            phases[i] -= 2 * Float64(π)
        elseif dp < -Float64(π)
            phases[i] += 2 * Float64(π)
        end
    end
    return
end

function CreateISDBTPhaseSlopeCorrector(; nfft::Int = 8192,
                                        pilot_spacing::Int = 12,
                                        pilot_offset0::Int = 3,
                                        pilot_offset_step::Int = 3,
                                        pilot_values::Union{Nothing,Vector{ComplexF32}} = nothing,
                                        samplerate::Real = 8_000_000,
                                        band_limit_hz::Real = 0.0,
                                        segment_carriers::Int = 432,
                                        segment_index::Int = 0,
                                        alpha::Real = 0.2,
                                        auto_sp_phase::Bool = true,
                                        symbol_index_ref::Union{Nothing,Base.Threads.Atomic{Int}} = nothing,
                                        log_stats::Bool = false,
                                        log_interval::Real = 1.0,
                                        poolsize::Int = 8)
    nfft < 32 && error("ISDBTPhaseSlopeCorrector: nfft must be >= 32.")
    pilot_spacing < 1 && error("ISDBTPhaseSlopeCorrector: pilot_spacing must be >= 1.")
    alpha <= 0 && error("ISDBTPhaseSlopeCorrector: alpha must be positive.")
    poolsize < 1 && error("ISDBTPhaseSlopeCorrector: poolsize must be at least 1.")
    log_interval <= 0 && error("ISDBTPhaseSlopeCorrector: log_interval must be > 0.")

    pos = segment_carriers > 0 ?
          seg0_pilot_bins(nfft, pilot_spacing, pilot_offset0, segment_carriers) :
          pilot_positions(nfft, pilot_spacing, pilot_offset0)
    pilot_values === nothing && (pilot_values = fill(ComplexF32(1, 0), length(pos)))
    length(pilot_values) != length(pos) && error("ISDBTPhaseSlopeCorrector: pilot_values length mismatch.")
    band_limit_hz < 0 && error("ISDBTPhaseSlopeCorrector: band_limit_hz must be >= 0.")
    segment_carriers < 1 && error("ISDBTPhaseSlopeCorrector: segment_carriers must be >= 1.")

    prbs_bits = band_limit_hz > 0 ? ISDBTPRBS.mode3_segment_prbs(segment_index; carriers = segment_carriers) : Int[]

    new_sinks = Channel{SignalFlowBlock}(4)
    sinks = Vector{SignalFlowBlock}()
    ctx = ISDBTPhaseSlopeCorrectorContext(Base.Threads.Atomic{Bool}(true),
                                          nfft,
                                          pilot_spacing,
                                          pilot_offset0,
                                          pilot_offset_step,
                                          pilot_values,
                                          Float64(samplerate),
                                          Float64(band_limit_hz),
                                          segment_carriers,
                                          prbs_bits,
                                          Float64(alpha),
                                          auto_sp_phase,
                                          symbol_index_ref,
                                          0.0,
                                          0.0,
                                          log_stats,
                                          Float64(log_interval),
                                          0.0,
                                          0,
                                          Vector{ComplexF32}(undef, nfft),
                                          RingFrameBuffer(ComplexF32, nfft, poolsize),
                                          nothing,
                                          nothing,
                                          new_sinks,
                                          sinks)
    ctx.worker = Threads.@spawn task!(ctx)
    return ctx
end

function task!(context::ISDBTPhaseSlopeCorrectorContext)
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                if rd_buffer.store_size == context.nfft
                    if context.symbol_index_ref !== nothing
                        context.symbol_index = context.symbol_index_ref[]
                    end
                    phase = (context.symbol_index % 4) + 1
                    if context.auto_sp_phase && !isempty(context.prbs_bits)
                        best_phase = 1
                        best_metric = -1.0
                        for phase_try in 0:3
                            offset_try = context.pilot_offset0 + context.pilot_offset_step * phase_try
                            pos_try = context.segment_carriers > 0 ?
                                      seg0_pilot_bins(context.nfft, context.pilot_spacing, offset_try, context.segment_carriers) :
                                      pilot_positions(context.nfft, context.pilot_spacing, offset_try)
                            s_re = 0.0
                            s_im = 0.0
                            @inbounds for i in 1:length(pos_try)
                                idx = pos_try[i]
                                carrier = ISDBTPRBS.seg0_bin_to_carrier(context.nfft, idx, context.segment_carriers)
                                if carrier >= 0
                                    bit = context.prbs_bits[carrier + 1]
                                    ref = ISDBTPRBS.pilot_value_unit_from_bit(bit)
                                    v = rd_buffer.buf[idx]
                                    s_re += real(v) * real(ref) + imag(v) * imag(ref)
                                    s_im += real(v) * imag(ref) - imag(v) * real(ref)
                                end
                            end
                            metric = sqrt(s_re * s_re + s_im * s_im)
                            if metric > best_metric
                                best_metric = metric
                                best_phase = phase_try + 1
                            end
                        end
                        phase = best_phase
                    end
                    offset = context.pilot_offset0 + context.pilot_offset_step * (phase - 1)
                    pos = context.segment_carriers > 0 ?
                          seg0_pilot_bins(context.nfft, context.pilot_spacing, offset, context.segment_carriers) :
                          pilot_positions(context.nfft, context.pilot_spacing, offset)
                    band_pos = pos
                    pilot_vals = context.pilot_values
                    if context.band_limit_hz > 0
                        band_pos = Int[]
                        pilot_vals = ComplexF32[]
                        @inbounds for i in 1:length(pos)
                            idx = pos[i]
                            carrier = ISDBTPRBS.seg0_bin_to_carrier(context.nfft, idx, context.segment_carriers)
                            if carrier >= 0
                                bit = context.prbs_bits[carrier + 1]
                                push!(band_pos, idx)
                                push!(pilot_vals, ISDBTPRBS.pilot_value_unit_from_bit(bit))
                            end
                        end
                    end
                    n = length(band_pos)
                    phases = Vector{Float64}(undef, n)
                    ks = Vector{Float64}(undef, n)
                    center = (context.nfft ÷ 2) + 1
                    @inbounds for i in 1:n
                        idx = band_pos[i]
                        v = rd_buffer.buf[idx] / pilot_vals[i]
                        phases[i] = atan(imag(v), real(v))
                        ks[i] = Float64(idx - center)
                    end
                    unwrap_phases!(phases)
                    sum_k = 0.0
                    sum_p = 0.0
                    sum_kp = 0.0
                    sum_k2 = 0.0
                    @inbounds for i in 1:n
                        k = ks[i]
                        p = phases[i]
                        sum_k += k
                        sum_p += p
                        sum_kp += k * p
                        sum_k2 += k * k
                    end
                    denom = n * sum_k2 - sum_k * sum_k
                    if denom != 0.0
                        slope = (n * sum_kp - sum_k * sum_p) / denom
                        intercept = (sum_p - slope * sum_k) / n
                        context.slope += context.alpha * (slope - context.slope)
                        context.intercept += context.alpha * (intercept - context.intercept)
                    end
                    if context.log_stats
                        now = time()
                        if now - context.last_log_time >= context.log_interval
                            lock(LOG_LOCK) do
                                println("PhaseSlope: slope=",
                                        round(context.slope, digits = 6),
                                        " rad/bin, intercept=",
                                        round(context.intercept, digits = 4))
                            end
                            context.last_log_time = now
                        end
                    end

                    @inbounds for k in 1:context.nfft
                        phi = context.slope * (k - center) + context.intercept
                        c = cos(phi)
                        s = -sin(phi)
                        v = rd_buffer.buf[k]
                        a = real(v)
                        b = imag(v)
                        context.outbuf[k] = ComplexF32(Float32(a * c - b * s),
                                                       Float32(a * s + b * c))
                    end
                    context.symbol_index += 1

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
            println("ISDBTPhaseSlopeCorrector error: ", e)
        end
    end
    return nothing
end

function input!(context::ISDBTPhaseSlopeCorrectorContext, samples::AbstractVector{ComplexF32}, samples_size::Int)
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

function stop!(context::ISDBTPhaseSlopeCorrectorContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

end
