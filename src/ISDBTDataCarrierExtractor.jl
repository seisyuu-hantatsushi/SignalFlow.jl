module ISDBTDataCarrierExtractor

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer
import ..ISDBTPRBS

mutable struct ISDBTDataCarrierExtractorContext <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    nfft::Int
    samplerate::Float64
    band_limit_hz::Float64
    pilot_spacing::Int
    pilot_offset0::Int
    pilot_offset_step::Int
    exclude_dc::Bool
    exclude_edge_bins::Int
    tps_positions::Vector{Int}
    exclude_carriers::Vector{Int}
    segment_carriers::Int
    prbs_bits::Vector{Int}
    auto_sp_phase::Bool
    output_indices::Vector{Vector{Int}}
    log_stats::Bool
    log_interval::Float64
    last_log_time::Float64
    symbol_index::Int
    symbol_index_ref::Union{Nothing,Base.Threads.Atomic{Int}}
    outbuf::Vector{ComplexF32}
    ringbuffer::RingFrameBuffer{ComplexF32}
    holdbuf::Union{Nothing, Int}
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
end

const LOG_LOCK = ReentrantLock()

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

function build_mask(nfft::Int,
                    samplerate::Float64,
                    band_limit_hz::Float64,
                    pilot_spacing::Int,
                    pilot_offset::Int,
                    exclude_dc::Bool,
                    exclude_edge_bins::Int,
                    tps_positions::Vector{Int},
                    segment_carriers::Int)
    mask = trues(nfft)
    if band_limit_hz > 0
        hz_per_bin = samplerate / nfft
        @inbounds for i in 1:nfft
            f = (i - 1) * hz_per_bin
            if f > samplerate / 2
                f -= samplerate
            end
            if abs(f) > band_limit_hz
                mask[i] = false
            end
        end
    end
    if exclude_edge_bins > 0
        lo = 1
        hi = min(nfft, exclude_edge_bins)
        for i in lo:hi
            mask[i] = false
        end
        lo2 = max(1, nfft - exclude_edge_bins + 1)
        hi2 = nfft
        for i in lo2:hi2
            mask[i] = false
        end
    end
    if exclude_dc && nfft >= 1
        mask[1] = false
    end
    pilot_bins = segment_carriers > 0 ?
                 seg0_pilot_bins(nfft, pilot_spacing, pilot_offset, segment_carriers) :
                 pilot_positions(nfft, pilot_spacing, pilot_offset)
    for p in pilot_bins
        mask[p] = false
    end
    for p in tps_positions
        if 1 <= p <= nfft
            mask[p] = false
        end
    end
    return findall(mask)
end

function build_indices_by_carrier(nfft::Int,
                                  samplerate::Float64,
                                  band_limit_hz::Float64,
                                  pilot_spacing::Int,
                                  pilot_offset::Int,
                                  segment_carriers::Int,
                                  exclude_carriers::Vector{Int},
                                  exclude_bins::Vector{Int})
    half = segment_carriers ÷ 2
    exclude_set = Set(exclude_carriers)
    exclude_bins_set = Set(exclude_bins)
    bins = Int[]
    hz_per_bin = samplerate / nfft
    @inbounds for carrier in 0:segment_carriers - 1
        if mod(carrier - pilot_offset, pilot_spacing) == 0
            continue
        end
        if carrier in exclude_set
            continue
        end
        bin = ISDBTPRBS.seg0_carrier_to_bin(nfft, carrier, segment_carriers)
        if bin <= 0
            continue
        end
        if bin in exclude_bins_set
            continue
        end
        if band_limit_hz > 0
            f = (bin - 1) * hz_per_bin
            if f > samplerate / 2
                f -= samplerate
            end
            if abs(f) > band_limit_hz
                continue
            end
        end
        push!(bins, bin)
    end
    # Order as negative-frequency (carrier 0..half-1) then positive (half..end).
    neg_bins = Int[]
    pos_bins = Int[]
    @inbounds for carrier in 0:segment_carriers - 1
        if mod(carrier - pilot_offset, pilot_spacing) == 0
            continue
        end
        if carrier in exclude_set
            continue
        end
        bin = ISDBTPRBS.seg0_carrier_to_bin(nfft, carrier, segment_carriers)
        if bin <= 0 || (bin in exclude_bins_set)
            continue
        end
        if band_limit_hz > 0
            f = (bin - 1) * hz_per_bin
            if f > samplerate / 2
                f -= samplerate
            end
            if abs(f) > band_limit_hz
                continue
            end
        end
        if carrier < half
            push!(neg_bins, bin)
        else
            push!(pos_bins, bin)
        end
    end
    return vcat(neg_bins, pos_bins)
end

function classify_indices(nfft::Int,
                          segment_carriers::Int,
                          pilot_spacing::Int,
                          pilot_offset::Int,
                          indices::Vector{Int},
                          exclude_carriers::Vector{Int})
    pilot_count = 0
    excluded_count = 0
    invalid_count = 0
    for idx in indices
        carrier = ISDBTPRBS.seg0_bin_to_carrier(nfft, idx, segment_carriers)
        if carrier < 0
            invalid_count += 1
            continue
        end
        if mod(carrier - pilot_offset, pilot_spacing) == 0
            pilot_count += 1
        end
        if carrier in exclude_carriers
            excluded_count += 1
        end
    end
    return pilot_count, excluded_count, invalid_count
end

function CreateISDBTDataCarrierExtractor(; nfft::Int = 8192,
                                         samplerate::Real = 8_000_000,
                                         band_limit_hz::Real = 0.0,
                                         pilot_spacing::Int = 12,
                                         pilot_offset0::Int = 3,
                                         pilot_offset_step::Int = 3,
                                         exclude_dc::Bool = true,
                                         exclude_edge_bins::Int = 0,
                                         tps_positions::Vector{Int} = Int[],
                                         exclude_carriers::Vector{Int} = Int[],
                                         segment_carriers::Int = 432,
                                         segment_index::Int = 0,
                                         auto_sp_phase::Bool = true,
                                         log_stats::Bool = false,
                                         log_interval::Real = 1.0,
                                         symbol_index_ref::Union{Nothing,Base.Threads.Atomic{Int}} = nothing,
                                         poolsize::Int = 8)
    nfft < 32 && error("ISDBTDataCarrierExtractor: nfft must be >= 32.")
    pilot_spacing < 1 && error("ISDBTDataCarrierExtractor: pilot_spacing must be >= 1.")
    exclude_edge_bins < 0 && error("ISDBTDataCarrierExtractor: exclude_edge_bins must be >= 0.")
    band_limit_hz < 0 && error("ISDBTDataCarrierExtractor: band_limit_hz must be >= 0.")
    segment_carriers < 1 && error("ISDBTDataCarrierExtractor: segment_carriers must be >= 1.")
    poolsize < 1 && error("ISDBTDataCarrierExtractor: poolsize must be at least 1.")
    log_interval <= 0 && error("ISDBTDataCarrierExtractor: log_interval must be > 0.")

    prbs_bits = band_limit_hz > 0 ? ISDBTPRBS.mode3_segment_prbs(segment_index; carriers = segment_carriers) : Int[]

    output_indices = Vector{Vector{Int}}(undef, 4)
    len0 = -1
    for phase in 0:3
        offset = pilot_offset0 + pilot_offset_step * phase
        if segment_carriers > 0
            idx = build_indices_by_carrier(nfft, Float64(samplerate), Float64(band_limit_hz),
                                           pilot_spacing, offset, segment_carriers,
                                           exclude_carriers, tps_positions)
        else
            idx = build_mask(nfft, Float64(samplerate), Float64(band_limit_hz),
                             pilot_spacing, offset, exclude_dc, exclude_edge_bins, tps_positions,
                             segment_carriers)
        end
        output_indices[phase + 1] = idx
        len0 = len0 == -1 ? length(idx) : min(len0, length(idx))
    end
    if len0 < 1
        error("ISDBTDataCarrierExtractor: output length must be >= 1.")
    end
    for phase in 1:4
        output_indices[phase] = output_indices[phase][1:len0]
    end

    if log_stats
        for phase in 0:3
            idx = output_indices[phase + 1]
            pilot_count, excluded_count, invalid_count =
                classify_indices(nfft, segment_carriers, pilot_spacing,
                                 pilot_offset0 + pilot_offset_step * phase, idx, exclude_carriers)
            lock(LOG_LOCK) do
                println("DataCarrierExtractor init: phase=", phase,
                        " len=", length(idx),
                        " pilot_overlap=", pilot_count,
                        " excluded_overlap=", excluded_count,
                        " invalid_bins=", invalid_count)
            end
        end
    end

    new_sinks = Channel{SignalFlowBlock}(64)
    sinks = Vector{SignalFlowBlock}()
    outbuf = Vector{ComplexF32}(undef, len0)

    ctx = ISDBTDataCarrierExtractorContext(Base.Threads.Atomic{Bool}(true),
                                           nfft,
                                           Float64(samplerate),
                                           Float64(band_limit_hz),
                                           pilot_spacing,
                                           pilot_offset0,
                                           pilot_offset_step,
                                           exclude_dc,
                                           exclude_edge_bins,
                                           tps_positions,
                                           exclude_carriers,
                                           segment_carriers,
                                           prbs_bits,
                                           auto_sp_phase,
                                           output_indices,
                                           log_stats,
                                           Float64(log_interval),
                                           0.0,
                                           0,
                                           symbol_index_ref,
                                           outbuf,
                                           RingFrameBuffer(ComplexF32, nfft, poolsize),
                                           nothing,
                                           nothing,
                                           new_sinks,
                                           sinks)
    ctx.worker = Threads.@spawn task!(ctx)
    return ctx
end

function task!(context::ISDBTDataCarrierExtractorContext)
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                if rd_buffer.store_size == context.nfft
                    if context.auto_sp_phase
                        best_phase = 1
                        best_metric = -1.0
                        for phase in 0:3
                            offset = context.pilot_offset0 + context.pilot_offset_step * phase
                            pos = context.segment_carriers > 0 ?
                                  seg0_pilot_bins(context.nfft, context.pilot_spacing, offset, context.segment_carriers) :
                                  pilot_positions(context.nfft, context.pilot_spacing, offset)
                            s_re = 0.0
                            s_im = 0.0
                            @inbounds for i in 1:length(pos)
                                idx = pos[i]
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
                                best_phase = phase + 1
                            end
                        end
                        phase = best_phase
                    else
                        if context.symbol_index_ref !== nothing
                            context.symbol_index = context.symbol_index_ref[]
                        end
                        phase = (context.symbol_index % 4) + 1
                    end
                    idx = context.output_indices[phase]
                    @inbounds for i in 1:length(idx)
                        context.outbuf[i] = rd_buffer.buf[idx[i]]
                    end
                    if context.log_stats
                        now = time()
                        if now - context.last_log_time >= context.log_interval
                            p_sum = 0.0
                            @inbounds for i in 1:length(context.outbuf)
                                v = context.outbuf[i]
                                p_sum += real(v) * real(v) + imag(v) * imag(v)
                            end
                            p_avg = p_sum / max(length(context.outbuf), 1)
                            lock(LOG_LOCK) do
                                println("DataCarrierExtractor: phase=", phase - 1,
                                        " len=", length(context.outbuf),
                                        " avgP=", round(10 * log10(max(p_avg, 1e-12)), digits = 2), " dB")
                            end
                            context.last_log_time = now
                        end
                    end
                    context.symbol_index += 1

                    while isready(context.new_sinks)
                        push!(context.sinks, take!(context.new_sinks))
                    end
                    for sink in context.sinks
                        input!(sink, context.outbuf, length(context.outbuf))
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
            println("ISDBTDataCarrierExtractor error: ", e)
        end
    end
    return nothing
end

function input!(context::ISDBTDataCarrierExtractorContext, samples::AbstractVector{ComplexF32}, samples_size::Int)
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

function stop!(context::ISDBTDataCarrierExtractorContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

end
