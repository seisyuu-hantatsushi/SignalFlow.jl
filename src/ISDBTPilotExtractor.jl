module ISDBTPilotExtractor

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer
import ..ISDBTPRBS

mutable struct ISDBTPilotExtractorContext <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    nfft::Int
    pilot_spacing::Int
    pilot_offset0::Int
    pilot_offset_step::Int
    segment_carriers::Int
    prbs_bits::Vector{Int}
    auto_sp_phase::Bool
    output_indices::Vector{Vector{Int}}
    output_refs::Vector{Vector{ComplexF32}}
    normalize::Bool
    symbol_index::Int
    symbol_index_ref::Union{Nothing,Base.Threads.Atomic{Int}}
    outbuf::Vector{ComplexF32}
    ringbuffer::RingFrameBuffer{ComplexF32}
    holdbuf::Union{Nothing, Int}
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
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

function CreateISDBTPilotExtractor(; nfft::Int = 8192,
                                   pilot_spacing::Int = 12,
                                   pilot_offset0::Int = 3,
                                   pilot_offset_step::Int = 3,
                                   segment_carriers::Int = 432,
                                   segment_index::Int = 0,
                                   auto_sp_phase::Bool = true,
                                   normalize::Bool = false,
                                   symbol_index_ref::Union{Nothing,Base.Threads.Atomic{Int}} = nothing,
                                   poolsize::Int = 8)
    nfft < 32 && error("ISDBTPilotExtractor: nfft must be >= 32.")
    pilot_spacing < 1 && error("ISDBTPilotExtractor: pilot_spacing must be >= 1.")
    segment_carriers < 1 && error("ISDBTPilotExtractor: segment_carriers must be >= 1.")
    poolsize < 1 && error("ISDBTPilotExtractor: poolsize must be at least 1.")

    prbs_bits = ISDBTPRBS.mode3_segment_prbs(segment_index; carriers = segment_carriers)
    output_indices = Vector{Vector{Int}}(undef, 4)
    output_refs = Vector{Vector{ComplexF32}}(undef, 4)
    len0 = -1
    for phase in 0:3
        offset = pilot_offset0 + pilot_offset_step * phase
        idx = seg0_pilot_bins(nfft, pilot_spacing, offset, segment_carriers)
        refs = ComplexF32[]
        @inbounds for i in 1:length(idx)
            carrier = ISDBTPRBS.seg0_bin_to_carrier(nfft, idx[i], segment_carriers)
            if carrier >= 0
                bit = prbs_bits[carrier + 1]
                push!(refs, ISDBTPRBS.pilot_value_from_bit(bit))
            else
                push!(refs, ComplexF32(1, 0))
            end
        end
        output_indices[phase + 1] = idx
        output_refs[phase + 1] = refs
        len0 = len0 == -1 ? length(idx) : min(len0, length(idx))
    end
    len0 < 1 && error("ISDBTPilotExtractor: output length must be >= 1.")
    for phase in 1:4
        output_indices[phase] = output_indices[phase][1:len0]
        output_refs[phase] = output_refs[phase][1:len0]
    end

    new_sinks = Channel{SignalFlowBlock}(64)
    sinks = Vector{SignalFlowBlock}()
    outbuf = Vector{ComplexF32}(undef, len0)

    ctx = ISDBTPilotExtractorContext(Base.Threads.Atomic{Bool}(true),
                                     nfft,
                                     pilot_spacing,
                                     pilot_offset0,
                                     pilot_offset_step,
                                     segment_carriers,
                                     prbs_bits,
                                     auto_sp_phase,
                                     output_indices,
                                     output_refs,
                                     normalize,
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

function task!(context::ISDBTPilotExtractorContext)
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
                            pos = seg0_pilot_bins(context.nfft, context.pilot_spacing, offset, context.segment_carriers)
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
                    refs = context.output_refs[phase]
                    if context.normalize
                        @inbounds for i in 1:length(idx)
                            context.outbuf[i] = rd_buffer.buf[idx[i]] / refs[i]
                        end
                    else
                        @inbounds for i in 1:length(idx)
                            context.outbuf[i] = rd_buffer.buf[idx[i]]
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
            println("ISDBTPilotExtractor error: ", e)
        end
    end
    return nothing
end

function input!(context::ISDBTPilotExtractorContext, samples::AbstractVector{ComplexF32}, samples_size::Int)
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
        buf.store_size = actual_size
        put!(context.ringbuffer.fullQ, idx)
    else
        return -1
    end

    return samples_size
end

function stop!(context::ISDBTPilotExtractorContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

end
