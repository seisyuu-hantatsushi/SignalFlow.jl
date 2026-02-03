module ISDBTPilotEqualizer

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer
import ..ISDBTPRBS

mutable struct ISDBTPilotEqualizerContext <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    nfft::Int
    pilot_spacing::Int
    pilot_offset0::Int
    pilot_offset_step::Int
    output_mode::Int
    symbol_index::Int
    pilot_values::Vector{ComplexF32}
    samplerate::Float64
    band_limit_hz::Float64
    segment_carriers::Int
    prbs_bits::Vector{Int}
    auto_sp_phase::Bool
    symbol_index_ref::Union{Nothing,Base.Threads.Atomic{Int}}
    h_est::Vector{ComplexF32}
    outbuf::Vector{ComplexF32}
    log_stats::Bool
    log_interval::Float64
    last_log_time::Float64
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

function linear_interp!(dst::Vector{ComplexF32}, pos::Vector{Int}, values::AbstractVector{ComplexF32})
    n = length(dst)
    np = length(pos)
    np < 2 && return
    @inbounds begin
        for i in 1:pos[1] - 1
            dst[i] = values[1]
        end
        for pi in 1:np - 1
            p0 = pos[pi]
            p1 = pos[pi + 1]
            v0 = values[pi]
            v1 = values[pi + 1]
            for k in p0:p1
                t = (k - p0) / max(p1 - p0, 1)
                dst[k] = v0 + (v1 - v0) * Float32(t)
            end
        end
        for i in pos[end] + 1:n
            dst[i] = values[end]
        end
    end
    return nothing
end

function linear_interp_range!(dst::Vector{ComplexF32},
                              pos::Vector{Int},
                              values::AbstractVector{ComplexF32},
                              lo::Int,
                              hi::Int)
    np = length(pos)
    np < 1 && return
    @inbounds begin
        if lo < pos[1]
            for i in lo:pos[1] - 1
                dst[i] = values[1]
            end
        end
        for pi in 1:np - 1
            p0 = pos[pi]
            p1 = pos[pi + 1]
            v0 = values[pi]
            v1 = values[pi + 1]
            for k in p0:p1
                t = (k - p0) / max(p1 - p0, 1)
                dst[k] = v0 + (v1 - v0) * Float32(t)
            end
        end
        if pos[end] < hi
            for i in pos[end] + 1:hi
                dst[i] = values[end]
            end
        end
    end
    return nothing
end

function CreateISDBTPilotEqualizer(; nfft::Int = 8192,
                                   pilot_spacing::Int = 12,
                                   pilot_offset0::Int = 3,
                                   pilot_offset_step::Int = 3,
                                   output_mode::Int = 2,
                                   pilot_values::Union{Nothing,Vector{ComplexF32}} = nothing,
                                   samplerate::Real = 8_000_000,
                                   band_limit_hz::Real = 0.0,
                                   segment_carriers::Int = 432,
                                   segment_index::Int = 0,
                                   auto_sp_phase::Bool = true,
                                   symbol_index_ref::Union{Nothing,Base.Threads.Atomic{Int}} = nothing,
                                   log_stats::Bool = false,
                                   log_interval::Real = 1.0,
                                   poolsize::Int = 8)
    nfft < 32 && error("ISDBTPilotEqualizer: nfft must be >= 32.")
    pilot_spacing < 1 && error("ISDBTPilotEqualizer: pilot_spacing must be >= 1.")
    poolsize < 1 && error("ISDBTPilotEqualizer: poolsize must be at least 1.")
    (output_mode == 1 || output_mode == 2) || error("ISDBTPilotEqualizer: output_mode must be 1 or 2.")
    band_limit_hz < 0 && error("ISDBTPilotEqualizer: band_limit_hz must be >= 0.")
    segment_carriers < 1 && error("ISDBTPilotEqualizer: segment_carriers must be >= 1.")
    log_interval <= 0 && error("ISDBTPilotEqualizer: log_interval must be > 0.")

    pos = segment_carriers > 0 ?
          seg0_pilot_bins(nfft, pilot_spacing, pilot_offset0, segment_carriers) :
          pilot_positions(nfft, pilot_spacing, pilot_offset0)
    pilot_values === nothing && (pilot_values = fill(ComplexF32(1, 0), length(pos)))
    length(pilot_values) != length(pos) && error("ISDBTPilotEqualizer: pilot_values length mismatch.")
    prbs_bits = band_limit_hz > 0 ? ISDBTPRBS.mode3_segment_prbs(segment_index; carriers = segment_carriers) : Int[]

    new_sinks = Channel{SignalFlowBlock}(4)
    sinks = Vector{SignalFlowBlock}()
    ctx = ISDBTPilotEqualizerContext(Base.Threads.Atomic{Bool}(true),
                                     nfft,
                                     pilot_spacing,
                                     pilot_offset0,
                                     pilot_offset_step,
                                     output_mode,
                                     0,
                                     pilot_values,
                                     Float64(samplerate),
                                     Float64(band_limit_hz),
                                     segment_carriers,
                                     prbs_bits,
                                     auto_sp_phase,
                                     symbol_index_ref,
                                     Vector{ComplexF32}(undef, nfft),
                                     Vector{ComplexF32}(undef, nfft),
                                     log_stats,
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

function task!(context::ISDBTPilotEqualizerContext)
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                if rd_buffer.store_size == context.nfft
                    fill!(context.h_est, ComplexF32(1, 0))
                    if context.symbol_index_ref !== nothing
                        context.symbol_index = context.symbol_index_ref[]
                    end
                    phase = (context.symbol_index % 4) + 1
                    if context.auto_sp_phase && !isempty(context.prbs_bits)
                        best_phase = 1
                        best_metric = -1.0
                        for phase_try in 0:3
                            offset = context.pilot_offset0 + context.pilot_offset_step * phase_try
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
                                best_phase = phase_try + 1
                            end
                        end
                        phase = best_phase
                    end
                    offset = context.pilot_offset0 + context.pilot_offset_step * (phase - 1)
                    pos = context.segment_carriers > 0 ?
                          seg0_pilot_bins(context.nfft, context.pilot_spacing, offset, context.segment_carriers) :
                          pilot_positions(context.nfft, context.pilot_spacing, offset)
                    use_pos = pos
                    use_vals = context.pilot_values
                    if context.band_limit_hz > 0
                        use_pos = Int[]
                        use_vals = ComplexF32[]
                        @inbounds for i in 1:length(pos)
                            idx = pos[i]
                            carrier = ISDBTPRBS.seg0_bin_to_carrier(context.nfft, idx, context.segment_carriers)
                            if carrier >= 0
                                bit = context.prbs_bits[carrier + 1]
                                push!(use_pos, idx)
                                push!(use_vals, ISDBTPRBS.pilot_value_from_bit(bit))
                            end
                        end
                    end
                    @inbounds for i in 1:length(use_pos)
                        idx = use_pos[i]
                        context.h_est[idx] = rd_buffer.buf[idx] / use_vals[i]
                    end
                    if length(use_pos) >= 2
                        if context.band_limit_hz > 0 && context.segment_carriers > 0
                            half = context.segment_carriers ÷ 2
                            pos_start = 2
                            pos_end = 1 + half
                            neg_start = context.nfft - (half - 1)
                            neg_end = context.nfft
                            pos_idx = Int[]
                            pos_vals = ComplexF32[]
                            neg_idx = Int[]
                            neg_vals = ComplexF32[]
                            @inbounds for i in 1:length(use_pos)
                                idx = use_pos[i]
                                if pos_start <= idx <= pos_end
                                    push!(pos_idx, idx)
                                    push!(pos_vals, context.h_est[idx])
                                elseif neg_start <= idx <= neg_end
                                    push!(neg_idx, idx)
                                    push!(neg_vals, context.h_est[idx])
                                end
                            end
                            if length(pos_idx) >= 2
                                linear_interp_range!(context.h_est, pos_idx, pos_vals, pos_start, pos_end)
                            end
                            if length(neg_idx) >= 2
                                linear_interp_range!(context.h_est, neg_idx, neg_vals, neg_start, neg_end)
                            end
                        else
                            linear_interp!(context.h_est, use_pos, view(context.h_est, use_pos))
                        end
                    end
                    if context.log_stats
                        now = time()
                        if now - context.last_log_time >= context.log_interval
                            sum_h = 0.0
                            @inbounds for i in 1:length(use_pos)
                                h = context.h_est[use_pos[i]]
                                sum_h += sqrt(real(h) * real(h) + imag(h) * imag(h))
                            end
                            mean_h = length(use_pos) > 0 ? sum_h / length(use_pos) : 0.0
                            lock(LOG_LOCK) do
                                println("PilotEQ: pilots=", length(use_pos),
                                        " mean|H|=", round(mean_h, digits = 3),
                                        " phase=", (context.symbol_index % 4))
                            end
                            context.last_log_time = now
                        end
                    end
                    if context.output_mode == 1
                        copyto!(context.outbuf, context.h_est)
                    else
                        @inbounds for k in 1:context.nfft
                            h = context.h_est[k]
                            context.outbuf[k] = h == 0f0 ? rd_buffer.buf[k] : rd_buffer.buf[k] / h
                        end
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
            println("ISDBTPilotEqualizer error: ", e)
        end
    end
    return nothing
end

function input!(context::ISDBTPilotEqualizerContext, samples::AbstractVector{ComplexF32}, samples_size::Int)
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

function stop!(context::ISDBTPilotEqualizerContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

end
