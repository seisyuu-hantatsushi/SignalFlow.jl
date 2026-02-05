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
    max_slope_step::Float64
    max_intercept_step::Float64
    pilot_min_mag::Float64
    pilot_trim_ratio::Float64
    max_fit_rms_on::Float64
    max_fit_rms_off::Float64
    update_enabled::Bool
    auto_sp_phase::Bool
    symbol_index_ref::Union{Nothing,Base.Threads.Atomic{Int}}
    gap_freeze_ref::Union{Nothing,Base.Threads.Atomic{Int}}
    slope::Float64
    intercept::Float64
    log_stats::Bool
    log_interval::Float64
    last_log_time::Float64
    symbol_index::Int
    phase_bins::Vector{Vector{Int}}
    phase_refs::Vector{Vector{ComplexF32}}
    phase_buf::Vector{Float64}
    k_buf::Vector{Float64}
    mag_buf::Vector{Float64}
    used_mask::BitVector
    residual_buf::Vector{Float64}
    residual_sorted_buf::Vector{Float64}
    outbuf::Vector{ComplexF32}
    ringbuffer::RingFrameBuffer{ComplexF32}
    holdbuf::Union{Nothing, Int}
    input_overrun_count::Int
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
end

@inline function wrap_phase(phase::Float64)
    if phase > Float64(π)
        return phase - 2.0 * Float64(π)
    elseif phase < -Float64(π)
        return phase + 2.0 * Float64(π)
    end
    return phase
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

function phase_tables(nfft::Int,
                      pilot_spacing::Int,
                      pilot_offset0::Int,
                      pilot_offset_step::Int,
                      pilot_values::Vector{ComplexF32},
                      segment_carriers::Int,
                      band_limit_hz::Float64,
                      prbs_bits::Vector{Int})
    bins_by_phase = Vector{Vector{Int}}(undef, 4)
    refs_by_phase = Vector{Vector{ComplexF32}}(undef, 4)
    default_pilot_values = all(v -> v == ComplexF32(1, 0), pilot_values)
    for phase_try in 0:3
        offset_try = pilot_offset0 + pilot_offset_step * phase_try
        pos_try = segment_carriers > 0 ?
                  seg0_pilot_bins(nfft, pilot_spacing, offset_try, segment_carriers) :
                  pilot_positions(nfft, pilot_spacing, offset_try)
        if band_limit_hz > 0
            bins = Int[]
            refs = ComplexF32[]
            @inbounds for i in 1:length(pos_try)
                idx = pos_try[i]
                carrier = ISDBTPRBS.seg0_bin_to_carrier(nfft, idx, segment_carriers)
                if carrier >= 0
                    bit = prbs_bits[carrier + 1]
                    push!(bins, idx)
                    push!(refs, ISDBTPRBS.pilot_value_unit_from_bit(bit))
                end
            end
            bins_by_phase[phase_try + 1] = bins
            refs_by_phase[phase_try + 1] = refs
        else
            bins_by_phase[phase_try + 1] = pos_try
            if default_pilot_values
                refs_by_phase[phase_try + 1] = fill(ComplexF32(1, 0), length(pos_try))
            elseif length(pilot_values) == length(pos_try)
                refs_by_phase[phase_try + 1] = copy(pilot_values)
            else
                error("ISDBTPhaseSlopeCorrector: pilot_values length mismatch at phase ", phase_try + 1)
            end
        end
    end
    return bins_by_phase, refs_by_phase
end

function unwrap_phases!(phases::AbstractVector{Float64})
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
                                        max_slope_step::Real = 0.0015,
                                        max_intercept_step_deg::Real = 12.0,
                                        pilot_min_mag::Real = 0.15,
                                        pilot_trim_ratio::Real = 0.2,
                                        max_fit_rms::Real = 0.4,
                                        max_fit_rms_off::Union{Nothing,Real} = nothing,
                                        auto_sp_phase::Bool = true,
                                        symbol_index_ref::Union{Nothing,Base.Threads.Atomic{Int}} = nothing,
                                        gap_freeze_ref::Union{Nothing,Base.Threads.Atomic{Int}} = nothing,
                                        log_stats::Bool = false,
                                        log_interval::Real = 1.0,
                                        poolsize::Int = 8)
    nfft < 32 && error("ISDBTPhaseSlopeCorrector: nfft must be >= 32.")
    pilot_spacing < 1 && error("ISDBTPhaseSlopeCorrector: pilot_spacing must be >= 1.")
    alpha <= 0 && error("ISDBTPhaseSlopeCorrector: alpha must be positive.")
    max_slope_step <= 0 && error("ISDBTPhaseSlopeCorrector: max_slope_step must be positive.")
    max_intercept_step_deg <= 0 && error("ISDBTPhaseSlopeCorrector: max_intercept_step_deg must be positive.")
    pilot_min_mag < 0 && error("ISDBTPhaseSlopeCorrector: pilot_min_mag must be >= 0.")
    (0.0 <= pilot_trim_ratio < 0.5) || error("ISDBTPhaseSlopeCorrector: pilot_trim_ratio must be in [0, 0.5).")
    max_fit_rms <= 0 && error("ISDBTPhaseSlopeCorrector: max_fit_rms must be > 0.")
    poolsize < 1 && error("ISDBTPhaseSlopeCorrector: poolsize must be at least 1.")
    log_interval <= 0 && error("ISDBTPhaseSlopeCorrector: log_interval must be > 0.")
    fit_rms_on = Float64(max_fit_rms)
    fit_rms_off = max_fit_rms_off === nothing ? fit_rms_on * 1.25 : Float64(max_fit_rms_off)
    (0.0 < fit_rms_on <= fit_rms_off) || error("ISDBTPhaseSlopeCorrector: require 0 < max_fit_rms <= max_fit_rms_off.")

    pos = segment_carriers > 0 ?
          seg0_pilot_bins(nfft, pilot_spacing, pilot_offset0, segment_carriers) :
          pilot_positions(nfft, pilot_spacing, pilot_offset0)
    pilot_values === nothing && (pilot_values = fill(ComplexF32(1, 0), length(pos)))
    length(pilot_values) != length(pos) && error("ISDBTPhaseSlopeCorrector: pilot_values length mismatch.")
    band_limit_hz < 0 && error("ISDBTPhaseSlopeCorrector: band_limit_hz must be >= 0.")
    segment_carriers < 1 && error("ISDBTPhaseSlopeCorrector: segment_carriers must be >= 1.")

    prbs_bits = band_limit_hz > 0 ? ISDBTPRBS.mode3_segment_prbs(segment_index; carriers = segment_carriers) : Int[]
    bins_by_phase, refs_by_phase = phase_tables(nfft,
                                                pilot_spacing,
                                                pilot_offset0,
                                                pilot_offset_step,
                                                pilot_values,
                                                segment_carriers,
                                                Float64(band_limit_hz),
                                                prbs_bits)
    max_pilots = max(1, maximum(length, bins_by_phase))

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
                                          Float64(max_slope_step),
                                          Float64(max_intercept_step_deg) * Float64(pi) / 180.0,
                                          Float64(pilot_min_mag),
                                          Float64(pilot_trim_ratio),
                                          fit_rms_on,
                                          fit_rms_off,
                                          false,
                                          auto_sp_phase,
                                          symbol_index_ref,
                                          gap_freeze_ref,
                                          0.0,
                                          0.0,
                                          log_stats,
                                          Float64(log_interval),
                                          0.0,
                                          0,
                                          bins_by_phase,
                                          refs_by_phase,
                                          Vector{Float64}(undef, max_pilots),
                                          Vector{Float64}(undef, max_pilots),
                                          Vector{Float64}(undef, max_pilots),
                                          falses(max_pilots),
                                          Vector{Float64}(undef, max_pilots),
                                          Vector{Float64}(undef, max_pilots),
                                          Vector{ComplexF32}(undef, nfft),
                                          RingFrameBuffer(ComplexF32, nfft, poolsize),
                                          nothing,
                                          0,
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
                        for phase_try in 1:4
                            bins_try = context.phase_bins[phase_try]
                            refs_try = context.phase_refs[phase_try]
                            s_re = 0.0
                            s_im = 0.0
                            @inbounds for i in 1:length(bins_try)
                                idx = bins_try[i]
                                ref = refs_try[i]
                                v = rd_buffer.buf[idx]
                                s_re += real(v) * real(ref) + imag(v) * imag(ref)
                                s_im += real(v) * imag(ref) - imag(v) * real(ref)
                            end
                            metric = sqrt(s_re * s_re + s_im * s_im)
                            if metric > best_metric
                                best_metric = metric
                                best_phase = phase_try
                            end
                        end
                        phase = best_phase
                    end
                    band_pos = context.phase_bins[phase]
                    pilot_vals = context.phase_refs[phase]
                    n = length(band_pos)
                    phases = context.phase_buf
                    ks = context.k_buf
                    mags = context.mag_buf
                    center = (context.nfft ÷ 2) + 1
                    @inbounds for i in 1:n
                        idx = band_pos[i]
                        v = rd_buffer.buf[idx] / pilot_vals[i]
                        phases[i] = atan(imag(v), real(v))
                        ks[i] = Float64(idx - center)
                        mags[i] = abs(v)
                    end
                    unwrap_phases!(@view(phases[1:n]))
                    used_mask = context.used_mask
                    fill!(used_mask, false)
                    @inbounds for i in 1:n
                        used_mask[i] = true
                    end
                    used = n
                    @inbounds for i in 1:n
                        if mags[i] < context.pilot_min_mag
                            used_mask[i] = false
                            used -= 1
                        end
                    end
                    if used < 6
                        while isready(context.new_sinks)
                            push!(context.sinks, take!(context.new_sinks))
                        end
                        copyto!(context.outbuf, 1, rd_buffer.buf, 1, context.nfft)
                        for sink in context.sinks
                            input!(sink, context.outbuf, context.nfft)
                        end
                        rd_buffer.store_size = 0
                        put!(context.ringbuffer.freeQ, rd_index)
                        continue
                    end

                    if context.pilot_trim_ratio > 0 && used >= 8
                        phi0 = context.intercept
                        rcount = 0
                        for i in 1:n
                            if used_mask[i]
                                rcount += 1
                                r = abs(wrap_phase(phases[i] - phi0))
                                context.residual_buf[rcount] = r
                                context.residual_sorted_buf[rcount] = r
                            end
                        end
                        thresh_idx = clamp(floor(Int, (1.0 - context.pilot_trim_ratio) * rcount), 1, rcount)
                        sort!(@view(context.residual_sorted_buf[1:rcount]))
                        thresh = context.residual_sorted_buf[thresh_idx]
                        riter = 0
                        @inbounds for i in 1:n
                            if used_mask[i]
                                riter += 1
                                r = context.residual_buf[riter]
                                if r > thresh
                                    used_mask[i] = false
                                    used -= 1
                                end
                            end
                        end
                    end

                    sum_k = 0.0
                    sum_p = 0.0
                    sum_kp = 0.0
                    sum_k2 = 0.0
                    wsum = 0.0
                    @inbounds for i in 1:n
                        if used_mask[i]
                            w = min(mags[i], 2.0)
                            k = ks[i]
                            p = phases[i]
                            sum_k += w * k
                            sum_p += w * p
                            sum_kp += w * k * p
                            sum_k2 += w * k * k
                            wsum += w
                        end
                    end
                    denom = wsum * sum_k2 - sum_k * sum_k
                    fit_rms = 0.0
                    update_applied = false
                    if denom != 0.0 && wsum > 0
                        slope = (wsum * sum_kp - sum_k * sum_p) / denom
                        intercept = (sum_p - slope * sum_k) / wsum
                        cnt_fit = 0
                        @inbounds for i in 1:n
                            if used_mask[i]
                                est = slope * ks[i] + intercept
                                e = wrap_phase(phases[i] - est)
                                fit_rms += e * e
                                cnt_fit += 1
                            end
                        end
                        fit_rms = cnt_fit > 0 ? sqrt(fit_rms / cnt_fit) : 0.0

                        freeze_active = context.gap_freeze_ref !== nothing && context.gap_freeze_ref[] > 0
                        if freeze_active
                            context.update_enabled = false
                        elseif context.update_enabled
                            if fit_rms > context.max_fit_rms_off
                                context.update_enabled = false
                            end
                        elseif fit_rms <= context.max_fit_rms_on
                            context.update_enabled = true
                        end

                        if context.update_enabled && !freeze_active
                            slope_delta = context.alpha * (slope - context.slope)
                            if slope_delta > context.max_slope_step
                                slope_delta = context.max_slope_step
                            elseif slope_delta < -context.max_slope_step
                                slope_delta = -context.max_slope_step
                            end
                            context.slope += slope_delta
                            err_i = wrap_phase(intercept - context.intercept)
                            int_delta = context.alpha * err_i
                            if int_delta > context.max_intercept_step
                                int_delta = context.max_intercept_step
                            elseif int_delta < -context.max_intercept_step
                                int_delta = -context.max_intercept_step
                            end
                            context.intercept = wrap_phase(context.intercept + int_delta)
                            update_applied = true
                        end
                    end
                    if context.log_stats
                        now = time()
                        if now - context.last_log_time >= context.log_interval
                            rms = 0.0
                            if used > 0
                                cnt = 0
                                @inbounds for i in 1:n
                                    if used_mask[i]
                                        est = context.slope * ks[i] + context.intercept
                                        e = wrap_phase(phases[i] - est)
                                        rms += e * e
                                        cnt += 1
                                    end
                                end
                                rms = cnt > 0 ? sqrt(rms / cnt) : 0.0
                            end
                            lock(LOG_LOCK) do
                                println("PhaseSlope: slope=",
                                        round(context.slope, digits = 6),
                                        " rad/bin, intercept=",
                                        round(context.intercept, digits = 4),
                                        " rms=", round(rms, digits = 3),
                                        " fit_rms=", round(fit_rms, digits = 3),
                                        " used=", used, "/", n,
                                        " gate=", context.update_enabled,
                                        " updated=", update_applied)
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

    wait_loops = 0
    while !isready(context.ringbuffer.freeQ) && context.running[]
        wait_loops += 1
        if wait_loops == 2000
            context.input_overrun_count += 1
            println("PhaseSlope: input_backpressure count=", context.input_overrun_count)
            wait_loops = 0
        end
        yield()
    end
    if !context.running[]
        return false
    end
    idx = take!(context.ringbuffer.freeQ)
    buf = context.ringbuffer.bufs[idx]
    copyto!(buf.buf, 1, samples, 1, actual_size)
    buf.store_size = actual_size
    put!(context.ringbuffer.fullQ, idx)

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
