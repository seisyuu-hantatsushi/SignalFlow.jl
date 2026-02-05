module ISDBTCPECorrector

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer
import ..ISDBTPRBS

const LOG_LOCK = ReentrantLock()

mutable struct ISDBTCPECorrectorContext <: SignalFlowBlock
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
    cpe_alpha::Float64
    cpe_max_step::Float64
    pilot_min_mag::Float64
    pilot_trim_ratio::Float64
    min_update_conf_on::Float64
    min_update_conf_off::Float64
    update_enabled::Bool
    auto_sp_phase::Bool
    auto_sp_phase_interval::Int
    auto_phase_countdown::Int
    selected_phase::Int
    symbol_index_ref::Union{Nothing,Base.Threads.Atomic{Int}}
    gap_freeze_ref::Union{Nothing,Base.Threads.Atomic{Int}}
    cpe_phase::Float64
    log_stats::Bool
    log_interval::Float64
    last_log_time::Float64
    symbol_index::Int
    phase_bins::Vector{Vector{Int}}
    phase_refs::Vector{Vector{ComplexF32}}
    angle_buf::Vector{Float64}
    mag_buf::Vector{Float64}
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
                error("ISDBTCPECorrector: pilot_values length mismatch at phase ", phase_try + 1)
            end
        end
    end
    return bins_by_phase, refs_by_phase
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

function CreateISDBTCPECorrector(; nfft::Int = 8192,
                                 pilot_spacing::Int = 12,
                                 pilot_offset0::Int = 3,
                                 pilot_offset_step::Int = 3,
                                 pilot_values::Union{Nothing,Vector{ComplexF32}} = nothing,
                                 samplerate::Real = 8_000_000,
                                 band_limit_hz::Real = 0.0,
                                 segment_carriers::Int = 432,
                                 segment_index::Int = 0,
                                 cpe_alpha::Real = 0.2,
                                 cpe_max_step_deg::Real = 12.0,
                                 pilot_min_mag::Real = 0.15,
                                 pilot_trim_ratio::Real = 0.2,
                                 min_update_conf::Real = 0.2,
                                 min_update_conf_off::Union{Nothing,Real} = nothing,
                                 auto_sp_phase::Bool = true,
                                 auto_sp_phase_interval::Int = 4,
                                 symbol_index_ref::Union{Nothing,Base.Threads.Atomic{Int}} = nothing,
                                 gap_freeze_ref::Union{Nothing,Base.Threads.Atomic{Int}} = nothing,
                                 log_stats::Bool = false,
                                 log_interval::Real = 1.0,
                                 poolsize::Int = 8)
    nfft < 32 && error("ISDBTCPECorrector: nfft must be >= 32.")
    pilot_spacing < 1 && error("ISDBTCPECorrector: pilot_spacing must be >= 1.")
    cpe_alpha <= 0 && error("ISDBTCPECorrector: cpe_alpha must be positive.")
    cpe_max_step_deg <= 0 && error("ISDBTCPECorrector: cpe_max_step_deg must be positive.")
    pilot_min_mag < 0 && error("ISDBTCPECorrector: pilot_min_mag must be >= 0.")
    (0.0 <= pilot_trim_ratio < 0.5) || error("ISDBTCPECorrector: pilot_trim_ratio must be in [0, 0.5).")
    (0.0 <= min_update_conf <= 1.0) || error("ISDBTCPECorrector: min_update_conf must be in [0, 1].")
    auto_sp_phase_interval < 1 && error("ISDBTCPECorrector: auto_sp_phase_interval must be >= 1.")
    poolsize < 1 && error("ISDBTCPECorrector: poolsize must be at least 1.")
    log_interval <= 0 && error("ISDBTCPECorrector: log_interval must be > 0.")
    update_conf_on = Float64(min_update_conf)
    update_conf_off = min_update_conf_off === nothing ? max(0.0, update_conf_on * 0.7) : Float64(min_update_conf_off)
    (0.0 <= update_conf_off <= update_conf_on <= 1.0) || error("ISDBTCPECorrector: require 0 <= min_update_conf_off <= min_update_conf <= 1.")

    pos = segment_carriers > 0 ?
          seg0_pilot_bins(nfft, pilot_spacing, pilot_offset0, segment_carriers) :
          pilot_positions(nfft, pilot_spacing, pilot_offset0)
    pilot_values === nothing && (pilot_values = fill(ComplexF32(1, 0), length(pos)))
    length(pilot_values) != length(pos) && error("ISDBTCPECorrector: pilot_values length mismatch.")
    band_limit_hz < 0 && error("ISDBTCPECorrector: band_limit_hz must be >= 0.")
    segment_carriers < 1 && error("ISDBTCPECorrector: segment_carriers must be >= 1.")

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
    ctx = ISDBTCPECorrectorContext(Base.Threads.Atomic{Bool}(true),
                                   nfft,
                                   pilot_spacing,
                                   pilot_offset0,
                                   pilot_offset_step,
                                   pilot_values,
                                   Float64(samplerate),
                                   Float64(band_limit_hz),
                                   segment_carriers,
                                   prbs_bits,
                                   Float64(cpe_alpha),
                                   Float64(cpe_max_step_deg) * Float64(pi) / 180.0,
                                   Float64(pilot_min_mag),
                                   Float64(pilot_trim_ratio),
                                   update_conf_on,
                                   update_conf_off,
                                   false,
                                   auto_sp_phase,
                                   auto_sp_phase_interval,
                                   0,
                                   1,
                                   symbol_index_ref,
                                   gap_freeze_ref,
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

function task!(context::ISDBTCPECorrectorContext)
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
                        if context.auto_phase_countdown <= 0
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
                            context.selected_phase = best_phase
                            context.auto_phase_countdown = context.auto_sp_phase_interval - 1
                        else
                            context.auto_phase_countdown -= 1
                        end
                        phase = context.selected_phase
                    end
                    band_pos = context.phase_bins[phase]
                    pilot_vals = context.phase_refs[phase]
                    s_re = 0.0
                    s_im = 0.0
                    mag_sum = 0.0
                    used = 0
                    @inbounds for i in 1:length(band_pos)
                        v = rd_buffer.buf[band_pos[i]] / pilot_vals[i]
                        m = abs(v)
                        if m >= context.pilot_min_mag
                            used += 1
                            context.angle_buf[used] = atan(imag(v), real(v))
                            context.mag_buf[used] = m
                            s_re += real(v)
                            s_im += imag(v)
                            mag_sum += m
                        end
                    end
                    conf = mag_sum > 0 ? sqrt(s_re * s_re + s_im * s_im) / mag_sum : 0.0
                    updated = false
                    freeze_active = context.gap_freeze_ref !== nothing && context.gap_freeze_ref[] > 0
                    if freeze_active
                        context.update_enabled = false
                    elseif context.update_enabled
                        if conf < context.min_update_conf_off
                            context.update_enabled = false
                        end
                    elseif conf >= context.min_update_conf_on
                        context.update_enabled = true
                    end
                    if used > 0 && context.update_enabled && !freeze_active
                        phi = atan(s_im, s_re)
                        if context.pilot_trim_ratio > 0 && used >= 8
                            @inbounds for i in 1:used
                                r = abs(wrap_phase(context.angle_buf[i] - phi))
                                context.residual_buf[i] = r
                                context.residual_sorted_buf[i] = r
                            end
                            thresh_idx = clamp(floor(Int, (1.0 - context.pilot_trim_ratio) * used), 1, used)
                            sort!(@view(context.residual_sorted_buf[1:used]))
                            thresh = context.residual_sorted_buf[thresh_idx]
                            s_re2 = 0.0
                            s_im2 = 0.0
                            mag_sum2 = 0.0
                            @inbounds for i in 1:used
                                if context.residual_buf[i] <= thresh
                                    w = min(context.mag_buf[i], 2.0)
                                    a = context.angle_buf[i]
                                    s_re2 += w * cos(a)
                                    s_im2 += w * sin(a)
                                    mag_sum2 += w
                                end
                            end
                            if mag_sum2 > 0
                                s_re = s_re2
                                s_im = s_im2
                                mag_sum = mag_sum2
                                phi = atan(s_im, s_re)
                            end
                        end
                        err = wrap_phase(phi - context.cpe_phase)
                        # Low-confidence symbols update more conservatively to reduce phase jitter.
                        conf_gain = clamp((conf - context.min_update_conf_off) / max(1e-6, 1.0 - context.min_update_conf_off), 0.2, 1.0)
                        delta = context.cpe_alpha * conf_gain * err
                        if delta > context.cpe_max_step
                            delta = context.cpe_max_step
                        elseif delta < -context.cpe_max_step
                            delta = -context.cpe_max_step
                        end
                        context.cpe_phase = wrap_phase(context.cpe_phase + delta)
                        updated = true
                    elseif context.auto_sp_phase
                        context.auto_phase_countdown = 0
                    end
                    if context.log_stats
                        now = time()
                        if now - context.last_log_time >= context.log_interval
                            deg = context.cpe_phase * 180 / Float64(π)
                            lock(LOG_LOCK) do
                                println("CPE: phase=", round(deg, digits = 2),
                                        " deg conf=", round(conf, digits = 3),
                                        " used=", used, "/", length(band_pos),
                                        " gate=", context.update_enabled,
                                        " updated=", updated)
                            end
                            context.last_log_time = now
                        end
                    end

                    c = cos(context.cpe_phase)
                    s = -sin(context.cpe_phase)
                    @inbounds for k in 1:context.nfft
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
            println("ISDBTCPECorrector error: ", e)
        end
    end
    return nothing
end

function input!(context::ISDBTCPECorrectorContext, samples::AbstractVector{ComplexF32}, samples_size::Int)
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
            println("CPE: input_backpressure count=", context.input_overrun_count)
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

function stop!(context::ISDBTCPECorrectorContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

end
