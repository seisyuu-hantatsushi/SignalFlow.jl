module ISDBTSymbolSync

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer

using GLMakie

#= 
ISDB-T OFDM symbol synchronization (CP correlation).
Searches the cyclic prefix alignment by maximizing normalized correlation
between CP and the end of the OFDM symbol, then outputs CP-removed symbols.
=#

mutable struct ISDBTSymbolSyncContext <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    mode::Int
    gi_ratio::Float64
    samplerate::Float64
    nfft::Int
    ncp::Int
    symbol_len::Int
    search_symbols::Int
    cfo_enabled::Bool
    cfo_alpha::Float64
    cfo_rad_per_sample::Float64
    cfo_phase::Float64
    enable_stats::Bool
    stats_len::Int
    stats_avg_len::Int
    stats_offset::Vector{Float64}
    stats_peak::Vector{Float64}
    stats_offset_avg::Vector{Float64}
    stats_peak_avg::Vector{Float64}
    stats_offset_obs::Union{Nothing,Observable{Vector{Float64}}}
    stats_peak_obs::Union{Nothing,Observable{Vector{Float64}}}
    stats_offset_avg_obs::Union{Nothing,Observable{Vector{Float64}}}
    stats_peak_avg_obs::Union{Nothing,Observable{Vector{Float64}}}
    stats_fig::Union{Nothing,Figure}
    stats_screen::Union{Nothing,GLMakie.Screen}
    stats_task::Union{Nothing,Task}
    statsQ::Union{Nothing,Channel{NTuple{2,Float64}}}
    log_stats::Bool
    log_interval::Float64
    last_log_time::Float64
    buffer::Vector{ComplexF32}
    buffer_size::Int
    outbuf::Vector{ComplexF32}
    ringbuffer::RingFrameBuffer{ComplexF32}
    holdbuf::Union{Nothing, Int}
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
end

function mode_to_nfft(mode::Int)
    mode == 1 && return 2048
    mode == 2 && return 4096
    mode == 3 && return 8192
    error("ISDBTSymbolSync: mode must be 1, 2, or 3.")
end

function gi_to_ratio(gi::Real)
    gi == 1 // 4 && return 0.25
    gi == 1 // 8 && return 0.125
    gi == 1 // 16 && return 0.0625
    gi == 1 // 32 && return 0.03125
    if gi isa Real
        gi_val = Float64(gi)
        for v in (0.25, 0.125, 0.0625, 0.03125)
            abs(gi_val - v) < 1e-6 && return v
        end
    end
    error("ISDBTSymbolSync: gi must be 1/4, 1/8, 1/16, or 1/32.")
end

function CreateISDBTSymbolSync(; mode::Int = 3,
                               gi = 1 // 8,
                               samplerate::Real = 8_000_000,
                               search_symbols::Int = 2,
                               cfo_enabled::Bool = false,
                               cfo_alpha::Real = 0.1,
                               log_interval::Real = 1.0,
                               enable_stats::Bool = false,
                               stats_len::Int = 300,
                               stats_avg_len::Int = 10,
                               stats_window_size = (900, 480),
                               log_stats::Bool = false,
                               poolsize::Int = 8,
                               frame_size::Int = 32768)
    samplerate <= 0 && error("ISDBTSymbolSync: samplerate must be positive.")
    search_symbols < 1 && error("ISDBTSymbolSync: search_symbols must be >= 1.")
    cfo_alpha <= 0 && error("ISDBTSymbolSync: cfo_alpha must be positive.")
    log_interval <= 0 && error("ISDBTSymbolSync: log_interval must be positive.")
    poolsize < 1 && error("ISDBTSymbolSync: poolsize must be at least 1.")
    frame_size < 1 && error("ISDBTSymbolSync: frame_size must be at least 1.")

    nfft = mode_to_nfft(mode)
    gi_ratio = gi_to_ratio(gi)
    ncp = Int(round(nfft * gi_ratio))
    ncp < 1 && error("ISDBTSymbolSync: CP length must be at least 1.")
    symbol_len = nfft + ncp
    stats_len < 2 && error("ISDBTSymbolSync: stats_len must be at least 2.")
    stats_avg_len < 1 && error("ISDBTSymbolSync: stats_avg_len must be at least 1.")

    new_sinks = Channel{SignalFlowBlock}(4)
    sinks = Vector{SignalFlowBlock}()
    buffer = Vector{ComplexF32}(undef, 0)
    outbuf = Vector{ComplexF32}(undef, nfft)

    ctx = ISDBTSymbolSyncContext(Base.Threads.Atomic{Bool}(true),
                                 mode,
                                 gi_ratio,
                                 Float64(samplerate),
                                 nfft,
                                 ncp,
                                 symbol_len,
                                 search_symbols,
                                 cfo_enabled,
                                 Float64(cfo_alpha),
                                 0.0,
                                 0.0,
                                 enable_stats,
                                 stats_len,
                                 stats_avg_len,
                                 fill(0.0, stats_len),
                                 fill(0.0, stats_len),
                                 fill(0.0, stats_len),
                                 fill(0.0, stats_len),
                                 nothing,
                                 nothing,
                                 nothing,
                                 nothing,
                                 nothing,
                                 nothing,
                                 nothing,
                                 nothing,
                                 log_stats,
                                 Float64(log_interval),
                                 time(),
                                 buffer,
                                 0,
                                 outbuf,
                                 RingFrameBuffer(ComplexF32, frame_size, poolsize),
                                 nothing,
                                 nothing,
                                 new_sinks,
                                 sinks)
    if ctx.enable_stats
        ctx.stats_offset_obs = Observable(copy(ctx.stats_offset))
        ctx.stats_peak_obs = Observable(copy(ctx.stats_peak))
        ctx.stats_offset_avg_obs = Observable(copy(ctx.stats_offset_avg))
        ctx.stats_peak_avg_obs = Observable(copy(ctx.stats_peak_avg))
        fig = Figure(size = stats_window_size)
        ax1 = Axis(fig[1, 1], xlabel = "Samples", ylabel = "Offset")
        ax2 = Axis(fig[2, 1], xlabel = "Samples", ylabel = "Peak (log10)")
        x = collect(1:ctx.stats_len)
        line_off = lines!(ax1, x, ctx.stats_offset_obs; linewidth = 1, color = :blue)
        line_off_avg = lines!(ax1, x, ctx.stats_offset_avg_obs; linewidth = 1, color = :red)
        line_peak = lines!(ax2, x, ctx.stats_peak_obs; linewidth = 1, color = :blue)
        line_peak_avg = lines!(ax2, x, ctx.stats_peak_avg_obs; linewidth = 1, color = :red)
        ylims!(ax1, 0.0, Float64(ctx.symbol_len))
        ylims!(ax2, -6.0, 0.0)
        axislegend(ax1, [line_off, line_off_avg], ["Offset", "Offset(avg)"])
        axislegend(ax2, [line_peak, line_peak_avg], ["Peak", "Peak(avg)"])
        main_task = current_task()
        renderloop = screen -> begin
            try
                GLMakie.renderloop(screen)
            catch e
                if e isa InterruptException
                    Base.throwto(main_task, e)
                    return
                end
                rethrow()
            end
        end
        ctx.stats_screen = GLMakie.Screen(; renderloop = renderloop)
        display(ctx.stats_screen, fig)
        if ctx.stats_screen !== nothing
            GLMakie.set_title!(ctx.stats_screen, "ISDB-T CP Sync Stats")
        end
        ctx.stats_fig = fig
        ctx.statsQ = Channel{NTuple{2,Float64}}(1)
        ctx.stats_task = @async stats_update_task!(ctx)
    end
    ctx.worker = Threads.@spawn task!(ctx)
    return ctx
end

function process_buffer!(context::ISDBTSymbolSyncContext)
    min_needed = context.symbol_len * context.search_symbols
    while context.buffer_size >= min_needed
        search_len = context.buffer_size - context.symbol_len + 1
        search_len > context.symbol_len && (search_len = context.symbol_len)
        best_offset = 1
        best_metric = -1.0
        @inbounds for offset in 1:search_len
            # Normalized CP correlation metric.
            s_re = 0.0
            s_im = 0.0
            p1 = 0.0
            p2 = 0.0
            base_a = offset
            base_b = offset + context.nfft
            for k in 0:context.ncp - 1
                a = context.buffer[base_a + k]
                b = context.buffer[base_b + k]
                s_re += real(a) * real(b) + imag(a) * imag(b)
                s_im += real(a) * imag(b) - imag(a) * real(b)
                p1 += real(a) * real(a) + imag(a) * imag(a)
                p2 += real(b) * real(b) + imag(b) * imag(b)
            end
            metric = (s_re * s_re + s_im * s_im) / (p1 * p2 + eps(Float64))
            if metric > best_metric
                best_metric = metric
                best_offset = offset
            end
        end

        # Extract CP-removed symbol.
        start = best_offset + context.ncp
        copyto!(context.outbuf, 1, context.buffer, start, context.nfft)

        if context.cfo_enabled
            base_a = best_offset
            base_b = best_offset + context.nfft
            s_re = 0.0
            s_im = 0.0
            @inbounds for k in 0:context.ncp - 1
                a = context.buffer[base_a + k]
                b = context.buffer[base_b + k]
                s_re += real(a) * real(b) + imag(a) * imag(b)
                s_im += real(a) * imag(b) - imag(a) * real(b)
            end
            phi = atan(s_im, s_re)
            est = phi / context.nfft
            context.cfo_rad_per_sample += context.cfo_alpha * (est - context.cfo_rad_per_sample)
            phase = context.cfo_phase
            @inbounds for k in 1:context.nfft
                c = cos(phase)
                s = sin(phase)
                v = context.outbuf[k]
                context.outbuf[k] = ComplexF32(Float32(real(v) * c + imag(v) * s),
                                               Float32(imag(v) * c - real(v) * s))
                phase += context.cfo_rad_per_sample
            end
            context.cfo_phase = phase
            if context.cfo_phase > Float64(π) || context.cfo_phase < -Float64(π)
                context.cfo_phase = mod(context.cfo_phase + Float64(π), 2 * Float64(π)) - Float64(π)
            end
        end

        while isready(context.new_sinks)
            push!(context.sinks, take!(context.new_sinks))
        end
        for sink in context.sinks
            input!(sink, context.outbuf, context.nfft)
        end
        if context.enable_stats && context.statsQ !== nothing
            if isready(context.statsQ)
                take!(context.statsQ)
            end
            put!(context.statsQ, (Float64(best_offset), Float64(best_metric)))
        end
        if context.log_stats
            now = time()
            if now - context.last_log_time >= context.log_interval
                println("ISDBTSymbolSync: offset=", best_offset, " peak=", best_metric)
                context.last_log_time = now
            end
        end

        # Drop up to the end of the selected symbol to avoid reusing samples.
        drop_count = best_offset + context.symbol_len - 1
        remaining = context.buffer_size - drop_count
        if remaining > 0
            copyto!(context.buffer, 1, context.buffer, drop_count + 1, remaining)
        end
        context.buffer_size = max(remaining, 0)
        resize!(context.buffer, context.buffer_size)
    end
    return nothing
end

function task!(context::ISDBTSymbolSyncContext)
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                n = rd_buffer.store_size
                if n > 0
                    new_size = context.buffer_size + n
                    resize!(context.buffer, new_size)
                    copyto!(context.buffer, context.buffer_size + 1, rd_buffer.buf, 1, n)
                    context.buffer_size = new_size
                    process_buffer!(context)
                end
                rd_buffer.store_size = 0
                put!(context.ringbuffer.freeQ, rd_index)
            else
                yield()
            end
        end
    catch e
        if !(e isa InterruptException)
            println("ISDBTSymbolSync error: ", e)
        end
    end
    return nothing
end

function stats_update_task!(context::ISDBTSymbolSyncContext)
    try
        while context.running[] && context.statsQ !== nothing
            if isready(context.statsQ)
                offset, peak = take!(context.statsQ)
                if context.stats_len > 1
                    @inbounds begin
                        copyto!(context.stats_offset, 1, context.stats_offset, 2, context.stats_len - 1)
                        copyto!(context.stats_peak, 1, context.stats_peak, 2, context.stats_len - 1)
                        context.stats_offset[end] = offset
                        context.stats_peak[end] = log10(peak + eps(Float64))
                        avg_len = min(context.stats_avg_len, context.stats_len)
                        start_idx = context.stats_len - avg_len + 1
                        off_avg = sum(view(context.stats_offset, start_idx:context.stats_len)) / avg_len
                        peak_avg = sum(view(context.stats_peak, start_idx:context.stats_len)) / avg_len
                        copyto!(context.stats_offset_avg, 1, context.stats_offset_avg, 2, context.stats_len - 1)
                        copyto!(context.stats_peak_avg, 1, context.stats_peak_avg, 2, context.stats_len - 1)
                        context.stats_offset_avg[end] = off_avg
                        context.stats_peak_avg[end] = peak_avg
                    end
                    context.stats_offset_obs[] = context.stats_offset
                    context.stats_peak_obs[] = context.stats_peak
                    if context.stats_offset_avg_obs !== nothing && context.stats_peak_avg_obs !== nothing
                        context.stats_offset_avg_obs[] = context.stats_offset_avg
                        context.stats_peak_avg_obs[] = context.stats_peak_avg
                    end
                end
            else
                yield()
            end
        end
    catch e
        if !(e isa InterruptException)
            println("ISDBTSymbolSync stats error: ", e)
        end
    end
    return nothing
end

function input!(context::ISDBTSymbolSyncContext, samples::AbstractVector{ComplexF32}, samples_size::Int)
    if !context.running[] || samples_size <= 0
        return false
    end

    actual_size = min(samples_size, length(samples))
    if actual_size <= 0
        return 0
    end

    remain_size = actual_size
    while remain_size > 0
        if context.holdbuf == nothing && isready(context.ringbuffer.freeQ)
            context.holdbuf = take!(context.ringbuffer.freeQ)
        end

        if context.holdbuf == nothing
            return -1
        else
            write_frame = context.ringbuffer.bufs[context.holdbuf]
            copy_size = min(remain_size, context.ringbuffer.frame_size - write_frame.store_size)
            copyto!(write_frame.buf, write_frame.store_size + 1, samples, actual_size - remain_size + 1, copy_size)
            write_frame.store_size += copy_size
            remain_size -= copy_size
            if write_frame.store_size >= context.ringbuffer.frame_size
                put!(context.ringbuffer.fullQ, context.holdbuf)
                context.holdbuf = nothing
            end
        end
    end

    return samples_size
end

function stop!(context::ISDBTSymbolSyncContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

end
