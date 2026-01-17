module BandSNREstimator

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer

using FFTW
using GLMakie

@enum WindowType begin
    Rectangular
    Hann
end

mutable struct BandSNREstimatorContext <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    samplerate::Float64
    fft_size::Int
    window::Vector{Float32}
    tmp::Vector{ComplexF32}
    spec::Vector{ComplexF32}
    sig_idx::Vector{Int}
    noise_idx::Vector{Int}
    enable_stats::Bool
    stats_len::Int
    stats_avg_len::Int
    stats_snr::Vector{Float64}
    stats_snr_avg::Vector{Float64}
    stats_snr_obs::Union{Nothing,Observable{Vector{Float64}}}
    stats_snr_avg_obs::Union{Nothing,Observable{Vector{Float64}}}
    stats_fig::Union{Nothing,Figure}
    stats_screen::Union{Nothing,GLMakie.Screen}
    stats_task::Union{Nothing,Task}
    statsQ::Union{Nothing,Channel{Float64}}
    stats_interval::Float64
    last_stats_time::Float64
    log_stats::Bool
    log_interval::Float64
    last_log_time::Float64
    ringbuffer::RingFrameBuffer{ComplexF32}
    holdbuf::Union{Nothing, Int}
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
end

function window_coeffs(n::Int, window::WindowType)
    if window == Rectangular
        return ones(Float32, n)
    else
        ns = 0:n - 1
        denom = Float32(n - 1)
        return 0.5f0 .* (1 .- cos.(2f0 * Float32(π) .* Float32.(ns) ./ denom))
    end
end

function band_indices(freqs::Vector{Float64}, bands::Vector{Tuple{Float64,Float64}})
    idx = Int[]
    for (f0, f1) in bands
        lo = min(f0, f1)
        hi = max(f0, f1)
        for (i, f) in enumerate(freqs)
            if f >= lo && f <= hi
                push!(idx, i)
            end
        end
    end
    return unique(idx)
end

function unshift_indices(idx::Vector{Int}, n::Int)
    out = Vector{Int}(undef, length(idx))
    half = n ÷ 2
    @inbounds for i in 1:length(idx)
        out[i] = ((idx[i] + half - 1) % n) + 1
    end
    return unique(out)
end

function CreateBandSNREstimator(; samplerate::Real,
                                 fft_size::Int = 4096,
                                 signal_band::Tuple{<:Real,<:Real},
                                 noise_bands::AbstractVector{<:Tuple{<:Real,<:Real}},
                                 window::WindowType = Hann,
                                 enable_stats::Bool = false,
                                 stats_len::Int = 300,
                                 stats_avg_len::Int = 10,
                                 stats_window_size = (900, 360),
                                 stats_interval::Real = 0.5,
                                 log_stats::Bool = false,
                                 log_interval::Real = 1.0,
                                 poolsize::Int = 8)
    samplerate <= 0 && error("BandSNREstimator: samplerate must be positive.")
    fft_size < 32 && error("BandSNREstimator: fft_size must be >= 32.")
    stats_len < 2 && error("BandSNREstimator: stats_len must be at least 2.")
    stats_avg_len < 1 && error("BandSNREstimator: stats_avg_len must be at least 1.")
    stats_interval <= 0 && error("BandSNREstimator: stats_interval must be positive.")
    log_interval <= 0 && error("BandSNREstimator: log_interval must be positive.")
    poolsize < 1 && error("BandSNREstimator: poolsize must be at least 1.")

    freqs = collect(range(-Float64(samplerate) / 2, Float64(samplerate) / 2; length = fft_size))
    sig_idx = band_indices(freqs, [(Float64(signal_band[1]), Float64(signal_band[2]))])
    noise_idx = band_indices(freqs, [(Float64(b[1]), Float64(b[2])) for b in noise_bands])
    isempty(sig_idx) && error("BandSNREstimator: signal_band has no bins.")
    isempty(noise_idx) && error("BandSNREstimator: noise_bands have no bins.")
    sig_idx = unshift_indices(sig_idx, fft_size)
    noise_idx = unshift_indices(noise_idx, fft_size)

    new_sinks = Channel{SignalFlowBlock}(4)
    sinks = Vector{SignalFlowBlock}()
    now = time()
    ctx = BandSNREstimatorContext(Base.Threads.Atomic{Bool}(true),
                                  Float64(samplerate),
                                  fft_size,
                                  window_coeffs(fft_size, window),
                                  Vector{ComplexF32}(undef, fft_size),
                                  Vector{ComplexF32}(undef, fft_size),
                                  sig_idx,
                                  noise_idx,
                                  enable_stats,
                                  stats_len,
                                  stats_avg_len,
                                  fill(-100.0, stats_len),
                                  fill(-100.0, stats_len),
                                  nothing,
                                  nothing,
                                  nothing,
                                  nothing,
                                  nothing,
                                  nothing,
                                  Float64(stats_interval),
                                  now,
                                  log_stats,
                                  Float64(log_interval),
                                  now,
                                  RingFrameBuffer(ComplexF32, fft_size, poolsize),
                                  nothing,
                                  nothing,
                                  new_sinks,
                                  sinks)
    if ctx.enable_stats
        ctx.stats_snr_obs = Observable(copy(ctx.stats_snr))
        ctx.stats_snr_avg_obs = Observable(copy(ctx.stats_snr_avg))
        fig = Figure(size = stats_window_size)
        ax = Axis(fig[1, 1], xlabel = "Samples", ylabel = "SNR (dB)")
        x = collect(1:ctx.stats_len)
        line_snr = lines!(ax, x, ctx.stats_snr_obs; linewidth = 1, color = :blue)
        line_snr_avg = lines!(ax, x, ctx.stats_snr_avg_obs; linewidth = 1, color = :red)
        ylims!(ax, -5, 40)
        axislegend(ax, [line_snr, line_snr_avg], ["SNR", "SNR(avg)"])
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
            GLMakie.set_title!(ctx.stats_screen, "Band SNR Estimate")
        end
        ctx.stats_fig = fig
        ctx.statsQ = Channel{Float64}(1)
        ctx.stats_task = @async stats_update_task!(ctx)
    end
    ctx.worker = Threads.@spawn task!(ctx)
    return ctx
end

function estimate_snr_db(context::BandSNREstimatorContext, samples::AbstractVector{ComplexF32})
    @inbounds for k in 1:context.fft_size
        context.tmp[k] = samples[k] * context.window[k]
    end
    context.spec .= fft(context.tmp)
    spec = context.spec
    n = context.fft_size
    sig = 0.0
    noise = 0.0
    @inbounds for idx in context.sig_idx
        s = spec[idx]
        sig += real(s) * real(s) + imag(s) * imag(s)
    end
    @inbounds for idx in context.noise_idx
        s = spec[idx]
        noise += real(s) * real(s) + imag(s) * imag(s)
    end
    sig /= length(context.sig_idx)
    noise /= length(context.noise_idx)
    snr = sig / (noise + eps(Float64))
    return 10.0 * log10(snr + eps(Float64))
end

function task!(context::BandSNREstimatorContext)
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                if rd_buffer.store_size == context.fft_size
                    snr_db = estimate_snr_db(context, rd_buffer.buf)
                    if context.enable_stats && context.statsQ !== nothing
                        now = time()
                        if now - context.last_stats_time >= context.stats_interval
                            if isready(context.statsQ)
                                take!(context.statsQ)
                            end
                            put!(context.statsQ, snr_db)
                            context.last_stats_time = now
                        end
                    end
                    if context.log_stats
                        now = time()
                        if now - context.last_log_time >= context.log_interval
                            println("BandSNREstimator: snr=", round(snr_db; digits = 2), " dB")
                            context.last_log_time = now
                        end
                    end
                    while isready(context.new_sinks)
                        push!(context.sinks, take!(context.new_sinks))
                    end
                    for sink in context.sinks
                        input!(sink, rd_buffer.buf, context.fft_size)
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
            println("BandSNREstimator error: ", e)
        end
    end
    return nothing
end

function stats_update_task!(context::BandSNREstimatorContext)
    try
        while context.running[] && context.statsQ !== nothing
            if isready(context.statsQ)
                snr_db = take!(context.statsQ)
                if context.stats_len > 1
                    @inbounds begin
                        copyto!(context.stats_snr, 1, context.stats_snr, 2, context.stats_len - 1)
                        context.stats_snr[end] = snr_db
                        avg_len = min(context.stats_avg_len, context.stats_len)
                        start_idx = context.stats_len - avg_len + 1
                        snr_avg = sum(view(context.stats_snr, start_idx:context.stats_len)) / avg_len
                        copyto!(context.stats_snr_avg, 1, context.stats_snr_avg, 2, context.stats_len - 1)
                        context.stats_snr_avg[end] = snr_avg
                    end
                    context.stats_snr_obs[] = copy(context.stats_snr)
                    if context.stats_snr_avg_obs !== nothing
                        context.stats_snr_avg_obs[] = copy(context.stats_snr_avg)
                    end
                end
            else
                yield()
            end
        end
    catch e
        if !(e isa InterruptException)
            println("BandSNREstimator stats error: ", e)
        end
    end
    return nothing
end

function input!(context::BandSNREstimatorContext, samples::AbstractVector{ComplexF32}, samples_size::Int)
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

function stop!(context::BandSNREstimatorContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    if context.stats_task !== nothing
        wait(context.stats_task)
    end
    return nothing
end

end
