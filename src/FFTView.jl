
module FFTView

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: FrameBuffer, RingFrameBuffer

using FFTW
using GLMakie

@enum WindowFunctions begin
    Hann
    Blackman
    BlackmanHaris
end

mutable struct Parameter
    inputSamplingRate::UInt64
    window::WindowFunctions
end

mutable struct ViewContext{T} <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    parameter::Parameter
    fft_buf::Vector{T}
    fft_pos::Int
    win::Vector{Float32}
    tmp::Vector{ComplexF32}
    idx_lo::Int
    idx_hi::Int
    yobs::Observable{Vector{Float64}}
    fig::Figure
    screen::Union{Nothing, GLMakie.Screen}
    freq_view::Vector{Float64}
    main_task::Union{Nothing,Task}
    worker::Union{Nothing,Task}
    update_task::Union{Nothing,Task}
    signal_datas::RingFrameBuffer{T}   
    holdbuf::Union{Nothing, Int}
    result_ringbuffer::RingFrameBuffer{Float64}
end

function hann_window_coeffs(ns::UnitRange{Int}, denom)
    return 0.5f0 .* (1 .- cos.(2f0 * Float32(π) .* Float32.(ns) ./ denom))
end

function blackman_window_coeffs(ns::UnitRange{Int}, denom)
    a0 = 0.42f0
    a1 = 0.5f0
    a2 = 0.08f0
    phase = 2f0 * Float32(π) .* Float32.(ns) ./ denom
    return a0 .- a1 .* cos.(phase) .+ a2 .* cos.(2f0 .* phase)
end

function blackman_haris_window_coeffs(ns::UnitRange{Int}, denom)
    a0 = 0.35875f0
        a1 = 0.48829f0
    a2 = 0.14128f0
    a3 = 0.01168f0
    phase = 2f0 * Float32(π) .* Float32.(ns) ./ denom
    return a0 .- a1 .* cos.(phase) .+ a2 .* cos.(2f0 .* phase) .- a3 .* cos.(3f0 .* phase)
end

function window_coeffs(fft_size::Int, window::WindowFunctions)
    ns = 0:fft_size - 1
    denom = Float32(fft_size - 1)
    if window == Hann
        return hann_window_coeffs(ns, denom)
    elseif window == Blackman
        return blackman_window_coeffs(ns, denom)
    else
        return blackman_haris_window_coeffs(ns, denom)
    end
end

function fftshift(v::AbstractVector)
    n = length(v)
    h = n ÷ 2
    return vcat(view(v, h + 1:n), view(v, 1:h))
end

function process_samples!(context::ViewContext{ComplexF32}, samples::AbstractVector{ComplexF32}, n::Integer)
    i = 1
    fft_size = length(context.fft_buf)
    while i <= n
        to_copy = min(fft_size - context.fft_pos + 1, n - i + 1)
        copyto!(context.fft_buf, context.fft_pos, samples, i, to_copy)
        context.fft_pos += to_copy
        i += to_copy
        if context.fft_pos > fft_size
            @inbounds for k in 1:fft_size
                context.tmp[k] = context.fft_buf[k] * context.win[k]
            end
            spec = abs.(fft(context.tmp))
            spec = fftshift(spec)
            spec_db = 20 .* log10.(spec ./ fft_size .+ eps(Float32))
            if isready(context.result_ringbuffer.freeQ)
                result_index = take!(context.result_ringbuffer.freeQ)
                result_buf = context.result_ringbuffer.bufs[result_index].buf
                @inbounds for k in 1:length(result_buf)
                    result_buf[k] = Float64(spec_db[context.idx_lo + k - 1])
                end
                put!(context.result_ringbuffer.fullQ, result_index)
            end
            context.fft_pos = 1
        end
    end
    return nothing
end

function process_samples!(context::ViewContext{Float32}, samples::AbstractVector{Float32}, n::Integer)
    i = 1
    fft_size = length(context.fft_buf)
    while i <= n
        to_copy = min(fft_size - context.fft_pos + 1, n - i + 1)
        copyto!(context.fft_buf, context.fft_pos, samples, i, to_copy)
        context.fft_pos += to_copy
        i += to_copy
        if context.fft_pos > fft_size
            @inbounds for k in 1:fft_size
                context.tmp[k] = ComplexF32(context.fft_buf[k] * context.win[k], 0f0)
            end
            spec = abs.(fft(context.tmp))
            spec = fftshift(spec)
            spec_db = 20 .* log10.(spec ./ fft_size .+ eps(Float32))
            if isready(context.result_ringbuffer.freeQ)
                result_index = take!(context.result_ringbuffer.freeQ)
                result_buf = context.result_ringbuffer.bufs[result_index].buf
                @inbounds for k in 1:length(result_buf)
                    result_buf[k] = Float64(spec_db[context.idx_lo + k - 1])
                end
                put!(context.result_ringbuffer.fullQ, result_index)
            end
            context.fft_pos = 1
        end
    end
    return nothing
end

function CreateView(::Type{T}, inputSamplingRate::UInt64, numberOfFFTSampling::UInt64, window::WindowFunctions;
                    frame_size::Int = Int(numberOfFFTSampling),
                    poolsize::Int = 16,
                    title::AbstractString = "FFT View",
                    fmin::Float64 = -Float64(inputSamplingRate) / 2,
                    fmax::Float64 = Float64(inputSamplingRate) / 2,
                    window_size = (900, 480)) where {T}

    fft_size = Int(numberOfFFTSampling)
    fft_size < 2 && error("FFTView: FFT size must be at least 2.")
    fmin >= fmax && error("FFTView: Frequency range must satisfy fmin < fmax.")
    frame_size < 1 && error("FFTView: Frame size must be at least 1.")
    poolsize < 1 && error("FFTView: Pool size must be at least 1.")

    freqs = range(-Float64(inputSamplingRate) / 2, Float64(inputSamplingRate) / 2; length = Int(fft_size))
    idx_lo = max(1, searchsortedfirst(freqs, fmin))
    idx_hi = min(fft_size, searchsortedlast(freqs, fmax))
    idx_lo > idx_hi && error("FFTView: Frequency range does not intersect FFT span.")
    freq_view = collect(freqs[idx_lo:idx_hi])

    fig = Figure(size = window_size)
    ax = Axis(fig[1, 1], xlabel = "Frequency (Hz)", ylabel = "Magnitude (dB)")
    tick_step = 10_000.0
    label_step = 100_000.0
    tick_vals = collect(range(fmin, fmax; step = tick_step))
    tick_labels = [mod(round(v), label_step) == 0 ? string(Int(round(v / 1000))) * "k" : "" for v in tick_vals]
    zero_idx = findfirst(==(0.0), tick_vals)
    if zero_idx !== nothing
        tick_labels[zero_idx] = "0"
    end
    ax.xticks = (tick_vals, tick_labels)
    yobs = Observable(fill(-120.0, length(freq_view)))
    lines!(ax, freq_view, yobs; linewidth = 1)
    ylims!(ax, -120, 20)
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

    screen = GLMakie.Screen(; renderloop = renderloop)
    display(screen, fig)
    if screen !== nothing
        GLMakie.set_title!(screen, title)
    end

    result_ringbuffer = RingFrameBuffer(Float64, length(freq_view), poolsize)
    
    view = ViewContext(Base.Threads.Atomic{Bool}(true),
                       Parameter(inputSamplingRate, window),
                       Vector{T}(undef, fft_size),
                       1,
                       window_coeffs(fft_size, window),
                       Vector{ComplexF32}(undef, fft_size),
                       idx_lo,
                       idx_hi,
                       yobs,
                       fig,
                       screen,
                       freq_view,
                       main_task,
                       nothing,
                       nothing,
                       RingFrameBuffer(T, fft_size, poolsize),
                       nothing,
                       result_ringbuffer)
    view.update_task = @async update_task!(view)
    view.worker = Threads.@spawn task!(view)
    return view
end

function isopen(context::ViewContext)
    return context.screen === nothing || GLMakie.isopen(context.screen)
end

function task!(context::ViewContext{T}) where {T}
    try
        while context.running[]
            if context.screen !== nothing && !isopen(context)
                context.running[] = false
                break
            end
            if isready(context.signal_datas.fullQ)
                rd_index = take!(context.signal_datas.fullQ)
                rd_buffer = context.signal_datas.bufs[rd_index]
                process_samples!(context, rd_buffer.buf, rd_buffer.store_size)
                rd_buffer.store_size = 0
                put!(context.signal_datas.freeQ, rd_index)
            else
                yield()
            end
        end
    catch e
        if !(e isa InterruptException)
            println("FFTView task error: ", e)
        end
    end
    return nothing
end

function update_task!(context::ViewContext{T}) where{T}
    try
        while context.running[]
            if isready(context.result_ringbuffer.fullQ)
                result_index = take!(context.result_ringbuffer.fullQ)
                result_buf = context.result_ringbuffer.bufs[result_index].buf
                copyto!(context.yobs[], result_buf)
                Base.notify(context.yobs)
                put!(context.result_ringbuffer.freeQ, result_index)
            else
                yield()
            end
        end
    catch e
        if !(e isa InterruptException)
            println("FFTView update error: ", e)
        end
    end
    return nothing
end

function input!(context::ViewContext{T}, samples::AbstractVector{T}, samples_size::Int) where {T}
    if !context.running[] ||  samples_size <= 0
        return false
    end

    actual_size = min(samples_size, length(samples))
    if actual_size <= 0
        return 0
    end
    remain_size = actual_size

    while remain_size > 0
        if context.holdbuf == nothing && isready(context.signal_datas.freeQ)
            context.holdbuf = take!(context.signal_datas.freeQ)
        end

        if context.holdbuf == nothing
            return -1
        else
            write_frame = context.signal_datas.bufs[context.holdbuf]
            copy_size = min(remain_size, context.signal_datas.frame_size - write_frame.store_size)
            copyto!(write_frame.buf, write_frame.store_size + 1, samples, actual_size-remain_size+1, copy_size)
            write_frame.store_size += copy_size
            remain_size -= copy_size
            if write_frame.store_size >= context.signal_datas.frame_size
                put!(context.signal_datas.fullQ, context.holdbuf)
                context.holdbuf = nothing
            end
        end
    end

    return samples_size
end

function stop!(context::ViewContext)
    println("FFTView.stop()")
    context.running[] = false
    wait(context.worker)
    wait(context.update_task)
    return nothing
end

end
