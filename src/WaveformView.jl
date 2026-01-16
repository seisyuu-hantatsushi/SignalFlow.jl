module WaveformView

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer

using GLMakie

@enum WaveformComplexMode begin
    RealPart
    ImagPart
    Magnitude
end

mutable struct ViewContext{T} <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    samplerate::Float64
    channels::Int
    complex_mode::WaveformComplexMode
    yobs::Vector{Observable{Vector{Float64}}}
    fig::Figure
    screen::Union{Nothing, GLMakie.Screen}
    main_task::Union{Nothing,Task}
    worker::Union{Nothing,Task}
    update_task::Union{Nothing,Task}
    ringbuffer::RingFrameBuffer{T}
    holdbuf::Union{Nothing, Int}
    result_ringbuffer::RingFrameBuffer{Float64}
end

function process_samples!(context::ViewContext{T}, samples::AbstractVector{T}, n::Integer) where {T<:Real}
    if !isready(context.result_ringbuffer.freeQ)
        return nothing
    end
    result_index = take!(context.result_ringbuffer.freeQ)
    result_buf = context.result_ringbuffer.bufs[result_index].buf
    frame_size = length(result_buf) ÷ context.channels
    @inbounds for i in 1:frame_size
        base = (i - 1) * context.channels
        for ch in 1:context.channels
            src_index = base + ch
            dst_index = (ch - 1) * frame_size + i
            if src_index <= n
                result_buf[dst_index] = Float64(samples[src_index])
            else
                result_buf[dst_index] = 0.0
            end
        end
    end
    put!(context.result_ringbuffer.fullQ, result_index)
    return nothing
end

function process_samples!(context::ViewContext{ComplexF32}, samples::AbstractVector{ComplexF32}, n::Integer)
    if !isready(context.result_ringbuffer.freeQ)
        return nothing
    end
    result_index = take!(context.result_ringbuffer.freeQ)
    result_buf = context.result_ringbuffer.bufs[result_index].buf
    frame_size = length(result_buf)
    len = min(n, frame_size)
    @inbounds for i in 1:len
        s = samples[i]
        if context.complex_mode == RealPart
            result_buf[i] = Float64(real(s))
        elseif context.complex_mode == ImagPart
            result_buf[i] = Float64(imag(s))
        else
            result_buf[i] = Float64(abs(s))
        end
    end
    @inbounds for i in len + 1:frame_size
        result_buf[i] = 0.0
    end
    put!(context.result_ringbuffer.fullQ, result_index)
    return nothing
end

function CreateView(::Type{T}, inputSamplingRate::UInt64;
                    frame_size::Int = 4096,
                    poolsize::Int = 16,
                    channels::Int = 1,
                    complex_mode::WaveformComplexMode = RealPart,
                    title::AbstractString = "Waveform View",
                    window_size = (900, 480)) where {T<:Real}

    frame_size < 1 && error("WaveformView: frame_size must be at least 1.")
    poolsize < 1 && error("WaveformView: poolsize must be at least 1.")
    channels < 1 && error("WaveformView: channels must be at least 1.")

    samplerate = Float64(inputSamplingRate)
    t = collect(range(0.0, step = 1.0 / samplerate, length = frame_size))
    yobs = [Observable(fill(0.0, frame_size)) for _ in 1:channels]

    fig = Figure(size = window_size)
    ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Amplitude")
    for ch in 1:channels
        lines!(ax, t, yobs[ch]; linewidth = 1)
    end

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

    result_ringbuffer = RingFrameBuffer(Float64, frame_size * channels, poolsize)

    view = ViewContext(Base.Threads.Atomic{Bool}(true),
                       samplerate,
                       channels,
                       complex_mode,
                       yobs,
                       fig,
                       screen,
                       main_task,
                       nothing,
                       nothing,
                       RingFrameBuffer(T, frame_size * channels, poolsize),
                       nothing,
                       result_ringbuffer)
    view.update_task = @async update_task!(view)
    view.worker = Threads.@spawn task!(view)
    return view
end

function CreateView(::Type{ComplexF32}, inputSamplingRate::UInt64;
                    frame_size::Int = 4096,
                    poolsize::Int = 16,
                    complex_mode::WaveformComplexMode = RealPart,
                    title::AbstractString = "Waveform View",
                    window_size = (900, 480))

    frame_size < 1 && error("WaveformView: frame_size must be at least 1.")
    poolsize < 1 && error("WaveformView: poolsize must be at least 1.")

    samplerate = Float64(inputSamplingRate)
    t = collect(range(0.0, step = 1.0 / samplerate, length = frame_size))
    yobs = [Observable(fill(0.0, frame_size))]

    fig = Figure(size = window_size)
    ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Amplitude")
    lines!(ax, t, yobs[1]; linewidth = 1)

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

    result_ringbuffer = RingFrameBuffer(Float64, frame_size, poolsize)

    view = ViewContext(Base.Threads.Atomic{Bool}(true),
                       samplerate,
                       1,
                       complex_mode,
                       yobs,
                       fig,
                       screen,
                       main_task,
                       nothing,
                       nothing,
                       RingFrameBuffer(ComplexF32, frame_size, poolsize),
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
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                process_samples!(context, rd_buffer.buf, rd_buffer.store_size)
                rd_buffer.store_size = 0
                put!(context.ringbuffer.freeQ, rd_index)
            else
                yield()
            end
        end
    catch e
        if !(e isa InterruptException)
            println("WaveformView task error: ", e)
        end
    end
    return nothing
end

function update_task!(context::ViewContext{T}) where {T}
    try
        while context.running[]
            if isready(context.result_ringbuffer.fullQ)
                result_index = take!(context.result_ringbuffer.fullQ)
                result_buf = context.result_ringbuffer.bufs[result_index].buf
                frame_size = length(result_buf) ÷ context.channels
                @inbounds for ch in 1:context.channels
                    src = view(result_buf, (ch - 1) * frame_size + 1:ch * frame_size)
                    copyto!(context.yobs[ch][], src)
                    Base.notify(context.yobs[ch])
                end
                put!(context.result_ringbuffer.freeQ, result_index)
            else
                yield()
            end
        end
    catch e
        if !(e isa InterruptException)
            println("WaveformView update error: ", e)
        end
    end
    return nothing
end

function input!(context::ViewContext{T}, samples::AbstractVector{T}, samples_size::Int) where {T}
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

function stop!(context::ViewContext)
    println("WaveformView.stop()")
    context.running[] = false
    wait(context.worker)
    wait(context.update_task)
    return nothing
end

end
