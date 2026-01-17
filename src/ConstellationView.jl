module ConstellationView

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer

using GLMakie

mutable struct ViewContext <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    frame_size::Int
    yobs::Observable{Vector{Point2f}}
    fig::Figure
    screen::Union{Nothing, GLMakie.Screen}
    main_task::Union{Nothing,Task}
    worker::Union{Nothing,Task}
    update_task::Union{Nothing,Task}
    ringbuffer::RingFrameBuffer{ComplexF32}
    holdbuf::Union{Nothing, Int}
    result_ringbuffer::RingFrameBuffer{Point2f}
end

function process_samples!(context::ViewContext, samples::AbstractVector{ComplexF32}, n::Integer)
    if !isready(context.result_ringbuffer.freeQ)
        return nothing
    end
    result_index = take!(context.result_ringbuffer.freeQ)
    result_buf = context.result_ringbuffer.bufs[result_index].buf
    len = min(n, context.frame_size)
    @inbounds for i in 1:len
        s = samples[i]
        result_buf[i] = Point2f(Float32(real(s)), Float32(imag(s)))
    end
    @inbounds for i in len + 1:context.frame_size
        result_buf[i] = Point2f(0f0, 0f0)
    end
    put!(context.result_ringbuffer.fullQ, result_index)
    return nothing
end

function CreateView(inputSamplingRate::UInt64;
                    frame_size::Int = 4096,
                    poolsize::Int = 16,
                    axis_limit::Real = 1.2,
                    title::AbstractString = "Constellation View",
                    window_size = (700, 700))

    frame_size < 1 && error("ConstellationView: frame_size must be at least 1.")
    poolsize < 1 && error("ConstellationView: poolsize must be at least 1.")
    axis_limit <= 0 && error("ConstellationView: axis_limit must be positive.")

    _ = inputSamplingRate

    yobs = Observable([Point2f(0f0, 0f0) for _ in 1:frame_size])

    fig = Figure(size = window_size)
    ax = Axis(fig[1, 1], xlabel = "I", ylabel = "Q")
    scatter!(ax, yobs; markersize = 4)
    xlims!(ax, -axis_limit, axis_limit)
    ylims!(ax, -axis_limit, axis_limit)

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

    result_ringbuffer = RingFrameBuffer(Point2f, frame_size, poolsize)

    view = ViewContext(Base.Threads.Atomic{Bool}(true),
                       frame_size,
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

function task!(context::ViewContext)
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
            println("ConstellationView task error: ", e)
        end
    end
    return nothing
end

function update_task!(context::ViewContext)
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
            println("ConstellationView update error: ", e)
        end
    end
    return nothing
end

function input!(context::ViewContext, samples::AbstractVector{ComplexF32}, samples_size::Int)
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
    println("ConstellationView.stop()")
    context.running[] = false
    wait(context.worker)
    wait(context.update_task)
    return nothing
end

end
