module AlsaSink

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer

using GLMakie
using Libdl

const LIBASOUND = "libasound"
const SND_PCM_STREAM_PLAYBACK = 0
const SND_PCM_ACCESS_RW_INTERLEAVED = 3
const SND_PCM_FORMAT_S16_LE = 2

mutable struct AlsaSinkParameter
    device::String
    ch::UInt32
    sampleRate::UInt32
    bufferTime::UInt32
    periodTime::UInt32
end

mutable struct AlsaSinkContext <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    params::AlsaSinkParameter
    pcm_handle::Ptr{Cvoid}
    hw_params::Ptr{Cvoid}
    period_frames::UInt64
    buffer_frames::UInt64
    enable_clock_correction::Bool
    target_delay_frames::Int64
    max_ppm::Float64
    correction_alpha::Float64
    clock_ratio::Float64
    enable_stats_view::Bool
    stats_len::Int
    stats_pos::Int
    stats_in::Vector{Float64}
    stats_out::Vector{Float64}
    stats_delay::Vector{Float64}
    stats_in_ratio::Vector{Float64}
    stats_out_ratio::Vector{Float64}
    stats_delay_ratio::Vector{Float64}
    stats_in_obs::Union{Nothing,Observable{Vector{Float64}}}
    stats_out_obs::Union{Nothing,Observable{Vector{Float64}}}
    stats_delay_obs::Union{Nothing,Observable{Vector{Float64}}}
    stats_in_ratio_obs::Union{Nothing,Observable{Vector{Float64}}}
    stats_out_ratio_obs::Union{Nothing,Observable{Vector{Float64}}}
    stats_delay_ratio_obs::Union{Nothing,Observable{Vector{Float64}}}
    stats_fig::Union{Nothing,Figure}
    stats_screen::Union{Nothing,GLMakie.Screen}
    stats_update_task::Union{Nothing,Task}
    statsQ::Union{Nothing,Channel{NTuple{3,Float64}}}
    ringbuffer::RingFrameBuffer{Int16}
    holdbuf::Union{Nothing, Int}
    input_samples::Base.Threads.Atomic{Int64}
    output_samples::Base.Threads.Atomic{Int64}
    last_report_ns::Int64
    prev_input_samples::Int64
    prev_output_samples::Int64
    prev_stats_in::Float64
    prev_stats_out::Float64
    prev_stats_delay::Float64
    resample_buf::Vector{Int16}
    ratio_update_frames::Int64
    frames_since_ratio_update::Int64
    worker::Union{Nothing,Task}
end

function check_ret(code::Cint, msg::AbstractString)
    code < 0 && error("AlsaSink: $msg (err=$code)")
    return nothing
end

function set_hw_params!(ctx::AlsaSinkContext)
    ret = ccall((:snd_pcm_hw_params_any, LIBASOUND), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}), ctx.pcm_handle, ctx.hw_params)
    check_ret(ret, "snd_pcm_hw_params_any failed")

    ret = ccall((:snd_pcm_hw_params_set_access, LIBASOUND), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Cint), ctx.pcm_handle, ctx.hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)
    check_ret(ret, "snd_pcm_hw_params_set_access failed")

    ret = ccall((:snd_pcm_hw_params_set_format, LIBASOUND), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Cint), ctx.pcm_handle, ctx.hw_params, SND_PCM_FORMAT_S16_LE)
    check_ret(ret, "snd_pcm_hw_params_set_format failed")

    ret = ccall((:snd_pcm_hw_params_set_channels, LIBASOUND), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Cuint), ctx.pcm_handle, ctx.hw_params, ctx.params.ch)
    check_ret(ret, "snd_pcm_hw_params_set_channels failed")

    ret = ccall((:snd_pcm_hw_params_set_rate_resample, LIBASOUND), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Cuint), ctx.pcm_handle, ctx.hw_params, 0)
    check_ret(ret, "snd_pcm_hw_params_set_rate_resample failed")

    rrate = Ref{Cuint}(ctx.params.sampleRate)
    ret = ccall((:snd_pcm_hw_params_set_rate_near, LIBASOUND), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Ref{Cuint}, Ptr{Cint}), ctx.pcm_handle, ctx.hw_params, rrate, C_NULL)
    check_ret(ret, "snd_pcm_hw_params_set_rate_near failed")
    rrate[] != ctx.params.sampleRate && error("AlsaSink: rate mismatch requested=$(ctx.params.sampleRate) actual=$(rrate[])")

    rbuffer_time = Ref{Cuint}(ctx.params.bufferTime)
    dir = Ref{Cint}(0)
    ret = ccall((:snd_pcm_hw_params_set_buffer_time_near, LIBASOUND), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Ref{Cuint}, Ref{Cint}), ctx.pcm_handle, ctx.hw_params, rbuffer_time, dir)
    check_ret(ret, "snd_pcm_hw_params_set_buffer_time_near failed")

    rperiod_time = Ref{Cuint}(ctx.params.periodTime)
    ret = ccall((:snd_pcm_hw_params_set_period_time_near, LIBASOUND), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Ref{Cuint}, Ref{Cint}), ctx.pcm_handle, ctx.hw_params, rperiod_time, dir)
    check_ret(ret, "snd_pcm_hw_params_set_period_time_near failed")

    ret = ccall((:snd_pcm_hw_params, LIBASOUND), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}), ctx.pcm_handle, ctx.hw_params)
    check_ret(ret, "snd_pcm_hw_params failed")

    buffer_frames = Ref{UInt64}(0)
    ret = ccall((:snd_pcm_hw_params_get_buffer_size, LIBASOUND), Cint,
                (Ptr{Cvoid}, Ref{UInt64}), ctx.hw_params, buffer_frames)
    check_ret(ret, "snd_pcm_hw_params_get_buffer_size failed")

    period_frames = Ref{UInt64}(0)
    ret = ccall((:snd_pcm_hw_params_get_period_size, LIBASOUND), Cint,
                (Ptr{Cvoid}, Ref{UInt64}, Ref{Cint}), ctx.hw_params, period_frames, dir)
    check_ret(ret, "snd_pcm_hw_params_get_period_size failed")

    ctx.buffer_frames = buffer_frames[]
    ctx.period_frames = period_frames[]
    return nothing
end

function CreateAlsaSink(device::AbstractString = "default";
                        ch::Int = 2,
                        sampleRate::Int = 48_000,
                        bufferTime::Int = 500_000,
                        periodTime::Int = 100_000,
                        enable_clock_correction::Bool = true,
                        target_delay_frames::Int = 0,
                        max_ppm::Real = 200.0,
                        correction_alpha::Real = 0.02,
                        ratio_update_frames::Int = 1024,
                        enable_stats_view::Bool = false,
                        stats_len::Int = 300,
                        stats_window_size = (900, 480),
                        poolsize::Int = 8,
                        frame_size::Int = 4096)
    ch < 1 && error("AlsaSink: ch must be at least 1.")
    sampleRate < 1 && error("AlsaSink: sampleRate must be positive.")
    bufferTime < 1 && error("AlsaSink: bufferTime must be positive.")
    periodTime < 1 && error("AlsaSink: periodTime must be positive.")
    max_ppm < 0 && error("AlsaSink: max_ppm must be >= 0.")
    correction_alpha <= 0 && error("AlsaSink: correction_alpha must be > 0.")
    correction_alpha > 1 && error("AlsaSink: correction_alpha must be <= 1.")
    ratio_update_frames < 1 && error("AlsaSink: ratio_update_frames must be >= 1.")
    stats_len < 2 && error("AlsaSink: stats_len must be at least 2.")
    poolsize < 1 && error("AlsaSink: poolsize must be at least 1.")
    frame_size < 1 && error("AlsaSink: frame_size must be at least 1.")

    params = AlsaSinkParameter(String(device), UInt32(ch), UInt32(sampleRate),
                               UInt32(bufferTime), UInt32(periodTime))

    pcm_handle = Ref{Ptr{Cvoid}}()
    ret = ccall((:snd_pcm_open, LIBASOUND), Cint,
                (Ref{Ptr{Cvoid}}, Cstring, Cint, Cint),
                pcm_handle, params.device, SND_PCM_STREAM_PLAYBACK, 0)
    check_ret(ret, "snd_pcm_open failed")

    hw_params = Ref{Ptr{Cvoid}}()
    ret = ccall((:snd_pcm_hw_params_malloc, LIBASOUND), Cint, (Ref{Ptr{Cvoid}},), hw_params)
    check_ret(ret, "snd_pcm_hw_params_malloc failed")

    ctx = AlsaSinkContext(Base.Threads.Atomic{Bool}(true),
                          params,
                          pcm_handle[],
                          hw_params[],
                          0,
                          0,
                          enable_clock_correction,
                          Int64(target_delay_frames),
                          Float64(max_ppm),
                          Float64(correction_alpha),
                          1.0,
                          enable_stats_view,
                          stats_len,
                          0,
                          fill(0.0, stats_len),
                          fill(0.0, stats_len),
                          fill(0.0, stats_len),
                          fill(1.0, stats_len),
                          fill(1.0, stats_len),
                          fill(1.0, stats_len),
                          nothing,
                          nothing,
                          nothing,
                          nothing,
                          nothing,
                          nothing,
                          nothing,
                          nothing,
                          nothing,
                          nothing,
                          RingFrameBuffer(Int16, frame_size, poolsize),
                          nothing,
                          Base.Threads.Atomic{Int64}(0),
                          Base.Threads.Atomic{Int64}(0),
                          time_ns(),
                          0,
                          0,
                          1.0,
                          1.0,
                          1.0,
                          Vector{Int16}(undef, 0),
                          Int64(ratio_update_frames),
                          0,
                          nothing)
    set_hw_params!(ctx)
    if ctx.target_delay_frames <= 0 && ctx.buffer_frames > 0
        ctx.target_delay_frames = Int64(ctx.buffer_frames ÷ 2)
    end
    if ctx.enable_stats_view
        ctx.stats_in_obs = Observable(copy(ctx.stats_in))
        ctx.stats_out_obs = Observable(copy(ctx.stats_out))
        ctx.stats_delay_obs = Observable(copy(ctx.stats_delay))
        ctx.stats_in_ratio_obs = Observable(copy(ctx.stats_in_ratio))
        ctx.stats_out_ratio_obs = Observable(copy(ctx.stats_out_ratio))
        ctx.stats_delay_ratio_obs = Observable(copy(ctx.stats_delay_ratio))

        fig = Figure(size = stats_window_size)
        ax1 = Axis(fig[1, 1], xlabel = "Samples", ylabel = "Rate (S/sec)")
        ax2 = Axis(fig[2, 1], xlabel = "Samples", ylabel = "Delay (frames)")
        ax3 = Axis(fig[3, 1], xlabel = "Samples", ylabel = "Ratio")
        x = collect(1:ctx.stats_len)
        line_in = lines!(ax1, x, ctx.stats_in_obs; linewidth = 1)
        line_out = lines!(ax1, x, ctx.stats_out_obs; linewidth = 1)
        lines!(ax2, x, ctx.stats_delay_obs; linewidth = 1)
        axislegend(ax1, [line_in, line_out], ["Input rate", "Output rate"])
        line_in_r = lines!(ax3, x, ctx.stats_in_ratio_obs; linewidth = 1)
        line_out_r = lines!(ax3, x, ctx.stats_out_ratio_obs; linewidth = 1)
        line_delay_r = lines!(ax3, x, ctx.stats_delay_ratio_obs; linewidth = 1)
        axislegend(ax3, [line_in_r, line_out_r, line_delay_r], ["In ratio", "Out ratio", "Delay ratio"])
        y_max_rate = Float64(ctx.params.sampleRate * ctx.params.ch) * 1.5
        y_max_delay = 2.6e4
        ylims!(ax1, 0.0, y_max_rate)
        ylims!(ax2, 0.0, y_max_delay)
        ylims!(ax3, 0.5, 1.5)

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
            GLMakie.set_title!(ctx.stats_screen, "AlsaSink Stats")
        end
        ctx.stats_fig = fig
        ctx.statsQ = Channel{NTuple{3,Float64}}(1)
        ctx.stats_update_task = @async stats_update_task!(ctx)
    end
    ctx.worker = Threads.@spawn task!(ctx)
    return ctx
end

function write_frames!(ctx::AlsaSinkContext, buf::Vector{Int16}, frames::Int)
    frames_left = frames
    frame_offset = 0
    while frames_left > 0 && ctx.running[]
        ptr = pointer(buf, frame_offset * Int(ctx.params.ch) + 1)
        written = ccall((:snd_pcm_writei, LIBASOUND), Clong,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Culong),
                        ctx.pcm_handle, ptr, frames_left)
        if written == -Libc.EPIPE
            ccall((:snd_pcm_prepare, LIBASOUND), Cint, (Ptr{Cvoid},), ctx.pcm_handle)
            continue
        end
        if written < 0
            ccall((:snd_pcm_prepare, LIBASOUND), Cint, (Ptr{Cvoid},), ctx.pcm_handle)
            continue
        end
        frames_left -= Int(written)
        frame_offset += Int(written)
    end
    return nothing
end

function resample_interleaved!(dst::Vector{Int16}, src::Vector{Int16}, frames_in::Int, ch::Int, ratio::Float64)
    ratio <= 0 && return 0
    frames_in < 2 && return 0
    frames_out = Int(floor(frames_in * ratio))
    frames_out < 1 && return 0
    needed = frames_out * ch
    if length(dst) < needed
        resize!(dst, needed)
    end
    step = 1.0 / ratio
    @inbounds for out_i in 0:frames_out - 1
        pos = out_i * step
        i0 = Int(floor(pos)) + 1
        i1 = min(i0 + 1, frames_in)
        frac = pos - (i0 - 1)
        base0 = (i0 - 1) * ch
        base1 = (i1 - 1) * ch
        out_base = out_i * ch
        for c in 1:ch
            s0 = Float64(src[base0 + c])
            s1 = Float64(src[base1 + c])
            v = s0 + (s1 - s0) * frac
            if v > 32767.0
                v = 32767.0
            elseif v < -32768.0
                v = -32768.0
            end
            dst[out_base + c] = Int16(round(Int, v))
        end
    end
    return frames_out
end

function stats_update_task!(context::AlsaSinkContext)
    try
        while context.running[] && context.statsQ !== nothing
            if isready(context.statsQ)
                in_rate, out_rate, delay = take!(context.statsQ)
                if context.stats_len > 1
                    @inbounds begin
                        copyto!(context.stats_in, 1, context.stats_in, 2, context.stats_len - 1)
                        copyto!(context.stats_out, 1, context.stats_out, 2, context.stats_len - 1)
                        copyto!(context.stats_delay, 1, context.stats_delay, 2, context.stats_len - 1)
                        copyto!(context.stats_in_ratio, 1, context.stats_in_ratio, 2, context.stats_len - 1)
                        copyto!(context.stats_out_ratio, 1, context.stats_out_ratio, 2, context.stats_len - 1)
                        copyto!(context.stats_delay_ratio, 1, context.stats_delay_ratio, 2, context.stats_len - 1)
                        context.stats_in[end] = in_rate
                        context.stats_out[end] = out_rate
                        context.stats_delay[end] = delay
                        context.stats_in_ratio[end] = context.prev_stats_in > 0 ? in_rate / context.prev_stats_in : 1.0
                        context.stats_out_ratio[end] = context.prev_stats_out > 0 ? out_rate / context.prev_stats_out : 1.0
                        context.stats_delay_ratio[end] = context.prev_stats_delay > 0 ? delay / context.prev_stats_delay : 1.0
                    end
                    context.stats_in_obs[] = context.stats_in
                    context.stats_out_obs[] = context.stats_out
                    context.stats_delay_obs[] = context.stats_delay
                    context.stats_in_ratio_obs[] = context.stats_in_ratio
                    context.stats_out_ratio_obs[] = context.stats_out_ratio
                    context.stats_delay_ratio_obs[] = context.stats_delay_ratio
                    context.prev_stats_in = in_rate
                    context.prev_stats_out = out_rate
                    context.prev_stats_delay = delay
                end
            else
                yield()
            end
        end
    catch e
        if !(e isa InterruptException)
            println("AlsaSink stats error: ", e)
        end
    end
    return nothing
end

function task!(context::AlsaSinkContext)
    silence = Vector{Int16}(undef, Int(context.period_frames) * Int(context.params.ch))
    fill!(silence, 0)
    ret = ccall((:snd_pcm_prepare, LIBASOUND), Cint, (Ptr{Cvoid},), context.pcm_handle)
    ret < 0 && println("AlsaSink error: snd_pcm_prepare failed (err=$(ret))")

    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                samples = rd_buffer.store_size
                frames = samples ÷ Int(context.params.ch)
                if frames > 0
                    frames_out = frames
                    if context.enable_clock_correction && context.buffer_frames > 0
                        context.frames_since_ratio_update += frames
                        delay = Ref{Int64}(0)
                        ret = ccall((:snd_pcm_delay, LIBASOUND), Cint,
                                    (Ptr{Cvoid}, Ref{Int64}), context.pcm_handle, delay)
                        if ret >= 0
                            if context.frames_since_ratio_update >= context.ratio_update_frames
                                context.frames_since_ratio_update = 0
                                err = delay[] - context.target_delay_frames
                                max_ratio = context.max_ppm * 1e-6
                                adj = clamp(err / Float64(context.buffer_frames), -1.0, 1.0) * max_ratio
                                target_ratio = 1.0 + adj
                                context.clock_ratio = (1.0 - context.correction_alpha) * context.clock_ratio +
                                    context.correction_alpha * target_ratio
                            end
                            max_ratio = context.max_ppm * 1e-6
                            ratio = clamp(context.clock_ratio, 1.0 - max_ratio, 1.0 + max_ratio)
                            frames_out = resample_interleaved!(context.resample_buf, rd_buffer.buf,
                                                               frames, Int(context.params.ch), ratio)
                            if frames_out > 0
                                write_frames!(context, context.resample_buf, frames_out)
                            end
                        else
                            write_frames!(context, rd_buffer.buf, frames)
                        end
                    else
                        write_frames!(context, rd_buffer.buf, frames)
                    end
                    Base.Threads.atomic_add!(context.output_samples, Int64(frames_out) * Int64(context.params.ch))
                end
                rd_buffer.store_size = 0
                put!(context.ringbuffer.freeQ, rd_index)
            else
                if context.period_frames > 0
                    write_frames!(context, silence, Int(context.period_frames))
                    Base.Threads.atomic_add!(context.output_samples, Int64(context.period_frames) * Int64(context.params.ch))
                else
                    yield()
                end
            end

            now_ns = time_ns()
            last_ns = context.last_report_ns
            if now_ns - last_ns >= 1_000_000_000
                in_samples = Base.Threads.atomic_add!(context.input_samples, 0)
                out_samples = Base.Threads.atomic_add!(context.output_samples, 0)
                dt = (now_ns - last_ns) / 1_000_000_000
                in_rate = (in_samples - context.prev_input_samples) / dt
                out_rate = (out_samples - context.prev_output_samples) / dt
                context.prev_input_samples = in_samples
                context.prev_output_samples = out_samples
                context.last_report_ns = now_ns
                # delay = Ref{Int64}(0)
                # dret = ccall((:snd_pcm_delay, LIBASOUND), Cint,
                #              (Ptr{Cvoid}, Ref{Int64}), context.pcm_handle, delay)
                # delay_str = dret >= 0 ? string(delay[]) : "n/a"
                # println("AlsaSink rate: in=", round(in_rate), "S/sec out=", round(out_rate),
                #         "S/sec delay=", delay_str, " frames")
                if context.enable_stats_view && context.statsQ !== nothing
                    if isready(context.statsQ)
                        take!(context.statsQ)
                    end
                    delay = Ref{Int64}(0)
                    dret = ccall((:snd_pcm_delay, LIBASOUND), Cint,
                                 (Ptr{Cvoid}, Ref{Int64}), context.pcm_handle, delay)
                    delay_val = dret >= 0 ? Float64(delay[]) : NaN
                    put!(context.statsQ, (Float64(in_rate), Float64(out_rate), delay_val))
                end
            end
        end
    catch e
        if !(e isa InterruptException)
            println("AlsaSink error: ", e)
        end
    end
    return nothing
end

function input!(context::AlsaSinkContext, samples::AbstractVector{Int16}, samples_size::Int)
    if !context.running[] || samples_size <= 0
        return false
    end

    actual_size = min(samples_size, length(samples))
    if actual_size <= 0
        return 0
    end

    Base.Threads.atomic_add!(context.input_samples, Int64(actual_size))

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

function stop!(context::AlsaSinkContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    if context.hw_params != C_NULL
        ccall((:snd_pcm_hw_params_free, LIBASOUND), Cvoid, (Ptr{Cvoid},), context.hw_params)
    end
    if context.pcm_handle != C_NULL
        ccall((:snd_pcm_close, LIBASOUND), Cint, (Ptr{Cvoid},), context.pcm_handle)
    end
    return nothing
end

end
