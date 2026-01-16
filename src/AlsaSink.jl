module AlsaSink

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer

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
    ringbuffer::RingFrameBuffer{Int16}
    holdbuf::Union{Nothing, Int}
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
                        poolsize::Int = 8,
                        frame_size::Int = 4096)
    ch < 1 && error("AlsaSink: ch must be at least 1.")
    sampleRate < 1 && error("AlsaSink: sampleRate must be positive.")
    bufferTime < 1 && error("AlsaSink: bufferTime must be positive.")
    periodTime < 1 && error("AlsaSink: periodTime must be positive.")
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
                          RingFrameBuffer(Int16, frame_size, poolsize),
                          nothing,
                          nothing)
    set_hw_params!(ctx)
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
                    write_frames!(context, rd_buffer.buf, frames)
                end
                rd_buffer.store_size = 0
                put!(context.ringbuffer.freeQ, rd_index)
            else
                if context.period_frames > 0
                    write_frames!(context, silence, Int(context.period_frames))
                else
                    yield()
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
