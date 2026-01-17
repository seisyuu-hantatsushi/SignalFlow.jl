module ISDBTPilotEqualizer

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer

mutable struct ISDBTPilotEqualizerContext <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    nfft::Int
    pilot_spacing::Int
    pilot_offset0::Int
    pilot_offset_step::Int
    output_mode::Int
    symbol_index::Int
    pilot_values::Vector{ComplexF32}
    h_est::Vector{ComplexF32}
    outbuf::Vector{ComplexF32}
    ringbuffer::RingFrameBuffer{ComplexF32}
    holdbuf::Union{Nothing, Int}
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
end

function pilot_positions(nfft::Int, spacing::Int, offset::Int)
    start = 1 + mod(offset, spacing)
    return collect(start:spacing:nfft)
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

function CreateISDBTPilotEqualizer(; nfft::Int = 8192,
                                   pilot_spacing::Int = 12,
                                   pilot_offset0::Int = 3,
                                   pilot_offset_step::Int = 3,
                                   output_mode::Int = 2,
                                   pilot_values::Union{Nothing,Vector{ComplexF32}} = nothing,
                                   poolsize::Int = 8)
    nfft < 32 && error("ISDBTPilotEqualizer: nfft must be >= 32.")
    pilot_spacing < 1 && error("ISDBTPilotEqualizer: pilot_spacing must be >= 1.")
    poolsize < 1 && error("ISDBTPilotEqualizer: poolsize must be at least 1.")
    (output_mode == 1 || output_mode == 2) || error("ISDBTPilotEqualizer: output_mode must be 1 or 2.")

    pos = pilot_positions(nfft, pilot_spacing, pilot_offset0)
    pilot_values === nothing && (pilot_values = fill(ComplexF32(1, 0), length(pos)))
    length(pilot_values) != length(pos) && error("ISDBTPilotEqualizer: pilot_values length mismatch.")

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
                                     Vector{ComplexF32}(undef, nfft),
                                     Vector{ComplexF32}(undef, nfft),
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
                    offset = context.pilot_offset0 + context.pilot_offset_step * (context.symbol_index % 4)
                    pos = pilot_positions(context.nfft, context.pilot_spacing, offset)
                    @inbounds for i in 1:length(pos)
                        idx = pos[i]
                        context.h_est[idx] = rd_buffer.buf[idx] / context.pilot_values[i]
                    end
                    linear_interp!(context.h_est, pos, view(context.h_est, pos))
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
