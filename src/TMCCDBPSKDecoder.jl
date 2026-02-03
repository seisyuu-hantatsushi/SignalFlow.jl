module TMCCDBPSKDecoder

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer

mutable struct TMCCDBPSKDecoderContext <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    nfft::Int
    bins::Vector{Int}
    label::String
    log_interval::Float64
    hist_len::Int
    last_log_time::Float64
    prev_symbols::Vector{ComplexF32}
    has_prev::Vector{Bool}
    bit_history::Vector{Int}
    outbuf::Vector{ComplexF32}
    ringbuffer::RingFrameBuffer{ComplexF32}
    holdbuf::Union{Nothing, Int}
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
end

const LOG_LOCK = ReentrantLock()

function CreateTMCCDBPSKDecoder(; nfft::Int = 8192,
                                bins::Vector{Int},
                                label::AbstractString = "TMCC",
                                log_interval::Real = 0.5,
                                hist_len::Int = 64,
                                poolsize::Int = 8)
    nfft < 32 && error("TMCCDBPSKDecoder: nfft must be >= 32.")
    isempty(bins) && error("TMCCDBPSKDecoder: bins must not be empty.")
    log_interval <= 0 && error("TMCCDBPSKDecoder: log_interval must be > 0.")
    hist_len < 1 && error("TMCCDBPSKDecoder: hist_len must be >= 1.")
    poolsize < 1 && error("TMCCDBPSKDecoder: poolsize must be at least 1.")
    @inbounds for b in bins
        (b < 1 || b > nfft) && error("TMCCDBPSKDecoder: bin out of range: $b")
    end

    new_sinks = Channel{SignalFlowBlock}(4)
    sinks = Vector{SignalFlowBlock}()
    ctx = TMCCDBPSKDecoderContext(Base.Threads.Atomic{Bool}(true),
                                  nfft,
                                  copy(bins),
                                  String(label),
                                  Float64(log_interval),
                                  Int(hist_len),
                                  0.0,
                                  fill(ComplexF32(0, 0), length(bins)),
                                  fill(false, length(bins)),
                                  Int[],
                                  Vector{ComplexF32}(undef, nfft),
                                  RingFrameBuffer(ComplexF32, nfft, poolsize),
                                  nothing,
                                  nothing,
                                  new_sinks,
                                  sinks)
    ctx.worker = Threads.@spawn task!(ctx)
    return ctx
end

function task!(context::TMCCDBPSKDecoderContext)
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                if rd_buffer.store_size == context.nfft
                    copyto!(context.outbuf, 1, rd_buffer.buf, 1, context.nfft)

                    bit_acc = 0.0
                    valid = 0
                    bits = Vector{Int}(undef, length(context.bins))
                    conf = 0.0
                    @inbounds for i in 1:length(context.bins)
                        idx = context.bins[i]
                        v = rd_buffer.buf[idx]
                        bits[i] = -1
                        if context.has_prev[i]
                            d = v * conj(context.prev_symbols[i])
                            mag = abs(d)
                            if mag > 0
                                m = real(d) / mag
                                bits[i] = m < 0 ? 1 : 0
                                bit_acc += m
                                conf += abs(m)
                                valid += 1
                            end
                        end
                        context.prev_symbols[i] = v
                        context.has_prev[i] = true
                    end

                    if valid > 0
                        bit = bit_acc < 0 ? 1 : 0
                        push!(context.bit_history, bit)
                        if length(context.bit_history) > context.hist_len
                            deleteat!(context.bit_history, 1)
                        end
                        now = time()
                        if now - context.last_log_time >= context.log_interval
                            bitstr = join((b < 0 ? "x" : string(b) for b in bits), "")
                            histstr = join(string.(context.bit_history), "")
                            lock(LOG_LOCK) do
                                println("TMCCDBPSK[", context.label, "]: bit=",
                                        bit,
                                        " metric=",
                                        round(bit_acc / valid, digits = 3),
                                        " conf=",
                                        round(conf / valid, digits = 3),
                                        " bins=",
                                        bitstr,
                                        " hist=",
                                        histstr)
                            end
                            context.last_log_time = now
                        end
                    end

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
            println("TMCCDBPSKDecoder error: ", e)
        end
    end
    return nothing
end

function input!(context::TMCCDBPSKDecoderContext, samples::AbstractVector{ComplexF32}, samples_size::Int)
    if !context.running[] || samples_size <= 0
        return false
    end

    actual_size = min(samples_size, length(samples))
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

function stop!(context::TMCCDBPSKDecoderContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

end
