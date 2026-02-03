module BinPowerMonitor

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer

mutable struct BinPowerMonitorContext <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    nfft::Int
    bins::Vector{Int}
    band_bins::Vector{Int}
    label::String
    log_interval::Float64
    last_log_time::Float64
    outbuf::Vector{ComplexF32}
    ringbuffer::RingFrameBuffer{ComplexF32}
    holdbuf::Union{Nothing, Int}
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
end

const LOG_LOCK = ReentrantLock()

function band_bins(nfft::Int, samplerate::Float64, band_limit_hz::Float64)
    band_limit_hz <= 0 && return collect(1:nfft)
    bins = Int[]
    hz_per_bin = samplerate / nfft
    @inbounds for i in 1:nfft
        f = (i - 1) * hz_per_bin
        if f > samplerate / 2
            f -= samplerate
        end
        if abs(f) <= band_limit_hz
            push!(bins, i)
        end
    end
    return bins
end

function CreateBinPowerMonitor(; nfft::Int = 8192,
                               bins::Vector{Int},
                               label::AbstractString = "Bins",
                               log_interval::Real = 1.0,
                               samplerate::Real = 8_000_000,
                               band_limit_hz::Real = 0.0,
                               poolsize::Int = 8)
    nfft < 32 && error("BinPowerMonitor: nfft must be >= 32.")
    poolsize < 1 && error("BinPowerMonitor: poolsize must be at least 1.")
    log_interval <= 0 && error("BinPowerMonitor: log_interval must be > 0.")
    samplerate <= 0 && error("BinPowerMonitor: samplerate must be > 0.")
    band_limit_hz < 0 && error("BinPowerMonitor: band_limit_hz must be >= 0.")

    bins_f = [b for b in bins if 1 <= b <= nfft]
    isempty(bins_f) && error("BinPowerMonitor: bins must include at least one valid bin.")
    bband = band_bins(nfft, Float64(samplerate), Float64(band_limit_hz))

    new_sinks = Channel{SignalFlowBlock}(4)
    sinks = Vector{SignalFlowBlock}()
    ctx = BinPowerMonitorContext(Base.Threads.Atomic{Bool}(true),
                                 nfft,
                                 bins_f,
                                 bband,
                                 String(label),
                                 Float64(log_interval),
                                 0.0,
                                 Vector{ComplexF32}(undef, nfft),
                                 RingFrameBuffer(ComplexF32, nfft, poolsize),
                                 nothing,
                                 nothing,
                                 new_sinks,
                                 sinks)
    ctx.worker = Threads.@spawn task!(ctx)
    return ctx
end

function task!(context::BinPowerMonitorContext)
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                if rd_buffer.store_size == context.nfft
                    now = time()
                    if now - context.last_log_time >= context.log_interval
                        p_bins = 0.0
                        @inbounds for i in 1:length(context.bins)
                            v = rd_buffer.buf[context.bins[i]]
                            p_bins += real(v) * real(v) + imag(v) * imag(v)
                        end
                        p_bins /= length(context.bins)
                        p_band = 0.0
                        @inbounds for i in 1:length(context.band_bins)
                            v = rd_buffer.buf[context.band_bins[i]]
                            p_band += real(v) * real(v) + imag(v) * imag(v)
                        end
                        p_band /= length(context.band_bins)
                        db_bins = 10 * log10(max(p_bins, 1e-12))
                        db_band = 10 * log10(max(p_band, 1e-12))
                        lock(LOG_LOCK) do
                            println("BinPowerMonitor[", context.label, "]: bins=",
                                    round(db_bins, digits = 2), " dB, band=",
                                    round(db_band, digits = 2), " dB")
                        end
                        context.last_log_time = now
                    end

                    while isready(context.new_sinks)
                        push!(context.sinks, take!(context.new_sinks))
                    end
                    for sink in context.sinks
                        input!(sink, rd_buffer.buf, context.nfft)
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
            println("BinPowerMonitor error: ", e)
        end
    end
    return nothing
end

function input!(context::BinPowerMonitorContext, samples::AbstractVector{ComplexF32}, samples_size::Int)
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

function stop!(context::BinPowerMonitorContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

end
